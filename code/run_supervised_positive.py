"""
Leak-free test of explicit supervised-contrastive class-pulling (SupCon-style
"positive" mode) vs the earlier "exclude" (debiased) mode and the raw-RF
baseline.

Leakage discipline: unlike simple exclusion (which only withholds an error
signal and cannot inject discriminative information about held-out drugs),
explicitly pulling same-class drugs together DOES inject information -- so a
held-out test drug must NEVER be given class_label>=0 during pretraining.
We reuse the exact seed=42 drug-disjoint split already used as the first of
the paper's 10 repeated-subsampling evaluation seeds: only that split's
SEEN-pool positives are ever labeled during pretraining; the TEST-pool
positives for that same split are the only drugs evaluated, and are always
labeled -1 (unlabeled) during pretraining. This is evaluated on a single
fixed split (not averaged over 10 subsampling seeds like the rest of the
paper) -- a smaller-scale, but still leak-free, follow-up experiment.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from models import CLIPStyleAligner
from splits import repeated_subsample_splits
from run_experiments import SigDataset, embed_drugs, BATCH_SIZE, LR, PRETRAIN_EPOCHS, DEVICE

ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose"
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = f"{ROOT}/ICLR2027_repurpose/results"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
N_PRETRAIN_SEEDS = 3


def pretrain_with_mode(sig_indices, drug_uid_of_sig, ecfp, expr, class_label_of_drug, class_mode,
                        epochs=PRETRAIN_EPOCHS, seed=0, log_tag=""):
    torch.manual_seed(seed)
    ds = SigDataset(sig_indices, drug_uid_of_sig, ecfp, expr, class_label_of_drug=class_label_of_drug)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    model = CLIPStyleAligner(structure_dim=ecfp.shape[1], expr_dim=expr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()
    t0 = time.time()
    losses = []
    for ep in range(epochs):
        ep_losses = []
        for batch in dl:
            structure = batch["structure"].to(DEVICE)
            expression = batch["expression"].to(DEVICE)
            drug_uid = batch["drug_uid"].to(DEVICE)
            class_label = batch["class_label"].to(DEVICE)
            loss, _, _ = model(structure, expression, drug_uid, class_label, class_mode=class_mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        losses.append(float(np.mean(ep_losses)))
    dt = time.time() - t0
    print(f"  [pretrain:{log_tag}] {len(ds)} sigs, {epochs} epochs in {dt:.1f}s, "
          f"final loss={losses[-1]:.4f} (init={losses[0]:.4f})", flush=True)
    return model


def eval_probe(train_X, train_y, test_X, test_y):
    if len(np.unique(train_y)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(train_X, train_y)
    score = clf.predict_proba(test_X)[:, 1]
    try:
        return {"auroc": roc_auc_score(test_y, score), "auprc": average_precision_score(test_y, score)}
    except ValueError:
        return None


def main():
    print("Loading dataset...", flush=True)
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = ecfp.shape[0]
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    results = []
    for cls in CLASS_COLS:
        pos_series = drug_level[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        n_pos = len(pos_drugs)
        if n_pos < 3:
            print(f"SKIP {cls}: too few positives ({n_pos})", flush=True)
            continue

        # Reuse the exact seed=42 split (first of the paper's 10 repeated-
        # subsampling seeds) so results are directly comparable to the main
        # results table at that seed.
        split = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=1, seed0=42)[0]
        seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
        seen_y = pos_series.reindex(seen).fillna(0).values
        seen_pos = seen[seen_y == 1]
        print(f"\n--- {cls}: n_pos={n_pos}, seen_pos(pretrain-visible)={len(seen_pos)}, "
              f"test_pos(held-out, never labeled during pretraining)={int(test_y.sum())} ---", flush=True)

        # class_label: 1 ONLY for seen-pool positives (safe to expose to
        # pretraining); -1 (unlabeled) for every held-out test drug and every
        # non-class drug. This is the leak-free guarantee.
        class_label_of_drug = np.full(n_drugs, -1, dtype=np.int64)
        for d in seen_pos:
            class_label_of_drug[d] = 1

        for mode in ["positive", "exclude"]:
            for pseed in range(N_PRETRAIN_SEEDS):
                m = pretrain_with_mode(sig_indices_all, drug_uid_of_sig, ecfp, expr, class_label_of_drug,
                                        class_mode=mode, seed=4000 + pseed, log_tag=f"{mode}:{cls}:{pseed}")
                z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
                assert has_e.all(), "BUG: zero-vector embedding artifact"
                for view_name, Z in [("structure", z_s), ("expression", z_e),
                                      ("concat", np.concatenate([z_s, z_e], axis=1))]:
                    r = eval_probe(Z[seen], seen_y, Z[test], test_y)
                    if r:
                        results.append({"class": cls, "mode": mode, "pretrain_seed": pseed,
                                         "view": view_name, **r})

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS}/supervised_positive_results.csv", index=False)
    print(f"\nSaved {len(df)} rows to {RESULTS}/supervised_positive_results.csv", flush=True)

    summary = df.groupby(["class", "mode", "view"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    print("\n=== SUMMARY (mean AUROC, single fixed seed=42 split, 3 pretrain seeds) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
