"""
Full experiment runner: contrastive pretraining (full / per-class-holdout /
random-holdout-control) + downstream probe evaluation (repeated drug-disjoint
subsampling) + non-contrastive baselines + alignment/uniformity analysis.

All results are real (no placeholders) -- every number here comes from an
actual run on real LINCS L1000 Phase 2 data with real PubChem-resolved
molecular structures.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from models import CLIPStyleAligner, alignment_loss, uniformity_loss
from splits import repeated_subsample_splits

ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose"
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = f"{ROOT}/ICLR2027_repurpose/results"
os.makedirs(RESULTS, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]

N_REPEATS = 10
PRETRAIN_EPOCHS = 25
BATCH_SIZE = 256
LR = 1e-3


class SigDataset(Dataset):
    def __init__(self, sig_indices, drug_uid_of_sig, ecfp_of_drug, expr, class_label_of_drug=None):
        self.sig_indices = sig_indices
        self.drug_uid_of_sig = drug_uid_of_sig
        self.ecfp_of_drug = ecfp_of_drug
        self.expr = expr
        self.class_label_of_drug = class_label_of_drug  # optional, int array indexed by drug_uid, -1 = unlabeled

    def __len__(self):
        return len(self.sig_indices)

    def __getitem__(self, i):
        s = self.sig_indices[i]
        d = self.drug_uid_of_sig[s]
        item = {
            "structure": torch.from_numpy(self.ecfp_of_drug[d]).float(),
            "expression": torch.from_numpy(self.expr[s]).float(),
            "drug_uid": torch.tensor(d, dtype=torch.long),
        }
        if self.class_label_of_drug is not None:
            item["class_label"] = torch.tensor(int(self.class_label_of_drug[d]), dtype=torch.long)
        return item


def pretrain(sig_indices, drug_uid_of_sig, ecfp_of_drug, expr, epochs=PRETRAIN_EPOCHS, seed=0, log_tag="",
             class_label_of_drug=None):
    torch.manual_seed(seed)
    ds = SigDataset(sig_indices, drug_uid_of_sig, ecfp_of_drug, expr, class_label_of_drug=class_label_of_drug)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    model = CLIPStyleAligner(structure_dim=ecfp_of_drug.shape[1], expr_dim=expr.shape[1]).to(DEVICE)
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
            class_label = batch["class_label"].to(DEVICE) if "class_label" in batch else None
            loss, _, _ = model(structure, expression, drug_uid, class_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        losses.append(float(np.mean(ep_losses)))
    dt = time.time() - t0
    print(f"  [pretrain:{log_tag}] {len(ds)} sigs, {epochs} epochs in {dt:.1f}s, "
          f"final loss={losses[-1]:.4f} (init={losses[0]:.4f})", flush=True)
    return model, losses


@torch.no_grad()
def embed_drugs(model, drug_uids, ecfp_of_drug, expr, drug_uid_of_sig, sig_indices):
    """Average per-drug structure embedding (deterministic, one per drug) and
    average per-drug expression embedding (mean over that drug's available
    signatures)."""
    model.eval()
    z_s_all = model.structure_encoder(torch.from_numpy(ecfp_of_drug).float().to(DEVICE)).cpu().numpy()

    sig_by_drug = {}
    for s in sig_indices:
        d = drug_uid_of_sig[s]
        sig_by_drug.setdefault(d, []).append(s)

    embed_dim = z_s_all.shape[1]
    z_e_all = np.zeros((ecfp_of_drug.shape[0], embed_dim), dtype=np.float32)
    counted = np.zeros(ecfp_of_drug.shape[0], dtype=bool)
    for d, sigs in sig_by_drug.items():
        x = torch.from_numpy(expr[sigs]).float().to(DEVICE)
        z = model.expression_encoder(x).cpu().numpy()
        z_e_all[d] = z.mean(axis=0)
        counted[d] = True

    return z_s_all, z_e_all, counted


def eval_probe(train_X, train_y, test_X, test_y, probe="logreg"):
    if len(np.unique(train_y)) < 2:
        return None
    if probe == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    elif probe == "knn":
        k = min(5, (train_y == 1).sum(), (train_y == 0).sum())
        k = max(1, k)
        clf = KNeighborsClassifier(n_neighbors=k)
    else:
        raise ValueError(probe)
    clf.fit(train_X, train_y)
    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(test_X)[:, 1]
    else:
        score = clf.decision_function(test_X)
    try:
        auroc = roc_auc_score(test_y, score)
        auprc = average_precision_score(test_y, score)
    except ValueError:
        return None
    return {"auroc": auroc, "auprc": auprc}


def raw_feature_baseline(train_X, train_y, test_X, test_y, model_type="rf"):
    if len(np.unique(train_y)) < 2:
        return None
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=0)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(train_X, train_y)
    score = clf.predict_proba(test_X)[:, 1]
    try:
        return {"auroc": roc_auc_score(test_y, score), "auprc": average_precision_score(test_y, score)}
    except ValueError:
        return None


def neighbor_connectivity_baseline(train_pos_X, test_X, test_y, k=5):
    """Parameter-free Connectivity-Map-style baseline: score = mean cosine
    similarity to the k nearest SEEN positive drugs in raw expression space.
    No learned parameters (mirrors the SE-TTA neighbor baseline)."""
    if len(train_pos_X) == 0:
        return None
    a = train_pos_X / (np.linalg.norm(train_pos_X, axis=1, keepdims=True) + 1e-8)
    b = test_X / (np.linalg.norm(test_X, axis=1, keepdims=True) + 1e-8)
    sims = b @ a.T  # [n_test, n_pos]
    k_eff = min(k, sims.shape[1])
    topk = np.sort(sims, axis=1)[:, -k_eff:]
    score = topk.mean(axis=1)
    try:
        return {"auroc": roc_auc_score(test_y, score), "auprc": average_precision_score(test_y, score)}
    except ValueError:
        return None


def main():
    print("Loading dataset...", flush=True)
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    drug_feat = pd.read_parquet(f"{DATA}/drug_features.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = ecfp.shape[0]
    print(f"  {len(meta)} signatures, {n_drugs} drugs, expr shape {expr.shape}, ecfp shape {ecfp.shape}", flush=True)

    drug_uid_of_sig = meta["drug_uid"].values.astype(int)  # index i -> drug_uid for signature row i
    sig_indices_all = np.arange(len(meta))

    class_labels_by_drug = {}
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    for cls in CLASS_COLS:
        class_labels_by_drug[cls] = drug_level[cls]

    all_drug_uids = np.array(sorted(drug_level.index.tolist()))
    print(f"  unique drugs w/ any signature: {len(all_drug_uids)}", flush=True)
    for cls in CLASS_COLS:
        n_pos = int(class_labels_by_drug[cls].reindex(all_drug_uids).fillna(0).sum())
        print(f"    {cls}: {n_pos} positive drugs", flush=True)

    results = []
    N_PRETRAIN_SEEDS = 3  # multiple pretraining seeds per regime so the
    # full-vs-cold-start comparison isn't confounded with single-seed
    # pretraining noise (the exact single-seed pitfall this paper flags
    # in prior work).

    # ---- 1. FULL pretraining runs (self-supervised, all drugs, 3 seeds) ----
    print(f"\n=== Pretraining: FULL (all drugs, {N_PRETRAIN_SEEDS} seeds) ===", flush=True)
    full_embeds = []
    align_uniform_log = []
    for pseed in range(N_PRETRAIN_SEEDS):
        model_full, losses_full = pretrain(sig_indices_all, drug_uid_of_sig, ecfp, expr, seed=pseed, log_tag=f"full{pseed}")
        z_s_full, z_e_full, has_expr_full = embed_drugs(model_full, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
        assert has_expr_full.all(), "BUG: some drugs never got an expression embedding (zero-vector artifact)"
        full_embeds.append({"z_s": z_s_full, "z_e": z_e_full})

        with torch.no_grad():
            idx_sample = np.random.RandomState(pseed).choice(sig_indices_all, size=min(4000, len(sig_indices_all)), replace=False)
            structure_b = torch.from_numpy(ecfp[drug_uid_of_sig[idx_sample]]).float().to(DEVICE)
            expr_b = torch.from_numpy(expr[idx_sample]).float().to(DEVICE)
            duid_b = torch.from_numpy(drug_uid_of_sig[idx_sample]).long().to(DEVICE)
            model_full.eval()
            zs, ze = model_full.encode(structure_b, expr_b)
            a = alignment_loss(zs, ze, duid_b)
            if a is None:
                a = float("nan")
            us = uniformity_loss(zs)
            ue = uniformity_loss(ze)
        print(f"  [full seed={pseed}] alignment={a:.4f}  uniformity(structure)={us:.4f}  uniformity(expr)={ue:.4f}", flush=True)
        align_uniform_log.append({"seed": pseed, "alignment": a, "uniformity_structure": us, "uniformity_expr": ue,
                                   "final_loss": losses_full[-1], "init_loss": losses_full[0]})

        if pseed == 0:
            np.save(f"{RESULTS}/z_s_full.npy", z_s_full)
            np.save(f"{RESULTS}/z_e_full.npy", z_e_full)

    with open(f"{RESULTS}/pretrain_full_curve.json", "w") as f:
        json.dump(align_uniform_log, f, indent=2)

    # ---- 2. Random-holdout CONTROL pretraining runs (3 seeds, matched size to avg class) ----
    avg_class_size = int(np.mean([int(class_labels_by_drug[c].reindex(all_drug_uids).fillna(0).sum()) for c in CLASS_COLS]))
    print(f"\n=== Pretraining: RANDOM-HOLDOUT control (holdout size={avg_class_size}) ===", flush=True)
    random_holdout_embeds = []
    for rseed in range(3):
        rng = np.random.RandomState(1000 + rseed)
        holdout_drugs = set(rng.choice(all_drug_uids, size=avg_class_size, replace=False).tolist())
        keep_sig_mask = ~np.isin(drug_uid_of_sig, list(holdout_drugs))
        sig_idx_keep = sig_indices_all[keep_sig_mask]
        m, _ = pretrain(sig_idx_keep, drug_uid_of_sig, ecfp, expr, seed=1000 + rseed, log_tag=f"randholdout{rseed}")
        # Embed with ALL signatures (plain forward pass, no gradient) so
        # held-out drugs get real embeddings from their real expression --
        # only the encoder WEIGHTS were trained without them.
        z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
        assert has_e.all(), "BUG: some drugs never got an expression embedding (zero-vector artifact)"
        random_holdout_embeds.append({"z_s": z_s, "z_e": z_e, "holdout": holdout_drugs})

    # ---- 3. Per-class CLASS-HOLDOUT (cold-start) pretraining runs, 3 seeds each ----
    print(f"\n=== Pretraining: CLASS-HOLDOUT (cold-start) per class, {N_PRETRAIN_SEEDS} seeds ===", flush=True)
    class_holdout_embeds = {cls: [] for cls in CLASS_COLS}
    for cls in CLASS_COLS:
        pos_drugs = class_labels_by_drug[cls]
        pos_drugs = set(pos_drugs[pos_drugs == 1].index.tolist())
        keep_sig_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs))
        sig_idx_keep = sig_indices_all[keep_sig_mask]
        for pseed in range(N_PRETRAIN_SEEDS):
            m, _ = pretrain(sig_idx_keep, drug_uid_of_sig, ecfp, expr, seed=2000 + pseed, log_tag=f"holdout:{cls}:{pseed}")
            # Same fix as above: embed on ALL signatures so held-out-class drugs
            # get real (non-zero) embeddings via a plain forward pass.
            z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
            assert has_e.all(), "BUG: some drugs never got an expression embedding (zero-vector artifact)"
            class_holdout_embeds[cls].append({"z_s": z_s, "z_e": z_e})

    # ---- 3b. Per-class DEBIASED-FULL pretraining runs, 3 seeds each ----
    # Same data as the FULL regime (all drugs, nothing removed), but the
    # contrastive loss excludes same-class-different-drug pairs from the
    # negative pool for the target class -- a causal test of the
    # false-negative-collision hypothesis: if that mechanism explains the
    # class-holdout structure-view advantage, removing the collision without
    # removing the data should recover (or beat) that advantage while
    # keeping the expression-view benefit of seeing the class's real signal.
    print(f"\n=== Pretraining: DEBIASED-FULL (same data, same-class pairs excluded from negatives), {N_PRETRAIN_SEEDS} seeds ===", flush=True)
    debiased_embeds = {cls: [] for cls in CLASS_COLS}
    for cls in CLASS_COLS:
        pos_drugs_set = class_labels_by_drug[cls]
        pos_drugs_set = set(pos_drugs_set[pos_drugs_set == 1].index.tolist())
        class_label_of_drug = np.full(n_drugs, -1, dtype=np.int64)
        for d in pos_drugs_set:
            class_label_of_drug[d] = 1
        for pseed in range(N_PRETRAIN_SEEDS):
            m, _ = pretrain(sig_indices_all, drug_uid_of_sig, ecfp, expr, seed=3000 + pseed,
                             log_tag=f"debiased:{cls}:{pseed}", class_label_of_drug=class_label_of_drug)
            z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
            assert has_e.all(), "BUG: some drugs never got an expression embedding (zero-vector artifact)"
            debiased_embeds[cls].append({"z_s": z_s, "z_e": z_e})

    # ---- 4. Raw feature matrices for non-contrastive baselines ----
    raw_expr_by_drug = np.zeros((n_drugs, expr.shape[1]), dtype=np.float32)
    sig_by_drug = {}
    for i, d in enumerate(drug_uid_of_sig):
        sig_by_drug.setdefault(d, []).append(i)
    for d, sigs in sig_by_drug.items():
        raw_expr_by_drug[d] = expr[sigs].mean(axis=0)

    # ---- 5. Downstream evaluation: repeated subsampling per class ----
    print("\n=== Downstream evaluation (repeated drug-disjoint subsampling) ===", flush=True)
    for cls in CLASS_COLS:
        pos_series = class_labels_by_drug[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        n_pos = len(pos_drugs)
        print(f"\n--- {cls} (n_pos={n_pos}) ---", flush=True)
        if n_pos < 3:
            print(f"  SKIP: too few positives ({n_pos})", flush=True)
            continue

        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=N_REPEATS, seed0=42)

        for split in splits:
            seed = split["seed"]
            seen = split["seen_drugs"]
            test = split["test_drugs"]
            test_y = split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values

            # contrastive FULL: structure-only, expression-only, concat -- logreg probe,
            # averaged over all N_PRETRAIN_SEEDS independent pretraining runs so the
            # full-vs-cold-start comparison isn't confounded with pretraining noise.
            for pseed, fe in enumerate(full_embeds):
                for view_name, Z in [("structure", fe["z_s"]), ("expression", fe["z_e"]),
                                      ("concat", np.concatenate([fe["z_s"], fe["z_e"]], axis=1))]:
                    r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                    if r:
                        results.append({"class": cls, "seed": seed, "pretrain_seed": pseed,
                                         "method": f"contrastive_full_{view_name}", **r})

            # contrastive CLASS-HOLDOUT (cold-start) for this class, same N_PRETRAIN_SEEDS averaging
            for pseed, ch in enumerate(class_holdout_embeds[cls]):
                for view_name, Z in [("structure", ch["z_s"]), ("expression", ch["z_e"]),
                                      ("concat", np.concatenate([ch["z_s"], ch["z_e"]], axis=1))]:
                    r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                    if r:
                        results.append({"class": cls, "seed": seed, "pretrain_seed": pseed,
                                         "method": f"contrastive_classholdout_{view_name}", **r})

            # contrastive DEBIASED-FULL for this class, same N_PRETRAIN_SEEDS averaging
            for pseed, de in enumerate(debiased_embeds[cls]):
                for view_name, Z in [("structure", de["z_s"]), ("expression", de["z_e"]),
                                      ("concat", np.concatenate([de["z_s"], de["z_e"]], axis=1))]:
                    r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                    if r:
                        results.append({"class": cls, "seed": seed, "pretrain_seed": pseed,
                                         "method": f"contrastive_debiased_{view_name}", **r})

            # contrastive RANDOM-HOLDOUT control (average over 3 control runs)
            for ri, rh in enumerate(random_holdout_embeds):
                Z = np.concatenate([rh["z_s"], rh["z_e"]], axis=1)
                r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                if r:
                    results.append({"class": cls, "seed": seed, "method": f"contrastive_randholdout{ri}_concat", **r})

            # raw feature baselines
            seen_idx, test_idx = seen, test
            r = raw_feature_baseline(raw_expr_by_drug[seen_idx], seen_y, raw_expr_by_drug[test_idx], test_y, "rf")
            if r:
                results.append({"class": cls, "seed": seed, "method": "raw_expression_RF", **r})
            r = raw_feature_baseline(ecfp[seen_idx], seen_y, ecfp[test_idx], test_y, "rf")
            if r:
                results.append({"class": cls, "seed": seed, "method": "raw_ecfp4_RF", **r})
            r = raw_feature_baseline(
                np.concatenate([raw_expr_by_drug, ecfp], axis=1)[seen_idx], seen_y,
                np.concatenate([raw_expr_by_drug, ecfp], axis=1)[test_idx], test_y, "rf")
            if r:
                results.append({"class": cls, "seed": seed, "method": "raw_concat_RF", **r})

            # parameter-free connectivity baseline
            seen_pos_mask = seen_y == 1
            r = neighbor_connectivity_baseline(raw_expr_by_drug[seen_idx][seen_pos_mask],
                                                raw_expr_by_drug[test_idx], test_y, k=5)
            if r:
                results.append({"class": cls, "seed": seed, "method": "neighbor_connectivity", **r})

            # random baseline
            rng = np.random.RandomState(seed)
            r = {"auroc": roc_auc_score(test_y, rng.rand(len(test_y))) if len(np.unique(test_y)) > 1 else None}
            if r["auroc"] is not None:
                r["auprc"] = average_precision_score(test_y, rng.rand(len(test_y)))
                results.append({"class": cls, "seed": seed, "method": "random", **r})

        n_done = sum(1 for r in results if r["class"] == cls)
        print(f"  {n_done} result rows collected for {cls}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS}/main_results.csv", index=False)
    print(f"\nSaved {len(df)} result rows to {RESULTS}/main_results.csv", flush=True)

    # ---- Summary table ----
    summary = df.groupby(["class", "method"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(f"{RESULTS}/summary_auroc.csv", index=False)
    print("\n=== SUMMARY (mean AUROC by class/method) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
