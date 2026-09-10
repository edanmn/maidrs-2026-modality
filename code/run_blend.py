"""
Cross-modal blend (paper Section VII): does a correctly specified expression
arm add anything to structure?

Best model per modality, chosen from the grid rather than reused across arms:
  structure  = random forest on ECFP4
  expression = logistic regression on 10 uM / 24 h signatures

The blend is a fixed 50:50 average of the two predicted probabilities. The
weight is fixed in advance, not selected on test: an earlier version of this
analysis picked the weight on test predictions and produced a +0.008 gain that
vanished under honest selection.

Writes results/blend_proper.log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits
from run_expression_grid import build_aggregations

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
CHECK_REPEATS = 6      # per-modality model comparison, matches the grid
BLEND_REPEATS = 20     # pooled n = 120 over six classes
SEED0 = 42


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  class_weight="balanced", random_state=0)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def sweep(X, make_model, drug_level, all_uids, n_repeats):
    vals = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pd_ = np.array(sorted(pos[pos == 1].index.tolist()))
        pd_ = pd_[np.isin(pd_, all_uids)]
        for sp in repeated_subsample_splits(pd_, all_uids, n_repeats=n_repeats, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue
            m = make_model()
            m.fit(X[seen], sy)
            vals.append(roc_auc_score(ty, m.predict_proba(X[test])[:, 1]))
    return float(np.mean(vals))


def main():
    logf = open(f"{RESULTS}/blend_proper.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    qc = pd.read_parquet(f"{DATA}/sig_qc.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_uids = np.array(sorted(drug_level.index.tolist()))
    n_drugs = int(meta["drug_uid"].max()) + 1
    aggs = build_aggregations(meta, expr, qc, n_drugs)
    hi, gold = aggs["A2_hidose24h_mean"], aggs["A5_gold_mean"]

    log("=== per-modality model check (mean AUROC over classes) ===")
    for name, X, mk in [("struct_RF", ecfp, rf), ("struct_LR", ecfp, lr),
                        ("expr_hi_RF", hi, rf), ("expr_hi_LR", hi, lr),
                        ("expr_gold_LR", gold, lr)]:
        log(f"  {name:<14}{sweep(X, mk, drug_level, all_uids, CHECK_REPEATS):.4f}")

    log("\n=== CROSS-MODAL BLEND: struct(RF/ECFP4) + expr(LogReg/hi-dose), fixed 50/50 ===")
    log(f"{'class':<16}{'struct':>9}{'expr':>9}{'blend':>9}{'gain':>9}{'p':>10}")
    S, E, B = [], [], []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pd_ = np.array(sorted(pos[pos == 1].index.tolist()))
        pd_ = pd_[np.isin(pd_, all_uids)]
        s, e, b = [], [], []
        for sp in repeated_subsample_splits(pd_, all_uids, n_repeats=BLEND_REPEATS, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue
            ms = rf(); ms.fit(ecfp[seen], sy)
            ps = ms.predict_proba(ecfp[test])[:, 1]
            me = lr(); me.fit(hi[seen], sy)
            pe = me.predict_proba(hi[test])[:, 1]
            s.append(roc_auc_score(ty, ps))
            e.append(roc_auc_score(ty, pe))
            b.append(roc_auc_score(ty, 0.5 * ps + 0.5 * pe))
        s, e, b = np.array(s), np.array(e), np.array(b)
        _, p = wilcoxon(b, s)
        log(f"{cls.replace('class_',''):<16}{s.mean():>9.3f}{e.mean():>9.3f}"
            f"{b.mean():>9.3f}{b.mean()-s.mean():>+9.3f}{p:>10.4f}")
        S += list(s); E += list(e); B += list(b)

    S, E, B = np.array(S), np.array(E), np.array(B)
    _, p = wilcoxon(B, S)
    log(f"\nPOOLED n={len(S)}: struct={S.mean():.4f} expr={E.mean():.4f} blend={B.mean():.4f}")
    log(f"gain={B.mean()-S.mean():+.4f}  p={p:.4g}  wins={(B>S).sum()}/{len(S)}")
    logf.close()


if __name__ == "__main__":
    main()
