"""
Phase 1 replication analyses (paper Section VI and the Phase 1 columns of
Table IV), plus the Phase 1 arm of the Wang et al. (2016) comparison.

LINCS Phase 1 (GSE92742) is a separate experimental campaign from Phase 2, with
different cell lines, doses and timepoints. `phase1_replication.py` extracts the
978 landmark genes for the Phase 1 signatures covering our compounds; this
script runs the aggregation-by-model grid and the SIDER comparison on them.

Phase 1 representations are the nearest available analogue of each Phase 2 one.
Phase 1 records `is_exemplar` rather than the gold QC fields, so the exemplar
mean stands in for the gold mean, and `distil_ss` stands in for TAS when picking
a single strongest signature.

As in `run_expression_grid.py`, every representation falls back to the compound's
naive mean when no signature meets its criterion, rather than leaving an all-zero
vector behind.

Writes results/p1_analyses.log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
REPEATS = 6
SEED0 = 42
N_JOBS = int(os.environ.get("MAIDRS_JOBS", "12"))
MIN_POS = MIN_NEG = 25


def _fit(X, key, seen, sy, test, ty):
    clf = (RandomForestClassifier(n_estimators=200, max_depth=6,
                                  class_weight="balanced", random_state=0)
           if key == "RF" else
           LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    clf.fit(X[seen], sy)
    return roc_auc_score(ty, clf.predict_proba(X[test])[:, 1])


def sweep(X, key, drug_level, uids):
    jobs = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pd_ = np.array(sorted(pos[pos == 1].index.tolist()))
        pd_ = pd_[np.isin(pd_, uids)]
        if len(pd_) < 3:
            continue
        for sp in repeated_subsample_splits(pd_, uids, n_repeats=REPEATS, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue
            jobs.append((seen, sy, test, ty))
    v = Parallel(n_jobs=N_JOBS)(delayed(_fit)(X, key, *j) for j in jobs)
    return float(np.mean(v))


def main():
    logf = open(f"{RESULTS}/p1_analyses.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    p1 = pd.read_parquet(f"{DATA}/p1_meta.parquet")
    X1 = np.load(f"{DATA}/p1_expr.npy")
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    name2uid = dict(zip(meta["pert_iname"], meta["drug_uid"].astype(int)))
    duid = p1["pert_iname"].map(name2uid)
    keep = duid.notna().values
    p1, X1, duid = p1[keep], X1[keep], duid[keep].astype(int).values
    n_drugs = ecfp.shape[0]
    log(f"Phase1: {X1.shape}, {len(set(duid))} compounds")

    hi = (p1["pert_idose"].values == "10 µM") & (p1["pert_itime"].values == "24 h")
    exemplar = p1["is_exemplar"].values == 1
    ss = p1["distil_ss"].values
    log(f"  hi-dose/24h: {int(hi.sum())}   exemplar: {int(exemplar.sum())}")

    sig_by_drug = {}
    for s, d in enumerate(duid):
        sig_by_drug.setdefault(d, []).append(s)
    G = X1.shape[1]
    naive = np.zeros((n_drugs, G), np.float32)
    for d, s in sig_by_drug.items():
        naive[d] = X1[s].mean(axis=0)

    def subset_mean(mask):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            sel = [i for i in s if mask[i]]
            if sel:
                out[d] = X1[sel].mean(axis=0)
        return out

    def best_single(score):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            out[d] = X1[s[int(np.argmax(score[s]))]]
        return out

    reps = {"naive_mean_all": naive,
            "hidose24h": subset_mean(hi),
            "strongest_distil_ss": best_single(ss),
            "exemplar_mean": subset_mean(exemplar)}

    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    uids = np.array(sorted(set(duid.tolist())))

    # positive control on the extraction: statins should move HMGCR
    # The Phase 1 matrix carries its own gene order, which is NOT the Phase 2
    # order, so the index has to come from p1_genes.csv.
    genes = pd.read_csv(f"{DATA}/p1_genes.csv").iloc[:, 0].astype(str).tolist()
    gi = genes.index("HMGCR") if "HMGCR" in genes else None
    if gi is not None:
        stat = drug_level["class_Statin"].reindex(uids).fillna(0).values.astype(int)
        hm = reps["hidose24h"][uids][:, gi]
        log(f"  positive control: HMGCR {hm[stat == 1].mean():+.2f} under statins "
            f"against {hm[stat == 0].mean():+.2f} elsewhere")

    log("\n=== (A) AGGREGATION x MODEL on PHASE 1 (expression-only AUROC) ===")
    log(f"{'representation':<26}{'RF':>7}{'LogReg':>9}")
    for name, X in reps.items():
        log(f"{name:<26}{sweep(X, 'RF', drug_level, uids):>7.3f}"
            f"{sweep(X, 'LR', drug_level, uids):>9.3f}")

    # ---- (B) Wang et al. 2016 comparison on Phase 1 -------------------------
    smap = pd.read_csv(f"{DATA}/lincs_sider_map.csv")
    pairs = pd.read_csv(f"{DATA}/sider_pairs.csv")
    cid2uid = dict(zip(smap["cid"], smap["drug_uid"].astype(int)))
    su = np.array(sorted(set(cid2uid.values()) & set(uids.tolist())))
    pos = {}
    for cid, se in zip(pairs["cid_flat"], pairs["se_name"]):
        u = cid2uid.get(cid)
        if u is not None and u in set(su.tolist()):
            pos.setdefault(se, set()).add(u)
    usable = [se for se, s in pos.items()
              if len(s) >= MIN_POS and len(su) - len(s) >= MIN_NEG]
    log(f"\n=== (B) WANG 2016 REPLICATION on PHASE 1 (SIDER) ===")
    log(f"  compounds {len(su)}, usable endpoints {len(usable)}")

    def endpoint(se):
        y = np.array([1 if u in pos[se] else 0 for u in su])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        ps, pe, px = (np.zeros(len(y)) for _ in range(3))
        for tr, te in skf.split(su, y):
            a, b = su[tr], su[te]
            m = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       class_weight="balanced", random_state=0)
            m.fit(ecfp[a], y[tr]); ps[te] = m.predict_proba(ecfp[b])[:, 1]
            for arr, rep in ((pe, "strongest_distil_ss"), (px, "exemplar_mean")):
                q = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                q.fit(reps[rep][a], y[tr]); arr[te] = q.predict_proba(reps[rep][b])[:, 1]
        return (roc_auc_score(y, ps), roc_auc_score(y, pe), roc_auc_score(y, px))

    out = Parallel(n_jobs=N_JOBS)(delayed(endpoint)(se) for se in usable)
    out = np.array(out)
    log(f"  n={len(out)}  structure {out[:,0].mean():.4f}  "
        f"expr(strongest distil_ss) {out[:,1].mean():.4f}  expr(exemplar) {out[:,2].mean():.4f}")
    for j, lab in ((1, "expr_strongest"), (2, "expr_exemplar")):
        _, p = wilcoxon(out[:, j], out[:, 0])
        log(f"  {lab} - structure: {out[:,j].mean()-out[:,0].mean():+.4f}  "
            f"p={p:.3g}  wins={(out[:,j]>out[:,0]).sum()}/{len(out)}")
    logf.close()


if __name__ == "__main__":
    main()
