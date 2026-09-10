"""
Expression aggregation x downstream model grid (paper Figure 1, Table III,
Table IV Phase 2 columns) and the 30-repeat corner re-estimation.

Aggregations, all producing one feature vector per compound:
  A1 naive_mean_all    mean over every Level-5 signature of the compound
  A2 hidose24h_mean    mean over 10 uM / 24 h signatures only
  A3 bestTAS_single    the single signature with the highest TAS
  A4 TASweighted       TAS-weighted mean over all signatures
  A5 gold_mean         mean over LINCS "gold" signatures
                       (distil_cc_q75 >= 0.2 and pct_self_rank_q25 <= 5)
  A6 percell_hidose    per-cell-line 10 uM / 24 h blocks, concatenated
  A7 percell_gold      per-cell-line gold blocks, concatenated

Each aggregation falls back to the naive mean for compounds with no signature
meeting its criterion, so every compound keeps a feature vector and the
comparison is over representations rather than over differing compound sets.

Writes results/grid_expression.csv, results/grid.log and results/stabilize.log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import repeated_subsample_splits

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
GRID_REPEATS = 6
CORNER_REPEATS = 30
SEED0 = 42
N_JOBS = int(os.environ.get('MAIDRS_JOBS', '12'))
N_CELL_BLOCKS = 6          # cell lines used for the per-cell-line concatenations


def models():
    return {
        "M1_RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=6,
                                                class_weight="balanced", random_state=0),
        "M2_LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "M3_kNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "M4_SVM_rbf": lambda: SVC(probability=True, class_weight="balanced", random_state=0),
        "M5_GBM": lambda: GradientBoostingClassifier(random_state=0),
    }


def build_aggregations(meta, expr, qc, n_drugs):
    duid = meta["drug_uid"].values.astype(int)
    G = expr.shape[1]
    sig_by_drug = {}
    for s, d in enumerate(duid):
        sig_by_drug.setdefault(d, []).append(s)

    tas = qc["tas"].values
    gold = (qc["distil_cc_q75"].values >= 0.2) & (qc["pct_self_rank_q25"].values <= 5)
    hidose = (meta["pert_idose"].values == "10.0 um") & (meta["pert_itime"].values == "24 h")
    cells = meta["cell_id"].values
    top_cells = pd.Series(cells).value_counts().index[:N_CELL_BLOCKS].tolist()

    naive = np.zeros((n_drugs, G), np.float32)
    for d, s in sig_by_drug.items():
        naive[d] = expr[s].mean(axis=0)

    def subset_mean(mask):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            sel = [i for i in s if mask[i]]
            if sel:
                out[d] = expr[sel].mean(axis=0)
        return out

    def best_single(score):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            out[d] = expr[s[int(np.argmax(score[s]))]]
        return out

    def weighted(w):
        out = naive.copy()
        for d, s in sig_by_drug.items():
            ww = np.maximum(w[s], 0)
            if ww.sum() > 0:
                out[d] = (expr[s] * ww[:, None]).sum(0) / ww.sum()
        return out

    def percell(mask):
        out = np.zeros((n_drugs, G * len(top_cells)), np.float32)
        for bi, c in enumerate(top_cells):
            blk = np.zeros((n_drugs, G), np.float32)
            cm = (cells == c) & mask
            for d, s in sig_by_drug.items():
                sel = [i for i in s if cm[i]]
                blk[d] = expr[sel].mean(axis=0) if sel else naive[d]
            out[:, bi * G:(bi + 1) * G] = blk
        return out

    return {
        "A1_naive_mean_all": naive,
        "A2_hidose24h_mean": subset_mean(hidose),
        "A3_bestTAS_single": best_single(tas),
        "A4_TASweighted": weighted(tas),
        "A5_gold_mean": subset_mean(gold),
        "A6_percell_hidose": percell(hidose),
        "A7_percell_gold": percell(gold),
    }


def _one_fit(X, model_key, seen, sy, test, ty):
    clf = models()[model_key]()
    clf.fit(X[seen], sy)
    return roc_auc_score(ty, clf.predict_proba(X[test])[:, 1])


def run_cell(X, drug_level, all_uids, model_key, n_repeats):
    """Evaluate one aggregation x model cell over every class and seed.

    Gradient boosting and the RBF SVM are single-threaded and the two
    per-cell-line representations are 5868-dimensional, so the fits are spread
    across cores. Every fit is independent, so this changes runtime and nothing
    else.
    """
    jobs = []
    for cls in CLASS_COLS:
        pos = drug_level[cls]
        pos_drugs = np.array(sorted(pos[pos == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_uids)]
        for sp in repeated_subsample_splits(pos_drugs, all_uids,
                                            n_repeats=n_repeats, seed0=SEED0):
            seen, test, ty = sp["seen_drugs"], sp["test_drugs"], sp["test_labels"]
            sy = pos.reindex(seen).fillna(0).values
            if len(np.unique(sy)) < 2:
                continue
            jobs.append((seen, sy, test, ty))
    vals = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one_fit)(X, model_key, *j) for j in jobs)
    return np.array(vals)


def main():
    logf = open(f"{RESULTS}/grid.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    qc = pd.read_parquet(f"{DATA}/sig_qc.parquet")
    assert (qc["sig_id"].values == meta["sig_id"].values).all(), "QC rows must align with meta"
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_uids = np.array(sorted(drug_level.index.tolist()))
    n_drugs = int(meta["drug_uid"].max()) + 1

    log(f"{expr.shape[0]} signatures, {len(all_uids)} drugs")
    aggs = build_aggregations(meta, expr, qc, n_drugs)
    for k, v in aggs.items():
        log(f"  {k:<20} dim={v.shape[1]}")

    rows = []
    log(f"\n=== grid ({GRID_REPEATS} repeats per cell) ===")
    for aname, X in aggs.items():
        for mname in models():
            v = run_cell(X, drug_level, all_uids, mname, GRID_REPEATS)
            rows.append({"agg": aname, "model": mname, "auroc": v.mean(), "n": len(v)})
            log(f"  {aname:<20} {mname:<12} {v.mean():.4f}  n={len(v)}")
    pd.DataFrame(rows).to_csv(f"{RESULTS}/grid_expression.csv", index=False)

    # variance decomposition over the grid
    df = pd.DataFrame(rows)
    gm = df["auroc"].mean()
    tot = ((df["auroc"] - gm) ** 2).sum()

    def ss(col):
        return sum(len(g) * (g["auroc"].mean() - gm) ** 2 for _, g in df.groupby(col))

    log(f"\n=== variance decomposition over the {len(df)}-cell grid ===")
    log(f"  between-model      {100*ss('model')/tot:.1f}%")
    log(f"  between-aggregation{100*ss('agg')/tot:.1f}%")

    # ---- corner cells at 30 repeats with bootstrap intervals ----------------
    with open(f"{RESULTS}/stabilize.log", "w") as sf:
        sf.write(f"{'cell':<20} {'mean':>7} {'95% CI':>18} {'n':>6}\n")
        log(f"\n=== corner cells ({CORNER_REPEATS} repeats, bootstrap CI) ===")
        corners = [("naive x RF", "A1_naive_mean_all", "M1_RF"),
                   ("naive x LogReg", "A1_naive_mean_all", "M2_LogReg"),
                   ("hidose x RF", "A2_hidose24h_mean", "M1_RF"),
                   ("hidose x LogReg", "A2_hidose24h_mean", "M2_LogReg")]
        rng = np.random.RandomState(0)
        for label, a, m in corners:
            v = run_cell(aggs[a], drug_level, all_uids, m, CORNER_REPEATS)
            bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            line = f"{label:<20} {v.mean():>7.3f}  [{lo:.3f}, {hi:.3f}] {len(v):>6}"
            sf.write(line + "\n")
            log("  " + line)
    logf.close()


if __name__ == "__main__":
    main()
