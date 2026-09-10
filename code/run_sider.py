"""
SIDER side-effect replication (paper Section VII).

Pharmacological class is close to structure-determined by construction, so the
redundancy result needs a second task family where prior work reports the
opposite. Wang et al. (2016) report expression outperforming structure for
drug-induced adverse events on L1000, so we repeat the structure/expression/
blend comparison on SIDER side-effect prediction.

The blend weight is selected by an inner cross-validation on the training folds
only. This matters: an earlier version of this analysis chose the weight on test
predictions and produced a +0.008 gain that disappeared once selection was made
honest.

Writes results/sider_nested.csv and results/sider_nested.log.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_expression_grid import build_aggregations

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

MIN_POS = MIN_NEG = 25
N_ENDPOINTS = 60
N_FOLDS = 5
SEED = 0
WEIGHTS = np.linspace(0.0, 1.0, 11)


def main():
    logf = open(f"{RESULTS}/sider_nested.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    qc = pd.read_parquet(f"{DATA}/sig_qc.parquet")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = int(meta["drug_uid"].max()) + 1
    aggs = build_aggregations(meta, expr, qc, n_drugs)
    hi, gold = aggs["A2_hidose24h_mean"], aggs["A7_percell_gold"]

    smap = pd.read_csv(f"{DATA}/lincs_sider_map.csv")
    pairs = pd.read_csv(f"{DATA}/sider_pairs.csv")
    cid2uid = dict(zip(smap["cid"], smap["drug_uid"].astype(int)))
    uids = np.array(sorted(set(cid2uid.values())))
    pos = {}
    for cid, se in zip(pairs["cid_flat"], pairs["se_name"]):
        u = cid2uid.get(cid)
        if u is not None:
            pos.setdefault(se, set()).add(u)
    log(f"{len(uids)} LINCS compounds map to SIDER; {len(pos)} side-effect terms")

    usable = [se for se, s in pos.items()
              if len(s) >= MIN_POS and len(uids) - len(s) >= MIN_NEG]
    log(f"{len(usable)} endpoints with >= {MIN_POS} positive and >= {MIN_NEG} negative drugs")
    rng = np.random.RandomState(SEED)
    chosen = sorted(usable)
    chosen = [chosen[i] for i in rng.choice(len(chosen), N_ENDPOINTS, replace=False)]

    rows = []
    for k, se in enumerate(chosen, 1):
        y = np.array([1 if u in pos[se] else 0 for u in uids])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        ps, pe, pr, pb = (np.zeros(len(y)) for _ in range(4))
        for tr, te in skf.split(uids, y):
            dtr, dte = uids[tr], uids[te]
            ms = RandomForestClassifier(n_estimators=200, max_depth=6,
                                        class_weight="balanced", random_state=0)
            ms.fit(ecfp[dtr], y[tr])
            ps[te] = ms.predict_proba(ecfp[dte])[:, 1]
            me = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            me.fit(hi[dtr], y[tr])
            pe[te] = me.predict_proba(hi[dte])[:, 1]
            mr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            mr.fit(gold[dtr], y[tr])
            pr[te] = mr.predict_proba(gold[dte])[:, 1]

            # inner CV on the training fold only, to pick the blend weight
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            scores = np.zeros(len(WEIGHTS))
            for itr, ite in inner.split(dtr, y[tr]):
                a, b = dtr[itr], dtr[ite]
                if len(np.unique(y[tr][itr])) < 2 or len(np.unique(y[tr][ite])) < 2:
                    continue
                s2 = RandomForestClassifier(n_estimators=200, max_depth=6,
                                            class_weight="balanced", random_state=0)
                s2.fit(ecfp[a], y[tr][itr])
                q_s = s2.predict_proba(ecfp[b])[:, 1]
                e2 = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                e2.fit(hi[a], y[tr][itr])
                q_e = e2.predict_proba(hi[b])[:, 1]
                for wi, w in enumerate(WEIGHTS):
                    scores[wi] += roc_auc_score(y[tr][ite], (1 - w) * q_s + w * q_e)
            w = WEIGHTS[int(np.argmax(scores))]
            pb[te] = (1 - w) * ps[te] + w * pe[te]

        rows.append({"se": se, "n_pos": int(y.sum()),
                     "struct": roc_auc_score(y, ps), "expr_mean": roc_auc_score(y, pe),
                     "expr_rich": roc_auc_score(y, pr), "blend_nested": roc_auc_score(y, pb)})
        log(f"[{k}/{N_ENDPOINTS}] {se}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/sider_nested.csv", index=False)
    _, p = wilcoxon(df["blend_nested"], df["struct"])
    log(f"\n=== NESTED (honest) over {len(df)} side effects ===")
    log(f"structure-only     : {df['struct'].mean():.4f}")
    log(f"expression (mean)  : {df['expr_mean'].mean():.4f}")
    log(f"expression (rich)  : {df['expr_rich'].mean():.4f}")
    log(f"blend (nested w)   : {df['blend_nested'].mean():.4f}")
    log(f"blend - struct     : {df['blend_nested'].mean() - df['struct'].mean():+.4f}")
    log(f"labels blend>struct: {(df['blend_nested'] > df['struct']).sum()}/{len(df)}")
    log(f"wilcoxon blend vs struct p = {p:.6g}")
    logf.close()


if __name__ == "__main__":
    main()
