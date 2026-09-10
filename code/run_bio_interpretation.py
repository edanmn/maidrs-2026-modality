"""
Gene-level check on the corrected expression features (paper Section V-A).

If the corrected pipeline carries real pharmacology, the landmark genes that
separate a drug class from the rest should be that class's known targets,
without the model being told what they are.

For each class we take the corrected (10 uM / 24 h) drug-level features, test
every one of the 978 landmark genes for a difference in response between class
members and all other compounds, and report the ten smallest p-values.

The test is an exact two-sided Mann-Whitney U. Classes have 5 to 16 members
against roughly 1,700 other compounds, so the exact null is the right one and
the normal approximation is noticeably anti-conservative here. We also report
Benjamini-Hochberg q-values across all 978 genes. These are not discovery
claims: the section is a positive control, and the q-values are given so the
reader can see how far the individual genes are from surviving correction.

Writes results/bio_interpretation.json and results/bio_interpretation.log.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_expression_grid import build_aggregations

ROOT = os.environ.get("MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose")
DATA = f"{ROOT}/ICLR2027_repurpose/data"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{ROOT}/BIBM_MAIDRS/results")
os.makedirs(RESULTS, exist_ok=True)

CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
TOP_K = 10


def bh(p):
    """Benjamini-Hochberg q-values."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def main():
    logf = open(f"{RESULTS}/bio_interpretation.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    qc = pd.read_parquet(f"{DATA}/sig_qc.parquet")
    genes = [l.strip() for l in open(f"{DATA}/gene_cols.txt") if l.strip()]
    n_drugs = int(meta["drug_uid"].max()) + 1

    hi = build_aggregations(meta, expr, qc, n_drugs)["A2_hidose24h_mean"]
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    uids = np.array(sorted(drug_level.index.tolist()))
    X = hi[uids]
    log(f"{len(uids)} drugs x {X.shape[1]} landmark genes, corrected (10 uM / 24 h) features")

    out = {}
    for cls in CLASS_COLS:
        y = drug_level[cls].reindex(uids).fillna(0).values.astype(int)
        a, b = X[y == 1], X[y == 0]
        delta = a.mean(0) - b.mean(0)
        p = np.array([stats.mannwhitneyu(a[:, i], b[:, i], method="exact").pvalue
                      for i in range(X.shape[1])])
        q = bh(p)
        top = np.argsort(p)[:TOP_K]
        out[cls] = {
            "n_members": int(y.sum()),
            "diff": [[genes[i], float(delta[i]), float(p[i]), float(q[i])] for i in top],
        }
        log(f"\n## {cls}  ({int(y.sum())} members)")
        for i in top:
            log(f"   {genes[i]:<10} delta={delta[i]:+.2f}  p={p[i]:.2e}  q={q[i]:.3f}")

    with open(f"{RESULTS}/bio_interpretation.json", "w") as f:
        json.dump(out, f, indent=1)
    log(f"\nwrote {RESULTS}/bio_interpretation.json")
    logf.close()


if __name__ == "__main__":
    main()
