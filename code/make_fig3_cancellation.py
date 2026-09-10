"""Figure 3: why naive aggregation destroys signal.

Recomputes, from the Level-5 landmark matrix itself, the signature-magnitude
statistics quoted in Section V: the median L2 norm of a drug's individual
signatures against the norm of its mean across all doses, cell lines and
timepoints, the resulting cancellation fraction, and the two activity checks
showing the loss is not a property of the underlying data.

Inputs (built by build_dataset.py):
    ICLR2027_repurpose/data/expr_signatures.npy   (n_sig x 978, float32)
    ICLR2027_repurpose/data/dataset_meta.parquet  (aligned row metadata)

Writes figures/fig3_cancellation.{pdf,png} and prints every number the paper
quotes from this analysis.
"""
import json

import numpy as np
import pandas as pd

from _figstyle import DATA, FIGS, apply_style, save
import matplotlib.pyplot as plt

STRONG_GENE_Z = 2.0
STRONG_GENE_COUNT = 50   # "strongly active" signature
WEAK_GENE_COUNT = 25     # "weak" signature; a deliberately lower cut


def main():
    apply_style()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    X = np.load(f"{DATA}/expr_signatures.npy", mmap_mode="r")
    assert len(meta) == X.shape[0], (len(meta), X.shape)
    print(f"signatures {X.shape[0]}, genes {X.shape[1]}, "
          f"compounds {meta['pert_iname'].nunique()}")

    # Per-signature quantities, computed in row blocks to keep memory flat.
    sig_norm = np.empty(X.shape[0], dtype=np.float64)
    sig_strong_genes = np.empty(X.shape[0], dtype=np.int32)
    for lo in range(0, X.shape[0], 20000):
        block = np.asarray(X[lo:lo + 20000], dtype=np.float32)
        sig_norm[lo:lo + len(block)] = np.linalg.norm(block, axis=1)
        sig_strong_genes[lo:lo + len(block)] = (
            np.abs(block) > STRONG_GENE_Z
        ).sum(axis=1)

    meta = meta.assign(sig_norm=sig_norm, strong_genes=sig_strong_genes)

    # Per-drug: typical individual magnitude vs magnitude of the naive mean.
    indiv, avg, names = [], [], []
    for name, idx in meta.groupby("pert_iname").indices.items():
        indiv.append(np.median(sig_norm[idx]))
        avg.append(np.linalg.norm(np.asarray(X[idx], dtype=np.float32).mean(axis=0)))
        names.append(name)
    indiv = np.array(indiv)
    avg = np.array(avg)

    med_indiv = float(np.median(indiv))
    med_avg = float(np.median(avg))
    cancelled = 1.0 - med_avg / med_indiv

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Clip the long right tail so the two modes stay legible; fewer than 0.5%
    # of drugs fall beyond the limit.
    hi = float(np.percentile(indiv, 99.5))
    bins = np.linspace(0, hi, 60)
    ax.hist(indiv, bins=bins, color="#3d8bd4", label="individual signatures")
    ax.hist(avg, bins=bins, color="#e8703a", label="drug-level mean")
    ax.set_xlabel("signature L2 norm")
    ax.set_ylabel("drugs")
    ax.set_title(f"Averaging cancels {cancelled * 100:.0f}% of magnitude")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0, hi)
    ax.legend()
    save(fig, "fig3_cancellation")

    # Activity checks: the signal loss is not an absence of signal.
    per_drug_max = meta.groupby("pert_iname")["strong_genes"].max()
    pct_drugs_active = 100.0 * (per_drug_max >= STRONG_GENE_COUNT).mean()
    pct_sigs_weak = 100.0 * (meta["strong_genes"] < WEAK_GENE_COUNT).mean()
    pct_sigs_below_strong = 100.0 * (meta["strong_genes"] < STRONG_GENE_COUNT).mean()

    stats = {
        "n_signatures": int(X.shape[0]),
        "n_compounds": int(meta["pert_iname"].nunique()),
        "median_individual_norm": round(med_indiv, 2),
        "median_drug_mean_norm": round(med_avg, 2),
        "cancelled_fraction_pct": round(100 * cancelled, 1),
        "pct_compounds_with_strong_signature": round(pct_drugs_active, 1),
        "pct_signatures_weak": round(pct_sigs_weak, 1),
        "pct_signatures_below_strong_cut": round(pct_sigs_below_strong, 1),
        "strong_signature_definition": (
            f">= {STRONG_GENE_COUNT} landmark genes at |z| > {STRONG_GENE_Z}"
        ),
        "weak_signature_definition": (
            f"< {WEAK_GENE_COUNT} landmark genes at |z| > {STRONG_GENE_Z}"
        ),
    }
    print("\n=== values quoted in Section V ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    with open(f"{FIGS}/fig3_cancellation_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  wrote {FIGS}/fig3_cancellation_stats.json")


if __name__ == "__main__":
    main()
