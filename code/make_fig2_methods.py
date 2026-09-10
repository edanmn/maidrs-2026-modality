"""Figure 2: downstream class recovery for six methods across six drug classes.

Source: ICLR2027_repurpose/results/summary_auroc.csv, the per-class mean AUROC
over 10 drug-disjoint subsampling seeds written by run_experiments.py.

Also prints the Wilcoxon signed-rank comparison of the contrastive model against
the strongest baseline per class (the p values quoted in Section IV) and the
count of classes in which the raw-concatenation baseline falls below structure
alone.
"""
import numpy as np
import pandas as pd

from _figstyle import CLASS_LABEL, CLASS_ORDER, RESULTS, apply_style, save
import matplotlib.pyplot as plt

METHODS = [
    ("random", "Random", "#1f77b4"),
    ("neighbor_connectivity", "Neighbor conn.", "#ff7f0e"),
    ("raw_expression_RF", "Expression+RF", "#2aa198"),
    ("contrastive_full_concat", "Contrastive", "#ffb000"),
    ("raw_concat_RF", "Concat+RF", "#e377a8"),
    ("raw_ecfp4_RF", "ECFP4+RF", "#178a17"),
]


def main():
    apply_style()
    df = pd.read_csv(f"{RESULTS}/summary_auroc.csv")
    lut = {(r["class"], r["method"]): r["mean"] for _, r in df.iterrows()}

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    n = len(METHODS)
    width = 0.8 / n
    x = np.arange(len(CLASS_ORDER))

    for k, (key, label, colour) in enumerate(METHODS):
        vals = [lut.get((c, key), np.nan) for c in CLASS_ORDER]
        ax.bar(x + (k - (n - 1) / 2) * width, vals, width, label=label, color=colour)

    chance = np.nanmean([lut.get((c, "random"), np.nan) for c in CLASS_ORDER])
    ax.axhline(chance, ls="--", lw=1.2, color="#888888", zorder=0)
    ax.text(len(CLASS_ORDER) - 0.42, chance + 0.006, "chance", fontsize=9,
            color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABEL[c] for c in CLASS_ORDER])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.32, 1.03)
    ax.set_title("A random forest on raw fingerprints wins in every class")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)

    save(fig, "fig2_methods")

    print("\n=== claims checked against the same table ===")
    struct = np.array([lut[(c, "raw_ecfp4_RF")] for c in CLASS_ORDER])
    concat = np.array([lut[(c, "raw_concat_RF")] for c in CLASS_ORDER])
    contra = np.array([lut[(c, "contrastive_full_concat")] for c in CLASS_ORDER])
    print(f"  concat below structure in {(concat < struct).sum()}/6 classes")
    print(f"  ECFP4+RF strongest in {(struct >= np.vstack([concat, contra]).max(0)).sum()}/6 classes")
    print(f"  structure - contrastive margin: {(struct - contra).min():.3f} "
          f"to {(struct - contra).max():.3f}")

    sig = pd.read_csv(f"{RESULTS}/significance.csv")
    print(f"  max Wilcoxon p vs best baseline: {sig['wilcoxon_p'].max():.4f}")


if __name__ == "__main__":
    main()
