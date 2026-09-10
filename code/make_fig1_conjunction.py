"""Figure 1: expression-only AUROC across aggregation strategies x model classes.

Source: ICLR2027_repurpose/results/grid_expression.csv, written by the grid
sweep (7 aggregation strategies x 5 downstream model classes, 6 drug-disjoint
subsampling repeats per cell, averaged over the six drug classes).

The red box marks the naive-aggregation/random-forest cell and the green box the
corrected pair. Both cell means shown here are 6-repeat estimates; the 30-repeat
re-estimates with bootstrap intervals are in Table III of the paper and come
from the 30-repeat sweep recorded in ICLR2027_repurpose/results/stabilize.log.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from _figstyle import RESULTS, apply_style, save
import matplotlib.pyplot as plt

AGG_LABEL = {
    "A1_naive_mean_all": "Naive mean\n(all conditions)",
    "A2_hidose24h_mean": "High-dose\n10 µM / 24 h",
    "A3_bestTAS_single": "Best-TAS\nsingle sig.",
    "A4_TASweighted": "TAS-weighted\nconsensus",
    "A5_gold_mean": "Gold-signature\nmean",
    "A6_percell_hidose": "Per-cell-line\nhigh-dose",
    "A7_percell_gold": "Per-cell-line\ngold",
}
MODEL_LABEL = {
    "M1_RF": "RF",
    "M2_LogReg": "LogReg",
    "M3_kNN": "kNN",
    "M4_SVM_rbf": "SVM-rbf",
    "M5_GBM": "GBM",
}
WORST = ("A1_naive_mean_all", "M1_RF")
BEST = ("A2_hidose24h_mean", "M2_LogReg")


def main():
    apply_style()
    df = pd.read_csv(f"{RESULTS}/grid_expression.csv")
    grid = df.pivot(index="agg", columns="model", values="auroc")
    grid = grid.reindex(index=list(AGG_LABEL), columns=list(MODEL_LABEL))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.grid(False)
    im = ax.imshow(grid.values, cmap="Blues", vmin=0.50, vmax=0.75, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if v > 0.66 else "#1a1a1a",
            )

    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([MODEL_LABEL[c] for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([AGG_LABEL[r] for r in grid.index], fontsize=9)
    ax.set_title("Expression-only AUROC: aggregation $\\times$ model class")

    for (agg, model), colour in ((WORST, "#d62728"), (BEST, "#2ca02c")):
        i = list(grid.index).index(agg)
        j = list(grid.columns).index(model)
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=colour, linewidth=2.5
            )
        )

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    worst_v = grid.loc[WORST[0], WORST[1]]
    best_v = grid.loc[BEST[0], BEST[1]]
    fig.text(
        0.02,
        -0.02,
        f"Red = naive aggregation $\\times$ RF (chance, {worst_v:.3f}). "
        f"Green = corrected pair ({best_v:.3f}).",
        fontsize=9,
        color="#555555",
    )

    save(fig, "fig1_conjunction")

    print("\n=== values used in the text ===")
    print(f"  naive x RF            {worst_v:.3f}")
    print(f"  hi-dose x LogReg      {best_v:.3f}")
    print(f"  naive x RF -> hi-dose x RF: "
          f"{grid.loc['A1_naive_mean_all','M1_RF']:.3f} -> "
          f"{grid.loc['A2_hidose24h_mean','M1_RF']:.3f}")
    for name, axis in (("aggregation", 1), ("model", 0)):
        means = grid.values.mean(axis=axis)
        ss = ((means - grid.values.mean()) ** 2).sum() * grid.shape[axis]
        tot = ((grid.values - grid.values.mean()) ** 2).sum()
        print(f"  between-{name} variance explained: {100 * ss / tot:.1f}%")


if __name__ == "__main__":
    main()
