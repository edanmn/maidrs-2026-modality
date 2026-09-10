"""Shared figure style for the MAI-DRS 2026 figures.

Every figure script writes both a PDF (used by the LaTeX source) and a PNG
(for quick inspection) into BIBM_MAIDRS/figures/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository layout. Override with MAIDRS_ROOT if the tree is checked out
# somewhere other than the original working directory.
ROOT = os.environ.get(
    "MAIDRS_ROOT", "/mnt/c/Users/rishik/epilepsy_repurpose"
)
PAPER_DIR = f"{ROOT}/BIBM_MAIDRS"
FIGS = f"{PAPER_DIR}/figures"
RESULTS = os.environ.get("MAIDRS_RESULTS", f"{PAPER_DIR}/results")
DATA = f"{ROOT}/ICLR2027_repurpose/data"

BG = "#fafafa"

CLASS_ORDER = [
    "class_AED",
    "class_Statin",
    "class_SSRI",
    "class_NSAID",
    "class_BetaBlocker",
    "class_Antipsychotic",
]
CLASS_LABEL = {
    "class_AED": "AED",
    "class_Statin": "Statin",
    "class_SSRI": "SSRI",
    "class_NSAID": "NSAID",
    "class_BetaBlocker": r"$\beta$-blocker",
    "class_Antipsychotic": "Antipsych.",
}


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#333333",
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "font.size": 11,
            "axes.titlesize": 13,
            "legend.frameon": False,
        }
    )


def save(fig, stem):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        path = f"{FIGS}/{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)
