"""
Turn results/main_results.csv into (a) LaTeX table snippets ready to paste
into paper/main.tex, and (b) figures, plus a Wilcoxon signed-rank test of the
best contrastive variant vs. the strongest baseline per class.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = "/mnt/c/Users/rishik/epilepsy_repurpose/ICLR2027_repurpose"
RESULTS = f"{ROOT}/results"
FIGS = f"{ROOT}/figures"
os.makedirs(FIGS, exist_ok=True)

CLASS_ORDER = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
               "class_BetaBlocker", "class_Antipsychotic"]
CLASS_LABEL = {"class_AED": "AED", "class_Statin": "Statin", "class_SSRI": "SSRI",
               "class_NSAID": "NSAID", "class_BetaBlocker": "Beta-blocker",
               "class_Antipsychotic": "Antipsychotic"}

MAIN_METHODS = ["random", "neighbor_connectivity", "raw_ecfp4_RF", "raw_expression_RF",
                "raw_concat_RF", "contrastive_full_concat"]
METHOD_LABEL = {
    "random": "Random",
    "neighbor_connectivity": "Neighbor connectivity (param.-free)",
    "raw_ecfp4_RF": "Raw ECFP4 + RF",
    "raw_expression_RF": "Raw expression + RF",
    "raw_concat_RF": "Raw concat + RF",
    "contrastive_full_concat": "Contrastive (full, concat)",
}

COLDSTART_METHODS = {
    "Full": "contrastive_full_concat",
    "Random-holdout (control, avg of 3)": None,  # handled specially
    "Class-holdout (cold-start)": "contrastive_classholdout_concat",
}


def fmt(mean, std):
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f}$\\pm${std:.3f}" if not pd.isna(std) else f"{mean:.3f}"


def main():
    df = pd.read_csv(f"{RESULTS}/main_results.csv")
    present_classes = [c for c in CLASS_ORDER if c in df["class"].unique()]

    # ---------------- Table 1: main results ----------------
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Main downstream results: mean$\pm$std AUROC over 10 repeated "
                 r"drug-disjoint subsampling seeds, full-data pretraining regime, "
                 r"concatenated (structure+expression) probe.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{center}\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(present_classes) + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(CLASS_LABEL[c] for c in present_classes) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for m in MAIN_METHODS:
        row = [METHOD_LABEL[m]]
        for c in present_classes:
            sub = df[(df["class"] == c) & (df["method"] == m)]
            if len(sub) == 0:
                row.append("--")
            else:
                row.append(fmt(sub["auroc"].mean(), sub["auroc"].std()))
        bold = m == "contrastive_full_concat"
        cells = " & ".join((f"\\textbf{{{v}}}" if bold else v) for v in row[1:])
        name = f"\\textbf{{{row[0]}}}" if bold else row[0]
        lines.append(f"{name} & {cells} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}\end{center}\end{table}")
    table1 = "\n".join(lines)

    # ---------------- Table 2: cold-start ----------------
    lines2 = []
    lines2.append(r"\begin{table}[t]")
    lines2.append(r"\caption{Cold-start generalization gap: class-holdout vs.\ full-data vs.\ "
                  r"random-holdout-control pretraining, mean AUROC (concat probe).}")
    lines2.append(r"\label{tab:coldstart}")
    lines2.append(r"\begin{center}\small")
    lines2.append(r"\begin{tabular}{l" + "c" * len(present_classes) + "}")
    lines2.append(r"\toprule")
    lines2.append("Regime & " + " & ".join(CLASS_LABEL[c] for c in present_classes) + r" \\")
    lines2.append(r"\midrule")

    rh_cols = [c for c in df["method"].unique() if c.startswith("contrastive_randholdout")]
    for regime_name, method in COLDSTART_METHODS.items():
        row = [regime_name]
        for c in present_classes:
            if method is not None:
                sub = df[(df["class"] == c) & (df["method"] == method)]
                row.append(fmt(sub["auroc"].mean(), sub["auroc"].std()) if len(sub) else "--")
            else:
                sub = df[(df["class"] == c) & (df["method"].isin(rh_cols))]
                row.append(fmt(sub["auroc"].mean(), sub["auroc"].std()) if len(sub) else "--")
        lines2.append(" & ".join(row) + r" \\")
    lines2.append(r"\bottomrule")
    lines2.append(r"\end{tabular}\end{center}\end{table}")
    table2 = "\n".join(lines2)

    with open(f"{RESULTS}/table_main.tex", "w") as f:
        f.write(table1)
    with open(f"{RESULTS}/table_coldstart.tex", "w") as f:
        f.write(table2)

    # ---------------- Significance test ----------------
    # contrastive_full_concat / contrastive_classholdout_concat have 3 rows per
    # subsample seed (one per independent pretraining seed) -- average over
    # pretrain_seed first so the paired test compares one value per subsample
    # seed against the baseline's one value per subsample seed.
    def per_subsample_seed_series(method_name, cls):
        sub = df[(df["class"] == cls) & (df["method"] == method_name)]
        if "pretrain_seed" in sub.columns and sub["pretrain_seed"].notna().any():
            sub = sub.groupby("seed")["auroc"].mean().sort_index()
            return sub.values
        return sub.sort_values("seed")["auroc"].values

    sig_report = []
    for c in present_classes:
        a = per_subsample_seed_series("contrastive_full_concat", c)
        best_baseline_name, best_val = None, -1
        for m in ["neighbor_connectivity", "raw_ecfp4_RF", "raw_expression_RF", "raw_concat_RF"]:
            sub = df[(df["class"] == c) & (df["method"] == m)]
            if len(sub) and sub["auroc"].mean() > best_val:
                best_val = sub["auroc"].mean()
                best_baseline_name = m
        b = per_subsample_seed_series(best_baseline_name, c)
        if len(a) == len(b) and len(a) > 1 and not np.allclose(a, b):
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        sig_report.append({"class": c, "contrastive_mean": a.mean() if len(a) else np.nan,
                            "best_baseline": best_baseline_name,
                            "best_baseline_mean": b.mean() if len(b) else np.nan, "wilcoxon_p": p})
    sig_df = pd.DataFrame(sig_report)
    sig_df.to_csv(f"{RESULTS}/significance.csv", index=False)
    print(sig_df.to_string())

    # ---------------- Core finding: full vs class-holdout, per view ----------------
    coldstart_sig = []
    for c in present_classes:
        for view in ["structure", "expression", "concat"]:
            full_v = per_subsample_seed_series(f"contrastive_full_{view}", c)
            cold_v = per_subsample_seed_series(f"contrastive_classholdout_{view}", c)
            if len(full_v) == len(cold_v) and len(full_v) > 1 and not np.allclose(full_v, cold_v):
                try:
                    stat, p = wilcoxon(full_v, cold_v)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            coldstart_sig.append({
                "class": c, "view": view,
                "full_mean": full_v.mean() if len(full_v) else np.nan,
                "classholdout_mean": cold_v.mean() if len(cold_v) else np.nan,
                "classholdout_minus_full": (cold_v.mean() - full_v.mean()) if len(full_v) and len(cold_v) else np.nan,
                "wilcoxon_p": p,
            })
    coldstart_sig_df = pd.DataFrame(coldstart_sig)
    coldstart_sig_df.to_csv(f"{RESULTS}/coldstart_significance.csv", index=False)
    print("\n=== Cold-start vs full, per view (core finding) ===")
    print(coldstart_sig_df.to_string())

    # ---------------- Figures ----------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(present_classes))
    width = 0.13
    for i, m in enumerate(MAIN_METHODS):
        means, stds = [], []
        for c in present_classes:
            sub = df[(df["class"] == c) & (df["method"] == m)]
            means.append(sub["auroc"].mean() if len(sub) else np.nan)
            stds.append(sub["auroc"].std() if len(sub) else 0)
        ax.bar(x + (i - len(MAIN_METHODS) / 2) * width, means, width, yerr=stds,
               label=METHOD_LABEL[m], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABEL[c] for c in present_classes], rotation=20)
    ax.set_ylabel("AUROC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Downstream class recovery by method")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/main_results.pdf")
    fig.savefig(f"{FIGS}/main_results.png", dpi=150)
    print(f"\nSaved figures to {FIGS}/main_results.{{pdf,png}}")

    # cold-start figure
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    regimes = list(COLDSTART_METHODS.keys())
    for i, r in enumerate(regimes):
        means = []
        for c in present_classes:
            method = COLDSTART_METHODS[r]
            if method is not None:
                sub = df[(df["class"] == c) & (df["method"] == method)]
            else:
                sub = df[(df["class"] == c) & (df["method"].isin(rh_cols))]
            means.append(sub["auroc"].mean() if len(sub) else np.nan)
        ax2.bar(x + (i - len(regimes) / 2) * 0.25, means, 0.25, label=r)
    ax2.set_xticks(x)
    ax2.set_xticklabels([CLASS_LABEL[c] for c in present_classes], rotation=20)
    ax2.set_ylabel("AUROC")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=8)
    ax2.set_title("Cold-start generalization gap")
    fig2.tight_layout()
    fig2.savefig(f"{FIGS}/coldstart.pdf")
    fig2.savefig(f"{FIGS}/coldstart.png", dpi=150)
    print(f"Saved figures to {FIGS}/coldstart.{{pdf,png}}")


if __name__ == "__main__":
    main()
