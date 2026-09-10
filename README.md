# How We Nearly Wrote Off a Modality

Code, result logs and figure scripts for the MAI-DRS 2026 paper *How We Nearly
Wrote Off a Modality: Model-Class Mismatch, Aggregation, and What Survived
Replication in Structure-Transcriptome Drug Repurposing* (workshop at IEEE BIBM
2026).

The paper is a negative result with a self-audit, so this repository is
organised around making each quoted number traceable rather than around a
reusable library.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Dataset construction, experiments, and figure scripts |
| `results/` | Every log and CSV the paper quotes from |
| `figures/` | The three paper figures as PDF and PNG, plus recomputed statistics |
| `paper/` | LaTeX source and the IEEE conference class |

## Reproducing the figures

The three figure scripts read from `results/` and, for Figure 3, from the
Level-5 landmark matrix built by `code/build_dataset.py`.

```bash
cd code
python make_fig1_conjunction.py   # reads results/grid_expression.csv
python make_fig2_methods.py       # reads results/summary_auroc.csv
python make_fig3_cancellation.py  # reads the Level-5 matrix, needs ~1 GB RAM
```

Each script prints the values the paper quotes from it, so a reader can check
the text against a fresh run rather than against the figure. Set `MAIDRS_ROOT`
if the tree is checked out somewhere other than the original path.

Figure 3 also writes `figures/fig3_cancellation_stats.json`, which records the
signature-magnitude statistics and the exact activity thresholds behind them.

## Rebuilding the dataset

`code/build_dataset.py` expects the raw LINCS releases, which are too large to
distribute here. Both are public:

- Phase 2, GSE70138: Level-5 matrix, `sig_info`, `gene_info`, `sig_metrics`
- Phase 1, GSE92742: Level-5 matrix, `sig_info`, `sig_metrics`

SMILES are resolved from PubChem by `code/fetch_smiles.py`. Side-effect labels
come from SIDER 4.1 (`meddra_all_se.tsv.gz`).

## Where each claim comes from

| Paper element | Source |
| --- | --- |
| Table I, main class recovery | `results/summary_auroc.csv`, `results/agg_test.log` |
| Table I significance, p <= 0.027 | `results/significance.csv` |
| Table II, stronger structural baselines | `results/strong_baselines.log` |
| Table III, 30-repeat corner cells | `results/stabilize.log` |
| Table IV, Phase 2 columns | `results/grid.log`, `results/grid_expression.csv` |
| Table IV, Phase 1 columns | `results/p1_analyses.log` |
| Variance decomposition, 68.3% and 12.5% | `results/grid.log`, recomputed by `make_fig1_conjunction.py` |
| Blend result, +0.0018 at p = 0.31 | `results/blend_proper.log` |
| SIDER nested blend, -0.018 | `results/sider_nested.log` |
| SIDER responsive/gold replication | `results/final_push.log` |
| Gene-level statin and SSRI findings | `results/bio_interpretation.json` |
| Phase 1 extraction and HMGCR control | `results/p1extract.log` |
| Signature cancellation statistics | `make_fig3_cancellation.py`, `figures/fig3_cancellation_stats.json` |

## Two notes on the numbers

Figure 1 and the Phase 2 columns of Table IV use 6 subsampling repeats per
cell. The four corner cells in Table III use 30 repeats with bootstrap
intervals, so the two differ by up to 0.015 for the same cell. Every corner
value quoted in the abstract and text comes from Table III.

The activity thresholds in Section V differ by design. A signature is
*strongly active* at 50 or more landmark genes with |z| > 2, and *weak* below
25 such genes. Both cuts are recorded in
`figures/fig3_cancellation_stats.json`.

## Citation

```
R. Kondadadi, "How We Nearly Wrote Off a Modality: Model-Class Mismatch,
Aggregation, and What Survived Replication in Structure-Transcriptome Drug
Repurposing," in Proc. IEEE BIBM Workshop on Multi-Modal Artificial
Intelligence in Drug Discovery, Repurposing, and Safety (MAI-DRS), 2026.
```
