# Model-Class Mismatch and Aggregation Hide Transcriptomic Signal

Code, result logs and figure scripts for the MAI-DRS 2026 paper *Model-Class
Mismatch and Aggregation Hide Transcriptomic Signal in LINCS Drug Repurposing*
(workshop at IEEE BIBM 2026).

The paper is a negative result with a self-audit, so this repository is
organised around making each quoted number traceable rather than around a
reusable library.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | Dataset construction, every experiment, and the figure scripts |
| `results/` | Every log and CSV the paper quotes |
| `figures/` | The three paper figures as PDF and PNG, plus recomputed statistics |
| `paper/` | LaTeX source and the IEEE conference class |

## Running the experiments

Each script writes its own log and CSV into `results/` and can be run on its
own. Set `MAIDRS_ROOT` if the tree is checked out somewhere other than the
original path, and `MAIDRS_JOBS` to change how many cores the sweeps use.

```bash
cd code
python run_experiments.py          # Table I, contrastive model and raw baselines
python run_strong_baselines.py     # Table II, four stronger structural methods
python run_finlayson_baseline.py   # Table II, the published alignment objective
python run_expression_grid.py      # Figure 1, Table III corners, Table IV Phase 2
python run_blend.py                # Section VII, cross-modal blend
python run_sider.py                # Section VII, SIDER replication
python run_bio_interpretation.py   # Section V-A, gene-level analysis
python run_qc_variants.py          # QC-filtered representation sweeps
python phase1_replication.py       # extracts the Phase 1 landmark matrix
python run_phase1_analysis.py      # Section VI, Table IV Phase 1 columns
```

## Reproducing the figures

```bash
cd code
python make_fig1_conjunction.py   # reads results/grid_expression.csv
python make_fig2_methods.py       # reads results/summary_auroc.csv
python make_fig3_cancellation.py  # reads the Level-5 matrix, needs ~1 GB RAM
```

Each script prints the values the paper quotes from it, so a reader can check
the text against a fresh run rather than against the figure.

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
| Table II, Finlayson quadruplet objective | `results/finlayson_baseline.log` |
| Table III, 30-repeat corner cells | `results/stabilize.log` |
| Table IV, Phase 2 columns | `results/grid.log`, `results/grid_expression.csv` |
| Table IV, Phase 1 columns | `results/p1_analyses.log` |
| Variance decomposition, 73.3% and 11.5% | `results/grid.log`, recomputed by `make_fig1_conjunction.py` |
| Blend result, +0.0013 at p = 0.15 | `results/blend_proper.log` |
| SIDER nested blend | `results/sider_nested.log` |
| Responsive-subset blend caveat | `results/final_push.log`, `results/qc_final.log` |
| Gene-level statin and SSRI findings | `results/bio_interpretation.json` |
| Phase 1 extraction and HMGCR control | `results/p1_analyses.log`, `results/p1extract.log` |
| Signature cancellation statistics | `make_fig3_cancellation.py`, `figures/fig3_cancellation_stats.json` |

## Three notes on the numbers

**Repeat counts.** Figure 1 and the Phase 2 columns of Table IV use 6
subsampling repeats per cell. The four corner cells in Table III use 30 repeats
with bootstrap intervals, so the two differ by up to 0.023 for the same cell.
The largest gap is the naive-mean/random-forest cell, 0.525 in Table IV against
0.548 in Table III; Table I reports the same cell at 0.507 over its 10 repeats.
All three are one estimator on the same seed sequence, and that cell has a
per-seed standard deviation of 0.060. Every corner value quoted in the abstract
and text comes from Table III.

**Subset aggregations fall back to the naive mean.** Any representation that
selects a subset of a compound's signatures (high dose, gold QC, per cell line)
has to decide what to do with compounds that have no qualifying signature: 49 of
1,725 for the high-dose subset, 50 for gold. An earlier version of this pipeline
left an all-zero vector behind, which favoured the random forest because it can
split on the all-zero pattern. Every representation now falls back to that
compound's own naive mean. This is defect 5 in the paper's self-audit.

**Activity thresholds.** The two cuts in the cancellation analysis differ by
design. A signature is *strongly active* at 50 or more landmark genes with
|z| > 2, and *weak* below 25 such genes. Both are recorded in
`figures/fig3_cancellation_stats.json`.

## Citation

```
R. Kondadadi and D. McCreary, "Model-Class Mismatch and Aggregation Hide
Transcriptomic Signal in LINCS Drug Repurposing," in Proc. IEEE BIBM Workshop on Multi-Modal Artificial
Intelligence in Drug Discovery, Repurposing, and Safety (MAI-DRS), 2026.
```
