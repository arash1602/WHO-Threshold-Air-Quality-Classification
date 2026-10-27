# WHO-Threshold Air Quality Classification from Low-Cost Sensors

Code and results for **"WHO-Threshold Air Quality Classification from Low-Cost Sensors: A Within-City and Cross-City Benchmark of Deep Sequence Models"** (J. Thyssen and M. Ghamari).

Acknowledgements: **Claude (Anthropic 2026) was leveraged for language refinement in the report as well as code optimization and code comments and this README.** 

The task is to predict whether the WHO 2021 24-hour mean limit for PM<sub>2.5</sub> (15 µg/m³) or PM<sub>10</sub> (45 µg/m³) will be exceeded. The prediction uses 24 hours of low-cost-sensor readings from the [SensEURCity](https://doi.org/10.5281/zenodo.17858205) campaign (Antwerp, Oslo and Zagreb), at forecast horizons of 1, 12 and 24 h. The benchmark covers three predictor types:

- **Deep sequence models:** a GRU, a TCN and a gated CNN–GRU hybrid (FUSION).
- **Zero-parameter WHO rule:** the WHO limits applied directly to the sensor's own readings.
- **Six-parameter logistic regression:** fitted on the same window means.

All are evaluated under leak-controlled nested cross-validation in two designs:
- **Single-City-Blocked:** purged, blocked folds within each city.
- **Multi-City:** leave-one-city-out.

## Repository layout

| Path | Role |
|---|---|
| `senseurcity_dataset.py` | Loads the SensEURCity CSVs: flag-based minute cleaning, hourly means, 24 h windows at 50 % overlap, WHO label, `logz` input scaling |
| `compare_models.py` | Fold geometry: 14-day blocks, serpentine dealing by positive rate, embargo `E = max(C + h, A)`, nested planners |
| `find_best_model.py` | Nested-CV grid search (32 configurations per architecture, re-run inside every outer fold), refit and outer-test scoring |
| `nested_worker.py`, `train_aqm.py` | GPU trainer (AdamW, class-weighted BCE, early stopping on validation loss), metrics, F2 threshold tuning |
| `AQM_GRU_Model.py`, `AQM_model.py` (TCN), `AQM_CNNGRU_Model.py` (FUSION) | The three architectures |
| `naive_baselines.py` | WHO rule and logistic regression on exactly the same folds |
| `make_tables.py` | Every results table in the paper (LaTeX) plus the statistics quoted in the text |
| `make_figures.py` | Figures 1–4 for both designs |
| `make_skew_figure.py` | Figure 5 (skewness of the PM channels before and after `log1p`) |
| `figures/pipeline/` | TikZ source of the pipeline diagram (compile `fig_pipeline_standalone.tex` with pdflatex or tectonic) |
| `test_nested_geometry.py` | Leak and embargo assertions on the fold plans and on materialized windows |
| `download_data.py` | Downloads and verifies the dataset from Zenodo |
| `reproduce.py` | Runs any or all of the stages below |
| `tuning_results/h{1,12,24}/` | The sweep results the paper reports (see [Shipped results](#shipped-results)) |
| `tables/`, `figures/h1-12-24/`, `figures/fig5_*` | Output of `make_tables.py` and the figure scripts on the shipped results |

## Setup

Python 3.13; the exact package versions used are in `requirements.txt`.

```bash
pip install -r requirements.txt       # install a CUDA build of torch first if needed
python download_data.py               # 5.7 GB download, ~21 GB unpacked -> ./senseurcity_data_v03/
```

To keep the data elsewhere, run `python download_data.py --dest <dir>` and set `AQM_DATA_DIR=<dir>/dataset`, or pass `--data-dir` to any script. The data is SensEURCity version 3.0.0 ([doi:10.5281/zenodo.17858205](https://doi.org/10.5281/zenodo.17858205), CC-BY-4.0). Please cite Van Poppel et al., *Scientific Data* 10, 322 (2023), when you use it.

On first use each script parses the minute-resolution CSVs (about 8 minutes, with a transient peak of a few GB of RAM). The result is cached in `.parse_cache/`, so later runs load in about a second.

## Reproducing the paper

**From the shipped results (minutes, no GPU, no data except figure 5):**

```bash
python reproduce.py                   # -> tables/*.tex, tables/paper_numbers.txt, figures/
```

**From scratch (retrains every model):**

```bash
python reproduce.py --stages all --out-root my_results
```

`--out-root` writes to a fresh directory. Resuming into `tuning_results/` would find the shipped caches complete and retrain nothing.

The sweep is about 2.5 h per horizon, so about 7.5 h for all three, on an RTX 4060 laptop GPU with 3 concurrent workers (`--workers`, about 1.4 GB of host RAM each). Individual stages can be run with `--stages geometry|sweep|baselines|tables|figures` and `--horizons`.

The stages map to these commands:

```bash
python test_nested_geometry.py --materialize --horizon 1
python find_best_model.py --experiments single-blocked --workers 3 --deterministic --scaling logz --horizon 1 --out-dir tuning_results/h1
python find_best_model.py --experiments multi          --workers 3 --deterministic --scaling logz --horizon 1 --out-dir tuning_results/h1
python naive_baselines.py --horizon 1                  # repeat both for --horizon 12 and 24
python make_tables.py
python make_figures.py --experiment single
python make_figures.py --experiment multi
python make_skew_figure.py
```

### Where each result comes from

| Paper | Produced by | File |
|---|---|---|
| Table V, single-city splits | `make_tables.py` | `tables/tab_cv_splits.tex` |
| Table VI, leave-one-city-out splits | `make_tables.py` | `tables/tab_cv_splits_multi.tex` |
| Table VII, architecture comparison within city | `make_tables.py` | `tables/tab_arch_within_city.tex` |
| Table VIII, deep models vs. baselines (with paired t-tests) | `make_tables.py` | `tables/tab_baselines.tex` |
| Table IX, hyperparameter identifiability | `make_tables.py` | `tables/tab_identifiability.tex` |
| Table X, leave-one-city-out generalization | `make_tables.py` | `tables/tab_multi_city.tex` |
| Numbers quoted in the text (paired differences, rank agreement, per-fold spans, transfer losses, ...) | `make_tables.py` | `tables/paper_numbers.txt` |
| Tables I–IV, fixed settings and search grids | constants in `find_best_model.py` (`NN_SPECS`, `EPOCHS`, `PATIENCE`, `DEFAULT_BATCH`, `SEED`, ...) | – |
| Fig. 1, pipeline diagram | TikZ | `figures/pipeline/` |
| Fig. 2, PM skewness | `make_skew_figure.py` | `figures/fig5_pm_skewness.pdf` |
| Fig. 3, AP by architecture (single city) | `make_figures.py --experiment single` | `figures/h1-12-24/fig1_pr_auc_by_arch_single.pdf` |
| Fig. 4, AP vs. prevalence (single city) | `make_figures.py --experiment single` | `figures/h1-12-24/fig2_prevalence_single.pdf` |
| Fig. 5, AP vs. parameter count | `make_figures.py --experiment single` | `figures/h1-12-24/fig4_pareto_single.pdf` |
| Fig. 6, AP vs. prevalence (leave-one-city-out) | `make_figures.py --experiment multi` | `figures/h1-12-24/fig2_prevalence_multi.pdf` |

`make_figures.py` also writes the figures the paper does not include, for both designs:
- AP by architecture and AP against parameter count for leave-one-city-out
- the threshold sweep with confusion matrices (`fig3_threshold_<arch>_<design>`; `--threshold-arch` picks the architecture)

## Shipped results

`tuning_results/h<H>/` holds, for each horizon, what the sweep and the baseline script wrote. Only the files needed to rebuild every table and figure are kept:

| File | Contents |
|---|---|
| `nested5_<arch>_<design>_ctx24_blocked.jsonl` | One `meta` line (every setting); one `inner` record per (outer fold, configuration) with the three inner-validation scores; one `outer` record per outer fold with the winner, its tuned threshold and every test metric |
| `nested5_*_inner.csv`, `nested5_*_outer.csv` | The same records as flat tables |
| `nested5_*_preds/<outer>.npz` | The refit model's validation and test logits and labels (used by the threshold figure) |
| `best_<arch>_<design>_ctx24_blocked_logz.json` | Pooled configuration, winner agreement, fold winners and nested-test means |
| `summary_nested5_ctx24.md` | The sweep's summary table |
| `baselines_ctx24.json`, `baselines_ctx24.md` | WHO-rule and logistic-regression results per fold |

A sweep also writes per-training artifacts, which are not shipped (see `.gitignore`):
- model weights
- inner-fold logits and learning curves
- per-fold window labels

`nested5` denotes cache schema 5, which is `logz` input scaling: `log1p` on the two OPC PM channels, then per-channel standardization fitted on each training split.

## Determinism

- **Fixed seeds:** every training uses seed 42.
- **Deterministic GPU kernels:** the sweep runs with `--deterministic`, so repeated runs on the same hardware and software stack reproduce every record, which is what the paper's numbers come from.
- **Other setups may differ:** a different GPU, driver or library version can change floating-point results slightly. That can flip a near-tied model selection, so expect small deviations there rather than bit-identical numbers.
- **Worker count doesn't matter:** `--workers` changes only wall-clock time, never results.

## Data license

The SensEURCity dataset is CC-BY-4.0, from its authors; it is not redistributed here.

## Revision analyses

Added for the journal revision. They reuse the fold plans and caches above, so every number is paired fold-by-fold with the deep-model results.

| Script | What it adds | Output |
| --- | --- | --- |
| `extra_baselines.py --horizon H` | Gradient-boosted trees on 30 window statistics, linear calibration followed by the WHO rule, and the logistic regression recomputed with per-window scores saved. Same inner-fold selection, refit and single outer-test score as the networks. Needs the dataset; CPU only, about 1 to 2 minutes per outer fold. | `tuning_results/h<H>/extra_baselines_ctx24.json`, `tuning_results/h<H>/extra_preds/` |
| `stats_extra.py` | Nadeau–Bengio corrected resampled t-test, exact Wilcoxon test and Holm adjustment for the within-city comparisons; calibration (Brier, ECE) of the deep models. No data needed. | `tables/stats_extra.txt`, `tables/tab_stats_corrected.tex` |
| `bootstrap_ci.py` | Day-block bootstrap 95 % intervals for each predictor's AP and its difference from the logistic regression and the trees, within each held-out city. | `tables/bootstrap_ci.{txt,json}` |
| `revision_numbers.py` | Every number the revision quotes: baseline means, all paired tests, multi-city tables, bootstrap intervals, which pollutant drives the label, calibration. | `tables/revision_numbers.txt` |

```
python extra_baselines.py --horizon 1     # repeat for 12 and 24
python bootstrap_ci.py
python revision_numbers.py
```

`make_figures.py` also had a bug fix: the prevalence figure's panels share one x-axis, and each panel used to set its own limit, so the last panel's range clipped high-prevalence folds (Zagreb-b1, prevalence 0.58) from the others.
