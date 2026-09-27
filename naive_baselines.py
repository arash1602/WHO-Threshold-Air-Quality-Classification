"""Naive baselines on the nested folds: does the deep learning earn its keep? (R17)

The label is an exceedance of a WHO 24-hour limit measured on the *reference*
instrument, and at ``horizon = 1`` its 24-hour window overlaps 23 of the 24 input
hours. The task is therefore much closer to "calibrate the low-cost sensor's
running 24-hour mean" than to forecasting, so a trivial predictor built from the
same window is the floor any model has to clear. Without that floor a nested test
PR-AUC of 0.75 means nothing on its own.

Both baselines are scored on **exactly** the folds ``find_best_model`` scores, with
the same geometry (context 24, horizon 1, overlap 0.5, 25 h embargo) and the same
inner/outer structure, so every number here is paired fold-by-fold with a
``nested{4,5}_*`` result. Labels are re-derived from the reference channels and
checked against the stored ``*_folds/<outer>.npz`` labels; a mismatch aborts rather
than reporting a plausible wrong number.

Baseline 1 -- WHO rule (**zero parameters**, nothing is fitted)
    Average the LCS PM channels over the input window, in physical ug/m3, and
    apply the WHO limits directly:

        harmful  <=>  mean(OPCN3PM25) > 15  or  mean(OPCN3PM10) > 45

    Reported at that fixed operating point (accuracy, precision, recall, F1, F2,
    confusion counts) and, because the models are compared on PR-AUC, also as a
    *ranking* of the same quantity via the exceedance ratio

        score = max(mean(OPCN3PM25) / 15, mean(OPCN3PM10) / 45)

    which mirrors the label's any-pollutant OR and still fits nothing. score > 1
    is exactly the rule above.

Baseline 2 -- logistic regression on window means (one hyperparameter, C)
    Five features: the window means of the two OPC PM channels plus the three
    meteorological channels (temperature, relative humidity, absolute humidity),
    which are the covariates LCS calibration actually depends on. Standardized on
    the training split only.

    It mirrors the model protocol rather than competing with it unfairly: C is
    chosen by mean inner-fold validation PR-AUC over the same inner folds, the
    decision threshold is the median of the winning C's inner-fold F2-maximising
    thresholds, and the chosen C is refit on the same ``refit_idx`` split the
    models refit on and scored **once** on the outer test. ``class_weight="balanced"``
    plays the role of the trainer's ``pos_weight``.

Outputs (to ``--out-dir``, default ``tuning_results/h<horizon>/``):
    baselines_ctx24.json   per-fold rows for both baselines
    baselines_ctx24.md     the tables, including a paired comparison against any
                           nested{schema} caches present

``--horizon`` must match the sweep being compared against: it changes the label
window, the usable windows and the embargo, so a baseline computed at one horizon
says nothing about a sweep run at another. Each horizon has its own directory.

Run:
    python naive_baselines.py                          # both experiments, h = 1
    python naive_baselines.py --horizon 24             # against tuning_results/h24/
    python naive_baselines.py --experiments single-blocked
    python naive_baselines.py --compare-schemas 4 5    # also pair against minmax runs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from compare_models import DEFAULT_BLOCK_HOURS, _resolve_embargo, nested_plan
from senseurcity_dataset import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    WHO_24H_LIMITS,
    SenseurCityDataset,
)

# Geometry: must match find_best_model's constants, or the folds are not the ones
# the models were scored on and nothing here is comparable.
CONTEXT_SIZE = 24
HORIZON = 1
OVERLAP = 0.5
K_OUTER = 5
N_INNER = 3
BLOCK_HOURS = DEFAULT_BLOCK_HOURS
SEED = 42

EXPERIMENTS = {"single-blocked": ("single", "blocked"), "multi": ("multi", "blocked")}
CITIES = ("Antwerp", "Oslo", "Zagreb")

# Features for baseline 2: the PM pair plus the meteorological channels.
LR_FEATURES = ("OPCN3PM25", "OPCN3PM10", "SHT31TE", "SHT31HE", "Absolute_humidity")
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)

THRESHOLD_BETA = 2.0


# --------------------------------------------------------------------------- #
# Metrics (same conventions as train_aqm/_classification_metrics: an undefined
# precision, recall or F-score is 0.0, not NaN)
# --------------------------------------------------------------------------- #
def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = THRESHOLD_BETA) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denom) if denom else 0.0


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": f1, "f2": fbeta_from_counts(tp, fp, fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """PR-AUC and ROC-AUC, NaN on a single-class split (both are undefined there)."""
    if len(np.unique(y)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score))}


def fbeta_max_threshold(score: np.ndarray, y: np.ndarray,
                        beta: float = THRESHOLD_BETA) -> tuple[float, float]:
    """Threshold on ``score`` maximising F-beta, and that F-beta.

    Candidates are the observed scores, so the optimum over all thresholds is
    reached; ties go to the lower threshold (higher recall), matching the
    trainer's preference under beta > 1.
    """
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="stable")
    ys = y[order].astype(np.int64)
    tp = np.cumsum(ys)                          # predicting the top k as positive
    fp = np.cumsum(1 - ys)
    total_pos = int(ys.sum())
    fn = total_pos - tp
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + b2) * tp / denom, 0.0)
    k = int(np.argmax(f))
    return float(score[order][k]), float(f[k])


# --------------------------------------------------------------------------- #
# Fold construction: unnormalized windows, so the WHO limits apply directly
# --------------------------------------------------------------------------- #
def _split(data_dir: str, cities, *, bands=None, **common) -> SenseurCityDataset:
    """One split as raw (unnormalized) windows.

    ``normalize=False`` on purpose: baseline 1 compares against WHO limits in
    physical ug/m3, and baseline 2 fits its own standardization on its training
    split. The band filter and window geometry are identical to the tuner's.
    """
    if bands is None:
        return SenseurCityDataset(data_dir, cities=list(cities), time_range=(0.0, 1.0),
                                  normalize=False, **common)
    return SenseurCityDataset(data_dir, cities=list(cities), time_bands=list(bands),
                              normalize=False, **common)


def window_means(ds: SenseurCityDataset) -> np.ndarray:
    """Mean of every raw input channel over each window's **input** hours.

    Only the input window is used -- the same 24 h the models see -- so the
    baseline is not handed the extra label hour.
    """
    out = np.empty((len(ds._windows), len(ds.input_features)), dtype=np.float64)
    for i, w in enumerate(ds._windows):
        X, _Y = ds._series[w.series_idx]
        out[i] = X[w.start: w.start + ds.context_size].mean(axis=0)
    return out


def fold_data(ds: SenseurCityDataset) -> tuple[np.ndarray, np.ndarray]:
    return window_means(ds), ds.compute_labels().astype(np.uint8)


def check_labels(out_dir, experiment, protocol, outer_id, y, schemas) -> str:
    """Compare re-derived labels with a stored fold file; '' when none is present.

    The fold files are arch-independent and identical across scalings, so any
    schema's copy is a valid check.
    """
    for s in schemas:
        p = os.path.join(out_dir, f"nested{s}_{experiment}_ctx{CONTEXT_SIZE}_{protocol}_folds",
                         f"{outer_id}.npz")
        if not os.path.exists(p):
            continue
        stored = np.load(p)["test_y"]
        if len(stored) != len(y) or not np.array_equal(stored, y):
            raise SystemExit(
                f"\n{outer_id}: re-derived labels do not match {p}\n"
                f"  n={len(y)} vs {len(stored)}; the folds are not the ones the models "
                "were scored on, so no comparison here would be valid."
            )
        return f"nested{s}"
    return ""


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def who_rule(means: np.ndarray, feature_names) -> tuple[np.ndarray, np.ndarray]:
    """``(prediction, ratio score)`` for the zero-parameter WHO rule."""
    i25 = list(feature_names).index("OPCN3PM25")
    i10 = list(feature_names).index("OPCN3PM10")
    lim25 = WHO_24H_LIMITS["Ref.PM2.5"]
    lim10 = WHO_24H_LIMITS["Ref.PM10"]
    pm25, pm10 = means[:, i25], means[:, i10]
    score = np.maximum(pm25 / lim25, pm10 / lim10)      # > 1 is exactly the rule
    return ((pm25 > lim25) | (pm10 > lim10)).astype(np.uint8), score


def _lr_columns(feature_names) -> list[int]:
    return [list(feature_names).index(f) for f in LR_FEATURES]


def fit_logreg(Xtr, ytr, C: float):
    """Standardize on train, then fit a balanced logistic regression."""
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                            random_state=SEED)
    lr.fit(sc.transform(Xtr), ytr)
    return sc, lr


def lr_score(sc, lr, X) -> np.ndarray:
    return lr.predict_proba(sc.transform(X))[:, 1]


# --------------------------------------------------------------------------- #
def run_experiment(exp_key, data_dir, out_dir, cities, schemas) -> list[dict]:
    experiment, protocol = EXPERIMENTS[exp_key]
    common = {"context_size": CONTEXT_SIZE, "overlap": OVERLAP, "horizon": HORIZON}
    print(f"\n{'#' * 70}\n{exp_key}  (experiment={experiment}, protocol={protocol})\n{'#' * 70}")

    # Planned per city for single-city designs: nested_plan already loops that way,
    # so this is identical to the tuner's plan while parsing one city at a time.
    specs = []
    if experiment == "single":
        for c in cities:
            specs += nested_plan(data_dir, experiment=experiment, protocol=protocol,
                                 cities=[c], k_outer=K_OUTER, n_inner=N_INNER,
                                 block_hours=BLOCK_HOURS, **common)
    else:
        specs = nested_plan(data_dir, experiment=experiment, protocol=protocol,
                            cities=list(cities), k_outer=K_OUTER, n_inner=N_INNER,
                            block_hours=BLOCK_HOURS, **common)

    rows = []
    for spec in specs:
        test_ds = _split(data_dir, spec.test_cities,
                         bands=None if spec.test_bands is None else spec.test_bands,
                         **common)
        Xte, yte = fold_data(test_ds)
        checked = check_labels(out_dir, experiment, protocol, spec.outer_id, yte, schemas)
        if len(np.unique(yte)) < 2:
            print(f"  [skip] {spec.outer_id}: single-class test split")
            continue

        row = {"experiment": exp_key, "outer": spec.outer_id, "n_test": len(yte),
               "test_pos_rate": float(yte.mean()), "labels_checked_against": checked}

        # ---- Baseline 1: the WHO rule, nothing fitted ----------------------
        pred, score = who_rule(Xte, test_ds.input_features)
        row["who"] = {**point_metrics(yte, pred), **ranking_metrics(yte, score)}

        # ---- Baseline 2: logistic regression, C chosen on the inner folds ---
        cols = _lr_columns(test_ds.input_features)
        inner = []
        for isp in spec.inner:
            tr = _split(data_dir, spec.train_cities, bands=isp.train_bands, **common)
            va = _split(data_dir, spec.train_cities, bands=isp.val_bands, **common)
            Xtr, ytr = fold_data(tr)
            Xva, yva = fold_data(va)
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue        # same single-class screen the tuner applies
            inner.append((isp.idx, Xtr[:, cols], ytr, Xva[:, cols], yva))
        if len(inner) < 2:
            print(f"  [skip] {spec.outer_id}: fewer than 2 usable inner folds")
            continue

        per_c = {}
        for C in LR_C_GRID:
            aps, thrs = [], []
            for _idx, Xtr, ytr, Xva, yva in inner:
                sc, lr = fit_logreg(Xtr, ytr, C)
                p = lr_score(sc, lr, Xva)
                aps.append(ranking_metrics(yva, p)["pr_auc"])
                thrs.append(fbeta_max_threshold(p, yva)[0])
            per_c[C] = (float(np.nanmean(aps)), float(np.nanstd(aps)),
                        [t for t in thrs if np.isfinite(t)])
        # Same ranking key as the tuner: mean inner PR-AUC, then lower std, then
        # the earlier grid point.
        best_C = sorted(per_c, key=lambda c: (-per_c[c][0], per_c[c][1],
                                              LR_C_GRID.index(c)))[0]
        thr = float(np.median(per_c[best_C][2])) if per_c[best_C][2] else 0.5

        refit = next((e for e in inner if e[0] == spec.refit_idx), None)
        if refit is None:
            print(f"  [skip] {spec.outer_id}: refit fold {spec.refit_idx} unusable")
            continue
        sc, lr = fit_logreg(refit[1], refit[2], best_C)
        pte = lr_score(sc, lr, Xte[:, cols])
        row["logreg"] = {
            "C": best_C, "inner_val_pr_auc_mean": per_c[best_C][0],
            "inner_val_pr_auc_std": per_c[best_C][1], "threshold_prob": thr,
            "n_inner_used": len(inner), "refit_fold": spec.refit_idx,
            "coef": dict(zip(LR_FEATURES, lr.coef_[0].round(4).tolist())),
            **ranking_metrics(yte, pte),
            **{f"{k}_at05": v for k, v in point_metrics(yte, pte >= 0.5).items()},
            **{f"{k}_at_threshold": v for k, v in point_metrics(yte, pte >= thr).items()},
        }
        rows.append(row)
        print(f"  {spec.outer_id:16s} n={len(yte):5d} pos={yte.mean():.3f} "
              f"labels={checked or 'unchecked':8s} | WHO rule PR-AUC "
              f"{row['who']['pr_auc']:.3f} F2 {row['who']['f2']:.3f} | logreg "
              f"C={best_C:<5g} PR-AUC {row['logreg']['pr_auc']:.3f} "
              f"F2@0.5 {row['logreg']['f2_at05']:.3f} "
              f"F2@thr {row['logreg']['f2_at_threshold']:.3f}")
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _model_pr_auc(out_dir, schema, arch, experiment, protocol) -> dict:
    p = os.path.join(out_dir,
                     f"nested{schema}_{arch}_{experiment}_ctx{CONTEXT_SIZE}_{protocol}.jsonl")
    if not os.path.exists(p):
        return {}
    got = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "outer":
                got[r["outer"]] = r.get("test", {}).get("pr_auc", float("nan"))
    return got


def write_report(rows, out_dir, schemas, path) -> str:
    embargo = _resolve_embargo(None, {"context_size": CONTEXT_SIZE, "horizon": HORIZON})
    out = [f"# Naive baselines on the nested folds (ctx {CONTEXT_SIZE}, horizon {HORIZON}, "
           f"{embargo} h embargo)", "",
           "Both baselines use only the **input window**, the same 24 h the models see.",
           "Labels were re-derived and checked against the stored fold files.", "",
           "- **WHO rule** (zero parameters): `mean(OPCN3PM25) > 15 or mean(OPCN3PM10) > 45`.",
           "  Its ranking view scores `max(pm25/15, pm10/45)`, which fits nothing either.",
           "- **Logistic regression** (one hyperparameter): window means of "
           + ", ".join(f"`{f}`" for f in LR_FEATURES) + ",",
           "  standardized on train, `class_weight=\"balanced\"`; `C` chosen by mean inner-fold",
           "  validation PR-AUC, threshold = median of that `C`'s inner F2-max thresholds,",
           "  refit on `refit_idx` and scored once on the outer test.", ""]

    for exp_key in EXPERIMENTS:
        sub = [r for r in rows if r["experiment"] == exp_key]
        if not sub:
            continue
        experiment, protocol = EXPERIMENTS[exp_key]
        out += [f"## {exp_key}", "",
                "| fold | n test | pos rate | WHO PR-AUC | WHO F2 | WHO precision | WHO recall "
                "| logreg C | logreg PR-AUC | logreg F2 @0.5 | logreg F2 @thr |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in sub:
            w, g = r["who"], r["logreg"]
            out.append(f"| {r['outer']} | {r['n_test']} | {r['test_pos_rate']:.3f} "
                       f"| {w['pr_auc']:.3f} | {w['f2']:.3f} | {w['precision']:.3f} "
                       f"| {w['recall']:.3f} | {g['C']:g} | {g['pr_auc']:.3f} "
                       f"| {g['f2_at05']:.3f} | {g['f2_at_threshold']:.3f} |")

        def ms(vals):
            a = np.array(vals, dtype=float)
            return f"{np.nanmean(a):.3f} +/- {np.nanstd(a):.3f}"

        out += ["", f"**Means over {len(sub)} folds.** "
                f"WHO rule PR-AUC {ms([r['who']['pr_auc'] for r in sub])}, "
                f"F2 {ms([r['who']['f2'] for r in sub])}. "
                f"Logistic regression PR-AUC {ms([r['logreg']['pr_auc'] for r in sub])}, "
                f"F2 @0.5 {ms([r['logreg']['f2_at05'] for r in sub])}, "
                f"F2 @threshold {ms([r['logreg']['f2_at_threshold'] for r in sub])}.", ""]

        # Paired comparison against whatever sweeps are on disk.
        folds = [r["outer"] for r in sub]
        base = {"WHO rule": np.array([r["who"]["pr_auc"] for r in sub], dtype=float),
                "logreg": np.array([r["logreg"]["pr_auc"] for r in sub], dtype=float)}
        head = False
        for schema in schemas:
            for arch in ("GRU", "TCN", "FUSION"):
                got = _model_pr_auc(out_dir, schema, arch, experiment, protocol)
                if not all(f in got for f in folds):
                    continue
                if not head:
                    out += ["### Paired per-fold test PR-AUC vs the models", "",
                            "| sweep | arch | model | baseline | mean diff | model wins |",
                            "| --- | --- | --- | --- | --- | --- |"]
                    head = True
                m = np.array([got[f] for f in folds], dtype=float)
                for name, b in base.items():
                    d = m - b
                    out.append(f"| nested{schema} | {arch} | {np.nanmean(m):.3f} | "
                               f"{name} {np.nanmean(b):.3f} | {np.nanmean(d):+.3f} | "
                               f"{int((d > 0).sum())}/{len(d)} |")
        out.append("")

    text = "\n".join(out) + "\n"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def main():
    # Declared up front: --horizon's default *reads* HORIZON while the parser is
    # built, and Python rejects a global statement that follows a use of the name.
    global HORIZON

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR")
                   or os.path.join(here, "senseurcity_data_v03", "dataset"),
                   help="SenseurCity v03 dataset/ folder (default: $AQM_DATA_DIR, "
                        "else ./senseurcity_data_v03/dataset)")
    p.add_argument("--horizon", type=int, default=HORIZON,
                   help="label horizon in hours; must match the sweep being compared "
                        "against (default 1)")
    # Resolved after parsing, because the default depends on --horizon.
    p.add_argument("--out-dir", default=None,
                   help="default: tuning_results/h<horizon>/")
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                   choices=list(EXPERIMENTS))
    p.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    p.add_argument("--compare-schemas", nargs="+", type=int, default=[5],
                   help="cache schemas to pair against, when their caches are present")
    args = p.parse_args()
    if args.horizon < 0:
        p.error("--horizon must be >= 0")
    HORIZON = args.horizon
    if args.out_dir is None:
        args.out_dir = os.path.join(here, "tuning_results", f"h{HORIZON}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"horizon {HORIZON} h  ->  out_dir {args.out_dir}")
    print(f"inputs {INPUT_FEATURES}\nreference {OUTPUT_FEATURES}  limits {WHO_24H_LIMITS}")
    print(f"logreg features {LR_FEATURES}  C grid {LR_C_GRID}")

    rows = []
    for exp_key in args.experiments:
        rows += run_experiment(exp_key, args.data_dir, args.out_dir, args.cities,
                               args.compare_schemas)

    jpath = os.path.join(args.out_dir, f"baselines_ctx{CONTEXT_SIZE}.json")
    tmp = jpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, jpath)
    mpath = write_report(rows, args.out_dir, args.compare_schemas,
                         os.path.join(args.out_dir, f"baselines_ctx{CONTEXT_SIZE}.md"))
    print(f"\nwrote {jpath}\nwrote {mpath}")


if __name__ == "__main__":
    main()
