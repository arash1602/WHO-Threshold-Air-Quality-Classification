"""Stronger baselines on the nested folds, added for the revision.

    python extra_baselines.py --horizon 1     # -> tuning_results/h1/extra_baselines_ctx24.json
                                              #    tuning_results/h1/extra_preds/<design>/<outer>.npz

Every predictor is scored on exactly the folds ``find_best_model`` and
``naive_baselines`` score (same plans, same bands, same windows), and selected by
the same inner-fold protocol: the configuration with the highest mean inner-fold
validation average precision (ties to the lower std), a decision threshold at the
median of its inner-fold F2-maximising thresholds, a refit on the ``refit_idx``
inner training split, and a single score on the outer test. Only the input
window's 24 hours are used.

Predictors
    gbm       Histogram gradient-boosted trees (scikit-learn, LightGBM-style) on 30
              window statistics: per channel the mean, standard deviation, minimum,
              maximum, last value and last-minus-first over the input window.
              ``class_weight="balanced"``; 8-point grid over tree count, leaf
              count and minimum leaf size (learning rate 0.05, L2 penalty 1).
    reg_lin   Regression then threshold, linear. Ridge regression of
              log(1 + reference 24 h mean) for each pollutant on the five window
              means (the classic linear LCS calibration), then the WHO rule on the
              calibrated estimates: score = max(PM2.5 / 15, PM10 / 45).
    logreg    The paper's logistic regression, recomputed here so its per-window
              test scores can be saved for the bootstrap; its average precision is
              checked against the shipped ``baselines_ctx24.json``.

Per-window test scores, labels, per-pollutant labels, window times and cities are
saved to ``extra_preds/`` for ``bootstrap_ci.py``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import naive_baselines as nb
from compare_models import nested_plan

# Regularization is searched as well as capacity: on these blocked folds the test
# block often lies partly outside the training ranges, where unregularized trees
# overfit (training AP ~0.99, test far lower), so the grid must let selection
# choose large leaves and few, shallow trees if the inner folds prefer them.
GBM_GRID = [dict(learning_rate=0.05, max_leaf_nodes=leaves, max_iter=iters,
                 min_samples_leaf=msl, l2_regularization=1.0)
            for leaves, iters, msl in itertools.product((7, 31), (100, 300), (20, 200))]
RIDGE_GRID = [dict(alpha=a) for a in (0.1, 1.0, 10.0, 100.0)]
STAT_NAMES = ("mean", "std", "min", "max", "last", "delta")
SEED = nb.SEED


# --------------------------------------------------------------------------- #
# Features and targets from one split's raw windows
# --------------------------------------------------------------------------- #
def window_arrays(ds):
    """``(stats, means, y, ref_means, pm_labels)`` for every window of ``ds``.

    stats      (n, 30)  six statistics per input channel over the input window
    means      (n, 5)   window means (the logistic regression's features)
    y          (n,)     the WHO label, identical to ``ds.compute_labels()``
    ref_means  (n, 2)   reference 24 h means over the label window, in
                        ``ds.output_features`` order
    pm_labels  (n, 2)   per-pollutant exceedance, same order
    """
    C = ds.context_size
    n = len(ds._windows)
    k = len(ds.input_features)
    stats = np.empty((n, 6 * k))
    ref = np.empty((n, len(ds.output_features)))
    for i, w in enumerate(ds._windows):
        X, Y = ds._series[w.series_idx]
        x = X[w.start: w.start + C].astype(np.float64)
        stats[i] = np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0),
                                   x[-1], x[-1] - x[0]])
        lo = w.start + ds._label_lo_rel
        ref[i] = Y[lo: lo + ds.label_hours].mean(axis=0)
    limits = np.asarray(ds._limits, dtype=np.float64)
    pm = (ref > limits).astype(np.uint8)
    y = pm.any(axis=1).astype(np.uint8)
    if not np.array_equal(y, ds.compute_labels().astype(np.uint8)):
        raise SystemExit("re-derived labels disagree with compute_labels(); aborting")
    return stats, stats[:, :k], y, ref, pm


def exceed_score(pred_ref, ds):
    """WHO exceedance ratio of calibrated estimates (> 1 is the rule)."""
    limits = np.asarray(ds._limits, dtype=np.float64)
    return np.max(np.clip(pred_ref, 0, None) / limits, axis=1)


# --------------------------------------------------------------------------- #
# Model families: fit on a training split, return a test-score function
# --------------------------------------------------------------------------- #
def fit_gbm(tr, cfg):
    stats, _m, y, _r, _p = tr
    clf = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                         random_state=SEED, **cfg)
    clf.fit(stats, y)
    return lambda d: clf.predict_proba(d[0])[:, 1]


def make_reg(kind, ds_ref):
    def fit(tr, cfg):
        stats, means, _y, ref, _p = tr
        target = np.log1p(np.clip(ref, 0, None))
        if kind == "lin":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(means)
            models = [Ridge(**cfg).fit(sc.transform(means), target[:, j])
                      for j in range(target.shape[1])]
            feats = lambda d: sc.transform(d[1])
        else:
            models = [HistGradientBoostingRegressor(early_stopping=False, random_state=SEED,
                                                    **cfg)
                      .fit(stats, target[:, j]) for j in range(target.shape[1])]
            feats = lambda d: d[0]

        def score(d):
            f = feats(d)
            pred = np.column_stack([np.expm1(m.predict(f)) for m in models])
            return exceed_score(pred, ds_ref)
        return score
    return fit


def fit_logreg(tr, cfg):
    _s, means, y, _r, _p = tr
    cols = [list(nb.INPUT_FEATURES).index(f) for f in nb.LR_FEATURES]
    sc, lr = nb.fit_logreg(means[:, cols], y, cfg["C"])
    return lambda d: nb.lr_score(sc, lr, d[1][:, cols])


# --------------------------------------------------------------------------- #
def select_and_score(fit, grid, inner, refit, test):
    """Inner-fold selection, refit, one outer-test score. Mirrors the tuner."""
    per = []
    for cfg in grid:
        aps, thrs = [], []
        for tr, va in inner:
            s = fit(tr, cfg)(va)
            aps.append(nb.ranking_metrics(va[2], s)["pr_auc"])
            thrs.append(nb.fbeta_max_threshold(s, va[2])[0])
        per.append((float(np.nanmean(aps)), float(np.nanstd(aps)),
                    [t for t in thrs if np.isfinite(t)]))
    best = sorted(range(len(grid)), key=lambda i: (-per[i][0], per[i][1], i))[0]
    thr = float(np.median(per[best][2])) if per[best][2] else float("nan")
    s = fit(refit, grid[best])(test)
    y = test[2]
    out = {"config": grid[best], "inner_val_pr_auc_mean": per[best][0],
           "inner_val_pr_auc_std": per[best][1], "threshold": thr,
           **nb.ranking_metrics(y, s),
           **{f"{k}_at_threshold": v for k, v in nb.point_metrics(y, s >= thr).items()}}
    return out, s


def run(design, data_dir, horizon, out_dir):
    experiment, protocol = nb.EXPERIMENTS[design]
    common = {"context_size": nb.CONTEXT_SIZE, "overlap": nb.OVERLAP, "horizon": horizon}
    specs = []
    if experiment == "single":
        for c in nb.CITIES:
            specs += nested_plan(data_dir, experiment=experiment, protocol=protocol,
                                 cities=[c], k_outer=nb.K_OUTER, n_inner=nb.N_INNER,
                                 block_hours=nb.BLOCK_HOURS, **common)
    else:
        specs = nested_plan(data_dir, experiment=experiment, protocol=protocol,
                            cities=list(nb.CITIES), k_outer=nb.K_OUTER, n_inner=nb.N_INNER,
                            block_hours=nb.BLOCK_HOURS, **common)

    pred_dir = os.path.join(out_dir, "extra_preds", design)
    os.makedirs(pred_dir, exist_ok=True)
    rows = []
    for spec in specs:
        t0 = time.time()
        test_ds = nb._split(data_dir, spec.test_cities, bands=spec.test_bands, **common)
        test = window_arrays(test_ds)
        if len(np.unique(test[2])) < 2:
            continue
        inner, refit = [], None
        for isp in spec.inner:
            tr_ds = nb._split(data_dir, spec.train_cities, bands=isp.train_bands, **common)
            va_ds = nb._split(data_dir, spec.train_cities, bands=isp.val_bands, **common)
            tr, va = window_arrays(tr_ds), window_arrays(va_ds)
            if len(np.unique(tr[2])) < 2 or len(np.unique(va[2])) < 2:
                continue
            inner.append((tr, va))
            if isp.idx == spec.refit_idx:
                refit = tr
        if len(inner) < 2 or refit is None:
            continue

        row = {"design": design, "outer": spec.outer_id, "n_test": int(len(test[2])),
               "pos_rate": float(test[2].mean()),
               "pos_rate_by_pollutant": dict(zip(test_ds.output_features,
                                                 test[4].mean(0).round(4).tolist())),
               "both_exceed_rate": float((test[4].sum(1) == 2).mean())}
        scores = {}
        families = {"logreg": (fit_logreg, [dict(C=c) for c in nb.LR_C_GRID]),
                    "gbm": (fit_gbm, GBM_GRID),
                    "reg_lin": (make_reg("lin", test_ds), RIDGE_GRID)}
        for name, (fit, grid) in families.items():
            row[name], scores[name] = select_and_score(fit, grid, inner, refit, test)
        np.savez_compressed(os.path.join(pred_dir, f"{spec.outer_id}.npz"),
                            y=test[2], pm=test[4], ref=test[3],
                            times=test_ds.window_times().asi8,
                            cities=test_ds.window_cities().astype(str),
                            **{f"score_{k}": v for k, v in scores.items()})
        rows.append(row)
        print(f"  {spec.outer_id:16s} n={row['n_test']:5d} pos={row['pos_rate']:.3f} | "
              + " ".join(f"{k} {row[k]['pr_auc']:.3f}" for k in families)
              + f" | {time.time() - t0:.0f}s", flush=True)
    return rows


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR",
                    os.path.join(here, "senseurcity_data_v03", "dataset")))
    ap.add_argument("--designs", nargs="+", default=list(nb.EXPERIMENTS))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    nb.HORIZON = args.horizon
    out_dir = args.out_dir or os.path.join(here, "tuning_results", f"h{args.horizon}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for design in args.designs:
        print(f"h={args.horizon} {design}", flush=True)
        rows += run(design, args.data_dir, args.horizon, out_dir)
    path = os.path.join(out_dir, f"extra_baselines_ctx{nb.CONTEXT_SIZE}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
