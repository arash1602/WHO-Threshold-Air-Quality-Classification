"""Day-block bootstrap confidence intervals for the leave-one-city-out results.

    python bootstrap_ci.py            # -> tables/bootstrap_ci.txt, tables/bootstrap_ci.json

With three held-out cities no fold-level test can reach significance, so the
uncertainty is estimated inside each held-out city instead. Test windows overlap
by 50 % and the city's devices share one reference series, so windows are not
independent; they are resampled in whole calendar days (UTC, from each window's
last touched hour), which keeps every device and every overlapping window of a
day together. For each resample the average precision of every predictor and its
difference from the logistic regression are recomputed; 95 % percentile
intervals are reported, for the difference from the logistic regression and from the gradient-boosted trees. Deep-model scores are the shipped refit test logits, and
their labels are checked window-for-window against the baseline predictions.

Calibration (Brier score, expected calibration error) of the logistic regression
and gradient-boosting baselines is reported alongside, from the same files.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

ARCHS = ("GRU", "TCN", "FUSION")
BASES = ("logreg", "gbm", "reg_lin")
HORIZONS = (1, 12, 24)
CITIES = ("Antwerp", "Oslo", "Zagreb")


def load_city(results_dir, h, city):
    d = os.path.join(results_dir, f"h{h}")
    z = np.load(os.path.join(d, "extra_preds", "multi", f"heldout-{city}.npz"))
    y = z["y"].astype(int)
    scores = {b: z[f"score_{b}"] for b in BASES}
    for a in ARCHS:
        p = np.load(os.path.join(d, f"nested5_{a}_multi_ctx24_blocked_preds",
                                 f"heldout-{city}.npz"))
        if not np.array_equal(p["test_y"].astype(int), y):
            raise SystemExit(f"h{h} {city} {a}: window labels differ from the baselines'")
        scores[a] = p["test_logits"].astype(float)
    # window_times() is datetime64[us] in this pandas version, so .asi8 is in
    # microseconds since the epoch.
    days = (z["times"] // (86_400 * 10**6)).astype(np.int64)
    if len(np.unique(days)) < 30:
        raise SystemExit(f"h{h} {city}: only {len(np.unique(days))} distinct days; "
                         "check the timestamp unit")
    return y, scores, days


def bootstrap(y, scores, days, B, rng):
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    names = list(scores)
    point = {n: average_precision_score(y, scores[n]) for n in names}
    draws = {n: [] for n in names}
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for n in names:
            draws[n].append(average_precision_score(yy, scores[n][idx]))
    draws = {n: np.array(v) for n, v in draws.items()}
    out = {}
    for n in names:
        lo, hi = np.percentile(draws[n], [2.5, 97.5])
        d = draws[n] - draws["logreg"]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        g = draws[n] - draws["gbm"]
        glo, ghi = np.percentile(g, [2.5, 97.5])
        out[n] = {"ap": point[n], "ci": [float(lo), float(hi)],
                  "diff_vs_lr": point[n] - point["logreg"],
                  "diff_ci": [float(dlo), float(dhi)],
                  "p_le_0": float((d <= 0).mean()),
                  "diff_vs_gbm": point[n] - point["gbm"],
                  "diff_gbm_ci": [float(glo), float(ghi)]}
    return out, len(uniq)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(here, "tuning_results"))
    ap.add_argument("--out", default=os.path.join(here, "tables"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--block-days", type=int, default=1,
                    help="resample blocks of this many consecutive calendar days (sensitivity to day-to-day dependence); output files get a _b<N>d suffix when N > 1")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    sfx = f"_b{args.block_days}d" if args.block_days > 1 else ""
    res, lines = {}, []
    for h in HORIZONS:
        for city in CITIES:
            y, scores, days = load_city(args.results_dir, h, city)
            if args.block_days > 1:
                days = (days - days.min()) // args.block_days
            r, n_days = bootstrap(y, scores, days, args.B, rng)
            res[f"h{h}-{city}"] = {"n_days": n_days, **r}
            lines.append(f"h={h:2d} {city:8s} ({n_days} days)")
            for n in list(ARCHS) + list(BASES):
                v = r[n]
                lines.append(f"   {n:8s} AP {v['ap']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
                             + ("" if n == "logreg" else
                                f"  vs LR {v['diff_vs_lr']:+.3f} "
                                f"[{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}]"))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"bootstrap_ci{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, f"bootstrap_ci{sfx}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
