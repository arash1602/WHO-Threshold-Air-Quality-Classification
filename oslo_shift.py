"""Why the linear models fail on held-out Oslo: covariate shift and linear extrapolation.

    python oslo_shift.py          # -> tables/oslo_shift.txt   (needs the dataset)

Diagnostic, not part of the benchmark: models here are fitted with fixed settings
(logistic regression C = 1; trees: 100 trees, 7 leaves, minimum leaf 200, learning
rate 0.05, L2 1) on the same refit split the leave-one-city-out baselines use, and
scored on held-out Oslo, so that only the input set changes between rows.

Reports, at h = 1 and 12:
* how far Oslo's input distribution lies outside the training cities' (percentiles,
  share of Oslo windows colder than the training 5th percentile, prevalence there);
* single-channel ranking of Oslo's PM2.5-driven exceedances (ROC-AUC, < 0.5 means
  the channel ranks them in reverse);
* linear versus tree models on nested input sets, from optical PM2.5 alone to all five.

Window means are cached in tables/oslo_shift_h<H>.npz after the first run.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import naive_baselines as nb
import extra_baselines as eb
from compare_models import nested_plan

FEAT = list(nb.INPUT_FEATURES)
SETS = (("OPCN3PM25",), ("OPCN3PM25", "OPCN3PM10"), ("OPCN3PM25", "SHT31TE"),
        ("OPCN3PM25", "OPCN3PM10", "SHT31TE"), tuple(nb.LR_FEATURES))
SHORT = {"OPCN3PM25": "PM2.5", "OPCN3PM10": "PM10", "SHT31TE": "T", "SHT31HE": "RH",
         "Absolute_humidity": "AH"}


def arrays(data_dir, h, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["tr"], z["ytr"], z["te"], z["y"], z["pm"]
    common = dict(context_size=nb.CONTEXT_SIZE, overlap=nb.OVERLAP, horizon=h)
    spec = next(s for s in nested_plan(data_dir, experiment="multi", protocol="blocked",
                                       cities=list(nb.CITIES), k_outer=nb.K_OUTER,
                                       n_inner=nb.N_INNER, block_hours=nb.BLOCK_HOURS, **common)
                if s.outer_id == "heldout-Oslo")
    te = eb.window_arrays(nb._split(data_dir, spec.test_cities, bands=spec.test_bands, **common))
    isp = next(i for i in spec.inner if i.idx == spec.refit_idx)
    tr = eb.window_arrays(nb._split(data_dir, spec.train_cities, bands=isp.train_bands, **common))
    np.savez_compressed(cache, tr=tr[1], ytr=tr[2], te=te[1], y=te[2], pm=te[4])
    return tr[1], tr[2], te[1], te[2], te[4]


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR",
                    os.path.join(here, "senseurcity_data_v03", "dataset")))
    args = ap.parse_args()
    out = []
    say = out.append
    for h in (1, 12):
        nb.HORIZON = h
        tr, ytr, te, y, pm = arrays(args.data_dir, h,
                                    os.path.join(here, "tables", f"oslo_shift_h{h}.npz"))
        pm25 = pm[:, 1] == 1
        neg = y == 0

        def sub(s):
            return roc_auc_score(np.r_[np.ones(pm25.sum()), np.zeros(neg.sum())],
                                 np.r_[s[pm25], s[neg]])

        t5 = np.percentile(tr[:, FEAT.index("SHT31TE")], 5)
        cold = te[:, FEAT.index("SHT31TE")] < t5
        ratio = lambda m: m[:, FEAT.index("OPCN3PM10")] / np.maximum(m[:, FEAT.index("OPCN3PM25")], 0.1)
        say(f"h={h}: train {len(ytr)} windows (Antwerp+Zagreb), Oslo test {len(y)}, prevalence {y.mean():.3f}")
        say(f"  Oslo windows colder than the training 5th percentile ({t5:.1f} C): {cold.mean():.1%}; "
            f"prevalence there {y[cold].mean():.3f}")
        tmin = tr[:, FEAT.index("SHT31TE")].min()
        below = te[:, FEAT.index("SHT31TE")] < tmin
        say(f"  Oslo windows colder than the coldest training window ({tmin:.1f} C): {below.mean():.1%}; "
            f"prevalence there {y[below].mean():.3f}; share of Oslo exceedances {y[below].sum() / y.sum():.1%}")
        say(f"  optical PM10/PM2.5 ratio, median: train {np.median(ratio(tr)):.2f}, Oslo {np.median(ratio(te)):.2f}")
        say("  channel   pct 5/50/95 train | Oslo          AUC Oslo PM2.5-driven | AUC train (all)")
        for j, f in enumerate(FEAT):
            say(f"  {SHORT[f]:6s} {np.percentile(tr[:, j], [5, 50, 95]).round(1)} | "
                f"{np.percentile(te[:, j], [5, 50, 95]).round(1)}   {sub(te[:, j]):.3f} | {roc_auc_score(ytr, tr[:, j]):.3f}")
        say("  inputs              linear AP / AUC     trees AP / AUC")
        for names in SETS:
            c = [FEAT.index(n) for n in names]
            sc, lr = nb.fit_logreg(tr[:, c], ytr, 1.0)
            s_lin = nb.lr_score(sc, lr, te[:, c])
            g = HistGradientBoostingClassifier(class_weight="balanced", max_iter=100,
                                               max_leaf_nodes=7, min_samples_leaf=200,
                                               learning_rate=0.05, l2_regularization=1.0,
                                               early_stopping=False, random_state=42)
            s_tree = g.fit(tr[:, c], ytr).predict_proba(te[:, c])[:, 1]
            say(f"  {'+'.join(SHORT[n] for n in names):18s}  "
                f"{average_precision_score(y, s_lin):.3f} / {roc_auc_score(y, s_lin):.3f}      "
                f"{average_precision_score(y, s_tree):.3f} / {roc_auc_score(y, s_tree):.3f}"
                f"   linear coef {np.round(lr.coef_[0], 2)}")
        say("")
    with open(os.path.join(here, "tables", "oslo_shift.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
