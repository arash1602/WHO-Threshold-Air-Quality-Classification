"""Tests of the explanation for the linear models' failure on held-out Oslo.

    python oslo_mechanism.py        # -> tables/oslo_mechanism.txt, tables/oslo_mechanism.json

Claim under test: Oslo is the only city with a substantial share of PM10-only
exceedances; a linear model trained on Antwerp and Zagreb learns a negative weight
on the optical PM10 channel and so ranks those episodes below clean air.

Three predictions, none needing retraining:

1. Sign. The logistic regression's PM10 coefficient is negative when trained on
   the other two cities and positive when trained within Oslo (single-city folds).
2. Removal. Dropping Oslo's PM10-only positive windows from the held-out test set
   should remove most of the logistic regression's deficit there. Because removing
   positives lowers prevalence, AP is reported together with its ratio to prevalence
   (lift) and with ROC-AUC, which does not depend on prevalence.
3. Separation. Ranking quality is computed separately for PM2.5-driven positives
   (PM2.5 above its limit, PM10 either way) and PM10-only positives, each against
   all negatives, as ROC-AUC. The prediction is an inverted ranking (AUC < 0.5) of
   PM10-only positives by the linear models and not by the nonlinear ones.

For comparison the same subset analysis is run on Antwerp and Zagreb.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ARCHS = ("GRU", "TCN", "FUSION")
BASES = ("logreg", "reg_lin", "gbm")
LABEL = {"logreg": "LR", "reg_lin": "LinCal", "gbm": "GBT",
         "GRU": "GRU", "TCN": "TCN", "FUSION": "FUSION"}
HORIZONS = (1, 12, 24)
CITIES = ("Antwerp", "Oslo", "Zagreb")


def load(rd, h, city):
    z = np.load(os.path.join(rd, f"h{h}", "extra_preds", "multi", f"heldout-{city}.npz"))
    y, pm = z["y"].astype(int), z["pm"].astype(int)       # pm columns: (PM10, PM2.5)
    scores = {b: z[f"score_{b}"] for b in BASES}
    for a in ARCHS:
        p = np.load(os.path.join(rd, f"h{h}", f"nested5_{a}_multi_ctx24_blocked_preds",
                                 f"heldout-{city}.npz"))
        assert np.array_equal(p["test_y"].astype(int), y)
        scores[a] = p["test_logits"].astype(float)
    return y, pm, scores


def auc_subset(y_pos_mask, neg_mask, s):
    yy = np.r_[np.ones(y_pos_mask.sum()), np.zeros(neg_mask.sum())]
    return float(roc_auc_score(yy, np.r_[s[y_pos_mask], s[neg_mask]]))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "tuning_results")
    out, res = [], {}
    say = out.append

    # 1. Sign of the PM10 coefficient
    say("1. Logistic regression coefficient on optical PM10 (standardized)")
    res["coef"] = {}
    for h in HORIZONS:
        with open(os.path.join(rd, f"h{h}", "baselines_ctx24.json"), encoding="utf-8") as f:
            b = json.load(f)
        within = [r["logreg"]["coef"]["OPCN3PM10"] for r in b if r["outer"].startswith("Oslo-")]
        across = [r["logreg"]["coef"]["OPCN3PM10"] for r in b if r["outer"] == "heldout-Oslo"]
        other = {r["outer"]: r["logreg"]["coef"]["OPCN3PM10"] for r in b
                 if r["outer"] in ("heldout-Antwerp", "heldout-Zagreb")}
        res["coef"][h] = {"within_oslo": within, "trained_without_oslo": across[0], "others": other}
        say(f"   h={h:2d} trained on Antwerp+Zagreb: {across[0]:+.3f}   "
            f"trained within Oslo (5 folds): " + " ".join(f"{c:+.3f}" for c in within)
            + f"   positive {sum(c > 0 for c in within)}/5   | held-out Antwerp {other['heldout-Antwerp']:+.3f}, "
            f"held-out Zagreb {other['heldout-Zagreb']:+.3f}")

    # 2 and 3
    res["subset"] = {}
    for city in CITIES:
        say(f"\n{city}")
        for h in HORIZONS:
            y, pm, scores = load(rd, h, city)
            pm10only = (pm[:, 0] == 1) & (pm[:, 1] == 0)
            pm25pos = pm[:, 1] == 1
            neg = y == 0
            keep = ~pm10only
            prev, prev_k = y.mean(), y[keep].mean()
            say(f"  h={h:2d} n={len(y)} positives={y.sum()} PM10-only={pm10only.sum()} "
                f"({pm10only.sum() / max(y.sum(), 1):.1%})  prevalence {prev:.3f} -> {prev_k:.3f} without PM10-only")
            rows = {}
            for n in ("logreg", "reg_lin", "gbm") + ARCHS:
                s = scores[n]
                r = {"ap": average_precision_score(y, s), "auc": roc_auc_score(y, s),
                     "ap_keep": average_precision_score(y[keep], s[keep]),
                     "auc_keep": roc_auc_score(y[keep], s[keep]),
                     "auc_pm25": auc_subset(pm25pos, neg, s),
                     "auc_pm10only": auc_subset(pm10only, neg, s) if pm10only.sum() >= 10 else float("nan")}
                r["lift"], r["lift_keep"] = r["ap"] / prev, r["ap_keep"] / prev_k
                rows[n] = r
                say(f"     {LABEL[n]:7s} AP {r['ap']:.3f} -> {r['ap_keep']:.3f}  "
                    f"(lift {r['lift']:.2f} -> {r['lift_keep']:.2f})  AUC {r['auc']:.3f} -> {r['auc_keep']:.3f}  "
                    f"| AUC PM2.5-driven {r['auc_pm25']:.3f}  PM10-only {r['auc_pm10only']:.3f}")
            # gap to logistic regression before and after removal
            for n in ("gbm",) + ARCHS:
                d0 = rows[n]["ap"] - rows["logreg"]["ap"]
                d1 = rows[n]["ap_keep"] - rows["logreg"]["ap_keep"]
                a0 = rows[n]["auc"] - rows["logreg"]["auc"]
                a1 = rows[n]["auc_keep"] - rows["logreg"]["auc_keep"]
                rows[n]["gap"] = {"ap": d0, "ap_keep": d1, "auc": a0, "auc_keep": a1}
                say(f"     gap {LABEL[n]}-LR  AP {d0:+.3f} -> {d1:+.3f}   AUC {a0:+.3f} -> {a1:+.3f}")
            res["subset"][f"h{h}-{city}"] = {"n": int(len(y)), "pos": int(y.sum()),
                                             "pm10only": int(pm10only.sum()),
                                             "prev": float(prev), "prev_keep": float(prev_k),
                                             "rows": rows}

    os.makedirs(os.path.join(here, "tables"), exist_ok=True)
    with open(os.path.join(here, "tables", "oslo_mechanism.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(here, "tables", "oslo_mechanism.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
