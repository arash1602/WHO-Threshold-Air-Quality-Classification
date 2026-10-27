"""Every number the revision adds to the paper, from the shipped sweep plus the new runs.

    python revision_numbers.py      # -> tables/revision_numbers.txt

Inputs: tuning_results/h*/ (shipped deep-model and baseline caches),
tuning_results/h*/extra_baselines_ctx24.json and extra_preds/ (extra_baselines.py),
tables/bootstrap_ci.json (bootstrap_ci.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import make_tables as mt
from stats_extra import corrected_t, ece, holm, sigmoid, test_train_ratio

HORIZONS = (1, 12, 24)
NEW = ("gbm", "reg_lin")
LABEL = {"logreg": "LR", "gbm": "GBM", "reg_lin": "LinCal"}


def load_extra(results_dir, h):
    with open(os.path.join(results_dir, f"h{h}", "extra_baselines_ctx24.json"),
              encoding="utf-8") as f:
        rows = json.load(f)
    return {(r["design"], r["outer"]): r for r in rows}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "tuning_results")
    data = mt.load(rd, HORIZONS, 5)
    out = []
    say = out.append

    extra = {h: load_extra(rd, h) for h in HORIZONS}

    # ------------------------------------------------ recomputed LR check
    worst = 0.0
    for h in HORIZONS:
        for exp, design in (("single", "single-blocked"), ("multi", "multi")):
            e = data[h][exp]
            for f in e["folds"]:
                worst = max(worst, abs(extra[h][(design, f)]["logreg"]["pr_auc"]
                                       - e["base"][f]["logreg"]["pr_auc"]))
    say(f"recomputed LR AP vs shipped: max |diff| = {worst:.2e}")

    def base_vec(h, exp, name, key):
        design = "single-blocked" if exp == "single" else "multi"
        e = data[h][exp]
        return np.array([extra[h][(design, f)][name][key] for f in e["folds"]])

    # ------------------------------------------------ single-city means
    say("\nSINGLE CITY, means over 15 folds (AP mean +/- sd; F2 at tuned threshold)")
    for h in HORIZONS:
        e = data[h]["single"]
        r = test_train_ratio(e)
        say(f" h={h}: prevalence {mt.prevalence(e).mean():.3f}; NB factor {1/15 + r:.3f}")
        rows = {"WHO": (mt.who(e, "pr_auc"), mt.who(e, "f2")),
                "LR": (mt.lr(e, "pr_auc"), mt.lr(e, "f2_at_threshold")),
                "LinCal": (base_vec(h, "single", "reg_lin", "pr_auc"),
                           base_vec(h, "single", "reg_lin", "f2_at_threshold")),
                "GBM": (base_vec(h, "single", "gbm", "pr_auc"),
                        base_vec(h, "single", "gbm", "f2_at_threshold"))}
        for a in mt.ARCHS:
            rows[a] = (mt.ap(e, a), mt.test_metric(e, a, "f2", tuned=True))
        for k, (apv, f2v) in rows.items():
            say(f"   {k:7s} AP {apv.mean():.3f} +/- {apv.std(ddof=0):.3f}   F2 {f2v.mean():.3f}")

    # ------------------------------------------------ single-city tests
    say("\nSINGLE CITY paired tests: row minus reference, Holm over the 9 cells of each family")
    for ref in ("LR", "GBM", "LinCal"):
        for metric in ("AP", "F2"):
            fam = []
            for h in HORIZONS:
                e = data[h]["single"]
                r = test_train_ratio(e)
                if ref == "LR":
                    b = mt.lr(e, "pr_auc") if metric == "AP" else mt.lr(e, "f2_at_threshold")
                else:
                    name = "gbm" if ref == "GBM" else "reg_lin"
                    b = base_vec(h, "single", name, "pr_auc" if metric == "AP" else "f2_at_threshold")
                for a in mt.ARCHS:
                    x = mt.ap(e, a) if metric == "AP" else mt.test_metric(e, a, "f2", tuned=True)
                    d = x - b
                    fam.append(dict(h=h, a=a, d=d.mean(), w=int((d > 0).sum()),
                                    pt=stats.ttest_rel(x, b).pvalue, pc=corrected_t(d, r),
                                    pw=stats.wilcoxon(d, method="exact").pvalue))
            for key in ("pt", "pc", "pw"):
                for row, v in zip(fam, holm([x[key] for x in fam])):
                    row[key + "h"] = v
            say(f" vs {ref}, {metric}:")
            for x in fam:
                say(f"   h={x['h']:2d} {x['a']:6s} {x['d']:+.3f} wins {x['w']:2d}/15  "
                    f"t {x['pt']:.3f}/{x['pth']:.3f}  NB {x['pc']:.3f}/{x['pch']:.3f}  "
                    f"W {x['pw']:.3f}/{x['pwh']:.3f}")
            say(f"   significant (Holm, 0.05): t {sum(x['pth'] < .05 for x in fam)}/9, "
                f"NB {sum(x['pch'] < .05 for x in fam)}/9, W {sum(x['pwh'] < .05 for x in fam)}/9")

    # baselines against LR
    say("\nSINGLE CITY baselines vs LR (AP):")
    for name in NEW:
        for h in HORIZONS:
            e = data[h]["single"]
            b = mt.lr(e, "pr_auc")
            x = base_vec(h, "single", name, "pr_auc")
            d = x - b
            say(f"   {LABEL[name]:7s} h={h:2d} {d.mean():+.3f} wins {(d > 0).sum():2d}/15 "
                f"NB p {corrected_t(d, test_train_ratio(e)):.3f}  "
                f"W p {stats.wilcoxon(d, method='exact').pvalue:.3f}")

    # ------------------------------------------------ multi-city
    say("\nMULTI CITY AP by held-out city (Antwerp, Oslo, Zagreb), mean, mean F2")
    for h in HORIZONS:
        e = data[h]["multi"]
        say(f" h={h}")
        rows = {"LR": (mt.lr(e, "pr_auc"), mt.lr(e, "f2_at_threshold")),
                "LinCal": (base_vec(h, "multi", "reg_lin", "pr_auc"),
                           base_vec(h, "multi", "reg_lin", "f2_at_threshold")),
                "GBM": (base_vec(h, "multi", "gbm", "pr_auc"),
                        base_vec(h, "multi", "gbm", "f2_at_threshold"))}
        for a in mt.ARCHS:
            rows[a] = (mt.ap(e, a), mt.test_metric(e, a, "f2", tuned=True))
        lr_ap = rows["LR"][0]
        gbm_ap = rows["GBM"][0]
        for k, (apv, f2v) in rows.items():
            say(f"   {k:7s} " + " ".join(f"{v:.3f}" for v in apv)
                + f"  mean {apv.mean():.3f}  F2 {f2v.mean():.3f}"
                + f"  >LR {(apv > lr_ap).sum()}/3  >GBM {(apv > gbm_ap).sum()}/3")

    # ------------------------------------------------ bootstrap
    p = os.path.join(here, "tables", "bootstrap_ci.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            bs = json.load(f)
        say("\nMULTI CITY day-block bootstrap: AP difference vs LR, 95% CI")
        for key, v in bs.items():
            parts = []
            for n in list(mt.ARCHS) + list(NEW):
                lo, hi = v[n]["diff_ci"]
                star = "*" if lo > 0 else ("-" if hi < 0 else " ")
                parts.append(f"{n} {v[n]['diff_vs_lr']:+.3f}[{lo:+.3f},{hi:+.3f}]{star}")
            say(f"   {key:12s} ({v['n_days']} d) " + "  ".join(parts))
        say("\nMULTI CITY day-block bootstrap: deep-model AP difference vs GBM, 95% CI")
        for key, v in bs.items():
            parts = []
            for n in mt.ARCHS:
                lo, hi = v[n]["diff_gbm_ci"]
                star = "*" if lo > 0 else ("-" if hi < 0 else " ")
                parts.append(f"{n} {v[n]['diff_vs_gbm']:+.3f}[{lo:+.3f},{hi:+.3f}]{star}")
            say(f"   {key:12s} " + "  ".join(parts))

    # ------------------------------------------------ pollutant shares
    say("\nWhich pollutant drives the label (positive windows, test sets of the multi design)")
    for h in HORIZONS:
        tot = np.zeros(3)
        for city in mt.CITIES:
            z = np.load(os.path.join(rd, f"h{h}", "extra_preds", "multi", f"heldout-{city}.npz"))
            pm, y = z["pm"], z["y"].astype(bool)
            # output_features order is (Ref.PM10, Ref.PM2.5)
            only25 = (pm[:, 1] == 1) & (pm[:, 0] == 0)
            only10 = (pm[:, 0] == 1) & (pm[:, 1] == 0)
            both = (pm[:, 0] == 1) & (pm[:, 1] == 1)
            c = np.array([only25.sum(), only10.sum(), both.sum()])
            tot += c
            say(f"   h={h:2d} {city:8s} positives {y.sum():5d}: PM2.5 only {c[0]/y.sum():.3f}, "
                f"PM10 only {c[1]/y.sum():.3f}, both {c[2]/y.sum():.3f}; "
                f"PM2.5 prevalence {pm[:, 1].mean():.3f}, PM10 prevalence {pm[:, 0].mean():.3f}")
        say(f"   h={h:2d} pooled: PM2.5 only {tot[0]/tot.sum():.3f}, PM10 only {tot[1]/tot.sum():.3f}, "
            f"both {tot[2]/tot.sum():.3f}")

    # ------------------------------------------------ calibration
    say("\nCalibration on outer test sets (mean over folds): Brier / ECE")
    for exp, design in (("single", "single-blocked"), ("multi", "multi")):
        for h in HORIZONS:
            parts = []
            for name in ("logreg", "gbm"):
                b_, c_ = [], []
                for f in data[h][exp]["folds"]:
                    z = np.load(os.path.join(rd, f"h{h}", "extra_preds", design, f"{f}.npz"))
                    pr, y = z[f"score_{name}"], z["y"].astype(float)
                    b_.append(np.mean((pr - y) ** 2))
                    c_.append(ece(pr, y))
                parts.append(f"{LABEL[name]} {np.mean(b_):.3f}/{np.mean(c_):.3f}")
            for a in mt.ARCHS:
                b_, c_ = [], []
                for f in data[h][exp]["folds"]:
                    z = np.load(os.path.join(rd, f"h{h}", f"nested5_{a}_{exp}_ctx24_blocked_preds",
                                             f"{f}.npz"))
                    pr, y = sigmoid(z["test_logits"].astype(float)), z["test_y"].astype(float)
                    b_.append(np.mean((pr - y) ** 2))
                    c_.append(ece(pr, y))
                parts.append(f"{a} {np.mean(b_):.3f}/{np.mean(c_):.3f}")
            say(f"   {exp:6s} h={h:2d}  " + "  ".join(parts))

    text = "\n".join(out) + "\n"
    with open(os.path.join(here, "tables", "revision_numbers.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
