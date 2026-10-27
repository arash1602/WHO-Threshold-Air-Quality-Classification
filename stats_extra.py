"""Supplementary statistics for the revision, computed from the shipped results.

    python stats_extra.py                       # -> tables/stats_extra.txt, tables/tab_stats_corrected.tex

No retraining. Adds what the paired t-tests in make_tables.py leave out:

* Nadeau--Bengio corrected resampled t-test. Outer folds of a city share most of
  their development data, so fold scores are positively correlated and the plain
  paired t-test understates the variance. The correction inflates it by
  (1/k + n_test/n_train) instead of 1/k (Nadeau and Bengio, Mach. Learn. 52, 2003).
* Exact Wilcoxon signed-rank test, which drops the normality assumption.
* Holm--Bonferroni adjustment over the nine architecture-vs-logistic-regression
  comparisons of a family (one family per test).
* Calibration of the deep models' test probabilities: Brier score and expected
  calibration error (10 equal-width bins), per fold, from the saved refit logits.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats

import make_tables as mt

HORIZONS = (1, 12, 24)


def holm(p):
    """Holm--Bonferroni adjusted p values, in the input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def corrected_t(d, ratio):
    """Two-sided p of the Nadeau--Bengio corrected resampled t-test."""
    d = np.asarray(d, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan")
    t = d.mean() / np.sqrt((1.0 / k + ratio) * var)
    return float(2 * stats.t.sf(abs(t), k - 1))


def test_train_ratio(e):
    """Mean n_test / n_dev over the outer folds of one design."""
    r = []
    for f in e["folds"]:
        dev_n, _, test_n, _ = mt.split_counts(e, f)
        r.append(test_n / dev_n)
    return float(np.mean(r))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def calibration(results_dir, h, arch, exp):
    d = os.path.join(results_dir, f"h{h}",
                     f"nested5_{arch}_{exp}_ctx24_blocked_preds")
    brier, ec = [], []
    for name in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, name))
        p, y = sigmoid(z["test_logits"].astype(float)), z["test_y"].astype(float)
        brier.append(float(np.mean((p - y) ** 2)))
        ec.append(ece(p, y))
    return np.array(brier), np.array(ec)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(here, "tuning_results"))
    ap.add_argument("--out", default=os.path.join(here, "tables"))
    args = ap.parse_args()

    data = mt.load(args.results_dir, HORIZONS, 5)
    lines = []
    say = lines.append

    # ---------------------------------------------------------- single city
    e1 = data[1]["single"]
    ratio = test_train_ratio(e1)
    say(f"Single-city mean n_test/n_dev = {ratio:.3f} "
        f"(corrected variance factor {1/15 + ratio:.3f} vs 1/k = {1/15:.3f})")
    rows = []
    for metric, getter in (("AP", None), ("F2@tuned", "f2")):
        fam = []
        for h in HORIZONS:
            e = data[h]["single"]
            base = mt.lr(e, "pr_auc") if getter is None else mt.lr(e, "f2_at_threshold")
            r = test_train_ratio(e)
            for a in mt.ARCHS:
                x = mt.ap(e, a) if getter is None else mt.test_metric(e, a, "f2", tuned=True)
                d = x - base
                p_t = float(stats.ttest_rel(x, base).pvalue)
                p_c = corrected_t(d, r)
                p_w = float(stats.wilcoxon(d, method="exact").pvalue)
                fam.append(dict(metric=metric, h=h, arch=a, diff=float(d.mean()),
                                wins=int((d > 0).sum()), p_t=p_t, p_c=p_c, p_w=p_w))
        for key in ("p_t", "p_c", "p_w"):
            adj = holm([r_[key] for r_ in fam])
            for r_, v in zip(fam, adj):
                r_[key + "_holm"] = float(v)
        rows += fam

    say("")
    say("Architecture minus logistic regression, single city, 15 folds per horizon")
    say(f"{'metric':9s} {'h':>3s} {'arch':7s} {'diff':>7s} {'wins':>5s} "
        f"{'t':>7s} {'t_holm':>7s} {'NB':>7s} {'NB_holm':>8s} {'W':>7s} {'W_holm':>7s}")
    for r_ in rows:
        say(f"{r_['metric']:9s} {r_['h']:3d} {r_['arch']:7s} {r_['diff']:+7.3f} "
            f"{r_['wins']:3d}/15 {r_['p_t']:7.3f} {r_['p_t_holm']:7.3f} "
            f"{r_['p_c']:7.3f} {r_['p_c_holm']:8.3f} {r_['p_w']:7.3f} {r_['p_w_holm']:7.3f}")
    for metric in ("AP", "F2@tuned"):
        fam = [r_ for r_ in rows if r_["metric"] == metric]
        say(f"{metric}: significant at 0.05 -- plain t {sum(r_['p_t'] < .05 for r_ in fam)}/9, "
            f"Holm t {sum(r_['p_t_holm'] < .05 for r_ in fam)}/9, "
            f"NB {sum(r_['p_c'] < .05 for r_ in fam)}/9, "
            f"Holm NB {sum(r_['p_c_holm'] < .05 for r_ in fam)}/9, "
            f"Wilcoxon {sum(r_['p_w'] < .05 for r_ in fam)}/9, "
            f"Holm Wilcoxon {sum(r_['p_w_holm'] < .05 for r_ in fam)}/9")

    # ---------------------------------------------------------- calibration
    say("")
    say("Calibration of deep-model test probabilities (mean over folds)")
    cal = {}
    for exp, label in (("single", "single"), ("multi", "multi")):
        for h in HORIZONS:
            for a in mt.ARCHS:
                b, c = calibration(args.results_dir, h, a, exp)
                cal[(exp, h, a)] = (b.mean(), c.mean())
                say(f"  {label:6s} h={h:2d} {a:6s} Brier {b.mean():.3f}  ECE {c.mean():.3f}")

    # ---------------------------------------------------------- LaTeX table
    tex = [r"\begin{tabular}{llccccc}", r"\toprule",
           r"$h$ & Architecture & $\Delta$AP & Wins & $p_{t}$ & $p_{\mathrm{NB}}$ & $p_{W}$ \\",
           r"\midrule"]
    for h in HORIZONS:
        fam = [r_ for r_ in rows if r_["metric"] == "AP" and r_["h"] == h]
        for i, r_ in enumerate(fam):
            lead = rf"\multirow{{3}}{{*}}{{{h}}}" if i == 0 else ""
            tex.append(f"{lead} & {r_['arch']} & {r_['diff']:+.3f} & {r_['wins']}/15 & "
                       f"{r_['p_t_holm']:.3f} & {r_['p_c_holm']:.3f} & {r_['p_w_holm']:.3f} \\\\")
        if h != HORIZONS[-1]:
            tex.append(r"\cmidrule(lr){1-7}")
    tex += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stats_extra.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "tab_stats_corrected.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
