"""Manuscript figures from a finished sweep. No retraining, no new model runs.

Produces the four figures identified as carrying the argument, each with one panel
per horizon so h = 1, 12 and 24 sit side by side:

The reported metric is average precision (sklearn ``average_precision_score``),
the standard estimator of the area under the precision-recall curve. It is stored
under the key ``pr_auc`` throughout the caches; the figures label it AP, since the
two are not the same quantity and reviewers distinguish them. "Prevalence" is used
everywhere for the positive-class rate, matching the manuscript tables.

    fig1_pr_auc_by_arch      per-fold test AP per architecture, with the WHO rule
                             and logistic regression drawn as reference lines at
                             their MEAN over folds (the value the tables report,
                             not the median) and each fold's three architectures
                             joined by a faint line (the folds are paired; the
                             lines show that fold identity moves the score more
                             than architecture does)
    fig2_prevalence          per-fold test AP against that fold's prevalence, with
                             the identity line -- a random ranker's AP *is* the
                             prevalence, so vertical distance above the line is the
                             only part of the score that is skill
    fig3_threshold           precision / recall / F2 against decision threshold
                             (median over folds, IQR band), with the tuned
                             threshold marked, over confusion matrices at that
                             threshold summed across folds
    fig4_pareto              test AP against trained-parameter count, from the
                             WHO rule and logistic regression up to the deep
                             winners -- the cost argument, now that the embedded
                             benchmark is descoped

Everything is read from the sweep's own artifacts:

    <dir>/nested<schema>_<arch>_<exp>_ctx24_blocked.jsonl   outer records
    <dir>/nested<schema>_<arch>_<exp>_ctx24_blocked_preds/  test logits + labels
    <dir>/baselines_ctx24.json                              WHO rule + logreg

Output is sized for IEEE two-column: 7.16 in wide (the double-column text width),
Times New Roman at 8 pt, saved as both PDF (vector, fonts embedded as Type 42 --
prefer this in LaTeX, it has no resolution) and PNG at --dpi, default 600.

    python make_figures.py                          # h1 h12 h24, single-blocked
    python make_figures.py --experiment multi       # leave-one-city-out
    python make_figures.py --horizons 1 --dpi 1200

A horizon with no cache is skipped with a warning rather than failing, so the
script still runs while a sweep is in flight.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARCHS = ("GRU", "TCN", "FUSION")
CONTEXT = 24
PROTOCOL = "blocked"

# baselines_ctx24.json spells the single-city experiment out; the cache stems abbreviate it.
EXPERIMENT_ALIASES = {"single": "single-blocked", "multi": "multi"}

# IEEE: 3.5 in single column, 7.16 in double. Every figure here is a 3-horizon row,
# so all of them are double-column and get \includegraphics[width=\textwidth].
COL_WIDTH = 3.5
PAGE_WIDTH = 7.16

ARCH_COLORS = {"GRU": "#0072B2", "TCN": "#D55E00", "FUSION": "#009E73"}
CITY_COLORS = {"Antwerp": "#0072B2", "Oslo": "#D55E00", "Zagreb": "#009E73"}
WHO_STYLE = dict(color="#666666", linestyle=":", linewidth=1.0)
LOGREG_STYLE = dict(color="#444444", linestyle="--", linewidth=1.0)


def style(base: float = 8.0) -> None:
    """IEEE-ish rcParams: serif text at caption size, embedded fonts, thin rules."""
    serif = "Times New Roman" if _has_font("Times New Roman") else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        # Type 42 = TrueType. IEEE requires embedded fonts; the default Type 3 is
        # rejected by some submission checkers.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # NOT bbox_inches="tight": that trims the page to its content, so a figure
        # declared at PAGE_WIDTH saves narrower than PAGE_WIDTH and LaTeX then
        # scales it up at \includegraphics[width=\textwidth], inflating every font
        # past the size set here. Constrained layout fits the content to the
        # declared size instead, so the PDF is 1:1 with the column.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    })


def _has_font(name: str) -> bool:
    from matplotlib import font_manager as fm
    return name in {f.name for f in fm.fontManager.ttflist}


# --------------------------------------------------------------------------- IO

def stem(out_dir: str, schema: int, arch: str, experiment: str) -> str:
    return os.path.join(
        out_dir, f"nested{schema}_{arch}_{experiment}_ctx{CONTEXT}_{PROTOCOL}")


def read_outer(out_dir: str, schema: int, arch: str, experiment: str) -> dict:
    """``{outer_id: outer record}`` for one architecture, or ``{}`` if no cache."""
    path = stem(out_dir, schema, arch, experiment) + ".jsonl"
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "outer":
                rows[rec["outer"]] = rec
    return rows


def read_baselines(out_dir: str, experiment: str) -> dict:
    """``{outer_id: row}`` from baselines_ctx24.json for one experiment."""
    path = os.path.join(out_dir, f"baselines_ctx{CONTEXT}.json")
    if not os.path.exists(path):
        return {}
    want = EXPERIMENT_ALIASES[experiment]
    with open(path, encoding="utf-8") as f:
        return {r["outer"]: r for r in json.load(f) if r.get("experiment") == want}


def horizon_dir(results_dir: str, horizon: int) -> str:
    return os.path.join(results_dir, f"h{horizon}")


def city_of(outer_id: str) -> str:
    """``Antwerp-b3`` -> Antwerp; ``heldout-Oslo`` -> Oslo (the city being tested)."""
    return outer_id.split("-", 1)[1] if outer_id.startswith("heldout-") \
        else outer_id.rsplit("-", 1)[0]


def finite(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    return a[np.isfinite(a)]


def load_horizon(results_dir: str, horizon: int, schema: int, experiment: str) -> dict | None:
    """Everything one panel needs, or None when that horizon has not been run."""
    d = horizon_dir(results_dir, horizon)
    per_arch = {a: read_outer(d, schema, a, experiment) for a in ARCHS}
    if not any(per_arch.values()):
        return None
    base = read_baselines(d, experiment)
    # Fold order is the ordered union, so a fold dropped by one architecture (a
    # single-class test split) still plots for the others rather than vanishing.
    folds: list[str] = []
    for a in ARCHS:
        for k in per_arch[a]:
            if k not in folds:
                folds.append(k)
    return {"dir": d, "horizon": horizon, "outer": per_arch,
            "baselines": base, "folds": folds}


# ------------------------------------------------------------------ parameters

_PARAM_CACHE: dict[tuple, int] = {}


def n_params(arch: str, hp: dict) -> int:
    """Trained-parameter count for one grid point. Cached: 45 folds share few hp."""
    key = (arch, json.dumps(hp, sort_keys=True))
    if key not in _PARAM_CACHE:
        import contextlib
        import io
        from nested_worker import build_model
        from compare_models import N_INPUT_CHANNELS
        hp = dict(hp)
        if "tcn_channels" in hp:          # JSON round-trips the tuple to a list
            hp["tcn_channels"] = tuple(hp["tcn_channels"])
        # Every constructor prints its own banner; 45 of them bury the real output.
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(arch, hp, N_INPUT_CHANNELS)
        _PARAM_CACHE[key] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _PARAM_CACHE[key]


# The logistic regression fits one coefficient per window-mean feature plus an
# intercept; the WHO rule fits nothing at all, and its single threshold is fixed by
# the standard rather than learned. Plotted at 1 so it has a place on a log axis,
# and labelled so the distinction is not lost.
LOGREG_PARAMS = 6
WHO_PARAMS_PLOTTED = 1


# ----------------------------------------------------------------------- panels

def panel_title(h: int) -> str:
    return f"$h$ = {h} h"


def below_legend(fig, handles, ncol: int) -> None:
    """One shared legend under the panel row. In-axes legends land on top of the
    data in at least one panel of every figure here, and IEEE readers scan the
    row as a unit anyway.

    ``outside lower center`` makes constrained layout *reserve* the strip rather
    than draw over the x-axis labels, so the figure keeps its declared size.
    """
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol,
               frameon=False, handlelength=1.6, columnspacing=1.4)


def _nanmedian_cols(m: np.ndarray):
    """Median / quartiles down the fold axis, leaving all-NaN columns as NaN.

    Precision is undefined where a threshold predicts nothing positive, so the
    top of the sweep is legitimately empty for every fold at once. numpy warns on
    those columns; the curve should simply stop there.
    """
    ok = np.isfinite(m).any(axis=0)
    out = [np.full(m.shape[1], np.nan) for _ in range(3)]
    if ok.any():
        sub = m[:, ok]
        out[0][ok] = np.nanmedian(sub, axis=0)
        out[1][ok] = np.nanpercentile(sub, 25, axis=0)
        out[2][ok] = np.nanpercentile(sub, 75, axis=0)
    return out


def save(fig, out_dir: str, name: str, dpi: int, formats) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi)
        print(f"  saved {path}")
    plt.close(fig)


def fig_pr_auc_by_arch(panels, out_dir, dpi, formats, experiment) -> None:
    """Per-fold PR-AUC per architecture, one panel per horizon, folds paired."""
    fig, axes = plt.subplots(1, len(panels), figsize=(PAGE_WIDTH, 2.7),
                             sharey=True, squeeze=False)
    axes = axes[0]
    xs = np.arange(len(ARCHS))

    for ax, p in zip(axes, panels):
        data = [[p["outer"][a][f]["test"]["pr_auc"]
                 for f in p["folds"] if f in p["outer"][a]] for a in ARCHS]

        # Paired lines first, so the points and boxes sit on top of them.
        for f in p["folds"]:
            ys = [p["outer"][a].get(f, {}).get("test", {}).get("pr_auc", np.nan)
                  for a in ARCHS]
            if np.isfinite(ys).sum() > 1:
                ax.plot(xs, ys, color="#999999", linewidth=0.35, alpha=0.55, zorder=1)

        bp = ax.boxplot([finite(d) for d in data], positions=xs, widths=0.5,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="black", linewidth=1.0),
                        boxprops=dict(linewidth=0.6),
                        whiskerprops=dict(linewidth=0.6),
                        capprops=dict(linewidth=0.6))
        for patch, a in zip(bp["boxes"], ARCHS):
            patch.set_color(ARCH_COLORS[a])

        for i, (a, d) in enumerate(zip(ARCHS, data)):
            ax.scatter(np.full(len(d), i), d, s=6, zorder=3,
                       color=ARCH_COLORS[a], edgecolors="none", alpha=0.85)

        who = finite([r["who"]["pr_auc"] for r in p["baselines"].values()])
        lr = finite([r["logreg"]["pr_auc"] for r in p["baselines"].values()])
        # Mean, not median, so the rules equal the values reported in the tables.
        # The two differ sharply for the WHO rule (0.570 against a median of 0.681
        # at h = 1, because three folds sit near zero), and a figure that disagrees
        # with its own table invites exactly the wrong reading.
        if who.size:
            ax.axhline(who.mean(), **WHO_STYLE, zorder=0)
        if lr.size:
            ax.axhline(lr.mean(), **LOGREG_STYLE, zorder=0)

        ax.set_xticks(xs)
        ax.set_xticklabels(ARCHS)
        ax.set_title(panel_title(p["horizon"]))
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test average precision")
    handles = [Line2D([], [], **LOGREG_STYLE, label="Logistic regression (mean)"),
               Line2D([], [], **WHO_STYLE, label="WHO rule (mean)"),
               Line2D([], [], color="#999999", linewidth=0.5, label="Same fold")]
    below_legend(fig, handles, 3)
    save(fig, out_dir, f"fig1_pr_auc_by_arch_{experiment}", dpi, formats)


def fig_prevalence(panels, out_dir, dpi, formats, experiment) -> None:
    """PR-AUC against fold prevalence, with the random-classifier identity line."""
    fig, axes = plt.subplots(1, len(panels), figsize=(PAGE_WIDTH, 2.9),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    markers = {"GRU": "o", "TCN": "s", "FUSION": "^"}

    # The panels share one x-axis, so the limit must cover every panel's folds;
    # setting it per panel lets the last panel's range clip the others.
    lim = [1.0, 0.0]
    for ax, p in zip(axes, panels):
        for a in ARCHS:
            for f, rec in p["outer"][a].items():
                x = rec["test"].get("pos_rate", np.nan)
                y = rec["test"].get("pr_auc", np.nan)
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                lim = [min(lim[0], x), max(lim[1], x)]
                ax.scatter(x, y, s=11, marker=markers[a], alpha=0.85,
                           color=CITY_COLORS.get(city_of(f), "#555555"),
                           edgecolors="none", zorder=3)
        # Average precision's chance level IS the prevalence, so this line is the floor.
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.7, linestyle="-.",
                zorder=1)
        ax.set_title(panel_title(p["horizon"]))
        ax.set_xlabel("Prevalence")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    pad = 0.04
    axes[0].set_xlim(max(0.0, lim[0] - pad), min(1.0, lim[1] + pad))
    axes[0].set_ylabel("Test average precision")
    handles = [Line2D([], [], marker=markers[a], linestyle="none", color="#555555",
                      markersize=3.5, label=a) for a in ARCHS]
    handles += [Line2D([], [], marker="o", linestyle="none", color=c, markersize=3.5,
                       label=city) for city, c in CITY_COLORS.items()]
    handles += [Line2D([], [], color="#999999", linewidth=0.7, linestyle="-.",
                       label="Random (AP = prevalence)")]
    below_legend(fig, handles, 4)
    save(fig, out_dir, f"fig2_prevalence_{experiment}", dpi, formats)


def _threshold_curves(logits, y, grid):
    """Precision, recall and F2 for ``prob > t`` at every t in ``grid``."""
    prob = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    t = np.asarray(y).astype(bool)
    n_pos = int(t.sum())
    prec, rec, f2 = [], [], []
    for cut in grid:
        pred = prob > cut
        tp = int((pred & t).sum())
        fp = int((pred & ~t).sum())
        fn = n_pos - tp
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / n_pos if n_pos else np.nan)
        den = 5 * tp + 4 * fn + fp
        f2.append(5 * tp / den if den else np.nan)
    return np.array(prec), np.array(rec), np.array(f2)


def fig_threshold(panels, out_dir, dpi, formats, experiment, arch) -> None:
    """Row 1: P/R/F2 against threshold, median over folds with IQR band, tuned
    threshold marked. Row 2: confusion matrix at that threshold, summed over folds."""
    fig, axes = plt.subplots(2, len(panels), figsize=(PAGE_WIDTH, 4.4),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [1.6, 1.0]})
    grid = np.linspace(0.005, 0.995, 200)
    colors = {"Precision": "#0072B2", "Recall": "#D55E00", "F2": "#009E73"}

    for col, p in enumerate(panels):
        ax, cax = axes[0][col], axes[1][col]
        curves = {"Precision": [], "Recall": [], "F2": []}
        thresholds, counts = [], np.zeros(4, dtype=np.int64)

        for f, rec in p["outer"][arch].items():
            rel = rec.get("predictions")
            path = os.path.join(p["dir"], rel) if rel else None
            if path and os.path.exists(path):
                with np.load(path) as z:
                    pr, rc, f2 = _threshold_curves(z["test_logits"], z["test_y"], grid)
                curves["Precision"].append(pr)
                curves["Recall"].append(rc)
                curves["F2"].append(f2)
            prob = (rec.get("threshold") or {}).get("prob")
            if prob is not None and np.isfinite(prob):
                thresholds.append(prob)
            tat = rec.get("test_at_threshold") or {}
            counts += np.array([tat.get(k, 0) for k in ("tn", "fp", "fn", "tp")],
                               dtype=np.int64)

        for name, stack in curves.items():
            if not stack:
                continue
            med, lo, hi = _nanmedian_cols(np.vstack(stack))
            ax.fill_between(grid, lo, hi, color=colors[name], alpha=0.16, linewidth=0)
            ax.plot(grid, med, color=colors[name], label=name)

        if thresholds:
            ax.axvline(float(np.median(thresholds)), color="black", linestyle="--",
                       linewidth=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(panel_title(p["horizon"]))
        ax.set_xlabel("Decision threshold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        cm = counts.reshape(2, 2).astype(float)          # rows actual, cols predicted
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        cax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                cax.text(j, i, f"{int(cm[i, j]):,}\n{norm[i, j]:.0%}",
                         ha="center", va="center", fontsize=6,
                         color="white" if norm[i, j] > 0.55 else "black")
        cax.set_xticks([0, 1]); cax.set_xticklabels(["pred 0", "pred 1"])
        cax.set_yticks([0, 1]); cax.set_yticklabels(["true 0", "true 1"])
        cax.tick_params(length=0)
        for s in cax.spines.values():
            s.set_visible(False)

    axes[0][0].set_ylabel("Score")
    axes[1][0].set_ylabel("Summed over folds")
    handles = [Line2D([], [], color=c, label=n) for n, c in colors.items()]
    # Architecture-specific on purpose. The P/R/F2 curves barely move between
    # architectures (F2 within 0.08 across the 0.1-0.6 decision band), but the
    # tuned threshold does: GRU 0.415 against TCN 0.286 at h = 12, GRU 0.202
    # against TCN 0.338 at h = 24. An unqualified label would imply a shared
    # operating point that does not exist.
    handles += [Line2D([], [], color="black", linestyle="--", linewidth=0.8,
                       label=f"{arch} tuned threshold (median)")]
    below_legend(fig, handles, 4)
    save(fig, out_dir, f"fig3_threshold_{arch}_{experiment}", dpi, formats)


def fig_pareto(panels, out_dir, dpi, formats, experiment) -> None:
    """PR-AUC against trained-parameter count: what the extra capacity buys."""
    fig, axes = plt.subplots(1, len(panels), figsize=(PAGE_WIDTH, 2.9),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, p in zip(axes, panels):
        who = finite([r["who"]["pr_auc"] for r in p["baselines"].values()])
        lr = finite([r["logreg"]["pr_auc"] for r in p["baselines"].values()])
        for xval, vals, color, marker in (
                (WHO_PARAMS_PLOTTED, who, "#666666", "x"),
                (LOGREG_PARAMS, lr, "#444444", "D")):
            if vals.size:
                ax.errorbar(xval, vals.mean(), yerr=vals.std(ddof=0), fmt=marker,
                            color=color, markersize=3.5, capsize=1.6, elinewidth=0.7,
                            markeredgewidth=0.9)

        for a in ARCHS:
            recs = list(p["outer"][a].values())
            if not recs:
                continue
            # Each fold selects its own winner, so the architecture occupies a
            # range of sizes, not a point. Plot the per-fold winners faintly and
            # the mean on top.
            px = np.array([n_params(a, r["hp"]) for r in recs], dtype=float)
            py = np.array([r["test"].get("pr_auc", np.nan) for r in recs], dtype=float)
            ok = np.isfinite(py)
            ax.scatter(px[ok], py[ok], s=5, color=ARCH_COLORS[a], alpha=0.35,
                       edgecolors="none", zorder=2)
            if ok.any():
                ax.errorbar(np.exp(np.log(px[ok]).mean()), py[ok].mean(),
                            yerr=py[ok].std(ddof=0), fmt="o", color=ARCH_COLORS[a],
                            markersize=3.5, capsize=1.6, elinewidth=0.7, zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(left=0.45)          # keep the WHO marker off the spine
        ax.set_title(panel_title(p["horizon"]))
        ax.set_xlabel("Trained parameters")
        ax.grid(True, alpha=0.3, which="major")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test average precision")
    handles = [Line2D([], [], marker="x", linestyle="none", color="#666666",
                      markersize=3.5, label="WHO rule (none learned)"),
               Line2D([], [], marker="D", linestyle="none", color="#444444",
                      markersize=3.5, label="Logistic regression")]
    handles += [Line2D([], [], marker="o", linestyle="none", color=ARCH_COLORS[a],
                       markersize=3.5, label=a) for a in ARCHS]
    below_legend(fig, handles, 5)
    save(fig, out_dir, f"fig4_pareto_{experiment}", dpi, formats)


# ------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "tuning_results"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
    ap.add_argument("--schema", type=int, default=5, help="4 = minmax, 5 = logz")
    ap.add_argument("--experiment", choices=("single", "multi"), default="single")
    ap.add_argument("--threshold-arch", choices=ARCHS, default="GRU",
                    help="architecture shown in figure 3")
    ap.add_argument("--out", default=os.path.join(here, "figures"))
    ap.add_argument("--dpi", type=int, default=600, help="raster dpi; PDF is vector")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--base-font", type=float, default=8.0)
    args = ap.parse_args()

    style(args.base_font)

    panels = []
    for h in args.horizons:
        p = load_horizon(args.results_dir, h, args.schema, args.experiment)
        if p is None:
            print(f"[skip] no schema-{args.schema} {args.experiment} cache in "
                  f"{horizon_dir(args.results_dir, h)}")
            continue
        n = {a: len(p["outer"][a]) for a in ARCHS}
        print(f"h={h}: folds {n}, baselines {len(p['baselines'])}")
        panels.append(p)
    if not panels:
        print("no horizons to plot")
        return 1

    out = os.path.join(args.out, f"h{'-'.join(str(p['horizon']) for p in panels)}")
    fig_pr_auc_by_arch(panels, out, args.dpi, args.formats, args.experiment)
    fig_prevalence(panels, out, args.dpi, args.formats, args.experiment)
    fig_threshold(panels, out, args.dpi, args.formats, args.experiment,
                  args.threshold_arch)
    fig_pareto(panels, out, args.dpi, args.formats, args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
