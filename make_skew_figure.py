"""Skewness of the low-cost-sensor PM channels, before and after log1p.

This figure exists to justify one pipeline decision: why ``scaling="logz"``
applies ``log1p`` to OPCN3PM25 and OPCN3PM10 (see ``LOG_FEATURES``) and to no
other channel. The argument is distributional, so the figure shows the two
channels as measured next to the same channels after the transform, with the
sample skewness printed on every panel.

The values plotted are not re-derived here. The script builds a
``SenseurCityDataset`` with ``normalize=False`` and reads its per-device hourly
arrays, so every number on the page has passed through the same flag-based
cleaning and hourly averaging the models were trained on. The transform applied
to the right-hand column is literally the pipeline's:

    np.log1p(np.clip(x, 0.0, None))

Layout is IEEE two-column: 7.16 in wide, Times New Roman at 8 pt, saved as PDF
(vector, Type 42 fonts embedded) and PNG. Use the PDF in LaTeX with
``\\includegraphics[width=\\textwidth]``; it has no resolution to lose.

    python make_skew_figure.py
    python make_skew_figure.py --dpi 1200
    python make_skew_figure.py --cities Antwerp Oslo --out figures

First run parses the CSVs (~8 min); afterwards the on-disk parse cache makes it
about a second.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from senseurcity_dataset import (
    CITIES,
    LOG_FEATURES,
    SenseurCityDataset,
)

# IEEE: 3.5 in single column, 7.16 in double. This figure is a 2x2 grid and so
# is double-column.
COL_WIDTH = 3.5
PAGE_WIDTH = 7.16

CITY_COLORS = {"Antwerp": "#0072B2", "Oslo": "#D55E00", "Zagreb": "#009E73"}
POOLED_FILL = "#BBBBBB"

# Display names for the two heavy-tailed channels.
PRETTY = {
    "OPCN3PM25": r"PM$_{2.5}$",
    "OPCN3PM10": r"PM$_{10}$",
}


def style(base: float = 8.0) -> None:
    """IEEE-ish rcParams: serif text at caption size, embedded fonts, thin rules.

    Kept byte-identical to ``make_figures.style`` so this figure sits beside the
    other four without a visible change of typeface or rule weight.
    """
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


# --------------------------------------------------------------------------- data

def collect(data_dir: str, cities, context_size: int = 24) -> dict:
    """``{feature: {city: 1-D array of valid hourly values}}``, unnormalized.

    Built through ``SenseurCityDataset`` rather than by re-reading the CSVs, so
    the flag-based minute cleaning and the hourly averaging match training
    exactly. ``normalize=False`` leaves ``_series`` in physical units.
    """
    ds = SenseurCityDataset(
        data_dir,
        context_size=context_size,
        cities=tuple(cities),
        normalize=False,
    )
    # Private attributes, deliberately: this is the only route to the cleaned
    # hourly rows without duplicating the cleaning. Asserted so a refactor that
    # renames them fails loudly here instead of silently plotting nothing.
    assert hasattr(ds, "_series") and hasattr(ds, "_series_city"), (
        "SenseurCityDataset no longer exposes _series/_series_city; "
        "update collect() to match."
    )
    feats = list(ds.input_features)
    cols = {f: feats.index(f) for f in LOG_FEATURES if f in feats}
    if not cols:
        raise RuntimeError(f"none of LOG_FEATURES={LOG_FEATURES} in {feats}")

    out = {f: {c: [] for c in cities} for f in cols}
    for (X, _y), city in zip(ds._series, ds._series_city):
        if city not in out[next(iter(cols))]:
            continue
        for f, j in cols.items():
            v = np.asarray(X[:, j], dtype=np.float64)
            out[f][city].append(v[np.isfinite(v)])

    for f in out:
        for c in list(out[f]):
            parts = out[f][c]
            out[f][c] = np.concatenate(parts) if parts else np.empty(0)
    return out


def moments(x: np.ndarray) -> dict:
    """Sample moments, computed directly so scipy is not a dependency.

    ``skew`` is the Fisher-Pearson coefficient g1 and ``kurt`` is excess
    kurtosis, both zero for a normal distribution.
    """
    n = x.size
    if n < 4:
        return dict(n=n, mean=np.nan, median=np.nan, sd=np.nan,
                    skew=np.nan, kurt=np.nan, p99=np.nan, mx=np.nan, neg=0)
    m = x.mean()
    d = x - m
    m2 = (d ** 2).mean()
    m3 = (d ** 3).mean()
    m4 = (d ** 4).mean()
    sd = np.sqrt(m2)
    return dict(
        n=n,
        mean=m,
        median=float(np.median(x)),
        sd=sd,
        skew=m3 / m2 ** 1.5 if m2 > 0 else np.nan,
        kurt=m4 / m2 ** 2 - 3.0 if m2 > 0 else np.nan,
        p99=float(np.percentile(x, 99)),
        mx=float(x.max()),
        neg=int((x < 0).sum()),
    )


def logt(x: np.ndarray) -> np.ndarray:
    """The pipeline's transform, verbatim (see LogZScaler)."""
    return np.log1p(np.clip(x, 0.0, None))


# --------------------------------------------------------------------------- plot

def _panel(ax, pooled, per_city, *, is_log, xlabel, nbins):
    """One histogram panel: pooled filled bars plus per-city step outlines."""
    lo, hi = 0.0, float(np.percentile(pooled, 99.9))
    if is_log:
        lo, hi = float(pooled.min()), float(pooled.max())
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, nbins + 1)

    ax.hist(pooled, bins=bins, density=True, color=POOLED_FILL,
            edgecolor="none", zorder=1)
    for city, v in per_city.items():
        if v.size:
            ax.hist(v, bins=bins, density=True, histtype="step",
                    color=CITY_COLORS.get(city, "#333333"), linewidth=0.8,
                    zorder=2, label=city)

    st = moments(pooled)
    if is_log:
        # Normal reference with the pooled mean and sd. Near-coincidence with the
        # bars is the whole claim: the transform delivers a channel a
        # standardizing scaler can work with.
        g = np.linspace(lo, hi, 400)
        pdf = np.exp(-0.5 * ((g - st["mean"]) / st["sd"]) ** 2) / (
            st["sd"] * np.sqrt(2 * np.pi))
        ax.plot(g, pdf, color="#222222", linestyle="--", linewidth=0.9,
                zorder=3, label="normal")
        note = f"$g_1$ = {st['skew']:.2f}\nexcess $\\kappa$ = {st['kurt']:.2f}"
    else:
        # Log y: the bulk sits near zero and the tail runs decades out, so a
        # linear count axis would render the tail as an invisible flat line.
        ax.set_yscale("log")
        note = (f"$g_1$ = {st['skew']:.1f}\n"
                f"excess $\\kappa$ = {st['kurt']:.0f}\n"
                f"max = {st['mx']:.0f}")

    ax.text(0.97, 0.94, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#999999", linewidth=0.4, alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_xlim(lo, hi)
    ax.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_skew(data, out_dir, dpi, formats, nbins) -> None:
    feats = [f for f in LOG_FEATURES if f in data]
    fig, axes = plt.subplots(len(feats), 2, figsize=(PAGE_WIDTH, 4.3),
                             squeeze=False)

    for r, f in enumerate(feats):
        per_city = data[f]
        pooled = np.concatenate([v for v in per_city.values() if v.size])
        name = PRETTY.get(f, f)

        _panel(axes[r][0], pooled, per_city, is_log=False,
               xlabel=f"OPC {name} ($\\mu$g m$^{{-3}}$)", nbins=nbins)
        axes[r][0].set_ylabel("density (log scale)")

        lp = {c: logt(v) for c, v in per_city.items()}
        _panel(axes[r][1], logt(pooled), lp, is_log=True,
               xlabel=f"$\\log(1 + $ OPC {name}$)$", nbins=nbins)
        axes[r][1].set_ylabel("density")

        if r == 0:
            axes[r][0].set_title("As measured")
            axes[r][1].set_title("After $\\log(1+x)$")

    handles, labels = axes[0][1].get_legend_handles_labels()
    # "outside lower center" is the only placement constrained layout reserves
    # space for; a plain "lower center" with a negative bbox_to_anchor is drawn
    # on top of the bottom row's x-axis label instead of below it.
    try:
        fig.legend(handles, labels, loc="outside lower center",
                   ncol=len(labels), frameon=False)
    except ValueError:      # matplotlib < 3.6 has no outside locs
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False)
        fig.subplots_adjust(bottom=0.14)

    save(fig, out_dir, "fig5_pm_skewness", dpi, formats)


def save(fig, out_dir: str, name: str, dpi: int, formats) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi)
        print(f"  saved {path}")
    plt.close(fig)


def report(data) -> None:
    """Stats table to stdout, for quoting in the text (kurt is excess kurtosis)."""
    print(f"\n{'channel':<11} {'city':<9} {'n':>8} {'median':>8} {'p99':>8} "
          f"{'max':>9} {'skew':>8} {'skew(log)':>10} {'kurt':>8} {'kurt(log)':>10}")
    for f in LOG_FEATURES:
        if f not in data:
            continue
        per_city = data[f]
        pooled = np.concatenate([v for v in per_city.values() if v.size])
        rows = list(per_city.items()) + [("pooled", pooled)]
        for city, v in rows:
            if not v.size:
                continue
            s, sl = moments(v), moments(logt(v))
            print(f"{f:<11} {city:<9} {s['n']:>8d} {s['median']:>8.1f} "
                  f"{s['p99']:>8.1f} {s['mx']:>9.1f} {s['skew']:>8.2f} "
                  f"{sl['skew']:>10.2f} {s['kurt']:>8.2f} {sl['kurt']:>10.2f}")
            if s["neg"]:
                print(f"{'':<21}{s['neg']} negative values clipped to 0 by log1p")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",
                    default=os.environ.get("AQM_DATA_DIR")
                    or os.path.join(here, "senseurcity_data_v03", "dataset"))
    ap.add_argument("--out", default=os.path.join(here, "figures"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES))
    # PDF is vector and ignores this; it sets the PNG raster only.
    ap.add_argument("--dpi", type=int, default=1200)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--bins", type=int, default=70)
    ap.add_argument("--base-font", type=float, default=8.0)
    args = ap.parse_args()

    style(args.base_font)
    print(f"loading {args.cities} from {args.data_dir}")
    data = collect(args.data_dir, args.cities)
    report(data)
    print()
    fig_skew(data, args.out, args.dpi, args.formats, args.bins)


if __name__ == "__main__":
    main()
