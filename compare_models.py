"""Fold geometry for the SenseurCity AQM classification task.

This module owns how the data is *cut*; ``find_best_model.py`` owns what is
trained on it and produces the reported numbers.

Nested planners::

    nested_blocked_plan  purged blocked CV within one city  (Single-City-Blocked)
    nested_loco_plan     leave-one-city-out, blocked inner  (Multi-City)
    nested_plan          dispatcher over (experiment, protocol)
    materialize_outer    OuterSpec -> datasets, with degenerate splits screened

See the "Nested cross-validation" section below for the purge and scaler
invariants that keep the outer test out of selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from senseurcity_dataset import WHO_AVG_HOURS, SenseurCityDataset


N_INPUT_CHANNELS = 5

# Blocked-CV geometry. The timeline is cut into blocks of this many hours, which
# are then dealt out to folds; the block must be short relative to a season (so a
# fold's blocks span the whole calendar) but long relative to a pollution episode
# (so a single episode does not smear across many folds). Two weeks sits between
# those scales for this data.
DEFAULT_BLOCK_HOURS = 14 * 24
# Blocks holding fewer distinct hours than this are dropped rather than dealt —
# they are coverage stubs at campaign edges and would add noise to a fold.
MIN_BLOCK_HOURS = 24


# --------------------------------------------------------------------------- #
# Dataset slices
# --------------------------------------------------------------------------- #
def _slice(data_dir, time_range, *, scaler=None, cities, **common):
    return SenseurCityDataset(
        data_dir, cities=cities, time_range=time_range,
        scaler=scaler, **common,
    )


def _slice_bands(data_dir, bands, *, scaler=None, cities, **common):
    return SenseurCityDataset(
        data_dir, cities=cities, time_bands=bands, scaler=scaler, **common,
    )






# --------------------------------------------------------------------------- #
# Purged blocked CV
# --------------------------------------------------------------------------- #
# A chronological split cuts the timeline into contiguous equal-wall-clock blocks,
# so with only ~1.2 seasonal cycles of data (2020-02 .. 2021-04) every fold
# boundary is also a season boundary: a 1/5 block is ~2.8 months. The harmful-air label is
# strongly seasonal (Zagreb runs 0.00 positive in June and 0.87 in November; Oslo
# has six consecutive months with no positives at all), so a fold's class prior is
# decided by which season its block landed in, and some folds hold one class only.
#
# Blocked CV breaks that confound: cut the calendar into blocks far shorter than a
# season, rank them by positive rate and deal them to folds, so every fold's
# train/val/test span the full seasonal range with comparable priors. The cost is
# that it no longer answers "can the model extrapolate forward in time".
#
# Purging: splitting on interleaved blocks puts train and test data adjacent in
# time at every block boundary, and consecutive windows are autocorrelated. So
# each block's band is contracted at its start by ``embargo_hours``, leaving a
# real gap between the last hour any window in one block touches and the first
# hour touched in the next.
#
# The default gap is one sample's full span. A window's inputs cover
# [s, s + context) and its label averages the WHO_AVG_HOURS ending ``horizon``
# hours past the last input hour, so the two overlap unless the horizon reaches
# past the input: at context 24 / horizon 1 the label starts 1 h into the input
# and the sample touches 25 h. Windows must already lie wholly inside one band,
# so no hour is ever shared across splits; the embargo is a buffer on top of that.
def _resolve_embargo(embargo_hours, common) -> int:
    """Purge gap in hours: one sample's full span, unless overridden."""
    if embargo_hours is not None:
        return int(embargo_hours)
    context = int(common.get("context_size", 24))
    horizon = int(common.get("horizon", 1))
    return context + horizon + max(0, WHO_AVG_HOURS - context - horizon)


def _probe_dataset(data_dir, cities, *, bands=None, **common) -> SenseurCityDataset:
    """Unnormalized dataset used only to read the label/time layout before folds
    exist. The module-level parse cache in senseurcity_dataset makes this ~free
    after the first build.

    ``bands`` restricts the probe to explicit wall-clock intervals instead of the
    whole calendar. Nested CV needs this: an inner loop must rank blocks using
    only the development region's hours, or a truncated final block gets ranked
    on partial data and survives the ``MIN_BLOCK_HOURS`` screen it should fail.
    """
    if bands is not None:
        return SenseurCityDataset(
            data_dir, cities=cities, time_bands=bands, normalize=False, **common
        )
    return SenseurCityDataset(
        data_dir, cities=cities, time_range=(0.0, 1.0), normalize=False, **common
    )


def _block_stats(probe, block_hours):
    """``(t0, {block_id: (n_unique_hours, n_positive_hours)})`` for the probe.

    Counts are over distinct ``(city, hour)`` pairs, not windows: a city's devices
    all share one reference series, so window counts weight an hour by how many
    devices happened to be co-located then (1x-34x over this campaign) rather than
    by the data. Blocks must be ranked on the honest quantity.
    """
    labels = probe.compute_labels()
    times = probe.window_times()
    cities = probe.window_cities()
    t0 = times.min().floor("h")
    width = pd.Timedelta(hours=block_hours)

    seen: dict[tuple, float] = {}
    for c, t, y in zip(cities, times, labels):
        seen[(c, t)] = y
    stats: dict[int, list] = {}
    for (_c, t), y in seen.items():
        b = int((t - t0) // width)
        cell = stats.setdefault(b, [0, 0])
        cell[0] += 1
        cell[1] += int(y)
    return t0, {b: tuple(v) for b, v in stats.items()}


def _deal_blocks(probe, n_groups, block_hours, min_block_hours=MIN_BLOCK_HOURS):
    """Deal the probe's blocks into ``n_groups`` balanced groups; return
    ``(groups, t0, stats)`` where ``groups[i]`` is a list of block ids and
    ``stats`` maps block id -> ``(n_unique_hours, n_positive_hours)``.

    Blocks are sorted by positive rate and dealt *serpentine* (0..n-1, n-1..0,
    ...). Straight round-robin biases the first group high whenever the block
    count is not a multiple of ``n_groups``; reversing on alternate passes pairs a
    high-rate block with a low-rate one in the same group, so every group ends up
    with a similar positive rate and a similar size.
    """
    t0, stats = _block_stats(probe, block_hours)
    usable = {b: v for b, v in stats.items() if v[0] >= min_block_hours}
    return _serpentine_deal(usable, n_groups, block_hours), t0, usable


def _serpentine_deal(usable, n_groups, block_hours=None) -> list[list[int]]:
    """Deal an explicit ``{block_id: (n_hours, n_pos)}`` mapping into ``n_groups``.

    Split out of ``_deal_blocks`` so nested CV can re-deal an arbitrary *subset*
    of blocks (a development region) without re-probing the dataset or
    recomputing statistics. The dealing rule lives here only, so the outer and
    inner loops are guaranteed to balance groups the same way.
    """
    if len(usable) < n_groups:
        width = f" of {block_hours}h" if block_hours is not None else ""
        raise RuntimeError(
            f"only {len(usable)} usable blocks{width} "
            f"(need >= {n_groups}); lower --block-hours or --k"
        )
    order = sorted(usable, key=lambda b: (-(usable[b][1] / usable[b][0]), b))
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, b in enumerate(order):
        cycle, pos = divmod(i, n_groups)
        groups[pos if cycle % 2 == 0 else n_groups - 1 - pos].append(b)
    return groups


def _group_positive_rate(blocks, stats) -> float:
    """Unique-hour positive rate of a set of blocks."""
    n = sum(stats[b][0] for b in blocks if b in stats)
    p = sum(stats[b][1] for b in blocks if b in stats)
    return (p / n) if n else float("nan")


def _most_representative_group(groups, stats) -> int:
    """Index of the group whose positive rate is closest to the pooled rate.

    Serpentine dealing equalises the groups well but not exactly — on a partial
    final pass the low-index groups keep taking the remaining high-rate blocks. So
    when only *one* group is singled out (the leave-one-city-out val slice), pick
    the one that actually matches the whole rather than group 0 by convention.
    """
    overall = _group_positive_rate([b for g in groups for b in g], stats)
    return min(range(len(groups)),
               key=lambda i: abs(_group_positive_rate(groups[i], stats) - overall))


def _bands_for_blocks(blocks, t0, block_hours, embargo_hours, clip=None):
    """``[(lo, hi), ...]`` wall-clock bands for a set of block ids.

    Each band starts ``embargo_hours`` into its block, so consecutive blocks are
    separated by that gap no matter which folds they were dealt to. Contracting
    only the start (not both ends) is enough: every boundary is then purged
    exactly once, at a cost of ``embargo_hours`` per block rather than twice that.
    """
    if embargo_hours >= block_hours:
        raise ValueError(
            f"embargo_hours={embargo_hours} must be smaller than block_hours="
            f"{block_hours}, or purging consumes the whole block and no window "
            "survives. Raise --block-hours or lower --embargo-hours."
        )
    width = pd.Timedelta(hours=block_hours)
    gap = pd.Timedelta(hours=embargo_hours)
    bands = []
    for b in sorted(blocks):
        lo = t0 + b * width
        band = (lo + gap, lo + width)
        if clip is not None:
            # A development region ends at an arbitrary instant, but blocks are a
            # fixed width, so the last block overhangs it. Without this clip the
            # band would reach past the boundary into the outer test region --
            # a direct leak of test hours into inner training.
            band = (max(band[0], clip[0]), min(band[1], clip[1]))
            if band[0] >= band[1]:
                continue
        bands.append(band)
    return bands




# --------------------------------------------------------------------------- #
# Tuned hyperparameters (produced by find_best_model.py)
# --------------------------------------------------------------------------- #
def tuned_hp_filename(model, experiment, context_size, protocol,
                      scaling="minmax") -> str:
    """Filename find_best_model writes its pooled winner to.

    The input size, the CV protocol and the input scaling all change which HPs
    win, so all three are tagged into the name (``"minmax"`` is left untagged).
    """
    tag = "" if scaling == "minmax" else f"_{scaling}"
    return f"best_{model}_{experiment}_ctx{context_size}_{protocol}{tag}.json"




# --------------------------------------------------------------------------- #
# Nested cross-validation
# --------------------------------------------------------------------------- #
# A flat protocol puts val and test at the SAME level, so one val split both
# picks the early-stopping epoch and ranks hyperparameters. Double-dipped, so the
# reported test metric is optimistic by an unknown amount.
#
# Nested CV separates the two jobs. Each OUTER fold holds out a test set that
# influences nothing but the final score. Within its development region an INNER
# loop re-deals the remaining blocks and picks hyperparameters; the winner is
# refit on dev and scored once on the outer test.
#
# Two-stage API: planning is cheap (wall-clock arithmetic only), materializing is
# expensive (builds datasets). Keeping them apart lets a resumed run skip a
# completed outer fold without paying to construct its datasets.
#
# The purge carries over for free. ``_bands_for_blocks`` contracts each band at
# its start, so the gap between consecutive blocks is the embargo regardless of
# which group each block landed in. Re-dealing the same block ids into inner
# groups is therefore purge-safe with no recomputation.
#
# Scaler invariant, enforced by ``materialize_outer`` and asserted in the tests:
#   every dataset a model TRAINS on fits its own scaler; every dataset a model is
#   only SCORED on is built with the scaler of the split it was trained on.
# Fitting one scaler on the whole dev region would leak dev-wide min/max into a
# training split that only saw part of it, and silently couple the inner folds.


@dataclass(frozen=True)
class InnerSpec:
    """One inner fold: purged wall-clock bands for its train and val splits."""
    idx: int
    train_bands: tuple
    val_bands: tuple


@dataclass(frozen=True)
class OuterSpec:
    """A complete outer fold, described without touching the data.

    ``refit_idx`` indexes ``inner``: that fold's (train, val) pair IS the refit
    pair, so the winning hyperparameters are refit under the same data budget the
    inner loop selected them with, and no extra dataset is built. Its val split
    serves only as the early-stopping signal during the refit.

    ``test_bands is None`` means "the whole of ``test_cities``" -- the
    leave-one-city-out case, where the split is geographic and no temporal purge
    applies, because the held-out city contributed nothing to any block
    statistic, scaler or inner fold.
    """
    outer_id: str
    experiment: str
    protocol: str
    train_cities: tuple
    test_cities: tuple
    inner: tuple
    refit_idx: int
    test_bands: tuple | None
    diag: dict


@dataclass
class OuterBundle:
    """Materialized datasets for one OuterSpec (screened; may drop inner folds)."""
    spec: OuterSpec
    inner: list
    refit: tuple
    test: object
    dists: list
    dropped_inner: list


def _inner_specs_from_blocks(dev_blocks, stats, t0, block_hours, embargo,
                             n_inner, clip=None):
    """Re-deal ``dev_blocks`` into ``n_inner`` balanced groups -> InnerSpecs.

    Returns ``(specs, refit_idx, groups)``. ``refit_idx`` is the group whose
    positive rate sits closest to the development region's overall rate, matching
    how leave-one-city-out already picks its val slice.
    """
    sub = {b: stats[b] for b in dev_blocks if b in stats}
    groups = _serpentine_deal(sub, n_inner, block_hours)
    specs = []
    for j in range(n_inner):
        train_blocks = [b for jj, g in enumerate(groups) if jj != j for b in g]
        specs.append(InnerSpec(
            idx=j,
            train_bands=tuple(_bands_for_blocks(train_blocks, t0, block_hours,
                                                embargo, clip=clip)),
            val_bands=tuple(_bands_for_blocks(groups[j], t0, block_hours,
                                              embargo, clip=clip)),
        ))
    return specs, _most_representative_group(groups, sub), groups


def nested_blocked_plan(data_dir, city, k_outer, n_inner, *,
                        block_hours=DEFAULT_BLOCK_HOURS, embargo_hours=None,
                        **common):
    """Purged blocked CV, nested. Outer fold o tests on group o; the remaining
    ``k_outer - 1`` groups are re-dealt into ``n_inner`` inner folds."""
    embargo = _resolve_embargo(embargo_hours, common)
    probe = _probe_dataset(data_dir, (city,), **common)
    outer_groups, t0, stats = _deal_blocks(probe, k_outer, block_hours)

    specs = []
    for o in range(k_outer):
        test_blocks = outer_groups[o]
        dev_blocks = [b for j, g in enumerate(outer_groups) if j != o for b in g]
        try:
            inner, refit_idx, groups = _inner_specs_from_blocks(
                dev_blocks, stats, t0, block_hours, embargo, n_inner)
        except RuntimeError as e:
            print(f"  [skip] {city} blocked-outer {o + 1}: {e}")
            continue
        specs.append(OuterSpec(
            outer_id=f"{city}-b{o + 1}", experiment="single", protocol="blocked",
            train_cities=(city,), test_cities=(city,),
            inner=tuple(inner), refit_idx=refit_idx,
            test_bands=tuple(_bands_for_blocks(test_blocks, t0, block_hours, embargo)),
            diag={"dev_blocks": sorted(dev_blocks),
                  "test_blocks": sorted(test_blocks),
                  "inner_groups": [sorted(g) for g in groups],
                  "dev_pos_rate": _group_positive_rate(dev_blocks, stats),
                  "test_pos_rate": _group_positive_rate(test_blocks, stats)},
        ))
    return specs


def nested_loco_plan(data_dir, cities, n_inner, *,
                     block_hours=DEFAULT_BLOCK_HOURS, embargo_hours=None,
                     **common):
    """Leave-one-city-out outside, blocked CV on the two dev cities inside.

    Fold count is ``len(cities)`` -- geography fixes it at 3, so there is no
    ``k_outer`` here.

    ``_block_stats`` counts distinct ``(city, hour)`` pairs, so one block id is
    one 14-day wall-clock window spanning BOTH dev cities, and both cities' hours
    always land in the same group. Every inner train and val therefore holds both
    cities, which keeps the scaler and pos_weight statistics stable across inner
    folds and stops an inner fold degenerating into a second, accidental
    leave-one-city-out split.

    The inner criterion is temporal while the outer is geographic. That mismatch
    is deliberate: with only two dev cities a geographic inner loop gives two
    folds, each training on a single city, so hyperparameters would be chosen
    under half the data budget the refit actually gets.
    """
    embargo = _resolve_embargo(embargo_hours, common)
    specs = []
    for held in cities:
        others = tuple(c for c in cities if c != held)
        if not others:
            continue
        try:
            probe = _probe_dataset(data_dir, others, **common)
            _, t0, stats = _deal_blocks(probe, n_inner, block_hours)
            inner, refit_idx, groups = _inner_specs_from_blocks(
                list(stats), stats, t0, block_hours, embargo, n_inner)
        except RuntimeError as e:
            print(f"  [skip] held-out {held}: {e}")
            continue
        specs.append(OuterSpec(
            outer_id=f"heldout-{held}", experiment="multi", protocol="blocked",
            train_cities=others, test_cities=(held,),
            inner=tuple(inner), refit_idx=refit_idx,
            test_bands=None,
            diag={"held": held, "dev_cities": list(others),
                  "inner_groups": [sorted(g) for g in groups]},
        ))
    return specs


def nested_plan(data_dir, *, experiment, protocol, cities, k_outer, n_inner,
                block_hours=DEFAULT_BLOCK_HOURS, embargo_hours=None, **common):
    """Dispatcher over the two evaluation designs.

    ``single``/``blocked`` pools the per-city plans over every city; ``multi`` is
    one fold per held-out city and ignores ``k_outer``.
    """
    if protocol != "blocked":
        raise ValueError(f"unknown protocol {protocol!r}; expected 'blocked'")
    kw = dict(block_hours=block_hours, embargo_hours=embargo_hours, **common)
    if experiment == "multi":
        return nested_loco_plan(data_dir, list(cities), n_inner, **kw)
    if experiment == "single":
        return [s for city in cities
                for s in nested_blocked_plan(data_dir, city, k_outer, n_inner, **kw)]
    raise ValueError(f"unknown experiment {experiment!r}; expected 'single' or 'multi'")


def materialize_outer(data_dir, spec, *, min_inner=2, **common):
    """Build the datasets for one OuterSpec, screening degenerate splits.

    Returns ``None`` when the outer fold is unusable: fewer than ``min_inner``
    inner folds survive the single-class screen, the refit fold is among the
    casualties, or the outer test is single-class. A single-class split leaves
    PR-AUC and ROC-AUC undefined and gives early stopping nothing to discriminate
    on, so scoring one would be meaningless rather than merely noisy.
    """
    inner_ds, dists, dropped = [], [], []
    for isp in spec.inner:
        try:
            tr = _slice_bands(data_dir, list(isp.train_bands),
                              cities=spec.train_cities, **common)
            va = _slice_bands(data_dir, list(isp.val_bands), scaler=tr.scaler,
                              cities=spec.train_cities, **common)
        except RuntimeError as e:
            print(f"  [skip] {spec.outer_id} inner {isp.idx}: {e}")
            dropped.append(isp.idx)
            continue
        row = _fold_distribution(spec.outer_id, f"inner{isp.idx}", (tr, va),
                                 names=("train", "val"))
        dists.append(row)
        if row["single_class"]:
            dropped.append(isp.idx)
            continue
        inner_ds.append((isp.idx, tr, va))

    kept = {i for i, _, _ in inner_ds}
    if len(kept) < min_inner:
        print(f"  [skip outer] {spec.outer_id}: only {len(kept)} usable inner folds")
        return None
    if spec.refit_idx not in kept:
        print(f"  [skip outer] {spec.outer_id}: refit fold {spec.refit_idx} unusable")
        return None

    refit = next((tr, va) for i, tr, va in inner_ds if i == spec.refit_idx)
    try:
        if spec.test_bands is None:
            test = _slice(data_dir, (0.0, 1.0), scaler=refit[0].scaler,
                          cities=spec.test_cities, **common)
        else:
            test = _slice_bands(data_dir, list(spec.test_bands),
                                scaler=refit[0].scaler,
                                cities=spec.test_cities, **common)
    except RuntimeError as e:
        print(f"  [skip outer] {spec.outer_id}: test slice empty ({e})")
        return None

    trow = _fold_distribution(spec.outer_id, "test", (test,), names=("test",))
    dists.append(trow)
    if trow["single_class"]:
        print(f"  [skip outer] {spec.outer_id}: single-class test split")
        return None

    return OuterBundle(spec=spec, inner=[(tr, va) for _, tr, va in inner_ds],
                       refit=refit, test=test, dists=dists, dropped_inner=dropped)


# --------------------------------------------------------------------------- #
# Fold class-distribution screening
# --------------------------------------------------------------------------- #
# A fold is "single-class" (degenerate) when ANY of its train/val/test splits
# holds only one class. Under a chronological split this happens routinely,
# because a contiguous block is one season and the label is seasonal: Oslo has six
# straight months with no exceedance at all. The blocked protocol removes the
# confound; this screen stays as the check that it did. Such folds are meaningless and unsafe:
#   * train single-class  -> the model cannot learn both classes, so the fold
#     teaches nothing;
#   * val single-class    -> early stopping has no discriminative signal;
#   * test single-class   -> precision/recall/F1 degenerate and ROC/PR-AUC are
#     undefined (NaN).
# So we screen every fold, print and store its per-split class distribution, and
# skip any single-class fold for *all* models (keeping the comparison on a common
# set of usable folds).
def _fold_distribution(group, fold, datasets, names=("train", "val", "test")) -> dict:
    """Per-split class distribution for one fold + a ``single_class`` flag.

    ``datasets`` lines up with ``names`` -- ``(train, val, test)`` by default, but
    nested CV screens ``(train, val)`` inner pairs that have no test split at all.
    Reports both denominators: window
    counts (``*_n``), which the model actually trains on, and distinct
    ``(city, hour)`` counts (``*_unique_n``), which is how much independent
    information the split holds. They differ by the device-duplication factor,
    which swings 1x-34x over the campaign, so a window positive rate can be far
    from the underlying hourly one. ``single_class`` is True when any split has
    zero positives or zero negatives (see the note above).
    """
    row: dict = {"group": group, "fold": fold}
    single = False
    for name, ds in zip(names, datasets):
        n_neg, n_pos = ds.class_counts()
        n = n_neg + n_pos
        u_n, u_pos = ds.unique_hour_counts()
        row[f"{name}_n"] = n
        row[f"{name}_neg"] = n_neg
        row[f"{name}_pos"] = n_pos
        row[f"{name}_pos_rate"] = (n_pos / n) if n else float("nan")
        row[f"{name}_unique_n"] = u_n
        row[f"{name}_unique_pos"] = u_pos
        row[f"{name}_unique_pos_rate"] = (u_pos / u_n) if u_n else float("nan")
        if n_pos == 0 or n_neg == 0:
            single = True
    row["single_class"] = single
    return row


def _print_fold_dist(experiment, row) -> None:
    """One line per fold. Iterates whichever splits are present, so it works for
    a nested ``(train, val)`` inner pair as well as a full ``(train, val, test)``."""
    tag = "   >>> SKIP (single-class fold)" if row["single_class"] else ""
    parts = [
        f"{name} pos {row[f'{name}_pos']}/{row[f'{name}_n']} "
        f"({row[f'{name}_unique_pos_rate']:.3f} of {row[f'{name}_unique_n']} h)"
        for name in ("train", "val", "test")
        if f"{name}_n" in row
    ]
    print(f"  [dist] {experiment} | {row['group']} fold {row['fold']}: "
          + " | ".join(parts) + tag)
