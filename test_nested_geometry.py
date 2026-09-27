"""Leak and geometry assertions for the nested-CV fold planners.

Runs on plans only -- no model is trained -- so it is the cheap, fast check that
the nested splits are actually clean. It catches every leak class this design can
produce:

  * inner train/val not partitioning the development region
  * development blocks overlapping the outer test blocks
  * any two bands from different splits sitting closer than the embargo
  * a scaler fitted on data the model was not trained on
  * (--materialize) a window in one split touching an hour within the embargo of
    a window in another -- checked on the real windows, inputs and label together

Run:  python test_nested_geometry.py            # plans only, seconds once cached
      python test_nested_geometry.py --materialize   # also builds one bundle
      python test_nested_geometry.py --horizon 24    # the other horizons' embargo
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from compare_models import (
    DEFAULT_BLOCK_HOURS,
    _resolve_embargo,
    materialize_outer,
    nested_blocked_plan,
    nested_loco_plan,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AQM_DATA_DIR") or os.path.join(HERE, "senseurcity_data_v03", "dataset")
COMMON = {"context_size": 24, "overlap": 0.5, "horizon": 1}
K_OUTER, N_INNER = 5, 3

_fails: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        _fails.append(label)


def min_gap(a: list, b: list):
    """Smallest positive wall-clock gap between two sets of bands, or None if
    they overlap (which is itself a failure the caller checks for)."""
    best = None
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            if lo1 < hi2 and lo2 < hi1:
                return pd.Timedelta(0)          # overlap
            gap = lo2 - hi1 if lo2 >= hi1 else lo1 - hi2
            best = gap if best is None else min(best, gap)
    return best


def check_partition(spec, label):
    """Every inner fold's train+val must cover the same block set (the dev
    region), and no inner band may touch the outer test bands."""
    covers = []
    for isp in spec.inner:
        covers.append(tuple(sorted(set(isp.train_bands) | set(isp.val_bands))))
        overlap = set(isp.train_bands) & set(isp.val_bands)
        check(not overlap, f"{label} inner{isp.idx}: train/val disjoint")
    check(len(set(covers)) == 1,
          f"{label}: all {len(spec.inner)} inner folds cover the same dev region")


def check_embargo(spec, embargo, label):
    """Dev and outer test must be separated by at least the embargo."""
    if spec.test_bands is None:
        print(f"  SKIP  {label}: geographic split, no temporal purge applies")
        return
    dev = sorted({b for isp in spec.inner for b in isp.train_bands + isp.val_bands})
    g = min_gap(dev, list(spec.test_bands))
    ok = g is not None and g >= pd.Timedelta(hours=embargo)
    check(ok, f"{label}: dev-to-test gap {g} >= embargo {embargo}h")


def touched_hours(ds) -> np.ndarray:
    """Sorted unique wall-clock hours touched by any window of ``ds``: its input hours
    and its label hours together, as integer hours since the epoch."""
    span = ds._end_offset - ds._begin_offset + 1
    ends = ds.window_times().values.astype("datetime64[h]").astype(np.int64)
    return np.unique((ends[:, None] - np.arange(span)[None, :]).ravel())


def nearest_gap_hours(a: np.ndarray, b: np.ndarray) -> int:
    """Smallest |x - y| over x in ``a`` and y in ``b`` (both sorted)."""
    i = np.searchsorted(b, a)
    left = np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)])
    right = np.abs(b[np.clip(i, 0, len(b) - 1)] - a)
    return int(min(left.min(), right.min()))


def check_window_gap(a_ds, b_ds, embargo, label):
    """No hour shared between two splits' windows, and at least ``embargo`` clear
    hours between the nearest pair: touched hours must sit more than ``embargo`` apart."""
    g = nearest_gap_hours(touched_hours(a_ds), touched_hours(b_ds))
    check(g > embargo, f"{label}: nearest touched hours {g} h apart, "
                       f"{g - 1} clear hours >= embargo {embargo}h")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Antwerp")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--horizon", type=int, default=COMMON["horizon"])
    ap.add_argument("--data-dir", default=DATA,
                    help="SenseurCity v03 dataset/ folder (default: $AQM_DATA_DIR, "
                         "else ./senseurcity_data_v03/dataset)")
    args = ap.parse_args()
    COMMON["horizon"] = args.horizon
    data = args.data_dir

    embargo = _resolve_embargo(None, COMMON)
    print(f"embargo = {embargo}h (one sample's full span: {COMMON['context_size']} h of "
          f"inputs and the 24 h label window ending {COMMON['horizon']} h past them), "
          f"block_hours = {DEFAULT_BLOCK_HOURS}\n")

    print(f"== blocked plan ({args.city}) ==")
    bplan = nested_blocked_plan(data, args.city, K_OUTER, N_INNER,
                                block_hours=DEFAULT_BLOCK_HOURS, **COMMON)
    check(len(bplan) == K_OUTER, f"produced {len(bplan)} outer folds (want {K_OUTER})")
    for spec in bplan:
        dev = set(spec.diag["dev_blocks"])
        test = set(spec.diag["test_blocks"])
        check(not (dev & test), f"{spec.outer_id}: dev and test blocks disjoint")
        union = set().union(*spec.diag["inner_groups"])
        check(union == dev, f"{spec.outer_id}: inner groups partition dev blocks")
        check(sum(len(g) for g in spec.diag["inner_groups"]) == len(dev),
              f"{spec.outer_id}: inner groups are disjoint")
        check_partition(spec, spec.outer_id)
        check_embargo(spec, embargo, spec.outer_id)

    print("\n== loco plan ==")
    lplan = nested_loco_plan(data, ["Antwerp", "Oslo", "Zagreb"], N_INNER,
                             block_hours=DEFAULT_BLOCK_HOURS, **COMMON)
    check(len(lplan) == 3, f"produced {len(lplan)} outer folds (want 3)")
    for spec in lplan:
        check(spec.test_bands is None, f"{spec.outer_id}: whole-city test")
        check(spec.test_cities[0] not in spec.train_cities,
              f"{spec.outer_id}: held-out city absent from train cities")
        check_partition(spec, spec.outer_id)
        check_embargo(spec, embargo, spec.outer_id)

    if args.materialize:
        print("\n== materialize one blocked outer fold ==")
        b = materialize_outer(data, bplan[0], **COMMON)
        check(b is not None, "bundle built")
        if b is not None:
            for j, (tr, va) in enumerate(b.inner):
                check(va.scaler is tr.scaler,
                      f"inner{j}: val uses its own train's scaler")
            check(b.test.scaler is b.refit[0].scaler,
                  "test uses the refit train's scaler")
            check(b.refit in b.inner, "refit pair is one of the inner pairs")
            span = b.test._end_offset - b.test._begin_offset + 1
            check(span == embargo, f"one sample touches {span} h, equal to the default embargo {embargo}h")
            for j, (tr, va) in enumerate(b.inner):
                check_window_gap(tr, va, embargo, f"inner{j}: train vs val windows")
            check_window_gap(b.refit[0], b.test, embargo, "refit train vs test windows")
            check_window_gap(b.refit[1], b.test, embargo, "refit val vs test windows")
            print(f"  sizes: inner train {len(b.inner[0][0])}, "
                  f"val {len(b.inner[0][1])}, test {len(b.test)}")

    print(f"\n{'ALL CHECKS PASSED' if not _fails else f'{len(_fails)} FAILURES'}")
    for f in _fails:
        print(f"  - {f}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
