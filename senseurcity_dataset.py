"""PyTorch Dataset for the SenseurCity v03 air-quality data (classification).

Each sample is a ``context_size``-hour history of low-cost-sensor (LCS)
features paired with a single binary "harmful air quality" label derived from
the segment's own reference 24-hour average against the WHO 2021 limits:

    inputs : [SHT31TE, SHT31HE, Absolute_humidity, OPCN3PM25, OPCN3PM10]
             shape (context_size, 5)
    label  : [harmful]  1.0 when the mean of *any* reference pollutant over a
             24-hour window ending ``horizon`` hours past the last input hour
             exceeds its WHO 2021 24-hour guideline limit, else 0.0
             shape (1,)

The label is always a WHO 24-hour mean. ``horizon`` positions that 24-hour window
relative to the *last input hour*: the window ends ``horizon`` hours past it and
covers the 24 hours leading up to that point. ``horizon=0`` labels the 24 hours
ending at the last input hour (a nowcast); the default ``horizon=1`` forecasts
the 24-hour mean ending one hour past the history; larger values forecast
further out.

Data layout (one CSV per LCS device, 1-minute resolution)::

    <data_dir>/<City>_<deviceid>.csv

where City is one of {Antwerp, Oslo, Zagreb}. Every device CSV carries both the
LCS columns and the co-located reference columns (prefixed ``Ref.``).

Cleaning uses the dataset's built-in per-channel ``*_flag`` columns: a minute is
flagged (suspect/invalid) whenever its flag column is non-null (e.g. ``W``,
``OutliersMax``, ``High_values``, ``T.max``). Flagged minutes are dropped to NaN
*before* the data is aggregated to hourly means, so a flagged reading never
contaminates an hourly average. Windows whose inputs or reference are not fully
observed after cleaning are dropped rather than imputed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from glob import glob
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CITIES = ("Antwerp", "Oslo", "Zagreb")

# LCS channels used as inputs.
#   SHT31TE           external temperature
#   SHT31HE           external relative humidity
#   Absolute_humidity derived absolute humidity
#   OPCN3PM25         optical particle counter PM2.5
#   OPCN3PM10         optical particle counter PM10
INPUT_FEATURES = (
    "SHT31TE",
    "SHT31HE",
    "Absolute_humidity",
    "OPCN3PM25",
    "OPCN3PM10",
)

# Reference pollutants whose 24-hour average defines the harmful-air label.
OUTPUT_FEATURES = ("Ref.PM10", "Ref.PM2.5")

# WHO 2021 global air-quality guideline 24-hour mean limits (µg/m³). Air quality
# is considered harmful for a 24-hour period when the mean concentration exceeds
# these values. Source: WHO global air quality guidelines, 2021.
#   https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines
WHO_24H_LIMITS = {
    "Ref.PM10": 45.0,
    "Ref.PM2.5": 15.0,
}

# The WHO guideline is a 24-hour mean, so the harmful-air label is always
# averaged over this many hours, regardless of the input ``context_size``.
WHO_AVG_HOURS = 24

# Input scalings (see MinMaxScaler / LogZScaler). "minmax" is the default so
# every result produced before "logz" existed stays reproducible.
SCALINGS = ("minmax", "logz")

# Channels ``scaling="logz"`` log-transforms before standardizing: the OPC
# particle counts, which are positive and heavy-tailed. Temperature and humidity
# are near-symmetric and would be *skewed* by a log, so they are left alone.
LOG_FEATURES = ("OPCN3PM25", "OPCN3PM10")

# Built-in flag columns guarding the input channels. A non-null value marks the
# corresponding minute as flagged/invalid. ``Absolute_humidity`` is derived and
# has no flag of its own; it is implicitly guarded by the SHT31 flags. The
# reference (``Ref.*``) columns have no flag column in this dataset.
FLAG_FOR_FEATURE = {
    "SHT31TE": "SHT31TE_flag",
    "SHT31HE": "SHT31HE_flag",
    "Absolute_humidity": ("SHT31TE_flag", "SHT31HE_flag"),
    "OPCN3PM25": "OPCN3PM25_flag",
    "OPCN3PM10": "OPCN3PM10_flag",
}

# Module-wide cache of cleaned hourly arrays, keyed by (path, input_features,
# output_features). Cleaning is deterministic in these, so datasets that share
# cities (e.g. k-fold CV folds) reuse the parse instead of re-reading the CSV.
_LOAD_CACHE: dict = {}

# The same cleaned arrays, cached on disk so a *new process* skips the CSVs
# entirely. The dataset is ~21 GB of minute-resolution CSV that reduces to ~30 MB
# of hourly arrays; parsing all three cities costs ~8 minutes per process and a
# transient multi-GB peak per file, while loading the cache costs about a second.
# Keyed by file identity (absolute path, size, mtime), so an edited or replaced
# CSV is re-parsed instead of served stale. Set AQM_PARSE_CACHE to a directory to
# relocate it, or to 0/off/none/false to disable it.
_DISK_CACHE_VERSION = 1
_DISK_CACHE_ENV = "AQM_PARSE_CACHE"
_DISK_CACHE_OFF = ("0", "off", "none", "false")


def _disk_cache_dir() -> "str | None":
    """Directory holding the on-disk parse cache, or None when it is disabled."""
    raw = os.environ.get(_DISK_CACHE_ENV, "").strip()
    if raw and raw.lower() in _DISK_CACHE_OFF:
        return None
    if raw:
        return raw
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".parse_cache")


def _disk_cache_path(cache_dir: str, path: str, in_feats, out_feats) -> str:
    """Cache filename for one (CSV, feature lists) pair.

    The digest covers exactly what ``_read_clean``'s output depends on: the file's
    identity (absolute path, size, mtime) and the two feature lists.
    """
    st = os.stat(path)
    sig = "|".join([os.path.abspath(path), str(st.st_size), str(st.st_mtime_ns),
                    ",".join(in_feats), ",".join(out_feats),
                    f"v{_DISK_CACHE_VERSION}"])
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_dir, f"{stem}_{digest}.npz")


def _disk_cache_load(cache_file: str):
    """``(X, Y, idx)`` from the cache, or None when absent or unreadable.

    Every failure mode -- missing, truncated, half-written, produced by another
    version -- returns None so the caller re-parses. The cache is an optimization,
    never a source of truth, so a corrupt entry must not be able to fail a run.
    """
    if not os.path.exists(cache_file):
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as z:
            if bool(z["empty"][()]):
                return (None, None, None)
            return z["X"], z["Y"], z["idx"]
    except Exception:       # noqa: BLE001 - any unreadable cache means "re-parse"
        return None


def _disk_cache_store(cache_file: str, arrays) -> None:
    """Write ``(X, Y, idx)`` atomically; a failure is logged nowhere and is not fatal.

    The temp name ends in .npz because ``np.savez`` appends that suffix otherwise,
    which would leave the real file unwritten and a stray temp behind.
    """
    X, Y, idx = arrays
    tmp = f"{cache_file}.{os.getpid()}.tmp.npz"
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if X is None:
            np.savez(tmp, empty=np.array(True))
        else:
            np.savez(tmp, empty=np.array(False), X=X, Y=Y, idx=idx)
        os.replace(tmp, cache_file)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


@dataclass
class _Window:
    series_idx: int
    start: int


@dataclass
class MinMaxScaler:
    """Per-channel min-max scaler fit on the *training* split and reused for the
    others. Maps each of the 5 input channels to ``[0, 1]`` via
    ``(x - min) / (max - min)``.

    The binary "harmful" label is never scaled, so this scaler only carries
    input statistics. Constant channels (``max == min``) get a range of 1 so
    they map to 0 instead of dividing by zero.

    Fitting on the observed min and max makes the scale outlier-sensitive: one
    1031 ug/m3 OPC hour in a train band compresses that channel's whole
    interquartile range into ~0.01 of [0, 1]. ``scaling="logz"``
    (``LogZScaler``) exists for that reason; this stays the default so every
    result produced before it remains reproducible.
    """

    feature_min: np.ndarray
    feature_max: np.ndarray

    kind = "minmax"

    def __post_init__(self):
        self.feature_min = np.asarray(self.feature_min, dtype=np.float32)
        self.feature_max = np.asarray(self.feature_max, dtype=np.float32)
        self.feature_range = self._safe_range(self.feature_min, self.feature_max)

    @staticmethod
    def _safe_range(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
        rng = (mx - mn).astype(np.float32)
        rng[rng == 0] = 1.0
        return rng

    def transform_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_min) / self.feature_range

    def inverse_transform_inputs(self, x: np.ndarray) -> np.ndarray:
        return x * self.feature_range + self.feature_min

    def params(self) -> dict:
        """Serializable description, for artifacts that record how inputs were scaled."""
        return {"kind": self.kind,
                "feature_min": self.feature_min.tolist(),
                "feature_max": self.feature_max.tolist()}


@dataclass
class LogZScaler:
    """``log1p`` on the heavy-tailed channels, then per-channel standardization.

    Fit on the *training* split and reused for the others, exactly like
    ``MinMaxScaler``. ``log_channels`` holds the input-channel indices that are
    log-transformed first (see ``LOG_FEATURES``); every channel is then mapped to
    ``(x - mean) / std`` with statistics taken *after* the log.

    The OPC particle counts are positive, multiplicative and heavy-tailed (skew
    ~4-6, p99.9 well below the maximum), so on the raw scale a min-max fit leaves
    them with ~20-40x less spread than temperature or humidity. Taking the log
    first brings their skew to <1 and gives every channel comparable spread,
    which is what the network's initialization assumes.

    The binary "harmful" label is never scaled, and the WHO comparison still
    happens in physical ug/m3 -- only the inputs are transformed.
    """

    feature_mean: np.ndarray
    feature_std: np.ndarray
    log_channels: tuple = ()

    kind = "logz"

    def __post_init__(self):
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float32)
        self.feature_std = self._safe_std(np.asarray(self.feature_std, dtype=np.float32))
        self.log_channels = tuple(int(i) for i in self.log_channels)

    @staticmethod
    def _safe_std(sd: np.ndarray) -> np.ndarray:
        """Constant or unobserved channels get a scale of 1, never a divide by zero."""
        return np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32)

    def _log(self, x: np.ndarray) -> np.ndarray:
        if not self.log_channels:
            return x
        cols = list(self.log_channels)
        out = x.copy()
        # clip at 0: the counts are non-negative, and a tiny negative from
        # averaging would otherwise make log1p produce NaN.
        out[..., cols] = np.log1p(np.clip(out[..., cols], 0.0, None))
        return out

    def transform_inputs(self, x: np.ndarray) -> np.ndarray:
        t = self._log(np.asarray(x, dtype=np.float32))
        return (t - self.feature_mean) / self.feature_std

    def inverse_transform_inputs(self, x: np.ndarray) -> np.ndarray:
        t = np.asarray(x, dtype=np.float32) * self.feature_std + self.feature_mean
        if self.log_channels:
            cols = list(self.log_channels)
            t = t.copy()
            t[..., cols] = np.expm1(t[..., cols])
        return t

    def params(self) -> dict:
        """Serializable description, for artifacts that record how inputs were scaled."""
        return {"kind": self.kind,
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
                "log_channels": list(self.log_channels)}


class SenseurCityDataset(Dataset):
    """Sliding-window ``context_size``-hour LCS history dataset.

    Each sample's target is a binary "harmful" flag from a WHO 24-hour reference
    mean: harmful (1.0) when the mean of *any* reference pollutant over a 24-hour
    window exceeds its WHO 2021 limit (see ``WHO_24H_LIMITS`` — PM2.5 > 15 µg/m³
    or PM10 > 45 µg/m³). ``horizon`` positions that 24-hour label window relative
    to the last input hour, turning the task into a forecast (see ``horizon``
    below).

    Args:
        data_dir: path to the SenseurCity ``dataset/`` folder (the directory
            holding the per-device ``<City>_<id>.csv`` files).
        context_size: number of hourly input time steps per sample (default 24).
        split: which split to expose, one of {"train", "val", "test"}.

            Temporal mode (``held_out_city=None``, the default): the pooled
            wall-clock span across all devices is cut into three contiguous time
            bands by ``split_fractions``; a window is assigned to a split by its
            real calendar time (the whole window must fall inside the band). The
            bands are global — shared by every device — so the earliest part of
            the calendar is train, the latest is test, independent of device.

            City-held-out mode (``held_out_city`` set): geographic split. train
            is every usable window from the *other* cities (full time span);
            ``held_out_city`` is split chronologically into val (its first
            ``held_out_val_frac``) and test (the rest). Train and val/test are
            disjoint cities, so training data may be shuffled without leakage.
        split_fractions: ``(train, val, test)`` fractions of the global calendar
            span, in chronological order. Must be positive and sum to 1. Used in
            temporal mode only (ignored when ``held_out_city`` is set).
        cities: which of {Antwerp, Oslo, Zagreb} to include (default: all).
        held_out_city: if set, switch to city-held-out mode (see ``split``).
            Must be one of ``cities`` and leave at least one other city to train
            on (default None = temporal mode).
        held_out_val_frac: in city-held-out mode, the leading chronological
            fraction of the held-out city used for val; the rest is test
            (default 0.25 = "first quarter").
        input_features: LCS columns used as inputs. Must not start with ``Ref.``.
        output_features: ``Ref.*`` columns whose 24-hour average defines the
            harmful-air label. Every one needs a WHO limit in ``WHO_24H_LIMITS``.
        horizon: how many hours past the *last input hour* the 24-hour label
            window ends. The label averages the reference over the 24 hours
            leading up to that point,
            ``[s + context_size + horizon - 24, s + context_size + horizon)``
            where ``s`` is the window's first input hour; its length is always
            24 h (the WHO guideline), regardless of ``context_size``. So
            ``horizon=0`` labels the 24 h ending at the last input hour (a
            nowcast); the default ``horizon=1`` forecasts the 24-h mean ending one
            hour past the history; ``horizon=k`` ends k hours past it. Must be
            >= 0. (When ``context_size < 24`` the label window reaches back before
            the inputs; such windows are kept only where that history exists.)
        overlap: fraction in ``[0, 1)`` of a window's ``context_size`` input
            hours that may be shared with the previous emitted window. ``0.0``
            (default) gives non-overlapping, back-to-back windows (stride =
            ``context_size``); ``0.5`` lets consecutive windows share half their
            hours (stride = ``context_size / 2``). Higher values yield more
            (denser, more autocorrelated) windows. Consecutive emitted windows
            are guaranteed to overlap by *no more* than this fraction. This
            controls overlap *within* a split only — the splits themselves never
            share any hour (see ``split``).
        time_range: ``(start, end)`` fractions of the pooled calendar span. When
            set, emit only windows whose whole span lies in that single band,
            overriding ``split``/``split_fractions``. The contiguous k-fold CV
            primitive. Mutually exclusive with ``held_out_city``/``time_bands``.
        time_bands: an explicit list of ``(lo, hi)`` wall-clock intervals (any
            value ``np.datetime64`` accepts; ``hi`` exclusive). A window is kept
            when its whole span lies inside *any* one of them, so a split can be
            a union of disjoint time blocks rather than one contiguous stretch.
            This is what blocked cross-validation uses to give every fold the
            same seasonal coverage. Mutually exclusive with ``time_range`` and
            ``held_out_city``. The no-straddle rule still applies per band, so no
            window ever spans a band boundary.
        normalize: when True, apply per-channel min-max normalization to the 5
            input channels. Stats are fit on the *training* split only. Build the
            train split first, then pass its ``.scaler`` to the val/test splits
            via ``scaler=`` (or use ``make_splits`` which wires this up). Raw
            arrays are kept internally — normalization is applied per sample — so
            the WHO label is still computed from physical µg/m³.
        scaler: a scaler fit on train (``MinMaxScaler`` or ``LogZScaler``, per
            ``scaling``). Required for non-train splits when ``normalize=True``;
            fitting on val/test would leak. If omitted on the train split, one is
            fit and exposed as ``self.scaler``. Its kind must match ``scaling``.
        dtype: tensor dtype for the returned samples.
        scaling: which scaler to fit when ``normalize=True`` -- ``"minmax"``
            (default, per-channel min-max to [0, 1]) or ``"logz"`` (``log1p`` on
            the ``LOG_FEATURES`` channels, then per-channel standardization).
            Only the transform changes; the fit is on the train bands either way
            and the WHO label is always computed from physical µg/m³.
    """

    def __init__(
        self,
        data_dir: str,
        context_size: int = 24,
        split: str = "train",
        split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        cities: Sequence[str] = CITIES,
        input_features: Sequence[str] = INPUT_FEATURES,
        output_features: Sequence[str] = OUTPUT_FEATURES,
        horizon: int = 1,
        overlap: float = 0.0,
        normalize: bool = True,
        scaler: "MinMaxScaler | None" = None,
        held_out_city: str | None = None,
        held_out_val_frac: float = 0.25,
        time_range: tuple[float, float] | None = None,
        time_bands: Sequence[tuple] | None = None,
        dtype: torch.dtype = torch.float32,
        scaling: str = "minmax",
    ):
        cities = list(cities)
        if not cities:
            raise ValueError("`cities` must be a non-empty sequence")
        for c in cities:
            if c not in CITIES:
                raise ValueError(f"Unknown city {c!r}; expected one of {CITIES}")
        if context_size < 1:
            raise ValueError("context_size must be >= 1")

        # Label geometry: a fixed 24-hour WHO averaging window (independent of
        # context_size) positioned by ``horizon`` relative to the *last* input
        # hour. The window ends ``horizon`` hours past the last input hour and
        # spans the 24 hours leading up to that point:
        #     label window = [s + context_size + horizon - 24,
        #                     s + context_size + horizon)
        # where ``s`` is the window's first input hour. horizon=0 labels the 24 h
        # ending at the last input hour (nowcast); the default horizon=1 ends one
        # hour past the history.
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        self.horizon = int(horizon)
        self.label_hours = WHO_AVG_HOURS
        # Label window bounds relative to the window start s (``_hi`` exclusive).
        # ``_label_lo_rel`` is negative when the 24 h window reaches back before
        # the inputs (context_size + horizon < 24).
        self._label_lo_rel = context_size + self.horizon - self.label_hours
        self._label_hi_rel = context_size + self.horizon

        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        self.overlap = float(overlap)
        # Minimum gap (in hours) between consecutive emitted window starts. With
        # overlap=0 windows are back-to-back (stride=context_size); higher
        # overlap shrinks the stride so windows share more hours.
        self.window_stride = max(1, int(round(context_size * (1.0 - overlap))))

        splits = ("train", "val", "test")
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; expected one of {splits}")
        if len(split_fractions) != 3 or any(f <= 0 for f in split_fractions):
            raise ValueError("split_fractions must be three positive numbers")
        total = float(sum(split_fractions))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_fractions must sum to 1, got {total}")
        self.split = split
        self.split_fractions = tuple(float(f) for f in split_fractions)

        # --- Explicit chronological slice (k-fold CV primitive) -------------
        # When set, emit only windows whose whole span falls in the fractional
        # band [start, end) of the pooled time span of the loaded `cities`. This
        # overrides split / split_fractions / held_out_city band logic and lets a
        # caller carve arbitrary train/val/test slices (e.g. forward-chaining
        # folds). Mutually exclusive with held_out_city.
        self.time_range = time_range
        if time_range is not None:
            s, e = time_range
            if not (0.0 <= s < e <= 1.0):
                raise ValueError(
                    f"time_range must satisfy 0 <= start < end <= 1, got {time_range}"
                )
            if held_out_city is not None:
                raise ValueError("time_range and held_out_city are mutually exclusive")

        # --- Explicit multi-interval bands (blocked-CV primitive) ------------
        # A union of disjoint wall-clock intervals instead of one contiguous
        # stretch, so a split can interleave blocks across the whole calendar and
        # inherit its full seasonal range. Resolved once here into datetime64 so
        # the band logic downstream is uniform with the fractional modes.
        self.time_bands = self._normalize_bands(time_bands)
        if self.time_bands is not None:
            if time_range is not None:
                raise ValueError("time_bands and time_range are mutually exclusive")
            if held_out_city is not None:
                raise ValueError("time_bands and held_out_city are mutually exclusive")

        # --- City-held-out (geographic) mode --------------------------------
        self.held_out_city = held_out_city
        if held_out_city is not None:
            if held_out_city not in cities:
                raise ValueError(
                    f"held_out_city {held_out_city!r} must be one of cities={cities}"
                )
            train_cities = [c for c in cities if c != held_out_city]
            if not train_cities:
                raise ValueError(
                    "held_out_city leaves no other city to train on; pass more cities"
                )
            if not (0.0 < held_out_val_frac < 1.0):
                raise ValueError(
                    f"held_out_val_frac must be in (0, 1), got {held_out_val_frac}"
                )
            # train pulls from the other cities; val/test from the held-out city.
            load_cities = train_cities if split == "train" else [held_out_city]
        else:
            load_cities = cities
        self._load_cities = load_cities
        self.held_out_val_frac = float(held_out_val_frac)

        self.input_features = list(input_features)
        self.output_features = list(output_features)
        for f in self.input_features:
            if f.startswith("Ref."):
                raise ValueError(f"Input feature {f!r} cannot be a Ref. column")
        for f in self.output_features:
            if not f.startswith("Ref."):
                raise ValueError(f"Output feature {f!r} must start with 'Ref.'")

        # Every reference channel needs a WHO 24h limit to build the label.
        missing = [f for f in self.output_features if f not in WHO_24H_LIMITS]
        if missing:
            raise ValueError(
                f"Every output feature needs a WHO 24h limit; no limit known "
                f"for {missing}. Known limits: {sorted(WHO_24H_LIMITS)}"
            )
        # WHO limits aligned to output_features order.
        self._limits = np.array(
            [WHO_24H_LIMITS[f] for f in self.output_features], dtype=np.float32
        )

        self.normalize = bool(normalize)
        if scaling not in SCALINGS:
            raise ValueError(f"unknown scaling {scaling!r}; expected one of {SCALINGS}")
        self.scaling = scaling
        # A supplied scaler must match the requested scaling, or the split would be
        # transformed by one rule while every record describes it as another.
        if scaler is not None and getattr(scaler, "kind", "minmax") != scaling:
            raise ValueError(
                f"scaling={scaling!r} but the supplied scaler is "
                f"{getattr(scaler, 'kind', '?')!r}. Build the train split and every "
                "split that reuses its scaler with the same `scaling`."
            )
        # In time_range mode the caller controls which slice is "train" (fits the
        # scaler) and must pass it to the others, so the split-based guard is off.
        if (
            self.normalize and scaler is None and split != "train"
            and time_range is None
        ):
            raise ValueError(
                f"normalize=True on split={split!r} requires a `scaler` fit on "
                "train (fitting on val/test would leak). Build the train split "
                "first and pass its `.scaler`, or use `make_splits`."
            )
        # When normalize is off the scaler stays None; transforms become no-ops.
        self.scaler = scaler if self.normalize else None

        self.context_size = context_size
        self.dtype = dtype
        # Span a window touches, relative to its start s: from the earlier of the
        # first input hour (0) and the label window's first hour (_label_lo_rel,
        # negative when the 24 h label reaches back before the inputs) to the
        # later of the last input hour (context_size - 1) and the last label hour
        # (_label_hi_rel - 1). The split-band check uses these so no part of a
        # window (input or its shifted label) leaks across a split boundary.
        self._begin_offset = min(0, self._label_lo_rel)
        self._end_offset = max(self.context_size - 1, self._label_hi_rel - 1)

        files = self._list_files(data_dir, self._load_cities)
        if not files:
            raise FileNotFoundError(
                f"No LCS CSVs found in {data_dir!r} for cities {self._load_cities}"
            )

        # Load and clean every device once, keeping each device's hourly array
        # together with its timestamp index so windows can be assigned to splits
        # by their real calendar time (a single global chronological split)
        # rather than per-device row fractions.
        loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        loaded_cities: list[str] = []
        for path in files:
            X, Y, idx = self._load_one(path)
            if X is not None:
                loaded.append((X, Y, idx))
                # City of the device that produced this series. Kept so windows
                # can be grouped by (city, hour) for honest unique-hour counts —
                # every device in a city shares one reference series, so the same
                # hour is otherwise counted once per co-located device.
                loaded_cities.append(self._city_of(path))

        if not loaded:
            raise RuntimeError(
                f"No co-located reference data for cities={self._load_cities}."
            )

        # Time band [lo, hi) for this split. time_range mode cuts an explicit
        # fractional slice; temporal mode cuts the global span by split_fractions;
        # city-held-out mode keeps the whole span for train and cuts only the
        # held-out city into val (first quarter) / test (rest).
        # Every mode resolves to a *list* of [lo, hi) bands; the fractional and
        # held-out modes simply produce a single-element list.
        if self.time_bands is not None:
            bands = list(self.time_bands)
        elif self.time_range is not None:
            bands = [self._fractional_band(loaded, *self.time_range)]
        elif self.held_out_city is not None:
            bands = [self._held_out_band(loaded)]
        else:
            bands = [self._split_time_band(loaded)]
        self.bands = bands

        # Fit the min-max scaler on the train band only (never val/test), using
        # these same global bands so it matches the data the train split exposes.
        if self.normalize and self.scaler is None:
            self.scaler = self._fit_scaler(loaded, bands)

        # Per-device hourly arrays plus the sliding windows that index into them.
        self._series: list[tuple[np.ndarray, np.ndarray]] = []
        self._series_city: list[str] = []
        windows: list[tuple[np.datetime64, _Window]] = []

        begin_off = self._begin_offset
        end_off = self._end_offset
        for (X, Y, idx), city in zip(loaded, loaded_cities):
            starts = self._usable_window_starts(X, Y)
            if len(starts) == 0:
                continue
            # Keep a window only when its *entire* span lies inside one of this
            # split's time bands — from the earlier of the first input hour and
            # the label window's first hour, through the later of the last input
            # hour and the last label hour — so no window straddles a boundary
            # and no split leaks. (The label window can begin before the inputs.)
            # With several bands the test is an OR: a window must fit wholly
            # within *some* band, never across two.
            begin_times = idx[starts + begin_off]
            end_times = idx[starts + end_off]
            keep = np.zeros(len(starts), dtype=bool)
            for lo, hi in bands:
                keep |= (begin_times >= lo) & (end_times < hi)
            kept = starts[keep]
            if len(kept) == 0:
                continue
            sid = len(self._series)
            self._series.append((X, Y))
            self._series_city.append(city)
            windows.extend(
                (et, _Window(sid, int(s)))
                for s, et in zip(kept, end_times[keep])
            )

        if not windows:
            raise RuntimeError(
                f"No usable {split!r} windows for cities={cities}, "
                f"context_size={context_size}. Every candidate window was "
                "flagged/missing or fell outside the split's time band."
            )

        # Global chronological ordering: sort every window across all devices by
        # its last-hour timestamp so iterating the dataset in index order yields
        # time-sequential samples (device boundaries are not preserved — sample
        # i+1 is never earlier in wall-clock time than sample i).
        windows.sort(key=lambda tw: tw[0])
        self._windows = [w for _, w in windows]
        # Each window's last touched hour, parallel to ``_windows`` and therefore
        # sorted. Retained (rather than discarded with the sort key as before) so
        # callers can group windows by wall-clock hour — blocked-CV block
        # assignment, unique-hour class counts and the leakage checks all need it.
        self._window_times = pd.DatetimeIndex([t for t, _ in windows], tz="UTC")

    # ------------------------------------------------------------------ #
    # Loading & cleaning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _city_of(path: str) -> str:
        """City name from a ``<City>_<deviceid>.csv`` device filename."""
        return os.path.basename(path).split("_", 1)[0]

    @staticmethod
    def _normalize_bands(
        time_bands: "Sequence[tuple] | None",
    ) -> "list[tuple[pd.Timestamp, pd.Timestamp]] | None":
        """Coerce user-supplied ``(lo, hi)`` pairs to sorted UTC ``pd.Timestamp``.

        The hourly index carries tz-aware UTC Timestamps, so the band bounds must
        too — comparing a tz-aware Timestamp against a tz-naive one raises. Naive
        inputs are read as UTC. Accepts anything ``pd.Timestamp`` understands (ISO
        strings, datetime64, Timestamps). Reversed or empty intervals are a caller
        bug, so they raise rather than silently emitting nothing.
        """
        if time_bands is None:
            return None

        def _as_utc(v) -> pd.Timestamp:
            ts = pd.Timestamp(v)
            return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for band in time_bands:
            lo, hi = _as_utc(band[0]), _as_utc(band[1])
            if not lo < hi:
                raise ValueError(f"time_bands entries need lo < hi, got ({lo}, {hi})")
            out.append((lo, hi))
        if not out:
            raise ValueError("time_bands must be a non-empty sequence of (lo, hi)")
        out.sort(key=lambda b: b[0])
        return out

    @staticmethod
    def _list_files(data_dir: str, cities: Sequence[str]) -> list[str]:
        out: list[str] = []
        for city in cities:
            out.extend(sorted(glob(os.path.join(data_dir, f"{city}_*.csv"))))
        return out

    def _flag_columns(self) -> list[str]:
        cols: set[str] = set()
        for f in self.input_features:
            flags = FLAG_FOR_FEATURE.get(f)
            if flags is None:
                continue
            if isinstance(flags, str):
                cols.add(flags)
            else:
                cols.update(flags)
        return sorted(cols)

    def _load_one(self, path: str):
        """Cleaned hourly arrays for one device CSV, memoized in RAM and on disk.

        Cleaning is a pure function of ``path`` and the in/out feature lists, so
        results are cached module-wide (``_LOAD_CACHE``) to avoid re-reading the
        large CSVs when many datasets share cities (e.g. k-fold CV). The cached
        arrays are treated read-only by callers, so sharing them is safe.

        The same arrays also persist on disk (see ``_disk_cache_dir``), which is
        what makes a fresh process cheap -- parsing every city costs ~8 minutes
        and a transient multi-GB peak, loading the cache about a second. The key
        includes the CSV's size and mtime, so editing the data invalidates it, and
        an unreadable entry is silently re-parsed.
        """
        key = (path, tuple(self.input_features), tuple(self.output_features))
        if key in _LOAD_CACHE:
            return _LOAD_CACHE[key]

        cache_dir = _disk_cache_dir()
        cache_file = None
        if cache_dir is not None:
            try:
                cache_file = _disk_cache_path(cache_dir, path, key[1], key[2])
            except OSError:
                cache_file = None       # unstattable file: just parse it
        if cache_file is not None:
            hit = _disk_cache_load(cache_file)
            if hit is not None:
                _LOAD_CACHE[key] = hit
                return hit

        arrays = self._read_clean(path)
        if cache_file is not None:
            _disk_cache_store(cache_file, arrays)
        _LOAD_CACHE[key] = arrays
        return arrays

    def _read_clean(self, path: str):
        """Load one device CSV, clean it with the built-in flags and aggregate
        to hourly means.

        Returns ``(X, Y, idx)``: the float32 input/reference arrays and the
        regular hourly ``datetime64`` index they are aligned on, or ``(None,
        None, None)`` when the device has no co-located reference data.
        """
        flag_cols = self._flag_columns()
        usecols = (
            ["date"] + self.input_features + self.output_features + flag_cols
        )

        df = pd.read_csv(path, usecols=lambda c: c in set(usecols), low_memory=False)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        for c in self.input_features + self.output_features:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- Flag-based cleaning (minute resolution) ---------------------
        # A flag column is null for a good minute and carries a code (e.g. "W",
        # "OutliersMax", "High_values") for a flagged one. Null out every
        # flagged minute so it cannot enter the hourly mean.
        for feat in self.input_features:
            flags = FLAG_FOR_FEATURE.get(feat)
            if flags is None:
                continue
            flags = [flags] if isinstance(flags, str) else list(flags)
            flagged = pd.Series(False, index=df.index)
            for fc in flags:
                if fc in df.columns:
                    flagged |= df[fc].notna()
            df.loc[flagged, feat] = np.nan

        # --- Aggregate to hourly means ----------------------------------
        df = df[self.input_features + self.output_features].resample("1h").mean()
        if df.empty:
            return None, None, None

        # Trim to the co-located span: outside it every Ref column is NaN
        # (the device was not next to a reference station).
        target_present = df[self.output_features].notna().any(axis=1)
        if not target_present.any():
            return None, None, None
        df = df.loc[target_present.idxmax(): target_present[::-1].idxmax()]

        X = df[self.input_features].to_numpy(dtype=np.float32)
        Y = df[self.output_features].to_numpy(dtype=np.float32)
        idx = df.index.to_numpy()  # datetime64[ns], regular hourly within span
        return X, Y, idx

    def _split_time_band(
        self, loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple[np.datetime64, np.datetime64]:
        """``[lo, hi)`` wall-clock bounds for ``self.split``.

        The pooled calendar span across every device (earliest first hour to
        latest last hour) is cut at the cumulative ``split_fractions``, giving
        contiguous, device-independent time bands shared by all devices.
        """
        tmin = min(idx[0] for _, _, idx in loaded)
        tmax = max(idx[-1] for _, _, idx in loaded)
        span_ns = (tmax - tmin) / np.timedelta64(1, "ns")  # float nanoseconds
        f_train, f_val, _ = self.split_fractions
        t1 = tmin + np.timedelta64(int(span_ns * f_train), "ns")
        t2 = tmin + np.timedelta64(int(span_ns * (f_train + f_val)), "ns")
        # ``hi`` is exclusive; nudge the final test bound past the last hour so
        # the very last window is retained.
        eps = np.timedelta64(1, "ns")
        bands = {
            "train": (tmin, t1),
            "val": (t1, t2),
            "test": (t2, tmax + eps),
        }
        return bands[self.split]

    def _held_out_band(
        self, loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple[np.datetime64, np.datetime64]:
        """``[lo, hi)`` bounds for city-held-out mode.

        ``loaded`` holds the *other* cities for train (kept whole) and the
        held-out city for val/test (cut chronologically at ``held_out_val_frac``:
        leading fraction -> val, remainder -> test).
        """
        tmin = min(idx[0] for _, _, idx in loaded)
        tmax = max(idx[-1] for _, _, idx in loaded)
        eps = np.timedelta64(1, "ns")
        if self.split == "train":
            return tmin, tmax + eps  # keep every window from the training cities
        span_ns = (tmax - tmin) / np.timedelta64(1, "ns")
        cut = tmin + np.timedelta64(int(span_ns * self.held_out_val_frac), "ns")
        return (tmin, cut) if self.split == "val" else (cut, tmax + eps)

    def _fractional_band(
        self,
        loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        start: float,
        end: float,
    ) -> tuple[np.datetime64, np.datetime64]:
        """``[lo, hi)`` for the fractional slice ``[start, end)`` of the pooled
        time span of the loaded cities (used by k-fold CV)."""
        tmin = min(idx[0] for _, _, idx in loaded)
        tmax = max(idx[-1] for _, _, idx in loaded)
        span_ns = (tmax - tmin) / np.timedelta64(1, "ns")
        lo = tmin + np.timedelta64(int(span_ns * start), "ns")
        hi = tmin + np.timedelta64(int(span_ns * end), "ns")
        if end >= 1.0:  # include the final hour when the slice runs to the end
            hi = hi + np.timedelta64(1, "ns")
        return lo, hi

    def _log_channel_indices(self) -> tuple[int, ...]:
        """Positions of ``LOG_FEATURES`` in this dataset's input channels."""
        return tuple(i for i, f in enumerate(self.input_features) if f in LOG_FEATURES)

    def _fit_scaler(
        self,
        loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        bands: list[tuple[np.datetime64, np.datetime64]],
    ) -> "MinMaxScaler | LogZScaler":
        """Fit the input scaler over the train bands' observed hours.

        Only rows inside the union of ``bands`` (the train split's time coverage)
        and only finite values contribute, so the stats never see val/test data or
        NaN gaps. Only the inputs are scaled — the binary label is left raw.

        ``self.scaling`` selects the statistics: min/max (``"minmax"``) or the
        mean/std of the log-transformed channels (``"logz"``). The rows used are
        identical either way, so the two differ only in the transform.
        """
        feat_rows = []
        for X, _Y, idx in loaded:
            m = np.zeros(len(idx), dtype=bool)
            for lo, hi in bands:
                m |= (idx >= lo) & (idx < hi)
            if m.any():
                feat_rows.append(X[m])
        if not feat_rows:
            raise RuntimeError(
                "No observed hours inside the train bands, so no scaler can be "
                "fit. Check the band bounds against the data's calendar span."
            )
        # concatenate allocates, so in-place work below cannot touch _LOAD_CACHE.
        feats = np.concatenate(feat_rows, axis=0)

        if self.scaling == "logz":
            log_idx = self._log_channel_indices()
            if log_idx:
                cols = list(log_idx)
                feats[:, cols] = np.log1p(np.clip(feats[:, cols], 0.0, None))
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(feats, axis=0)
                std = np.nanstd(feats, axis=0)
            # Guard fully-missing channels (all-NaN in the band) -> identity scaling.
            mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
            return LogZScaler(mean, std, log_idx)

        fmin = np.nanmin(feats, axis=0)
        fmax = np.nanmax(feats, axis=0)
        # Guard fully-missing channels (all-NaN in the band) -> identity scaling.
        fmin = np.where(np.isfinite(fmin), fmin, 0.0).astype(np.float32)
        fmax = np.where(np.isfinite(fmax), fmax, 1.0).astype(np.float32)
        return MinMaxScaler(fmin, fmax)

    @staticmethod
    def _rolling_all(ok: np.ndarray, width: int, last_start: int) -> np.ndarray:
        """Boolean array over starts ``0..last_start`` that is True where every
        one of the ``width`` hours ``[s, s + width)`` is True in ``ok``."""
        csum = np.concatenate(([0], np.cumsum(ok.astype(np.int64))))
        return (csum[width: last_start + width + 1] - csum[: last_start + 1]) == width

    def _apply_stride(self, starts: np.ndarray) -> np.ndarray:
        """Thin out usable ``starts`` so consecutive emitted windows are at least
        ``window_stride`` hours apart, i.e. overlap by no more than ``overlap``.

        Greedy from the first usable start (so the earliest data is kept): emit a
        window, then skip every candidate closer than ``window_stride`` to it.
        NaN gaps simply act as larger skips. With ``overlap`` large enough that
        ``window_stride == 1`` this returns ``starts`` unchanged (dense windows).
        """
        stride = self.window_stride
        if stride <= 1 or len(starts) == 0:
            return starts
        kept = []
        next_ok = starts[0]
        for s in starts:
            if s >= next_ok:
                kept.append(s)
                next_ok = s + stride
        return np.asarray(kept, dtype=np.int64)

    def _usable_window_starts(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Start indices of fully-observed windows after cleaning, thinned to the
        configured ``overlap`` (see ``_apply_stride``).

        A window occupies inputs ``[s, s + context_size)`` and a 24-hour label
        window ``[s + context_size + horizon - label_hours, s + context_size +
        horizon)`` (ending ``horizon`` hours past the last input hour); both must
        be fully observed, so it is usable when every input hour *and* every
        reference hour in the label window is non-NaN. The label window may reach
        before ``s`` (when context_size < label_hours), so early starts that lack
        the required history are excluded too.

        Windows with any NaN are dropped (not imputed), so the model never trains
        on flagged or fabricated values.
        """
        n = len(X)
        input_ok = ~np.isnan(X).any(axis=1)          # per-hour input completeness
        label_ok = ~np.isnan(Y).any(axis=1)          # per-hour reference completeness

        C, h, L = self.context_size, self.horizon, self.label_hours
        # Candidate starts s: the label window [s + C + h - L, s + C + h) must fit
        # in [0, n), and so must the inputs [s, s + C). The label end (s + C + h)
        # is the binding upper bound; the label start (s + C + h - L) the binding
        # lower one when it is negative.
        first_start = max(0, L - C - h)     # keep label start p = s + C + h - L >= 0
        last_start = n - C - h              # inclusive; keep label end s + C + h <= n
        if last_start < first_start:
            return np.empty(0, dtype=np.int64)

        s_arr = np.arange(first_start, last_start + 1)
        # Inputs fully observed over [s, s + C), aligned to s_arr.
        window_input_ok = self._rolling_all(input_ok, C, last_start)[first_start:]
        # Reference fully observed over the label window [p, p + L), p = s + C + h - L.
        # _rolling_all(width=L) is indexed by window start p; index it at each p.
        p_arr = s_arr + (C + h - L)
        ref_roll = self._rolling_all(label_ok, L, int(p_arr[-1]))
        window_ref_ok = ref_roll[p_arr]
        starts = s_arr[window_input_ok & window_ref_ok]

        return self._apply_stride(starts)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        w = self._windows[idx]
        X, Y = self._series[w.series_idx]
        x = X[w.start: w.start + self.context_size].copy()

        # Reference 24-hour average over the label window
        # [start + C + horizon - L, start + C + horizon), i.e. the 24 hours ending
        # ``horizon`` hours past the last input hour. Taken from the *raw*
        # (physical) hours so the WHO comparison is in µg/m³ regardless of input
        # normalization.
        lo = w.start + self._label_lo_rel
        seg = Y[lo: lo + self.label_hours]
        means = seg.mean(axis=0)
        harmful = float(np.any(means > self._limits))
        y = np.array([harmful], dtype=np.float32)

        # Apply per-channel normalization (fit on train) to inputs; the binary
        # label is never scaled.
        if self.scaler is not None:
            x = self.scaler.transform_inputs(x)

        return (
            torch.as_tensor(x, dtype=self.dtype),
            torch.as_tensor(y, dtype=self.dtype),
        )

    def compute_labels(self) -> np.ndarray:
        """Binary ``{0., 1.}`` label for every window.

        Computed directly from the stored raw targets — same rule as
        ``__getitem__`` but skipping the input normalization — so it is cheap to
        use for class-balance / ``pos_weight`` calculations.
        """
        labels = np.empty(len(self._windows), dtype=np.float32)
        for i, w in enumerate(self._windows):
            _, Y = self._series[w.series_idx]
            lo = w.start + self._label_lo_rel
            seg = Y[lo: lo + self.label_hours]
            labels[i] = float(np.any(seg.mean(axis=0) > self._limits))
        return labels

    def class_counts(self) -> tuple[int, int]:
        """``(n_negative, n_positive)`` window counts."""
        labels = self.compute_labels()
        n_pos = int(labels.sum())
        return len(labels) - n_pos, n_pos

    def window_times(self) -> pd.DatetimeIndex:
        """Each window's last touched hour (UTC), sorted, parallel to the samples.

        This is the wall-clock key a window is split on, so it is also the key to
        group windows by when reporting or assigning them to blocks.
        """
        return self._window_times

    def window_cities(self) -> np.ndarray:
        """City name per window, parallel to ``window_times()``."""
        return np.array(
            [self._series_city[w.series_idx] for w in self._windows], dtype=object
        )

    def unique_hour_counts(self) -> tuple[int, int]:
        """``(n_unique, n_unique_positive)`` over distinct ``(city, hour)`` pairs.

        Every device in a city is co-located with the *same* reference station, so
        the same hour yields one window per device and ``len(self)`` counts it
        many times over. That multiplier is not constant: it tracks the
        co-location campaigns (all 34 Antwerp devices sit at the reference station
        during a campaign, ~12 during deployment), so window counts and window
        positive rates are weighted by campaign geometry rather than by the data.

        These counts collapse that duplication — they are the honest denominator
        for reading a split's class balance. The label is a property of the
        (city, hour) pair, so all duplicates of an hour share one label.
        """
        if not self._windows:
            return 0, 0
        labels = self.compute_labels()
        cities = self.window_cities()
        seen: dict[tuple, float] = {}
        for city, t, y in zip(cities, self._window_times, labels):
            seen[(city, t)] = y
        n_pos = int(sum(seen.values()))
        return len(seen), n_pos


def make_splits(
    data_dir: str,
    *,
    normalize: bool = True,
    **kwargs,
) -> tuple[SenseurCityDataset, SenseurCityDataset, SenseurCityDataset]:
    """Build the train/val/test ``SenseurCityDataset`` splits with a shared,
    train-fit min-max scaler.

    The scaler is fit on the train split only and reused for val/test, so no
    normalization statistics leak across the split boundary. All ``kwargs``
    (``context_size``, ``cities``, ``overlap``, ...) are passed through unchanged
    to every split. Returns ``(train, val, test)``.
    """
    train = SenseurCityDataset(data_dir, split="train", normalize=normalize, **kwargs)
    val = SenseurCityDataset(
        data_dir, split="val", normalize=normalize, scaler=train.scaler, **kwargs
    )
    test = SenseurCityDataset(
        data_dir, split="test", normalize=normalize, scaler=train.scaler, **kwargs
    )
    return train, val, test


if __name__ == "__main__":
    # Quick smoke test against the bundled dataset folder.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "senseurcity_data_v03", "dataset")

    # make_splits fits the train min-max scaler and shares it with val/test.
    train_ds, val_ds, test_ds = make_splits(data_dir, context_size=24, cities=("Antwerp",))
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        print(f"{name:>5} samples: {len(ds)}")

    x, y = train_ds[0]
    print(
        f"\nclassifier  input {tuple(x.shape)}  label {tuple(y.shape)}  value {y.item()}"
        f"  (horizon={train_ds.horizon}, label_hours={train_ds.label_hours})"
    )
    print("x[0] (normalized) =", [round(v, 3) for v in x[0].tolist()])

    n_neg, n_pos = train_ds.class_counts()
    print(f"harmful fraction over {len(train_ds)} train windows: {n_pos / len(train_ds):.3f}")
