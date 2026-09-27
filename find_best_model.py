"""Nested-CV hyperparameter search for the SenseurCity AQM classification task.

Each OUTER fold holds out a test set that influences nothing but the final
score. Inside its development region an INNER loop (``n_inner`` folds, re-dealt
from the dev blocks) ranks the grid; the winner is refit on dev -- with one inner
group kept back purely for early stopping -- and scored once on the outer test.
Inner training jobs are never given the outer-test file at all, so the leak is
not merely unselected but unrepresentable.

Two evaluation designs, both at context size 24, both run by default:

    single-blocked   purged blocked k-fold within each city   (5 outer/city)
    multi            leave-one-city-out                       (3 outer total)

Blocked outer folds put test blocks *between* training blocks, so the model
trains on hours after the ones it is scored on: the single-blocked scores
measure generalization across conditions within a city, not forward-in-time
extrapolation.

What the outer scores estimate is the *procedure* ("select hyperparameters by
inner CV on whatever data is available, then refit"), not any single
hyperparameter set -- different outer folds may pick different winners. The
single set reported in ``best_*.json`` is the argmax of each combo's inner-fold
score pooled over every outer fold, and ``hp_agreement`` records how often the
per-fold winners agreed with it. Low agreement is a finding, not a defect.

Results land in ``tuning_results/h<horizon>/`` -- one directory per label horizon,
because cache filenames carry the context size but **not** the horizon, while the
horizon changes which windows are usable and widens the purge embargo. Mixing two
horizons in one directory would collide on filenames with settings the resume guard
refuses. Each session appends to a JSONL cache as it goes and can resume after an
interruption; see the cache section for the corruption-resistance measures.

Trainings run through ``nested_worker``: each outer fold is stacked into tensors
once, and every (arch, combo, inner fold) becomes one job. ``--workers K`` runs K
jobs at a time on spawn-based worker processes sharing the GPU; ``--workers 1``
runs the same jobs inline. Only this process writes the cache, one record per
completed (outer, combo), so the worker count changes wall-clock time -- not what
is computed, and not how a resume behaves. Each worker costs ~1.2 GB of host RAM.
``--deterministic`` selects deterministic cuDNN kernels, making TCN/FUSION results
bit-reproducible (GRU is either way) at a measured ~+30% TCN / ~+10% FUSION time.

Run:
    python find_best_model.py                                   # single-blocked + multi
    python find_best_model.py --workers 3                       # same, 3 trainings at once
    python find_best_model.py --archs GRU --experiments single-blocked
    python find_best_model.py --horizon 12                      # -> tuning_results/h12/
    python find_best_model.py --dry-run-folds                   # geometry only
    python find_best_model.py --cities Zagreb --max-outer 1 --max-combos 2 --epochs 3

The dataset is read from --data-dir, else $AQM_DATA_DIR, else
./senseurcity_data_v03/dataset next to this file.

A smoke run -- any of --max-combos, --max-outer, --epochs other than the default, or a
subset of --cities -- writes to tuning_results/h<horizon>/smoke/ unless --out-dir is given, so
it can never leave records in the caches a full sweep resumes from. Resuming also
refuses a cache written under any different result-affecting setting (grid, epochs,
cities, embargo, trainer, ...), wherever it lives.

Input scaling: ``--scaling logz`` (the default, and what the paper reports)
applies log1p to the OPC PM channels and then standardizes every channel;
``--scaling minmax`` is per-channel min-max. The OPC counts are positive and
heavy-tailed, so a min-max fit -- whose denominator is one extreme hour -- leaves
them roughly 20-40x narrower than temperature and humidity. Scaling selects the
cache schema (minmax -> 4, logz -> 5), which tags every cache, artifact and
best_*.json name, so the two cannot mix and a cross-scaling resume is refused.

The summary table (summary_nested{schema}_ctx{ctx}.md) is rebuilt from every
best_*.json of the current schema in the output directory, so experiments run by
separate invocations (as the run_sweep scripts do) all appear in it.

Decision threshold: metrics under their existing names are at probability 0.5. Each
outer fold is also scored at a tuned threshold -- the median of the winning combo's
inner-fold F2-maximising val thresholds -- in the *_at_threshold fields and columns.
Each refit's val and test logits are saved next to the cache
(nested{schema}_*_preds/<outer>.npz) so another criterion can be applied later.

Artifacts. So figures and post-hoc analyses never need a rerun, every training's
outputs are saved next to its cache, each written atomically before the record it
belongs to (``<stem>`` is the cache name without ``.jsonl``):

    <stem>_inner/<outer>/cNN.npz   combo NN, per inner fold f: f{f}_val_logits (best
                                   epoch), f{f}_train_loss / f{f}_val_loss (per epoch),
                                   f{f}_best_epoch, f{f}_epochs_run
    <stem>_inner/<outer>/cNN.pt    combo NN: best-epoch state_dicts, one per inner fold
    <stem>_preds/<outer>.npz       refit: val/test logits + labels, and its curve
    <stem>_models/<outer>.pt       refit: state_dict, hp, tuned threshold, train scaler
    nested{schema}_<experiment>_ctx<ctx>_<protocol>_folds/<outer>.npz
                                   arch-independent: labels, window end hour (UTC) and
                                   city of every inner train/val split and the test split

Inner fold f is the f-th *kept* inner fold; ``inner_idx`` in the fold file maps it back
to the plan's fold index when a single-class fold was dropped. Load .pt files with
``torch.load(path, weights_only=True)``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from glob import glob
from typing import Any

import numpy as np
import torch

from nested_worker import TRAINER_ID, Job, init_worker, run_job, set_determinism
from senseurcity_dataset import CITIES
from compare_models import (
    DEFAULT_BLOCK_HOURS,
    N_INPUT_CHANNELS,
    materialize_outer,
    nested_plan,
    tuned_hp_filename,
    _print_fold_dist,
    _resolve_embargo,
)

# --------------------------------------------------------------------------- #
# Fixed regimen
# --------------------------------------------------------------------------- #
SEED = 42
CONTEXT_SIZE = 24        # WHO 24h-mean classification window; NOT tuned
OVERLAP = 0.5            # geometry constant, not a hyperparameter (see module docs)
HORIZON = 1              # label ends h hours past the last input hour; NOT tuned.
# Set from --horizon at startup (like CACHE_SCHEMA). It is not a hyperparameter:
# it defines the task. Changing it moves the label window relative to the inputs
# (at h = 1 the two overlap by 23 of 24 hours; at h = 24 they are disjoint) and it
# widens the purge embargo, which is one sample's span -- so folds, windows and
# results from different horizons are NOT comparable and never share a directory.
K_OUTER = 5              # outer folds per city (single-city experiments only)
N_INNER = 3              # inner folds carved from each development region
BLOCK_HOURS = DEFAULT_BLOCK_HOURS
EPOCHS = 100
PATIENCE = 10
DEFAULT_BATCH = 64
# Chunk size for validation/test forward passes. Equal to DEFAULT_BATCH, so the
# trainer reproduces the DataLoader path exactly. Checked on real folds: 1024 was
# only a few percent faster, and moved val loss by up to ~1e-4 -- enough to flip an
# early-stopping decision between two near-equal epochs.
EVAL_BATCH = DEFAULT_BATCH
# 3: trainings moved to nested_worker's GPU-tensor trainer.
# 4: the default purge embargo is one sample's span (25 h at context 24 / horizon 1)
#    instead of context + horizon + 24 (49 h), so every fold holds different windows.
# The file prefix follows the schema (nested4_*), so an older cache is never resumed into.
# 5: inputs scaled with log1p(OPC PM) + per-channel standardization rather than
#    per-channel min-max (--scaling logz). Same folds, same trainer, different
#    input values, so its records must never mix with schema 4's.
CACHE_SCHEMA = 5

# Input scaling -> cache schema. The schema tags every cache, artifact directory
# and best_*.json name, so the two scalings cannot write to the same files and a
# resume across them is refused. main() sets CACHE_SCHEMA from --scaling.
SCALING_SCHEMA = {"minmax": 4, "logz": 5}
DEFAULT_SCALING = "logz"

METRICS = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc")

# Where a smoke run writes when --out-dir is not given: a subdirectory of
# tuning_results/, so its caches, best_*.json and summary never mix with a sweep's.
SMOKE_SUBDIR = "smoke"

# Combos are ranked by the mean of this validation metric across inner folds.
# PR-AUC (average precision) rather than F1: F1 is read off a fixed 0.5
# threshold, so its value depends on the fold's class prior, and these priors
# differ by design. PR-AUC is rank-based and threshold-free, so it is comparable
# fold to fold. F1 stays in every table for continuity.
SELECT_METRIC = "pr_auc"

# Decision threshold. Ranking is threshold-free (PR-AUC), so the threshold is not a
# grid axis: it cannot change which combo wins, and every threshold would repeat the
# same trainings. Instead every inner training records the logit threshold that
# maximises F-beta on its own val predictions; a refit uses the median of its winning
# combo's inner-fold thresholds, and the outer test is scored at both probability
# 0.5 (the existing fields) and that threshold (the *_at_threshold fields). The refit
# val split never picks the threshold, so the choice is not tuned to the val split
# that also picked the refit's early-stopping epoch. Part of the resume guard.
#
# beta = 2 (F2): a missed harmful-air period costs more than a false alarm, so recall
# is weighted over precision. F2 is reported alongside F1 at both thresholds.
THRESHOLD_BETA = 2.0
THRESHOLD_RULE = (f"f{THRESHOLD_BETA:g}-max on val, median over the winning combo's "
                  "inner folds (v1)")
THRESHOLD_METRICS = ("accuracy", "precision", "recall", "f1", "f2")

# The two evaluation designs, as (experiment, protocol) pairs. These map onto
# the filename convention compare_models.tuned_hp_filename uses.
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "single-blocked": ("single", "blocked"),
    "multi": ("multi", "blocked"),
}
# What runs when --experiments is omitted.
DEFAULT_EXPERIMENTS: list[str] = ["single-blocked", "multi"]


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _grid(**axes: list) -> list[dict]:
    """Cartesian product of the named axes -> list of hyperparameter dicts.

    Keys keep their declared order, so every combo dict (and every results
    table) lists the hyperparameters in a stable, readable order.
    """
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


# Every NN grid must carry ``learning_rate`` and ``weight_decay``: they are
# trainer arguments, read in ``nested_worker.run_job``. Models
# are built by ``nested_worker.build_model``, so a worker needs no grid.
#
# Axes pinned rather than searched were measured inert on earlier (flat-CV) ctx24
# and ctx12 sessions, which are not part of this repository. Effect on mean val
# PR-AUC, against a fold-to-fold std of 0.231:
#   weight_decay  0.6-1.1e-3 for all three NNs  -> pinned at 0.0
#   dropout       2.0e-3 for GRU                -> pinned at 0.1
#   dropout       8.4e-3 for TCN, but the best combo used 0.3 in all four
#                 sessions, so pinning at 0.3 costs exactly 0.0000
NN_SPECS: dict[str, dict] = {
    # 4 x 4 x 1 x 2 x 1 = 32
    "GRU": {
        "grid": _grid(
            hidden_size=[16, 32, 64, 128],
            num_layers=[1, 2, 3, 4],
            dropout=[0.1],
            learning_rate=[1e-3, 3e-4],
            weight_decay=[0.0],
        ),
    },
    # 8 x 2 x 1 x 2 x 1 = 32
    "TCN": {
        "grid": _grid(
            tcn_channels=[
                (16, 16), (32, 32), (16, 16, 16), (32, 32, 32),
                (64, 64), (64, 64, 64), (32, 32, 32, 32), (64, 64, 64, 64),
            ],
            kernel_size=[2, 3],
            dropout=[0.3],
            learning_rate=[1e-3, 3e-4],
            weight_decay=[0.0],
        ),
    },
    # 2 x 2 x 2 x 2 x 1 x 1 x 2 x 1 = 32.  The proposed model: a causal CNN
    # branch and a GRU branch fused by a learned gate.
    "FUSION": {
        "grid": _grid(
            fusion_dim=[32, 64],
            cnn_channels=[32, 64],
            kernel_size=[3, 5],
            gru_hidden=[32, 64],
            num_layers=[1],
            dropout=[0.1],
            learning_rate=[1e-3, 3e-4],
            weight_decay=[0.0],
        ),
    },
}

ALL_ARCHS = list(NN_SPECS)


def _grid_for(arch: str) -> list[dict]:
    return NN_SPECS[arch]["grid"]


# Sanity: grid sizes are fixed per architecture and are part of the cache key
# (via the grid hash). A new arch missing from this dict fails loudly at import
# rather than silently producing an unresumable cache.
EXPECTED_GRID = {"GRU": 32, "TCN": 32, "FUSION": 32}
for _a in ALL_ARCHS:
    _n = len(_grid_for(_a))
    assert _n == EXPECTED_GRID[_a], f"{_a} grid has {_n} combos, expected {EXPECTED_GRID[_a]}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _nanmean(vals) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")


def _nanstd(vals) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.nanstd(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")


def _jsonify(obj: Any) -> Any:
    """JSON-safe copy of a hyperparameter dict (tuples -> lists)."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonify(v) for v in obj]
    return obj


def _grid_hash(grid: list[dict]) -> str:
    """Identity of a grid, stored in the cache meta line.

    Combo *index* is the cache key, so any change to the grid re-maps every
    index and silently invalidates a resume. Hashing the whole grid turns that
    into a loud refusal instead of corrupt results.
    """
    payload = json.dumps([_jsonify(h) for h in grid], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _atomic_write(path: str, text: str) -> None:
    """Write via a temp file + os.replace so a crash mid-write cannot leave a
    half-written JSON/CSV behind for a downstream reader to choke on."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
# One JSONL per (arch, experiment, protocol). Three record kinds:
#   meta   one per cache file, written when it is created: schema, grid hash, every knob
#   inner  one per (outer_fold, combo): the inner-fold val scores. No test
#          fields exist here at all -- that is the structural leak fix.
#   outer  one per outer fold: winning combo + the single outer-test score
#
# Resume keys: inner -> (outer, combo); outer -> outer. An outer fold is complete
# iff it has an ``outer`` record.
def _cache_path(out_dir, arch, experiment, context_size, protocol) -> str:
    return os.path.join(
        out_dir, f"nested{CACHE_SCHEMA}_{arch}_{experiment}_ctx{context_size}_{protocol}.jsonl"
    )


def _append_cache(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())      # flush alone can still be lost on a hard reboot


def _load_cache(path: str, grid: list[dict]) -> tuple[dict, dict, dict | None]:
    """Return ``(inner_by_key, outer_by_id, meta)``.

    Tolerates a truncated final line: the file is append-only, so a malformed
    line can only be the last one and skipping it loses at most one training's
    record. The previous implementation raised and made the session unresumable
    -- exactly the failure mode a forced Windows reboot produces.

    The meta line is read before any record is mapped. When its grid hash differs
    from ``grid``'s, no record is mapped at all and the caller's guard refuses the
    cache: mapping first would index ``grid`` with combo numbers from another grid
    (an IndexError under ``--max-combos``) before that refusal could be reached.
    """
    inner: dict[tuple[str, int], dict] = {}
    outer: dict[str, dict] = {}
    meta = None
    if not os.path.exists(path):
        return inner, outer, meta
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                kind = rec["kind"]
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"  [cache] skipping malformed line in {os.path.basename(path)}")
                continue
            if kind == "meta":
                meta = meta or rec          # the first meta describes the file
            else:
                records.append(rec)
    if meta is not None and meta.get("grid_hash") != _grid_hash(grid):
        return inner, outer, meta
    for rec in records:
        if rec["kind"] == "inner":
            # Guard the index->hp mapping per record, not just per session.
            c = rec.get("combo")
            if not (isinstance(c, int) and 0 <= c < len(grid)) \
                    or _jsonify(grid[c]) != rec.get("hp"):
                print(f"  [cache] combo {c} hp mismatch; dropping record")
                continue
            inner[(rec["outer"], c)] = rec
        elif rec["kind"] == "outer":
            outer[rec["outer"]] = rec
    return inner, outer, meta


# Settings introduced after some caches were already written, mapped to the value
# those older caches necessarily used. Without this, adding a setting would refuse
# every resume of every cache written before it existed -- e.g. nested4's metas
# predate "scaling", and min-max is the only thing they could have run under.
# Keys NOT listed here still count as a mismatch when missing, which is the point
# of the guard: an unrecognised absence means the cache cannot be vouched for.
_LEGACY_SETTING_DEFAULTS = {"scaling": "minmax"}


def _resume_mismatches(meta: dict, settings: dict) -> list[str]:
    """``key: cached -> now`` for every result-affecting setting a cache's meta line
    disagrees with. A key missing from the meta counts as a mismatch, unless it is
    in ``_LEGACY_SETTING_DEFAULTS`` (added later, with one possible past value)."""
    def cached(k):
        return meta[k] if k in meta else _LEGACY_SETTING_DEFAULTS.get(k)
    return [f"{k}: {cached(k)!r} -> {v!r}" for k, v in settings.items()
            if cached(k) != v]


# --------------------------------------------------------------------------- #
# Inner-fold records
# --------------------------------------------------------------------------- #
def _summarize_inner(outer_id, combo, hp, results, n_inner_used, secs) -> dict:
    """One cache record for (outer fold, combo) from its inner jobs' results, in fold
    order. Deliberately has no `test_*` field: the signature makes a test leak into
    ranking impossible.

    Also records each fold's F-beta-maximising val threshold (logit) and their median,
    which becomes the decision threshold if this combo wins the outer fold.
    """
    per_inner = [r["best_val_metrics"] or {} for r in results]
    thresholds = [r.get("val_threshold", float("nan")) for r in results]
    fb_at = [r.get("val_fbeta_at_threshold", float("nan")) for r in results]
    rec = {
        "kind": "inner", "outer": outer_id, "combo": combo, "hp": _jsonify(hp),
        "n_inner_used": n_inner_used, "secs": round(secs, 1),
        "inner": [{"fold": i, "val": v, "threshold": t, "val_fbeta_at_threshold": f}
                  for i, (v, t, f) in enumerate(zip(per_inner, thresholds, fb_at))],
    }
    for m in METRICS:
        vals = [v.get(m, float("nan")) for v in per_inner]
        rec[f"val_{m}_mean"] = _nanmean(vals)
        rec[f"val_{m}_std"] = _nanstd(vals)
    finite = [t for t in thresholds if t is not None and np.isfinite(t)]
    rec["thresholds"] = thresholds
    rec["threshold_median"] = float(np.median(finite)) if finite else None
    rec["val_fbeta_at_threshold_mean"] = _nanmean(fb_at)
    return rec


def _sigmoid(x):
    return None if x is None or not np.isfinite(x) else float(1.0 / (1.0 + np.exp(-x)))


def _preds_path(cache_path: str, outer_id: str) -> str:
    """Where a refit's val/test predictions live: next to its cache, one file per outer."""
    return os.path.join(cache_path[: -len(".jsonl")] + "_preds", f"{outer_id}.npz")


def _save_predictions(path: str, preds: dict) -> None:
    """Write atomically, before the outer record that points at it is appended."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[: -len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **preds)
    os.replace(tmp, path)


def _save_torch(path: str, obj) -> None:
    """``torch.save`` atomically, like ``_save_predictions``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _inner_artifact_paths(cache_path: str, outer_id: str, combo: int) -> tuple[str, str]:
    """``(npz, pt)`` for one combo's inner trainings in one outer fold."""
    base = os.path.join(cache_path[: -len(".jsonl")] + "_inner", outer_id, f"c{combo:02d}")
    return base + ".npz", base + ".pt"


def _model_path(cache_path: str, outer_id: str) -> str:
    return os.path.join(cache_path[: -len(".jsonl")] + "_models", f"{outer_id}.pt")


def _folds_path(out_dir, experiment, context_size, protocol, outer_id) -> str:
    return os.path.join(
        out_dir, f"nested{CACHE_SCHEMA}_{experiment}_ctx{context_size}_{protocol}_folds",
        f"{outer_id}.npz")


def _curve_arrays(res: dict, prefix: str = "") -> dict:
    return {f"{prefix}train_loss": np.asarray(res["train_loss"], dtype=np.float64),
            f"{prefix}val_loss": np.asarray(res["val_loss"], dtype=np.float64),
            f"{prefix}best_epoch": np.int64(res["best_epoch"]),
            f"{prefix}epochs_run": np.int64(res["epochs_run"])}


def _state_dict(res: dict) -> dict | None:
    state = res.get("best_state")
    return None if state is None else {k: torch.from_numpy(v) for k, v in state.items()}


def _save_inner_artifacts(cache_path, outer_id, arch, combo, hp, results) -> None:
    """Val logits, learning curves and best-epoch weights of one combo's inner
    trainings, ``results`` in kept-inner-fold order."""
    npz, pt = _inner_artifact_paths(cache_path, outer_id, combo)
    arrays = {}
    for f, r in enumerate(results):
        if r.get("val_logits") is not None:
            arrays[f"f{f}_val_logits"] = r["val_logits"]
        arrays.update(_curve_arrays(r, f"f{f}_"))
    _save_predictions(npz, arrays)
    _save_torch(pt, {"arch": arch, "hp": _jsonify(hp), "combo": combo, "outer": outer_id,
                     "input_channels": N_INPUT_CHANNELS,
                     "state_dicts": [_state_dict(r) for r in results]})


def _inner_key(rec):
    """Rank combos on inner-fold validation only.

    Primary: mean inner val PR-AUC, descending. Secondary: lower spread across
    inner folds, preferring a combo that is consistently good over one carried
    by a single lucky fold. Final tie-break is the combo index -- deterministic,
    reproducible, and carrying no information from any held-out split.
    """
    v = rec[f"val_{SELECT_METRIC}_mean"]
    s = rec[f"val_{SELECT_METRIC}_std"]
    return (
        -(v if not np.isnan(v) else -np.inf),
        s if not np.isnan(s) else np.inf,
        rec["combo"],
    )


# --------------------------------------------------------------------------- #
# Parallel execution
# --------------------------------------------------------------------------- #
@dataclass
class _FoldFiles:
    """One outer fold, stacked to tensors on disk for the workers."""
    dev_path: str           # inner train/val pairs (the refit pair is one of them)
    test_path: str          # outer test split; handed to refit jobs only
    refit_fold: int         # position of the refit pair among the inner pairs
    n_inner: int
    dropped_inner: list
    scalers: list           # per inner pair: the train-fit scaler's params(), or None
    fold_arrays: dict       # the arch-independent fold file (see module docs)


def _window_meta(ds, prefix: str, y: torch.Tensor) -> dict:
    """Labels, window end hour and city per sample of ``ds``, in sample order."""
    return {f"{prefix}_y": y.reshape(-1).numpy().astype(np.uint8),
            f"{prefix}_time": ds.window_times().values.astype("datetime64[s]"),
            f"{prefix}_city": ds.window_cities().astype(str)}


def _stack(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Every sample of ``ds`` as (X, Y). Built through ``__getitem__`` like DataLoader
    collation, so the values are exactly what the DataLoader trainer saw."""
    xs, ys = zip(*(ds[i] for i in range(len(ds))))
    return torch.stack(xs), torch.stack(ys)


def _export_outer(bundle, tmp_dir: str, tag: str) -> _FoldFiles:
    """Write one materialized outer fold as a dev file and a separate test file.

    Two files keep the structural leak guarantee: an inner job is only ever given
    the dev path, so outer-test tensors are never loaded where a candidate trains.
    """
    dev, refit_fold, fold, scalers = {}, None, {}, []
    for i, (tr, va) in enumerate(bundle.inner):
        dev[f"tr{i}_X"], dev[f"tr{i}_Y"] = _stack(tr)
        dev[f"va{i}_X"], dev[f"va{i}_Y"] = _stack(va)
        fold.update(_window_meta(tr, f"inner{i}_train", dev[f"tr{i}_Y"]))
        fold.update(_window_meta(va, f"inner{i}_val", dev[f"va{i}_Y"]))
        sc = tr.scaler
        # params() describes whichever scaler kind this run fit, so the artifact
        # records what was actually applied instead of assuming min/max.
        scalers.append(None if sc is None else sc.params())
        if tr is bundle.refit[0]:
            refit_fold = i
    if refit_fold is None:
        raise RuntimeError(f"{tag}: refit pair is not among the inner pairs")
    te_x, te_y = _stack(bundle.test)
    fold.update(_window_meta(bundle.test, "test", te_y))
    fold["refit_fold"] = np.int64(refit_fold)
    fold["inner_idx"] = np.array([s.idx for s in bundle.spec.inner
                                  if s.idx not in bundle.dropped_inner], dtype=np.int64)
    fold["dropped_inner"] = np.array(bundle.dropped_inner, dtype=np.int64)
    dev_path = os.path.join(tmp_dir, f"{tag}_dev.pt")
    test_path = os.path.join(tmp_dir, f"{tag}_test.pt")
    torch.save(dev, dev_path)
    torch.save({"X": te_x, "Y": te_y}, test_path)
    return _FoldFiles(dev_path, test_path, refit_fold, len(bundle.inner),
                      list(bundle.dropped_inner), scalers, fold)


class _Runner:
    """Runs Jobs inline (``workers == 1``) or on a spawn-based process pool.

    Both modes call the same ``run_job``, so the worker count decides where a
    training runs, never what it computes.
    """

    def __init__(self, workers: int, deterministic: bool = False):
        self.capacity = max(1, workers)
        self._pool = None
        if workers > 1:
            # spawn: CUDA cannot run in forked children, and Windows has no fork.
            self._pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=init_worker, initargs=(deterministic,))
        else:
            set_determinism(deterministic)

    def submit(self, job: Job) -> Future:
        if self._pool is not None:
            return self._pool.submit(run_job, job)
        fut: Future = Future()
        try:
            fut.set_result(run_job(job))
        except Exception as e:      # re-raised by fut.result(), as a pool would
            fut.set_exception(e)
        return fut

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)


def _run_outer(spec, exp_key, archs, grids, paths, caches, runner, ff, epochs):
    """Train every pending (arch, combo, inner fold) of one outer fold, then each
    arch's refit on the outer test.

    Jobs finish in any order. A combo's record is written only once all of its
    inner folds are back, and an arch's refit is queued -- ahead of remaining work
    -- only once all of its combos are recorded. The records and the resume units
    are therefore the same as the sequential loop's.
    """
    def job(arch, combo, fold, test_path=None, threshold=None):
        return Job(arch=arch, combo=combo, hp=grids[arch][combo], fold=fold,
                   dev_path=ff.dev_path, test_path=test_path,
                   input_channels=N_INPUT_CHANNELS, seed=SEED, epochs=epochs,
                   patience=PATIENCE, batch_size=DEFAULT_BATCH, eval_batch=EVAL_BATCH,
                   threshold=threshold, threshold_beta=THRESHOLD_BETA)

    def refit_job(arch):
        rows = [r for (o, _), r in caches[arch]["inner"].items() if o == spec.outer_id]
        winner = sorted(rows, key=_inner_key)[0]
        return job(arch, winner["combo"], ff.refit_fold, test_path=ff.test_path,
                   threshold=winner.get("threshold_median"))

    queue: deque = deque()
    missing: dict[str, set[int]] = {}
    for a in archs:
        if spec.outer_id in caches[a]["outer"]:
            continue
        missing[a] = {ci for ci in range(len(grids[a]))
                      if (spec.outer_id, ci) not in caches[a]["inner"]}
        if not missing[a]:
            queue.append(refit_job(a))
        for ci in sorted(missing[a]):
            queue.extend(job(a, ci, f) for f in range(ff.n_inner))

    partial: dict[tuple[str, int], dict[int, dict]] = {}
    running: set = set()
    try:
        while queue or running:
            while queue and len(running) < runner.capacity:
                running.add(runner.submit(queue.popleft()))
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                jb, res = fut.result()
                a = jb.arch

                if jb.test_path is not None:            # refit on the outer test
                    win = caches[a]["inner"][(spec.outer_id, jb.combo)]
                    ppath = _preds_path(paths[a], spec.outer_id)
                    if res.get("predictions") is not None:
                        _save_predictions(ppath, {**res["predictions"], **_curve_arrays(res)})
                    mpath = _model_path(paths[a], spec.outer_id)
                    if res.get("best_state") is not None:
                        sp = ff.scalers[ff.refit_fold] or {}
                        _save_torch(mpath, {
                            "arch": a, "hp": _jsonify(jb.hp), "combo": jb.combo,
                            "outer": spec.outer_id, "input_channels": N_INPUT_CHANNELS,
                            "refit_fold": ff.refit_fold, "threshold_logit": jb.threshold,
                            "scaling": sp.get("kind"), "scaler": sp or None,
                            # Min-max artifacts keep these legacy field names, so every
                            # loader written against schema 4 still works. A logz
                            # artifact leaves them None and carries its mean/std in
                            # "scaler" instead.
                            "scaler_min": sp.get("feature_min"),
                            "scaler_max": sp.get("feature_max"),
                            "state_dict": _state_dict(res),
                        })
                    orec = {
                        "kind": "outer", "outer": spec.outer_id,
                        "winner_combo": jb.combo, "hp": _jsonify(jb.hp),
                        f"inner_val_{SELECT_METRIC}_mean": win[f"val_{SELECT_METRIC}_mean"],
                        "refit_idx": spec.refit_idx, "n_inner_used": ff.n_inner,
                        "dropped_inner": ff.dropped_inner,
                        "test": res["test_metrics"] or {},
                        "threshold": {
                            "rule": THRESHOLD_RULE,
                            "logit": jb.threshold, "prob": _sigmoid(jb.threshold),
                            "inner_logits": win.get("thresholds"),
                            "refit_val_fbeta_max_logit": res.get("val_threshold"),
                        },
                        "test_at_threshold": res.get("test_metrics_at_threshold") or {},
                        "val_at_threshold": res.get("val_metrics_at_threshold") or {},
                        "predictions": os.path.relpath(ppath, os.path.dirname(paths[a]))
                        if res.get("predictions") is not None else None,
                        "model": os.path.relpath(mpath, os.path.dirname(paths[a]))
                        if res.get("best_state") is not None else None,
                        "best_epoch": res["best_epoch"], "best_val_loss": res["best_val_loss"],
                        "secs": round(res["secs"], 1),
                    }
                    _append_cache(paths[a], orec)
                    caches[a]["outer"][spec.outer_id] = orec
                    tuned = orec["test_at_threshold"]
                    print(f"  [{a}|{exp_key}|{spec.outer_id}] winner combo {jb.combo}  "
                          f"-> OUTER TEST {SELECT_METRIC}="
                          f"{orec['test'].get(SELECT_METRIC, float('nan')):.3f}  "
                          f"f2@0.5={orec['test'].get('f2', float('nan')):.3f}  "
                          f"f2@p{orec['threshold']['prob'] or float('nan'):.3f}="
                          f"{tuned.get('f2', float('nan')):.3f}")
                    continue

                got = partial.setdefault((a, jb.combo), {})
                got[jb.fold] = res
                if len(got) < ff.n_inner:
                    continue
                del partial[(a, jb.combo)]
                ordered = [got[f] for f in range(ff.n_inner)]
                _save_inner_artifacts(paths[a], spec.outer_id, a, jb.combo, jb.hp, ordered)
                # secs = summed training time of the combo's inner jobs, as before
                rec = _summarize_inner(spec.outer_id, jb.combo, jb.hp, ordered, ff.n_inner,
                                       sum(r["secs"] for r in got.values()))
                _append_cache(paths[a], rec)
                caches[a]["inner"][(spec.outer_id, jb.combo)] = rec
                print(f"  [{a}|{exp_key}|{spec.outer_id}] combo {jb.combo + 1:>3}/{len(grids[a])}  "
                      f"val_{SELECT_METRIC}={rec[f'val_{SELECT_METRIC}_mean']:.3f}  "
                      f"({rec['secs']:.1f}s)")
                missing[a].discard(jb.combo)
                if not missing[a]:
                    queue.appendleft(refit_job(a))
    finally:
        for fut in running:
            fut.cancel()


# --------------------------------------------------------------------------- #
# One session = one (arch, experiment, protocol)
# --------------------------------------------------------------------------- #
def run_experiment(exp_key, data_dir, out_dir, *, archs, cities, context_size,
                   epochs, max_combos, max_outer, resume, runner=None,
                   deterministic=False, scaling=DEFAULT_SCALING, dry_run=False):
    """Run every arch over one evaluation design.

    Outer-major: a completed outer fold is never re-materialized on resume, and all
    archs share one (expensive) materialization per outer fold -- their jobs run
    interleaved when ``runner`` has several workers. ``runner`` defaults to inline.
    """
    experiment, protocol = EXPERIMENTS[exp_key]
    common = {"context_size": context_size, "overlap": OVERLAP, "horizon": HORIZON}
    embargo = _resolve_embargo(None, common)
    # Scaling belongs to materialization only. The planners build unnormalized
    # datasets purely to read fold geometry, so passing it down there would change
    # nothing and only widen the surface for a kwarg mismatch.
    mat_common = {**common, "scaling": scaling}
    print(f"\n{'#' * 70}\n{exp_key}  (experiment={experiment}, protocol={protocol})"
          f"  archs={archs}  embargo={embargo}h  scaling={scaling}"
          f"  out_dir={out_dir}\n{'#' * 70}")

    grids = {a: _grid_for(a) for a in archs}
    if max_combos is not None:
        grids = {a: g[:max_combos] for a, g in grids.items()}
    paths = {a: _cache_path(out_dir, a, experiment, context_size, protocol) for a in archs}
    # Everything that changes what a cached record means. A resume must match all of
    # it: the filename already fixes (arch, experiment, protocol, ctx, schema), and
    # an outer id such as "heldout-Zagreb" names a different fold under another city
    # set, so cities are compared too. --max-outer only truncates the plan and needs
    # no guard; --max-combos changes the grid hash.
    settings = {
        "trainer": TRAINER_ID, "cudnn_deterministic": deterministic,
        "embargo_hours": embargo, "epochs": epochs, "patience": PATIENCE, "seed": SEED,
        "k_outer": K_OUTER, "n_inner": N_INNER, "block_hours": BLOCK_HOURS,
        "overlap": OVERLAP, "horizon": HORIZON, "batch": DEFAULT_BATCH,
        "eval_batch": EVAL_BATCH, "cities": sorted(cities), "threshold_rule": THRESHOLD_RULE,
        "scaling": scaling,
    }

    # Check every cache before planning, which parses the dataset (minutes), so a
    # refused resume fails immediately. Nothing is written until after the plan.
    caches, metas = {}, {}
    for a in archs if not dry_run else ():
        want = _grid_hash(grids[a])
        if not resume:
            if os.path.exists(paths[a]) and os.path.getsize(paths[a]) > 0:
                raise SystemExit(
                    f"\n{paths[a]}\n  already holds records, and --no-resume would append a "
                    "second session to it; the next resume would then mix both.\n  Move "
                    "or delete that file first, or use a different --out-dir.")
            inner, outer, meta = {}, {}, None
        else:
            inner, outer, meta = _load_cache(paths[a], grids[a])
        if meta is not None and meta.get("grid_hash") != want:
            raise SystemExit(
                f"\n{paths[a]}\n  was written under a different grid "
                f"({str(meta.get('grid_hash', '?'))[:12]} with {meta.get('n_grid')} combos "
                f"!= {want[:12]} with {len(grids[a])}).\n"
                "  Combo index is the cache key, so resuming would mis-map every "
                "record (--max-combos changes the grid).\n  Archive that file and rerun, "
                "use a different --out-dir, or restore the old grid."
            )
        if meta is not None:
            diff = _resume_mismatches(meta, settings)
            if diff:
                raise SystemExit(
                    f"\n{paths[a]}\n  was written under different settings:\n    "
                    + "\n    ".join(diff)
                    + "\n  Its records would not mean the same thing under this run, so "
                    "resuming would mix the two.\n  Archive that file, use a different "
                    "--out-dir, or rerun with the cached settings."
                )
        caches[a] = {"inner": inner, "outer": outer}
        metas[a] = meta
        if inner or outer:
            bare = sum(1 for o, c in inner
                       if not os.path.exists(_inner_artifact_paths(paths[a], o, c)[0]))
            print(f"resume [{a}]: {len(outer)} outer folds complete, "
                  f"{len(inner)} inner records cached"
                  + (f" -- {bare} of them have no saved artifacts (logits, curves, "
                     "weights); archive the cache and rerun if those are needed" if bare else ""))

    plan = nested_plan(data_dir, experiment=experiment, protocol=protocol,
                       cities=cities, k_outer=K_OUTER, n_inner=N_INNER,
                       block_hours=BLOCK_HOURS, **common)
    if max_outer is not None:
        plan = plan[:max_outer]
    print(f"plan: {len(plan)} outer folds -> {[s.outer_id for s in plan]}")

    if dry_run:
        for spec in plan:
            print(f"\n  {spec.outer_id}: refit_idx={spec.refit_idx} "
                  f"inner={len(spec.inner)} test_bands="
                  f"{'whole city' if spec.test_bands is None else len(spec.test_bands)}")
            for k, v in spec.diag.items():
                print(f"      {k}: {v}")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    for a in archs:
        if metas[a] is None:
            _append_cache(paths[a], {
                "kind": "meta", "schema": CACHE_SCHEMA, "arch": a,
                "experiment": experiment, "protocol": protocol, "ctx": context_size,
                "grid_hash": _grid_hash(grids[a]), "n_grid": len(grids[a]),
                **settings,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    runner = runner or _Runner(1, deterministic)
    tmp_dir = tempfile.mkdtemp(prefix="aqm_nested_")
    try:
        for spec in plan:
            if all(spec.outer_id in caches[a]["outer"] for a in archs):
                print(f"\n[{spec.outer_id}] complete for every arch; not materializing")
                continue

            print(f"\n[{spec.outer_id}] materializing...")
            bundle = materialize_outer(data_dir, spec, **mat_common)
            if bundle is None:
                continue
            for row in bundle.dists:
                _print_fold_dist(exp_key, row)
            ff = _export_outer(bundle, tmp_dir, f"{exp_key}_{spec.outer_id}")
            del bundle              # jobs read the stacked tensors; free the datasets
            _save_predictions(_folds_path(out_dir, experiment, context_size, protocol,
                                          spec.outer_id), ff.fold_arrays)

            _run_outer(spec, exp_key, archs, grids, paths, caches, runner, ff, epochs)
            for p in (ff.dev_path, ff.test_path):
                with contextlib.suppress(OSError):
                    os.remove(p)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {a: _finalize(a, exp_key, experiment, protocol, grids[a], caches[a],
                         out_dir, context_size, scaling) for a in archs}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _finalize(arch, exp_key, experiment, protocol, grid, cache, out_dir, context_size,
              scaling=DEFAULT_SCALING):
    """Aggregate one session: pick the reported HP set, write tables + best_*.json."""
    inner_rows = list(cache["inner"].values())
    outer_rows = list(cache["outer"].values())
    if not inner_rows or not outer_rows:
        print(f"[{arch}|{exp_key}] nothing to finalize")
        return None

    # Aggregate-then-argmax: pool each combo's inner scores over every outer
    # fold. Free (all numbers already computed), paired across combos, and it
    # never touches a test split. The alternative -- rerunning inner selection on
    # all the data -- costs another full pass for a marginally better surrogate.
    by_combo: dict[int, list[float]] = {}
    for r in inner_rows:
        by_combo.setdefault(r["combo"], []).append(r[f"val_{SELECT_METRIC}_mean"])
    agg = [{"combo": c, f"val_{SELECT_METRIC}_mean": _nanmean(v),
            f"val_{SELECT_METRIC}_std": _nanstd(v), "n": len(v)}
           for c, v in by_combo.items()]
    agg.sort(key=_inner_key)
    best = agg[0]
    best_hp = grid[best["combo"]]

    fold_winners = [{"outer": r["outer"], "combo": r["winner_combo"], "hp": r["hp"]}
                    for r in outer_rows]
    agreement = (sum(1 for w in fold_winners if w["combo"] == best["combo"])
                 / len(fold_winners))

    test_vals = {m: [r["test"].get(m, float("nan")) for r in outer_rows] for m in METRICS}
    thr_vals = {m: [(r.get("test_at_threshold") or {}).get(m, float("nan")) for r in outer_rows]
                for m in THRESHOLD_METRICS}
    thr_probs = [p for p in ((r.get("threshold") or {}).get("prob") for r in outer_rows)
                 if p is not None]
    stem = f"nested{CACHE_SCHEMA}_{arch}_{experiment}_ctx{context_size}_{protocol}"

    # per-(outer, combo) inner table
    def num(v, fmt=".6f"):
        return "nan" if v is None or not np.isfinite(v) else format(v, fmt)

    hdr = (["outer", "combo", "n_inner_used"] + [f"val_{m}_mean" for m in METRICS]
           + ["threshold_median_logit", "val_fbeta_at_threshold_mean"])
    lines = [",".join(hdr)]
    for r in sorted(inner_rows, key=lambda r: (r["outer"], r["combo"])):
        lines.append(",".join([r["outer"], str(r["combo"]), str(r["n_inner_used"])]
                              + [num(r.get(f"val_{m}_mean")) for m in METRICS]
                              + [num(r.get("threshold_median")),
                                 num(r.get("val_fbeta_at_threshold_mean"))]))
    _atomic_write(os.path.join(out_dir, f"{stem}_inner.csv"), "\n".join(lines) + "\n")

    # per-outer-fold table -- the headline numbers
    counts = ("tp", "fp", "tn", "fn")
    hdr = (["outer", "winner_combo", "best_epoch"] + [f"test_{m}" for m in METRICS]
           + [f"test_{c}" for c in counts]
           + ["threshold_logit", "threshold_prob"]
           + [f"test_{m}_at_threshold" for m in THRESHOLD_METRICS]
           + [f"test_{c}_at_threshold" for c in counts] + ["hp"])
    lines = [",".join(hdr)]
    for r in sorted(outer_rows, key=lambda r: r["outer"]):
        thr = r.get("threshold") or {}
        tt = r.get("test_at_threshold") or {}
        lines.append(",".join(
            [r["outer"], str(r["winner_combo"]), str(r["best_epoch"])]
            + [f"{r['test'].get(m, float('nan')):.6f}" for m in METRICS]
            + [str(r["test"].get(c, "")) for c in counts]
            + [num(thr.get("logit")), num(thr.get("prob"))]
            + [num(tt.get(m)) for m in THRESHOLD_METRICS]
            + [str(tt.get(c, "")) for c in counts]
            + [json.dumps(r["hp"]).replace(",", ";")]))
    _atomic_write(os.path.join(out_dir, f"{stem}_outer.csv"), "\n".join(lines) + "\n")

    payload = {
        "schema": CACHE_SCHEMA, "arch": arch, "experiment": experiment,
        "protocol": protocol, "context_size": context_size, "scaling": scaling,
        "overlap": OVERLAP, "horizon": HORIZON,
        "embargo_hours": _resolve_embargo(None, {"context_size": context_size, "horizon": HORIZON}),
        "selected_by": (f"argmax of mean inner-fold validation {SELECT_METRIC}, "
                        "pooled over all outer folds"),
        "hyperparameters": _jsonify(best_hp),
        f"inner_val_{SELECT_METRIC}_mean": best[f"val_{SELECT_METRIC}_mean"],
        f"inner_val_{SELECT_METRIC}_std": best[f"val_{SELECT_METRIC}_std"],
        "n_outer": len(outer_rows), "n_inner": N_INNER,
        "outer_fold_winners": fold_winners,
        "hp_agreement": agreement,
        f"nested_test_{SELECT_METRIC}_mean": _nanmean(test_vals[SELECT_METRIC]),
        f"nested_test_{SELECT_METRIC}_std": _nanstd(test_vals[SELECT_METRIC]),
        "nested_test_f1_mean": _nanmean(test_vals["f1"]),
        # Threshold-dependent metrics at probability 0.5 (as above) and at each outer
        # fold's tuned threshold. PR-AUC, ROC-AUC and the ranking do not depend on it.
        "threshold_rule": THRESHOLD_RULE, "threshold_beta": THRESHOLD_BETA,
        "nested_test_at05_means": {m: _nanmean(test_vals[m]) for m in THRESHOLD_METRICS},
        "nested_test_at_threshold_means": {m: _nanmean(thr_vals[m]) for m in THRESHOLD_METRICS},
        "nested_test_at_threshold_stds": {m: _nanstd(thr_vals[m]) for m in THRESHOLD_METRICS},
        "nested_test_f2_mean": _nanmean(test_vals["f2"]),
        "nested_test_f2_at_threshold_mean": _nanmean(thr_vals["f2"]),
        "nested_test_f2_at_threshold_std": _nanstd(thr_vals["f2"]),
        "nested_test_f1_at_threshold_mean": _nanmean(thr_vals["f1"]),
        "nested_test_f1_at_threshold_std": _nanstd(thr_vals["f1"]),
        "outer_thresholds": [{"outer": r["outer"],
                              "logit": (r.get("threshold") or {}).get("logit"),
                              "prob": (r.get("threshold") or {}).get("prob")}
                             for r in sorted(outer_rows, key=lambda r: r["outer"])],
        "threshold_prob_median": float(np.median(thr_probs)) if thr_probs else None,
        "threshold_prob_min": min(thr_probs) if thr_probs else None,
        "threshold_prob_max": max(thr_probs) if thr_probs else None,
        "nested_test_note": (
            "Unbiased estimate of the tuning+training PROCEDURE, not of the single "
            "'hyperparameters' set above. Different outer folds may select different "
            "hyperparameters; see outer_fold_winners and hp_agreement."
        ),
    }
    _atomic_write(os.path.join(out_dir, tuned_hp_filename(arch, experiment, context_size,
                                                          protocol, scaling)),
                  json.dumps(payload, indent=2) + "\n")

    print(f"\n[{arch}|{exp_key}] reported hp = {best_hp}")
    print(f"  inner val {SELECT_METRIC} = {best[f'val_{SELECT_METRIC}_mean']:.4f}"
          f"  | hp_agreement = {agreement:.2f} over {len(fold_winners)} outer folds")
    print(f"  NESTED TEST {SELECT_METRIC} = {payload[f'nested_test_{SELECT_METRIC}_mean']:.4f}"
          f" +/- {payload[f'nested_test_{SELECT_METRIC}_std']:.4f}")
    print(f"  NESTED TEST f2 @0.5 = {payload['nested_test_f2_mean']:.4f} | "
          f"@tuned threshold = {payload['nested_test_f2_at_threshold_mean']:.4f}"
          f" +/- {payload['nested_test_f2_at_threshold_std']:.4f}  "
          f"(f1 {payload['nested_test_f1_mean']:.4f} -> "
          f"{payload['nested_test_f1_at_threshold_mean']:.4f}; "
          f"median threshold prob {payload['threshold_prob_median']})")
    return payload


def _summary_rows(out_dir, context_size, schema):
    """``[(exp_key, arch, payload)]`` for every ``best_*.json`` in ``out_dir`` written
    under ``schema`` at ``context_size``.

    Read from disk rather than from one invocation's results: run_nested_sequential.cmd
    runs each experiment in its own process, and a summary built from in-memory results
    held only the last one. Files from other schemas are skipped, so an older sweep's
    winners never appear in this schema's table.
    """
    exp_keys = {v: k for k, v in EXPERIMENTS.items()}
    rows = []
    for path in sorted(glob(os.path.join(out_dir, f"best_*_ctx{context_size}_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                p = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [summary] skipping unreadable {os.path.basename(path)}: {e}")
            continue
        if not isinstance(p, dict):
            continue
        key = exp_keys.get((p.get("experiment"), p.get("protocol")))
        if p.get("schema") != schema or p.get("context_size") != context_size or key is None:
            continue
        rows.append((key, p["arch"], p))
    return rows


def write_summary(out_dir, context_size, schema=None):
    """One matrix of arch x experiment nested outer-test scores. The actual result.

    Rebuilt from every ``best_*.json`` of ``schema`` in ``out_dir`` (see
    ``_summary_rows``), so experiments finished by separate invocations accumulate.
    Returns the written path, or None when there is nothing to summarize (an existing
    summary is then left untouched).
    """
    # Resolved here, not in the signature: a default argument binds CACHE_SCHEMA at
    # import time and would ignore the schema --scaling selects.
    schema = CACHE_SCHEMA if schema is None else schema
    rows = _summary_rows(out_dir, context_size, schema)
    if not rows:
        return None
    done = {(k, a) for k, a, _ in rows}
    missing = [f"{k}/{a}" for k in DEFAULT_EXPERIMENTS for a in ALL_ARCHS
               if (k, a) not in done]
    # Tuned-threshold columns appear only when some payload has them, so an older
    # schema's table (no thresholds) keeps its original layout.
    tuned = any("nested_test_f2_at_threshold_mean" in p for _, _, p in rows)

    def fmt(v, spec=".4f"):
        return "n/a" if v is None or not np.isfinite(v) else format(v, spec)

    at_thr = lambda p, m: (p.get("nested_test_at_threshold_means") or {}).get(m)
    cols = ["experiment", "arch", f"nested_test_{SELECT_METRIC}_mean",
            f"nested_test_{SELECT_METRIC}_std", "nested_test_f1_mean"]
    if tuned:
        cols += ["nested_test_f2_mean", "nested_test_f2_at_threshold_mean",
                 "nested_test_f1_at_threshold_mean", "recall_at_threshold",
                 "precision_at_threshold", "threshold_prob_median"]
    cols += [f"inner_val_{SELECT_METRIC}_mean", "hp_agreement", "n_outer", "hyperparameters"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for k, a, p in sorted(rows, key=lambda r: (r[0], r[1])):
        cells = [k, a,
                 f"{p[f'nested_test_{SELECT_METRIC}_mean']:.4f}",
                 f"{p[f'nested_test_{SELECT_METRIC}_std']:.4f}",
                 f"{p['nested_test_f1_mean']:.4f}"]
        if tuned:
            cells += [fmt(p.get("nested_test_f2_mean")),
                      fmt(p.get("nested_test_f2_at_threshold_mean")),
                      fmt(p.get("nested_test_f1_at_threshold_mean")),
                      fmt(at_thr(p, "recall")), fmt(at_thr(p, "precision")),
                      fmt(p.get("threshold_prob_median"), ".3f")]
        cells += [f"{p[f'inner_val_{SELECT_METRIC}_mean']:.4f}",
                  f"{p['hp_agreement']:.2f}", str(p["n_outer"]),
                  json.dumps(p["hyperparameters"]).replace("|", "/")]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    note = ("\n\nNot yet in this directory (default experiments): " + ", ".join(missing)
            if missing else "")
    if tuned:
        note += ("\n\n`nested_test_f1_mean` and `nested_test_f2_mean` are at probability 0.5. "
                 "The `*_at_threshold` columns are at each outer fold's tuned threshold: the "
                 f"median of the winning combo's inner-fold F{THRESHOLD_BETA:g}-maximising val "
                 "thresholds (test data never involved). `threshold_prob_median` is the median "
                 "of those thresholds over outer folds, as a probability. PR-AUC does not depend "
                 "on the threshold.")
    path = os.path.join(out_dir, f"summary_nested{schema}_ctx{context_size}.md")
    _atomic_write(
        path,
        f"# Nested CV results (schema {schema}, context_size={context_size})\n\n"
        f"Outer-fold test {SELECT_METRIC}, mean +/- std across outer folds. Each outer "
        "fold selected its own hyperparameters by inner CV and was scored once; these "
        "numbers estimate the tuning+training procedure and are not conditioned on the "
        "single hyperparameter set listed. `hp_agreement` is the fraction of outer folds "
        "whose own winner matched that set. Built from every schema-"
        f"{schema} `best_*.json` in this directory.\n\n" + text + note + "\n")
    print(f"\n{'#' * 70}\nNESTED RESULTS ({os.path.basename(path)})\n{'#' * 70}\n{text}{note}")
    return path


def _smoke_flags(*, max_combos, max_outer, cities, epochs) -> list[str]:
    """The options that make a run a partial smoke run rather than a sweep."""
    flags = []
    if max_combos is not None:
        flags.append(f"--max-combos {max_combos}")
    if max_outer is not None:
        flags.append(f"--max-outer {max_outer}")
    if set(cities) != set(CITIES):
        flags.append("--cities " + " ".join(cities))
    if epochs != EPOCHS:
        flags.append(f"--epochs {epochs}")
    return flags


def _resolve_out_dir(out_dir, here, smoke_flags) -> str:
    """An explicit --out-dir is used as given. Otherwise a sweep writes to
    tuning_results/h{HORIZON}/ and a smoke run to tuning_results/h{HORIZON}/smoke/,
    away from the caches a sweep resumes from.

    The horizon is in the path because cache filenames carry the context size but
    **not** the horizon, while the horizon changes the fold windows and the embargo.
    Without the split, a second horizon would land on the first one's filenames with
    incompatible settings: the resume guard would refuse (correctly) and a fresh run
    would refuse too, because the cache is non-empty.
    """
    if out_dir is not None:
        return out_dir
    base = os.path.join(here, "tuning_results", f"h{HORIZON}")
    return os.path.join(base, SMOKE_SUBDIR) if smoke_flags else base


# --------------------------------------------------------------------------- #
def main():
    # Declared up front because --horizon's default *reads* HORIZON while building
    # the parser: Python rejects a global statement that follows a use of the name.
    # Both are set from the parsed args below, and every path that names a file or
    # builds a fold reads them at call time.
    global CACHE_SCHEMA, HORIZON

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("AQM_DATA_DIR")
                   or os.path.join(here, "senseurcity_data_v03", "dataset"),
                   help="SenseurCity v03 dataset/ folder (default: $AQM_DATA_DIR, "
                        "else ./senseurcity_data_v03/dataset)")
    p.add_argument("--out-dir", default=None,
                   help="default: tuning_results/h<horizon>/, or .../h<horizon>/smoke/ "
                        "for a smoke run (--max-combos, --max-outer, non-default "
                        "--epochs, city subset)")
    p.add_argument("--archs", nargs="+", default=ALL_ARCHS, choices=ALL_ARCHS)
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS,
                   choices=list(EXPERIMENTS),
                   help="default: single-blocked multi")
    p.add_argument("--cities", nargs="+", default=list(CITIES),
                   help="restrict to a subset of cities (a fast smoke test)")
    p.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--max-combos", type=int, default=None,
                   help="cap grid points per session")
    p.add_argument("--max-outer", type=int, default=None,
                   help="cap outer folds per experiment; needed for a fast smoke "
                        "test, since outer-fold count dominates the cost")
    p.add_argument("--dry-run-folds", action="store_true",
                   help="print fold geometry and exit without training")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="trainings run at once on worker processes sharing the GPU "
                        "(default 1 = inline); ~1.2 GB host RAM per worker")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic cuDNN kernels: bit-reproducible TCN/FUSION results "
                        "(GRU is either way); measured ~+30%% TCN, ~+10%% FUSION time")
    p.add_argument("--scaling", default=DEFAULT_SCALING, choices=list(SCALING_SCHEMA),
                   help="input scaling: logz (default, cache schema 5) -- log1p on the "
                        "OPC PM channels then per-channel standardization -- or minmax "
                        "(schema 4). The schema tags every cache, artifact and "
                        "best_*.json, so the two never mix")
    p.add_argument("--horizon", type=int, default=HORIZON,
                   help="hours past the last input hour at which the 24 h label window "
                        "ends (default 1). Defines the task, not a hyperparameter: it "
                        "moves the label window and widens the embargo, so each horizon "
                        "gets its own tuning_results/h<H>/ directory and results from "
                        "different horizons are not comparable")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.horizon < 0:
        p.error("--horizon must be >= 0")
    # Keeps a logz run's caches separate from a minmax run's (via the schema) and
    # each horizon's from every other horizon's (via the out-dir and the embargo).
    CACHE_SCHEMA = SCALING_SCHEMA[args.scaling]
    HORIZON = args.horizon
    smoke = _smoke_flags(max_combos=args.max_combos, max_outer=args.max_outer,
                         cities=args.cities, epochs=args.epochs)
    args.out_dir = _resolve_out_dir(args.out_dir, here, smoke)
    if smoke:
        print(f"smoke run ({' '.join(smoke)}) -> out_dir {args.out_dir}")

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | "
          f"archs={args.archs} | experiments={args.experiments} | "
          f"cities={args.cities} | ctx={args.context_size} | epochs={args.epochs} | "
          f"max_combos={args.max_combos} | max_outer={args.max_outer} | "
          f"workers={args.workers} | deterministic={args.deterministic} | "
          f"scaling={args.scaling} (cache schema {CACHE_SCHEMA}) | "
          f"horizon={HORIZON}h (embargo "
          f"{_resolve_embargo(None, {'context_size': args.context_size, 'horizon': HORIZON})}h)")

    runner = _Runner(args.workers, args.deterministic)
    try:
        for key in args.experiments:
            run_experiment(
                key, args.data_dir, args.out_dir, archs=args.archs, cities=args.cities,
                context_size=args.context_size, epochs=args.epochs,
                max_combos=args.max_combos, max_outer=args.max_outer,
                resume=not args.no_resume, runner=runner,
                deterministic=args.deterministic, scaling=args.scaling,
                dry_run=args.dry_run_folds,
            )
    finally:
        runner.close()
    if not args.dry_run_folds:
        write_summary(args.out_dir, args.context_size)


if __name__ == "__main__":
    # Leave without interpreter teardown. By the time main() returns every result
    # is on disk (cache lines fsynced, tables written atomically, pool shut down),
    # but a process that has parsed the dataset CSVs and trained on CUDA has been
    # seen to die in teardown with a native fail-fast (0xC0000409) and no Python
    # traceback -- turning a finished run into a failed exit code.
    code = 0
    try:
        main()
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            code = e.code or 0
        else:
            print(e.code, file=sys.stderr)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
