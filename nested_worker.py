"""GPU-tensor trainer and process-pool worker for the nested-CV tuner.

``find_best_model`` materializes each outer fold once, stacks every split into
tensors, and writes two files: a *dev* file (the inner train/val pairs, which
include the refit pair) and a *test* file. A ``Job`` names one training by file
path; ``run_job`` loads the tensors onto the device -- cached per process, so
only the first job of an outer fold pays for it -- and trains.

The leak structure survives the move: an inner job carries ``test_path=None``, so
outer-test tensors are never loaded in the scope that trains a selection
candidate. Only the refit job is given a test path.

``fast_train`` ports ``train_aqm.train_aqm_model`` for caller-supplied folds,
minus the per-batch host syncs that dominate the cost of these small models. It
replays the global-RNG draws the DataLoader path makes, so it visits batches in
the same order and consumes the same random stream:

  * every DataLoader iterator draws one base seed when created;
  * RandomSampler draws one seed on the first batch and shuffles with a private
    generator seeded from it;
  * train_aqm_model's output-shape check takes one batch before epoch 1.

With ``eval_batch`` equal to the training batch size (the tuner's setting),
evaluation is chunked identically too, and the trainer reproduces the DataLoader
path exactly -- verified bit for bit on real folds for GRU, TCN and FUSION, refit
test metrics included, under deterministic cuDNN. A larger ``eval_batch`` moved
val loss by up to ~1e-4 in the same check: enough to flip an early-stopping
decision between two near-equal epochs.

Decision threshold. Training and every threshold-free metric are untouched, and
the metrics under their existing names stay at logit > 0 (probability 0.5). On top
of that, each training reports the F-beta-maximising logit threshold (beta = 2 by
default: recall weighted over precision) on its best-epoch validation predictions
(``fbeta_max_threshold``). The tuner gives a refit job the
median of its winning combo's inner-fold thresholds; the job then also scores its
val and test predictions at that threshold, and returns the raw logits so another
criterion can be applied later without retraining. Thresholds never see test data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from AQM_CNNGRU_Model import AQM_CNNGRU_Fusion_Model
from AQM_GRU_Model import AQM_GRU_Model
from AQM_model import AQM_TCN_Model
from train_aqm import EarlyStopping, _classification_metrics, _probability_metrics

# Stored in every cache meta line and checked on resume. Change it whenever the
# training procedure changes in a way that can move results.
TRAINER_ID = "gpu-tensor-v1"


def build_model(arch: str, hp: dict, input_channels: int) -> nn.Module:
    """Instantiate one architecture from a grid point (single-logit output)."""
    if arch == "GRU":
        return AQM_GRU_Model(
            input_channels=input_channels,
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            output_channels=1,
        )
    if arch == "TCN":
        return AQM_TCN_Model(
            input_channels=input_channels,
            tcn_channels=hp["tcn_channels"],
            kernel_size=hp["kernel_size"],
            dropout=hp["dropout"],
            output_channels=1,
        )
    if arch == "FUSION":
        return AQM_CNNGRU_Fusion_Model(
            input_channels=input_channels,
            cnn_channels=hp["cnn_channels"],
            kernel_size=hp["kernel_size"],
            gru_hidden=hp["gru_hidden"],
            num_layers=hp["num_layers"],
            fusion_dim=hp["fusion_dim"],
            dropout=hp["dropout"],
            output_channels=1,
        )
    raise ValueError(f"unknown arch {arch!r}")


@dataclass(frozen=True)
class Job:
    """One training. Picklable, so it can cross to a worker process."""
    arch: str
    combo: int
    hp: dict
    fold: int                  # position of the train/val pair in the dev file
    dev_path: str
    test_path: str | None      # set for the outer refit only
    input_channels: int
    seed: int
    epochs: int
    patience: int
    batch_size: int
    eval_batch: int
    threshold: float | None = None  # refit only: logit threshold chosen on the inner folds
    threshold_beta: float = 2.0     # F-beta the val threshold search maximises


# Per-process tensor cache: slot -> (path, tensors on device). One outer fold at a
# time, so device memory stays bounded however long the worker lives.
_cache: dict[str, tuple[str | None, dict | None]] = {"dev": (None, None), "test": (None, None)}


def _tensors(slot: str, path: str, device: torch.device) -> dict:
    cur_path, cur = _cache[slot]
    if cur_path != path:
        _cache[slot] = (None, None)         # release the previous fold before loading
        cur = {k: v.to(device) for k, v in torch.load(path, weights_only=True).items()}
        _cache[slot] = (path, cur)
    return cur


def set_determinism(deterministic: bool) -> None:
    """cuDNN kernel choice for this process. The default kernels are faster but not
    bit-reproducible for convolutions: the same TCN/FUSION training can land on a
    different best epoch from run to run. GRU is reproducible either way."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def init_worker(deterministic: bool = False) -> None:
    """Pool initializer. One CPU thread per worker: the GPU does the arithmetic,
    and several processes each spawning a full thread pool would only contend."""
    torch.set_num_threads(1)
    set_determinism(deterministic)


def run_job(job: Job) -> tuple[Job, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if _cache["dev"][0] != job.dev_path:
        _cache["test"] = (None, None)       # new outer fold: its test file differs too
    dev = _tensors("dev", job.dev_path, device)
    test = None
    if job.test_path is not None:
        t = _tensors("test", job.test_path, device)
        test = (t["X"], t["Y"])

    # Same order as train_aqm.train_aqm_model: seed, build on CPU, move.
    torch.manual_seed(job.seed)
    model = build_model(job.arch, job.hp, job.input_channels).to(device)
    res = fast_train(
        model,
        dev[f"tr{job.fold}_X"], dev[f"tr{job.fold}_Y"],
        dev[f"va{job.fold}_X"], dev[f"va{job.fold}_Y"],
        test=test,
        learning_rate=job.hp["learning_rate"],
        weight_decay=job.hp["weight_decay"],
        epochs=job.epochs,
        patience=job.patience,
        batch_size=job.hp.get("batch_size", job.batch_size),
        eval_batch=job.eval_batch,
        threshold=job.threshold,
        threshold_beta=job.threshold_beta,
        return_predictions=job.test_path is not None,
    )
    res["secs"] = time.time() - t0
    return job, res


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def _draw_seed() -> int:
    """One global-RNG draw, made exactly the way DataLoader and RandomSampler do."""
    return int(torch.empty((), dtype=torch.int64).random_().item())


@torch.no_grad()
def _evaluate(model, X, Y, loss_fn, eval_batch) -> tuple[float, torch.Tensor]:
    """Chunked forward pass -> (mean loss, logits).

    The loss is accumulated as ``train_aqm._run_epoch`` does it -- per-chunk mean
    times chunk size, summed in float64 in order -- so ``eval_batch == batch_size``
    reproduces its value bit for bit. One host sync per call.
    """
    model.eval()
    logits, means, sizes = [], [], []
    for i in range(0, len(X), eval_batch):
        out = model(X[i:i + eval_batch])
        logits.append(out)
        means.append(loss_fn(out, Y[i:i + eval_batch]))
        sizes.append(out.shape[0])
    loss_sum = 0.0
    for m, bs in zip(torch.stack(means).tolist(), sizes):
        loss_sum += m * bs
    return loss_sum / max(len(X), 1), torch.cat(logits)


def _metrics(loss: float, logits: torch.Tensor, Y: torch.Tensor) -> dict[str, float]:
    """The metric dict ``train_aqm._run_epoch`` returns, key for key."""
    p = (logits > 0).view(-1)
    t = Y.bool().view(-1)
    tp, fp, tn, fn = torch.stack([(p & t).sum(), (p & ~t).sum(),
                                  (~p & ~t).sum(), (~p & t).sum()]).tolist()
    out = {"loss": loss}
    out.update(_classification_metrics(tp, fp, tn, fn))
    out["f2"] = fbeta_from_counts(tp, fp, fn, 2.0)
    out.update(_probability_metrics(Y.reshape(-1).float().cpu().numpy(),
                                    logits.reshape(-1).float().cpu().numpy()))
    return out


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    """F-beta = (1 + b^2) tp / ((1 + b^2) tp + b^2 fn + fp); 0 when undefined."""
    b2 = beta * beta
    den = (1.0 + b2) * tp + b2 * fn + fp
    return (1.0 + b2) * tp / den if den else 0.0


def fbeta_max_threshold(logits, y, beta: float = 2.0) -> tuple[float, float]:
    """``(logit threshold, F-beta)`` maximising F-beta for the rule ``logit > threshold``.

    With the top ``k`` scores predicted positive and ``P`` positives in all,
    F-beta = (1 + b^2) tp / (k + b^2 P). Only cuts between two distinct scores are
    considered, so tied scores always get the same prediction, and the threshold is
    the midpoint of that gap (below the lowest score when every window is predicted
    positive). Among equal-score cuts the highest threshold -- fewest alarms -- wins.
    Computed in float64 on CPU: the midpoint of two adjacent float32 logits need not
    be representable in float32. Returns NaNs when ``y`` holds one class, where the
    maximiser carries no information.
    """
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    t = np.asarray(y).reshape(-1) > 0.5
    n_pos = int(t.sum())
    if n_pos == 0 or n_pos == len(t):
        return float("nan"), float("nan")
    b2 = beta * beta
    order = np.argsort(-s, kind="stable")
    s_sorted, t_sorted = s[order], t[order]
    tp = np.cumsum(t_sorted)
    k = np.arange(1, len(s) + 1)
    boundary = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    score = np.where(boundary, (1.0 + b2) * tp / (k + b2 * n_pos), -1.0)
    i = int(np.argmax(score))
    thr = (s_sorted[i] + s_sorted[i + 1]) / 2.0 if i + 1 < len(s) else s_sorted[i] - 1.0
    return float(thr), float(score[i])


def metrics_at_threshold(logits, y, threshold: float) -> dict[str, float]:
    """Confusion-matrix metrics (accuracy, precision, recall, F1, F2, counts, pos_rate)
    for the rule ``logit > threshold``, compared in float64 (see ``fbeta_max_threshold``)."""
    p = np.asarray(logits, dtype=np.float64).reshape(-1) > float(threshold)
    t = np.asarray(y).reshape(-1) > 0.5
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    tn, fn = int((~p & ~t).sum()), int((~p & t).sum())
    out = _classification_metrics(tp, fp, tn, fn)
    out["f2"] = fbeta_from_counts(tp, fp, fn, 2.0)
    out["threshold"] = float(threshold)
    return out


def fast_train(model, Xtr, Ytr, Xva, Yva, *, test=None, learning_rate, weight_decay,
               epochs, patience, batch_size, eval_batch, min_delta=0.0,
               threshold=None, threshold_beta=2.0, return_predictions=False) -> dict:
    """Train with early stopping on val loss; restore the best epoch's weights.

    Mirrors ``train_aqm_model(datasets=...)``: AdamW, BCEWithLogits with
    ``pos_weight = n_neg / n_pos`` from the train split, reshuffled train batches,
    patience on val loss. Train-split metrics are not computed -- the tuner never
    reads them. Returns best epoch, best val loss, best val metrics, optional test
    metrics and the number of epochs run.

    Also returned: ``val_threshold`` / ``val_fbeta_at_threshold``, the logit threshold
    maximising F-``threshold_beta`` on the best epoch's val predictions. With ``threshold`` given
    (a refit), ``val_metrics_at_threshold`` and ``test_metrics_at_threshold`` score
    the same predictions at that threshold. ``return_predictions`` adds the best
    epoch's val logits/labels and the test logits/labels as float32/uint8 arrays.

    Every call also returns what the tuner saves as artifacts: the learning curve --
    ``train_loss`` (sample-weighted mean batch loss over each epoch, dropout active)
    and ``val_loss``, one entry per epoch run -- the best epoch's ``val_logits``, and
    its weights as numpy arrays (``best_state``). The per-epoch train loss is summed
    on the device and read once at the end.

    None of this touches the RNG or the training arithmetic.
    """
    device = Xtr.device
    n = len(Xtr)
    n_pos = int(Ytr.sum().item())
    pos_weight = torch.tensor([(n - n_pos) / n_pos if n_pos else 1.0],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    _draw_seed()        # output-shape check: iter(train_loader) ...
    _draw_seed()        # ... and its first batch

    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    best_state, best_val_logits = None, None
    best_epoch, best_val_loss, best_val_metrics = -1, float("inf"), {}
    epochs_run = 0
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        _draw_seed()                                # train iterator base seed
        gen = torch.Generator()
        gen.manual_seed(_draw_seed())               # RandomSampler seed
        perm = torch.randperm(n, generator=gen).to(device)

        model.train()
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().double() * idx.numel()
        train_losses.append(epoch_loss / max(n, 1))

        _draw_seed()                                # val iterator base seed
        val_loss, val_logits = _evaluate(model, Xva, Yva, loss_fn, eval_batch)
        val_losses.append(val_loss)
        if stopper(val_loss):
            best_epoch, best_val_loss = epoch, val_loss
            best_val_metrics = _metrics(val_loss, val_logits, Yva)
            best_val_logits = val_logits.reshape(-1).float().cpu().numpy()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if stopper.early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_y = Yva.reshape(-1).cpu().numpy().astype(np.uint8)
    val_thr, val_fb = (fbeta_max_threshold(best_val_logits, val_y, threshold_beta)
                       if best_val_logits is not None else (float("nan"), float("nan")))
    out = {
        "val_threshold": val_thr, "val_fbeta_at_threshold": val_fb,
        "train_loss": (torch.stack(train_losses).cpu().numpy() if train_losses
                       else np.zeros(0, dtype=np.float64)),
        "val_loss": np.asarray(val_losses, dtype=np.float64),
        "val_logits": best_val_logits,
        # numpy, not tensors: plain pickling to the parent, no shared-memory handles
        "best_state": (None if best_state is None
                       else {k: v.cpu().numpy() for k, v in best_state.items()}),
    }

    test_metrics = None
    if test is not None:
        test_loss, test_logits = _evaluate(model, test[0], test[1], loss_fn, eval_batch)
        test_metrics = _metrics(test_loss, test_logits, test[1])
        test_np = test_logits.reshape(-1).float().cpu().numpy()
        test_y = test[1].reshape(-1).cpu().numpy().astype(np.uint8)
        if threshold is not None:
            out["test_metrics_at_threshold"] = metrics_at_threshold(test_np, test_y, threshold)
            if best_val_logits is not None:
                out["val_metrics_at_threshold"] = metrics_at_threshold(best_val_logits, val_y,
                                                                       threshold)
        if return_predictions:
            out["predictions"] = {"val_logits": best_val_logits, "val_y": val_y,
                                  "test_logits": test_np, "test_y": test_y}

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "epochs_run": epochs_run,
        **out,
    }
