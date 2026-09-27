"""Generic training entry point for SenseurCity AQM classification models.

``train_aqm_model`` takes a *pre-built* PyTorch model and runs a full
train/validate loop with early stopping over the SenseurCity data:

    * predict a single "harmful air quality" flag from the segment's reference
      24-hour average vs the WHO 2021 limits (model must output 1 channel, a raw
      logit). Loss: BCEWithLogitsLoss. Metric: accuracy / precision / recall /
      F1.

Optimisation is AdamW. The function builds the train/val/test
``SenseurCityDataset`` splits itself, so the caller only supplies the model and
the data directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from senseurcity_dataset import SenseurCityDataset, CITIES, make_splits


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stop when the monitored score (validation loss, lower-is-better) has not
    improved by at least ``min_delta`` for ``patience`` consecutive epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Return True when ``score`` is a new best."""
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float
    best_val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    history: list[dict[str, Any]] = field(default_factory=list)
    # Kept for reporting parity across model families.
    problem_type: str = "classification"
    # Train-fit min-max scaler for the inputs (None if normalize off).
    scaler: Any = None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _classification_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_rate": (tp + fn) / total if total else 0.0,
        # Raw confusion-matrix counts (for aggregating confusion matrices).
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _probability_metrics(y_true, y_score) -> dict[str, float]:
    """Threshold-free ROC-AUC and PR-AUC (average precision) from raw scores.

    Both are rank-based, so any monotonic score works (raw logits or
    probabilities). Returns NaN when the split has a single class, since the
    AUCs are undefined then.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evals.
    Returns the average loss plus classification metrics: 0.5-threshold confusion
    metrics and the threshold-free ROC-AUC / PR-AUC.
    """
    train = optimizer is not None
    model.train(train)

    loss_sum = 0.0
    n_samples = 0
    tp = fp = tn = fn = 0
    scores: list[torch.Tensor] = []   # raw logits (rank-equivalent to prob)
    labels: list[torch.Tensor] = []

    torch.set_grad_enabled(train)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # pred, y: [batch, 1] logits-vs-{0,1}. BCEWithLogitsLoss handles the
        # shape directly.
        loss = loss_fn(pred, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = y.shape[0]
        loss_sum += loss.item() * bs
        n_samples += bs

        # Metrics never need gradients — detach to avoid retaining the graph.
        pred = pred.detach()
        p = (pred > 0).view(-1)             # logit > 0  <=>  prob > 0.5
        t = y.bool().view(-1)
        tp += int((p & t).sum())
        fp += int((p & ~t).sum())
        tn += int((~p & ~t).sum())
        fn += int((~p & t).sum())
        # Keep scores/labels for the threshold-free AUCs (ROC-AUC & PR-AUC are
        # rank-based, so the raw logit works as the score — no sigmoid needed).
        scores.append(pred.reshape(-1).float().cpu())
        labels.append(y.detach().reshape(-1).float().cpu())
    torch.set_grad_enabled(True)

    metrics = {"loss": loss_sum / max(n_samples, 1)}
    metrics.update(_classification_metrics(tp, fp, tn, fn))
    if scores:
        metrics.update(_probability_metrics(
            torch.cat(labels).numpy(), torch.cat(scores).numpy()
        ))
    else:
        metrics.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
    return metrics


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
        f"f1={metrics['f1']:.3f} roc={metrics.get('roc_auc', float('nan')):.3f} "
        f"pr={metrics.get('pr_auc', float('nan')):.3f}"
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def train_aqm_model(
    model: nn.Module,
    data_dir: str,
    *,
    context_size: int = 24,
    horizon: int = 1,
    cities: tuple[str, ...] = CITIES,
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    shuffle_train: bool | None = None,
    num_workers: int = 0,
    overlap: float = 0.0,
    held_out_city: str | None = None,
    held_out_val_frac: float = 0.25,
    pos_weight: float | str | None = "auto",
    device: torch.device | str | None = None,
    eval_test: bool = True,
    dataset_kwargs: dict[str, Any] | None = None,
    datasets: tuple[Any, Any, Any] | None = None,
) -> TrainResult:
    """Train ``model`` on SenseurCity to classify harmful air quality.

    Args:
        model: a pre-built ``nn.Module`` taking ``[batch, context_size, 5]`` and
            returning ``[batch, 1]`` raw logits. The output width is validated
            before training.
        data_dir: SenseurCity ``dataset/`` folder.
        context_size, horizon, cities, split_fractions, dataset_kwargs: forwarded
            to ``SenseurCityDataset`` (ignored when ``datasets`` is supplied — the
            caller sets those on the pre-built splits). ``horizon`` positions the
            24-hour label window to end ``horizon`` hours past the last input hour
            (default 1 = forecast ending 1 h past the history; 0 = nowcast).
            ``split_fractions`` defines the global chronological train/val/test
            bands.
        batch_size, learning_rate, weight_decay, epochs, patience, min_delta:
            standard training knobs. Optimiser is AdamW.
        shuffle_train: shuffle the training set each epoch. Default None =
            "auto": True in city-held-out mode (train/val/test are disjoint
            cities, so shuffling cannot leak), False in temporal mode (preserve
            chronological order). Pass an explicit bool to override.
        held_out_city, held_out_val_frac: enable city-held-out evaluation —
            train on the other cities, validate on the first ``held_out_val_frac``
            (chronologically) of ``held_out_city`` and test on the rest.
            Forwarded to ``SenseurCityDataset`` (default None = temporal split).
        pos_weight: positive-class weight for ``BCEWithLogitsLoss``. ``"auto"``
            (default) sets it to ``n_negative / n_positive`` over the train split
            so the under-represented "harmful" class is up-weighted; a float
            overrides it explicitly; ``None`` disables weighting (standard BCE).
        eval_test: evaluate the best-val checkpoint on the test split at the end.
        device: torch device; auto-selects CUDA when available if None.

    Returns:
        ``TrainResult`` with the trained model (best-val weights restored),
        per-epoch history, best validation metrics and optional test metrics.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print(f"Using device: {device}")

    ds_kwargs = dict(dataset_kwargs or {})

    if datasets is not None:
        # Caller-supplied pre-built splits (e.g. k-fold CV folds). They must
        # already share a train-fit scaler (val/test built with train.scaler).
        train_ds, val_ds, test_ds = datasets
        split_label = "custom"
    else:
        # make_splits fits the min-max scaler on train and shares it with
        # val/test, so normalization never leaks across the split boundary.
        train_ds, val_ds, test_ds = make_splits(
            data_dir,
            context_size=context_size,
            horizon=horizon,
            split_fractions=split_fractions,
            cities=cities,
            overlap=overlap,
            held_out_city=held_out_city,
            held_out_val_frac=held_out_val_frac,
            **ds_kwargs,
        )
        split_label = (
            f"city-held-out ({held_out_city})" if held_out_city else "temporal"
        )
    if not eval_test:
        test_ds = None

    # Auto-pick shuffling: safe (and helpful) whenever train and val/test are
    # disjoint by construction (city-held-out or caller-supplied folds); off in
    # the plain temporal split to keep chronological order. Explicit overrides.
    if shuffle_train is None:
        shuffle_train = held_out_city is not None or datasets is not None
    print(f"split mode: {split_label} | shuffle_train={shuffle_train}")
    print(
        f"classification | train={len(train_ds)} val={len(val_ds)}"
        + (f" test={len(test_ds)}" if test_ds is not None else "")
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = model.to(device)
    _validate_model_output(model, train_loader, device)

    # Loss + optimiser. Weight the under-represented "harmful" class via
    # BCEWithLogitsLoss(pos_weight).
    pw_tensor = _resolve_pos_weight(pos_weight, train_ds, device)
    loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    print(
        f"BCE pos_weight = "
        + ("none" if pw_tensor is None else f"{pw_tensor.item():.3f}")
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        improved = stopper(val_metrics["loss"])
        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train {_fmt(train_metrics)} | "
            f"val {_fmt(val_metrics)}"
            + ("  *" if improved else "")
        )

        if improved:
            best_epoch = epoch
            best_val_loss = val_metrics["loss"]
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if stopper.early_stop:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    # Restore the best-validation weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_metrics = _run_epoch(model, test_loader, loss_fn, device, optimizer=None)
        print(f"test  | {_fmt(test_metrics)}")

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        history=history,
        scaler=train_ds.scaler,
    )


def _resolve_pos_weight(
    pos_weight: float | str | None,
    train_ds,
    device: torch.device,
) -> torch.Tensor | None:
    """Turn the ``pos_weight`` argument into a 1-element tensor (or None).

    ``"auto"`` -> ``n_negative / n_positive`` from the train split; a float ->
    that value; ``None`` -> unweighted BCE.
    """
    if pos_weight is None:
        return None
    if isinstance(pos_weight, str):
        if pos_weight != "auto":
            raise ValueError(f"pos_weight string must be 'auto', got {pos_weight!r}")
        n_neg, n_pos = train_ds.class_counts()
        if n_pos == 0:
            print("pos_weight='auto' but train has no positive windows; using 1.0")
            value = 1.0
        else:
            value = n_neg / n_pos
    else:
        value = float(pos_weight)
    return torch.tensor([value], dtype=torch.float32, device=device)


def _validate_model_output(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Fail fast if the model's output width is not a single logit."""
    X, _ = next(iter(loader))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(X[:1].to(device))
    model.train(was_training)
    if out.ndim != 2 or out.shape[-1] != 1:
        raise ValueError(
            f"Model output shape {tuple(out.shape)} is incompatible with "
            "classification; expected last dim 1. Build the model with "
            "output_channels=1."
        )
