"""Calibration-controlled training and ground-truth latent readout metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from research.cmae.arithmetic_data import ArithmeticDataset, ArithmeticSplit
from research.cmae.arithmetic_transformer import (
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
)

TrainingArm = Literal["standard", "regularized", "auxiliary", "auxiliary_regularized"]
AUXILIARY_ARMS: frozenset[TrainingArm] = frozenset({"auxiliary", "auxiliary_regularized"})
REGULARIZED_ARMS: frozenset[TrainingArm] = frozenset({"regularized", "auxiliary_regularized"})
TRAINING_ARMS: frozenset[TrainingArm] = frozenset(
    {"standard", "regularized", "auxiliary", "auxiliary_regularized"}
)


@dataclass(frozen=True)
class CalibrationTrainingResult:
    arm: TrainingArm
    best_epoch: int
    best_calibration_loss: float
    best_calibration_accuracy: float
    weight_decay: float
    auxiliary_weight: float
    history: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "best_epoch": self.best_epoch,
            "best_calibration_loss": self.best_calibration_loss,
            "best_calibration_accuracy": self.best_calibration_accuracy,
            "weight_decay": self.weight_decay,
            "auxiliary_weight": self.auxiliary_weight,
            "history": list(self.history),
        }


def readout_rank_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("readout logits and targets must align as [n,v] and [n]")
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    positions = (order == targets[:, None]).to(torch.int64).argmax(dim=1) + 1
    ranks = positions.to(torch.float64)
    return {
        "top1_accuracy": float((positions == 1).to(torch.float64).mean().cpu().item()),
        "recall_at_10": float((positions <= 10).to(torch.float64).mean().cpu().item()),
        "mean_rank": float(ranks.mean().cpu().item()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean().cpu().item()),
    }


def evaluate_latent_readouts(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    split: ArithmeticSplit,
    *,
    batch_size: int = 128,
) -> dict[str, float]:
    tokens_np, _, intermediates_np = dataset.arrays(split)
    tokens = torch.from_numpy(tokens_np)
    intermediates = torch.from_numpy(intermediates_np)
    u_logits_batches: list[Tensor] = []
    v_logits_batches: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start : start + batch_size]
                _, u_logits, v_logits = model.forward_with_latents(batch)
                u_logits_batches.append(u_logits.cpu())
                v_logits_batches.append(v_logits.cpu())
    finally:
        model.train(was_training)
    u_metrics = readout_rank_metrics(torch.cat(u_logits_batches), intermediates[:, 0])
    v_metrics = readout_rank_metrics(torch.cat(v_logits_batches), intermediates[:, 1])
    return {f"u_{key}": value for key, value in u_metrics.items()} | {
        f"v_{key}": value for key, value in v_metrics.items()
    }


def train_with_calibration(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    arm: TrainingArm,
    *,
    epochs: int = 250,
    evaluation_every: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    weight_decay: float | None = None,
    auxiliary_weight: float = 0.5,
    seed: int = 101,
) -> CalibrationTrainingResult:
    """Select checkpoints using calibration final-answer loss only."""

    if arm not in TRAINING_ARMS:
        raise ValueError("unknown training arm")
    effective_weight_decay = (
        (1e-2 if arm in REGULARIZED_ARMS else 1e-4) if weight_decay is None else weight_decay
    )
    effective_auxiliary_weight = auxiliary_weight if arm in AUXILIARY_ARMS else 0.0
    if (
        epochs <= 0
        or evaluation_every <= 0
        or batch_size <= 0
        or learning_rate <= 0
        or effective_weight_decay < 0
        or auxiliary_weight < 0
    ):
        raise ValueError("invalid calibration-controlled training configuration")
    train_tokens_np, train_targets_np, train_intermediates_np = dataset.arrays("train")
    train_tokens = torch.from_numpy(train_tokens_np)
    train_targets = torch.from_numpy(train_targets_np)
    train_intermediates = torch.from_numpy(train_intermediates_np)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=effective_weight_decay
    )
    loss_function = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best_epoch = 0
    initial_calibration = evaluate_arithmetic_transformer(model, dataset, "calibration")
    best_loss = float(initial_calibration["loss"])
    best_accuracy = float(initial_calibration["accuracy"])
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(len(train_tokens))
        weighted_loss = 0.0
        for start in range(0, len(train_tokens), batch_size):
            indices = torch.from_numpy(permutation[start : start + batch_size].copy())
            batch_tokens = train_tokens[indices]
            batch_targets = train_targets[indices]
            optimizer.zero_grad(set_to_none=True)
            if arm in AUXILIARY_ARMS:
                logits, u_logits, v_logits = model.forward_with_latents(batch_tokens)
                final_loss = loss_function(logits[:, -1, :], batch_targets)
                batch_intermediates = train_intermediates[indices]
                auxiliary_loss = loss_function(u_logits, batch_intermediates[:, 0]) + loss_function(
                    v_logits, batch_intermediates[:, 1]
                )
                loss = final_loss + effective_auxiliary_weight * auxiliary_loss
            else:
                logits = model(batch_tokens)
                final_loss = loss_function(logits[:, -1, :], batch_targets)
                loss = final_loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite calibration-controlled training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            weighted_loss += float(loss.detach().cpu().item()) * len(indices)
        if epoch % evaluation_every == 0 or epoch == epochs:
            calibration = evaluate_arithmetic_transformer(model, dataset, "calibration")
            calibration_loss = float(calibration["loss"])
            calibration_accuracy = float(calibration["accuracy"])
            latent = evaluate_latent_readouts(model, dataset, "calibration")
            history.append(
                {
                    "epoch": epoch,
                    "training_objective": weighted_loss / len(train_tokens),
                    "calibration_loss": calibration_loss,
                    "calibration_accuracy": calibration_accuracy,
                    "calibration_u_recall_at_10": latent["u_recall_at_10"],
                    "calibration_v_recall_at_10": latent["v_recall_at_10"],
                }
            )
            if calibration_loss < best_loss - 1e-9:
                best_loss = calibration_loss
                best_accuracy = calibration_accuracy
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    return CalibrationTrainingResult(
        arm=arm,
        best_epoch=best_epoch,
        best_calibration_loss=best_loss,
        best_calibration_accuracy=best_accuracy,
        weight_decay=effective_weight_decay,
        auxiliary_weight=effective_auxiliary_weight,
        history=tuple(history),
    )
