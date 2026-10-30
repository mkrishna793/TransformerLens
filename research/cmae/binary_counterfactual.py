"""Fixed-denominator binary objectives and exact small-model GGN influence.

The first-order scores are hypotheses to check by retraining, not guarantees
about nonconvex optimization or finite-step leave-one-out behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.metrics import pearson, spearman

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def binary_logits(model: nn.Module, inputs: Tensor) -> Tensor:
    """Return one logit per row or reject an unsupported output contract."""

    output = model(inputs)
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != inputs.shape[0]:
        raise ValueError("binary model must return [batch] or [batch, 1] logits")
    return output


def fixed_denominator_bce_loss(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    *,
    original_count: int,
) -> Tensor:
    """Compute ``sum_b w_b * loss_b / N`` with the original denominator N."""

    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or weights.shape != logits.shape
        or original_count != logits.shape[0]
        or original_count <= 0
    ):
        raise ValueError("fixed-denominator loss requires aligned nonempty vectors")
    if not bool(torch.all((weights == 0) | (weights == 1)).item()):
        raise ValueError("fixed-denominator weights must be binary")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(dtype=logits.dtype), reduction="none"
    )
    return (losses * weights.to(dtype=logits.dtype)).sum() / original_count


def fit_binary_fixed_denominator(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    weights: Tensor,
    *,
    epochs: int = 120,
    learning_rate: float = 0.03,
    original_count: int | None = None,
) -> list[float]:
    """Train synchronously under the fixed-weight objective."""

    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("training epochs and learning rate must be positive")
    if inputs.ndim != 2 or inputs.shape[0] != labels.shape[0]:
        raise ValueError("binary training expects a row-major feature matrix")
    count = inputs.shape[0] if original_count is None else original_count
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = fixed_denominator_bce_loss(
            binary_logits(model, inputs), labels, weights, original_count=count
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite fixed-denominator training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def fit_binary_physical_subset(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    *,
    epochs: int = 120,
    learning_rate: float = 0.03,
) -> list[float]:
    """Train on an already physically constructed accepted-only dataset."""

    if inputs.shape[0] == 0 or inputs.shape[0] != labels.shape[0]:
        raise ValueError("physical accepted-only training data must be nonempty and aligned")
    weights = torch.ones_like(labels)
    return fit_binary_fixed_denominator(
        model,
        inputs,
        labels,
        weights,
        epochs=epochs,
        learning_rate=learning_rate,
        original_count=inputs.shape[0],
    )


def query_logit(model: nn.Module, query_inputs: Tensor) -> Tensor:
    """Fixed query measurement: mean raw target-class logit."""

    if query_inputs.ndim != 2 or query_inputs.shape[0] == 0:
        raise ValueError("query inputs must be a nonempty feature matrix")
    return binary_logits(model, query_inputs).mean()


def _flat_gradient(scalar: Tensor, parameters: tuple[Tensor, ...]) -> Tensor:
    gradients = torch.autograd.grad(scalar, parameters, retain_graph=False, allow_unused=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).detach()


@dataclass(frozen=True)
class GGNArtifact:
    """Exact scoped GGN matrix and per-sample fixed-normalization gradients."""

    logit_jacobian: Tensor
    loss_gradients: Tensor
    curvature: Tensor
    query_gradient: Tensor
    training_losses: Tensor

    @property
    def parameter_count(self) -> int:
        return int(self.logit_jacobian.shape[1])


def capture_binary_ggn(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    query_inputs: Tensor,
) -> GGNArtifact:
    """Capture exact logit Jacobians and the binary GGN at the current weights."""

    if (
        inputs.ndim != 2
        or labels.ndim != 1
        or inputs.shape[0] != labels.shape[0]
        or inputs.shape[0] < 2
        or query_inputs.ndim != 2
        or query_inputs.shape[1] != inputs.shape[1]
        or query_inputs.shape[0] == 0
    ):
        raise ValueError("invalid binary GGN data shapes")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("GGN capture requires trainable parameters")
    was_training = model.training
    model.eval()
    jacobian_rows: list[Tensor] = []
    curvature_weights: list[Tensor] = []
    losses: list[Tensor] = []
    try:
        for value, label in zip(inputs, labels, strict=True):
            logit = binary_logits(model, value.unsqueeze(0))[0]
            jacobian_rows.append(_flat_gradient(logit, parameters))
            probability = torch.sigmoid(logit.detach())
            curvature_weights.append(probability * (1 - probability))
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logit.detach(), label.detach().to(dtype=logit.dtype)
                )
            )
        query_gradient = _flat_gradient(query_logit(model, query_inputs), parameters)
    finally:
        model.train(was_training)
    jacobian = torch.stack(jacobian_rows)
    count = inputs.shape[0]
    residual_scale = torch.stack(curvature_weights).to(dtype=jacobian.dtype) / count
    curvature = jacobian.T @ (residual_scale[:, None] * jacobian)
    loss_scale = (torch.sigmoid(binary_logits(model, inputs).detach()) - labels.detach()).to(
        dtype=jacobian.dtype
    ) / count
    loss_gradients = loss_scale[:, None] * jacobian
    artifact = GGNArtifact(
        logit_jacobian=jacobian,
        loss_gradients=loss_gradients,
        curvature=curvature,
        query_gradient=query_gradient,
        training_losses=torch.stack(losses),
    )
    for tensor in (
        artifact.logit_jacobian,
        artifact.loss_gradients,
        artifact.curvature,
        artifact.query_gradient,
        artifact.training_losses,
    ):
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise FloatingPointError("nonfinite binary GGN artifact")
    return artifact


def damped_ggn_influence(artifact: GGNArtifact, *, damping: float = 1e-3) -> FloatArray:
    """Return signed full-minus-reduced first-order upweighting influence."""

    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("GGN damping must be finite and positive")
    dtype = artifact.curvature.dtype
    identity = torch.eye(artifact.parameter_count, dtype=dtype, device=artifact.curvature.device)
    solved = torch.linalg.solve(
        artifact.curvature + damping * identity, artifact.query_gradient.to(dtype=dtype)
    )
    scores = -(artifact.loss_gradients @ solved).detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite damped GGN influence")
    return scores


def gradient_similarity_influence(artifact: GGNArtifact) -> FloatArray:
    """Uncurvatured ``-g_q^T g_b`` baseline with the same sign and scale."""

    scores = (
        -(artifact.loss_gradients @ artifact.query_gradient)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite gradient-similarity influence")
    return scores


def training_loss_scores(artifact: GGNArtifact) -> FloatArray:
    scores = artifact.training_losses.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite training-loss baseline")
    return scores


def fixed_group_partition(count: int, *, groups: int = 8, seed: int = 7001) -> list[IntArray]:
    """Frozen score-independent partition covering every training row exactly once."""

    if count < groups or groups < 2:
        raise ValueError("group partition requires at least two nonempty groups")
    permutation = np.random.default_rng(seed).permutation(count)
    partition = [np.sort(group).astype(np.int64) for group in np.array_split(permutation, groups)]
    covered = np.concatenate(partition)
    if sorted(covered.tolist()) != list(range(count)):
        raise RuntimeError("group partition failed coverage conservation")
    return partition


def stratified_loo_indices(
    attack_labels: IntArray, *, per_stratum: int = 12, seed: int = 7002
) -> IntArray:
    """Fixed diagnostic LOO subset; labels stratify evaluation, never score orientation."""

    if attack_labels.ndim != 1 or per_stratum <= 0:
        raise ValueError("invalid LOO stratification")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for value in (0, 1):
        eligible = np.where(attack_labels == value)[0]
        if eligible.size == 0:
            raise ValueError("LOO stratification needs both attack classes")
        selected.extend(rng.permutation(eligible)[: min(per_stratum, eligible.size)].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def aggregate_scores(scores: FloatArray, groups: list[IntArray]) -> FloatArray:
    covered = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
    if scores.ndim != 1 or sorted(covered.tolist()) != list(range(scores.size)):
        raise ValueError("groups must partition the score vector exactly once")
    return np.asarray([scores[group].sum() for group in groups], dtype=np.float64)


def counterfactual_metrics(predicted: FloatArray, actual: FloatArray) -> dict[str, float]:
    """Rank, sign, and raw-effect error; correlation alone is not sufficient evidence."""

    if (
        predicted.ndim != 1
        or actual.shape != predicted.shape
        or predicted.size < 2
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(actual))
    ):
        raise ValueError("counterfactual vectors must be finite, aligned, and nonempty")
    return {
        "pearson": pearson(predicted, actual),
        "spearman": spearman(predicted, actual),
        "sign_agreement": float(np.mean((predicted > 0) == (actual > 0))),
        "mean_absolute_error_raw_scale": float(np.mean(np.abs(predicted - actual))),
        "actual_mean_absolute_effect": float(np.mean(np.abs(actual))),
    }
