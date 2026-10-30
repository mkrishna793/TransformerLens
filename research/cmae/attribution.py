"""Retraining-checkable attribution references for a differentiable linear model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate_classification(features: FloatArray, labels: FloatArray) -> None:
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] == 0:
        raise ValueError("features must have shape [n>=2, p>0]")
    if labels.shape != (features.shape[0],) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be a binary vector aligned with features")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels)):
        raise ValueError("classification inputs must be finite")


def sigmoid(value: FloatArray) -> FloatArray:
    """Numerically stable logistic sigmoid."""

    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def logistic_terms(
    features: FloatArray, labels: FloatArray, weights: FloatArray
) -> dict[str, FloatArray]:
    """Return signed-logit features, loss gradients, and correct-class residuals."""

    _validate_classification(features, labels)
    if weights.shape != (features.shape[1],) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite vector matching features")
    logits = features @ weights
    probabilities = sigmoid(logits)
    signed_labels = 2.0 * labels - 1.0
    correct_residual = sigmoid(-signed_labels * logits)
    output_gradients = probabilities - labels
    return {
        "signed_features": signed_labels[:, None] * features,
        "loss_gradients": output_gradients[:, None] * features,
        "output_gradients": output_gradients,
        "correct_residual": correct_residual,
        "probabilities": probabilities,
        "curvature": probabilities * (1.0 - probabilities),
    }


def trak_output_attribution(
    features: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    query_feature: FloatArray,
    *,
    projection: FloatArray | None = None,
    damping: float = 1e-3,
) -> FloatArray:
    """Damped TRAK-style attribution for full-minus-retrained output changes.

    Signed-logit gradients are used so the sign convention is explicit. The query is
    a raw logit gradient. Projection is applied to both training and query features.
    """

    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    terms = logistic_terms(features, labels, weights)
    signed_features = terms["signed_features"]
    if query_feature.shape != (features.shape[1],):
        raise ValueError("query feature dimension does not match")
    if projection is not None:
        if projection.ndim != 2 or projection.shape[0] != features.shape[1]:
            raise ValueError("projection must have shape [parameters, rank]")
        signed_features = signed_features @ projection
        query_feature = query_feature @ projection
    normal = signed_features.T @ signed_features
    normal += damping * np.eye(normal.shape[0])
    solved = np.linalg.solve(normal, query_feature)
    return terms["correct_residual"] * (signed_features @ solved)


def ekfac_output_attribution(
    features: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    query_feature: FloatArray,
    *,
    damping: float = 1e-3,
) -> FloatArray:
    """Single-linear-layer EK-FAC output attribution in the KFE."""

    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    terms = logistic_terms(features, labels, weights)
    loss_gradients = terms["loss_gradients"]
    if query_feature.shape != (features.shape[1],):
        raise ValueError("query feature dimension does not match")
    activation_covariance = features.T @ features / features.shape[0]
    activation_eigenbasis = np.linalg.eigh(activation_covariance)[1]
    output_eigenbasis = np.array([[1.0]])
    eigenbasis = np.kron(activation_eigenbasis, output_eigenbasis)
    projected_train = loss_gradients @ eigenbasis
    projected_query = eigenbasis.T @ query_feature
    # Use summed-loss curvature: for a mean-loss leave-one-out counterfactual,
    # the 1/n removal factor cancels the 1/n in the Hessian.
    moments = np.sum(projected_train**2, axis=0)
    return -np.sum(projected_query[None, :] * projected_train / (moments + damping), axis=1)


def exact_ggn_output_attribution(
    features: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    query_feature: FloatArray,
    *,
    damping: float = 1e-6,
) -> FloatArray:
    """Exact logistic Gauss--Newton reference for the same output convention."""

    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    terms = logistic_terms(features, labels, weights)
    gradients = terms["loss_gradients"]
    # For a logistic linear model the GGN is X^T diag(p(1-p)) X. Use summed-loss
    # curvature so the mean-loss leave-one-out factor is correctly scaled.
    hessian = (features.T * terms["curvature"]) @ features
    hessian += damping * np.eye(hessian.shape[0])
    return -gradients @ np.linalg.solve(hessian, query_feature)


def gradient_output_attribution(
    features: FloatArray,
    labels: FloatArray,
    weights: FloatArray,
    query_feature: FloatArray,
) -> FloatArray:
    """Uncurvature-corrected gradient baseline with the same sign convention."""

    terms = logistic_terms(features, labels, weights)
    return -terms["loss_gradients"] @ query_feature


def fit_logistic_newton(
    features: FloatArray,
    labels: FloatArray,
    *,
    l2: float = 1e-3,
    max_steps: int = 100,
    tolerance: float = 1e-9,
) -> tuple[FloatArray, int]:
    """Fit a small logistic model by deterministic damped Newton iterations."""

    _validate_classification(features, labels)
    if l2 < 0 or not np.isfinite(l2) or max_steps <= 0 or tolerance <= 0:
        raise ValueError("invalid logistic optimizer settings")
    weights = np.zeros(features.shape[1], dtype=np.float64)
    penalty = np.ones(features.shape[1], dtype=np.float64)
    penalty[-1] = 0.0  # The final column is reserved for an unregularized bias.
    for iteration in range(1, max_steps + 1):
        logits = features @ weights
        probabilities = sigmoid(logits)
        curvature = probabilities * (1.0 - probabilities)
        gradient = features.T @ (probabilities - labels) / features.shape[0]
        gradient += l2 * penalty * weights
        hessian = (features.T * curvature) @ features / features.shape[0]
        hessian += np.diag(l2 * penalty + 1e-9)
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if np.linalg.norm(update) <= tolerance * (1.0 + np.linalg.norm(weights)):
            return weights, iteration
    return weights, max_steps


def logistic_loss(features: FloatArray, labels: FloatArray, weights: FloatArray) -> float:
    """Mean binary cross-entropy, with a stable logaddexp implementation."""

    _validate_classification(features, labels)
    logits = features @ weights
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
