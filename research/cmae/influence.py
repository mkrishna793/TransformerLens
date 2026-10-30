"""Small, explicit influence calculations used to test CMAE approximations.

The O(k) identity is a sparse diagonal Kronecker/K-FAC-style calculation. The
``ekfac_reference`` function separately implements the defining EK-FAC diagonal
rescaling in a Kronecker-factored eigenbasis. Neither routine claims exact deep-model
leave-one-out retraining influence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _vector(value: FloatArray, name: str) -> None:
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _positive_vector(value: FloatArray, name: str) -> None:
    _vector(value, name)
    if np.any(value <= 0):
        raise ValueError(f"{name} must be strictly positive")


def rank_one_sparse_diagonal_influence(
    activation_b: FloatArray,
    output_gradient_b: FloatArray,
    activation_q: FloatArray,
    output_gradient_q: FloatArray,
    activation_precision: FloatArray,
    output_precision: FloatArray,
) -> float:
    """Return ``-g_q^T (P_A kron P_S) g_b`` for rank-one gradients.

    The supplied precision vectors are already inverse/damped diagonal factors.
    With sparse supports, each dot product needs only active coordinates. The leading
    minus sign is the standard upweighting-influence convention.
    """

    for value, name in (
        (activation_b, "activation_b"),
        (activation_q, "activation_q"),
        (activation_precision, "activation_precision"),
    ):
        _vector(value, name)
    for value, name in (
        (output_gradient_b, "output_gradient_b"),
        (output_gradient_q, "output_gradient_q"),
    ):
        _vector(value, name)
    _positive_vector(output_precision, "output_precision")
    _positive_vector(activation_precision, "activation_precision")
    if not activation_b.shape == activation_q.shape == activation_precision.shape:
        raise ValueError("activation factors have inconsistent dimensions")
    if not output_gradient_b.shape == output_gradient_q.shape == output_precision.shape:
        raise ValueError("output-gradient factors have inconsistent dimensions")
    activation_term = np.sum(activation_precision * activation_q * activation_b)
    output_term = np.sum(output_precision * output_gradient_q * output_gradient_b)
    return float(-activation_term * output_term)


def rank_one_explicit_inverse_influence(
    activation_b: FloatArray,
    output_gradient_b: FloatArray,
    activation_q: FloatArray,
    output_gradient_q: FloatArray,
    activation_precision: FloatArray,
    output_precision: FloatArray,
) -> float:
    """Materialize the small Kronecker inverse as an independent test reference."""

    gradient_b = np.kron(activation_b, output_gradient_b)
    gradient_q = np.kron(activation_q, output_gradient_q)
    inverse = np.kron(np.diag(activation_precision), np.diag(output_precision))
    return float(-gradient_q @ inverse @ gradient_b)


def position_sum_sparse_diagonal_influence(
    activations_b: FloatArray,
    output_gradients_b: FloatArray,
    activations_q: FloatArray,
    output_gradients_q: FloatArray,
    activation_precision: FloatArray,
    output_precision: FloatArray,
) -> float:
    """Generalize the identity to gradients summed over token/spatial positions.

    Cross-position terms are required. Replacing the sums by one pooled outer product
    is a different approximation and is not algebraically exact.
    """

    if activations_b.ndim != 2 or activations_q.ndim != 2:
        raise ValueError("activations must have shape [positions, dimension]")
    if output_gradients_b.ndim != 2 or output_gradients_q.ndim != 2:
        raise ValueError("output gradients must have shape [positions, dimension]")
    activation_cross = np.einsum("td,d,ud->tu", activations_b, activation_precision, activations_q)
    output_cross = np.einsum(
        "td,d,ud->tu", output_gradients_b, output_precision, output_gradients_q
    )
    return float(-np.sum(activation_cross * output_cross))


def position_sum_explicit_inverse_influence(
    activations_b: FloatArray,
    output_gradients_b: FloatArray,
    activations_q: FloatArray,
    output_gradients_q: FloatArray,
    activation_precision: FloatArray,
    output_precision: FloatArray,
) -> float:
    """Explicit reference for the position-sum identity, including cross terms."""

    gradient_b = sum(
        np.kron(activation, gradient)
        for activation, gradient in zip(activations_b, output_gradients_b, strict=True)
    )
    gradient_q = sum(
        np.kron(activation, gradient)
        for activation, gradient in zip(activations_q, output_gradients_q, strict=True)
    )
    inverse = np.kron(np.diag(activation_precision), np.diag(output_precision))
    return float(-gradient_q @ inverse @ gradient_b)


def ekfac_reference(
    gradient_b: FloatArray,
    gradient_q: FloatArray,
    activation_eigenbasis: FloatArray,
    output_eigenbasis: FloatArray,
    eigenbasis_second_moments: FloatArray,
    damping: float,
) -> float:
    """Compute influence with EK-FAC's diagonal correction in the KFE.

    ``eigenbasis_second_moments`` has one entry per Kronecker eigenbasis direction,
    not merely one entry per activation and output concept. Dense eigenbases generally
    destroy sparsity, which is why the O(k) routine above is not exact EK-FAC.
    """

    _vector(gradient_b, "gradient_b")
    _vector(gradient_q, "gradient_q")
    _positive_vector(eigenbasis_second_moments, "eigenbasis_second_moments")
    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and strictly positive")
    if activation_eigenbasis.ndim != 2 or output_eigenbasis.ndim != 2:
        raise ValueError("eigenbases must be matrices")
    eigenbasis = np.kron(activation_eigenbasis, output_eigenbasis)
    if gradient_b.shape != gradient_q.shape:
        raise ValueError("query and training gradients must have the same shape")
    if eigenbasis.shape[0] != gradient_b.size or eigenbasis.shape[1] != gradient_b.size:
        raise ValueError("Kronecker eigenbasis does not match the gradient dimension")
    if eigenbasis_second_moments.shape != (gradient_b.size,):
        raise ValueError("EK-FAC needs one second moment per KFE coordinate")
    projected_b = eigenbasis.T @ gradient_b
    projected_q = eigenbasis.T @ gradient_q
    return float(-np.sum(projected_q * projected_b / (eigenbasis_second_moments + damping)))
