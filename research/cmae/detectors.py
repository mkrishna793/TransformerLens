"""Robust spectral evidence with explicit health and convergence metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SpectralEvidence:
    scores: FloatArray
    robust_z: FloatArray
    direction: FloatArray
    iterations: int
    residual: float
    healthy: bool
    reason: str


def robust_mad_spectral(
    features: FloatArray,
    *,
    max_steps: int = 30,
    min_steps: int = 3,
    tolerance: float = 1e-7,
    epsilon: float = 1e-12,
) -> SpectralEvidence:
    """Compute a custom median/MAD spectral score, not ART's exact algorithm.

    Three power iterations are only a lower bound here. Failure to converge produces
    unhealthy evidence, allowing the controller to abstain instead of silently accepting.
    A robust coordinate median alone does not make the principal direction robust.
    """

    if features.ndim != 2 or features.shape[0] < 4 or features.shape[1] == 0:
        raise ValueError("features must have shape [n>=4, d>0]")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite")
    if max_steps < min_steps or min_steps < 1:
        raise ValueError("iteration bounds are inconsistent")
    if tolerance <= 0 or epsilon <= 0:
        raise ValueError("tolerance and epsilon must be positive")

    centered = features - np.median(features, axis=0, keepdims=True)
    covariance = centered.T @ centered / features.shape[0]
    direction = np.linspace(1.0, 2.0, features.shape[1], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    residual = float("inf")
    iterations = 0
    reason = "power_iteration_not_converged"

    for iterations in range(1, max_steps + 1):
        product = covariance @ direction
        norm = float(np.linalg.norm(product))
        if norm <= epsilon:
            reason = "degenerate_covariance"
            break
        updated = product / norm
        eigenvalue = float(updated @ covariance @ updated)
        residual = float(np.linalg.norm(covariance @ updated - eigenvalue * updated))
        direction = updated
        if iterations >= min_steps and residual <= tolerance:
            reason = "ok"
            break

    scores = (centered @ direction) ** 2
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = 1.4826 * mad
    robust_z = (scores - median) / (scale + epsilon)
    if reason == "ok" and scale <= epsilon:
        reason = "degenerate_mad"

    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return SpectralEvidence(
        scores=scores,
        robust_z=robust_z,
        direction=direction,
        iterations=iterations,
        residual=residual,
        healthy=reason == "ok",
        reason=reason,
    )


def art_spectral_scores(features: FloatArray) -> FloatArray:
    """Call IBM ART's own spectral-signature implementation as an optional baseline."""

    if features.ndim != 2 or features.shape[0] < 2 or not np.all(np.isfinite(features)):
        raise ValueError("ART features must be a finite matrix with at least two rows")
    try:
        from art.defences.detector.poison.spectral_signature_defense import (
            SpectralSignatureDefense,
        )
    except ImportError as error:
        raise RuntimeError("IBM ART and its runtime dependencies are optional") from error
    scores = SpectralSignatureDefense.spectral_signature_scores(features)
    return np.asarray(scores, dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class ActivationClusterEvidence:
    scores: FloatArray
    report: Mapping[str, Any]
    healthy: bool
    reason: str


def art_activation_cluster_scores(
    classifier: Any,
    inputs: FloatArray,
    labels: IntArray,
    *,
    nb_dims: int = 2,
) -> ActivationClusterEvidence:
    """Run ART ActivationDefence on a trained classifier and its input batch.

    ART performs its own hidden-activation extraction and per-class clustering. The
    returned score is binary suspicion (1 means ART marked the row as poison), so it
    is a baseline decision feature, not a calibrated probability.
    """

    if inputs.ndim != 2 or inputs.shape[0] < 4 or not np.all(np.isfinite(inputs)):
        raise ValueError("ART activation inputs must be a finite matrix with at least four rows")
    if labels.shape != (inputs.shape[0],) or np.any(labels < 0):
        raise ValueError("ART activation labels must be aligned nonnegative integers")
    if nb_dims < 1:
        raise ValueError("activation reduction dimensions must be positive")
    try:
        from art.defences.detector.poison import ActivationDefence
    except ImportError as error:
        raise RuntimeError("IBM ART and its runtime dependencies are optional") from error
    try:
        defence = ActivationDefence(classifier, inputs, labels)
        report, is_clean = defence.detect_poison(
            nb_clusters=2,
            nb_dims=nb_dims,
            reduce="PCA",
            cluster_analysis="smaller",
        )
        clean = np.asarray(is_clean, dtype=np.int64).reshape(-1)
        if clean.shape != labels.shape or not np.all((clean == 0) | (clean == 1)):
            return ActivationClusterEvidence(
                np.zeros(labels.shape[0], dtype=np.float64), report, False, "invalid_art_output"
            )
        return ActivationClusterEvidence(1.0 - clean.astype(np.float64), report, True, "ok")
    except Exception as error:
        return ActivationClusterEvidence(
            np.zeros(labels.shape[0], dtype=np.float64),
            {"exception": type(error).__name__},
            False,
            f"art_activation_failed:{type(error).__name__}",
        )
