"""Small dependency-free metrics for sequential CMAE validation experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _labels_scores(labels: NDArray[np.int64], scores: FloatArray) -> None:
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and scores must be aligned nonempty vectors")
    if not np.all((labels == 0) | (labels == 1)) or not np.all(np.isfinite(scores)):
        raise ValueError("labels must be binary and scores finite")


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def roc_auc(labels: NDArray[np.int64], scores: FloatArray) -> float:
    _labels_scores(labels, scores)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC needs both classes")
    ranks = _ranks(scores)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def average_precision(labels: NDArray[np.int64], scores: FloatArray) -> float:
    _labels_scores(labels, scores)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("average precision needs a positive class")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    positions = np.arange(1, labels.size + 1)
    return float(np.sum((cumulative / positions) * sorted_labels) / positives)


def pearson(first: FloatArray, second: FloatArray) -> float:
    if first.ndim != 1 or second.shape != first.shape or first.size < 2:
        raise ValueError("Pearson inputs must be aligned vectors with at least two values")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Pearson inputs must be finite")
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = float(np.linalg.norm(centered_first) * np.linalg.norm(centered_second))
    return float(centered_first @ centered_second / denominator) if denominator > 0 else 0.0


def spearman(first: FloatArray, second: FloatArray) -> float:
    return pearson(_ranks(first), _ranks(second))


def threshold_for_clean_fpr(
    scores: FloatArray, clean_labels: NDArray[np.int64], fpr: float = 0.1
) -> float:
    """Choose a high-score threshold using only clean calibration examples."""

    if not 0 <= fpr <= 1:
        raise ValueError("target FPR must be between zero and one")
    if scores.ndim != 1 or clean_labels.shape != scores.shape:
        raise ValueError("calibration scores and labels must align")
    clean = scores[clean_labels == 0]
    if clean.size == 0:
        raise ValueError("clean calibration examples are required")
    return float(np.quantile(clean, 1.0 - fpr, method="higher"))


def confusion(labels: NDArray[np.int64], scores: FloatArray, threshold: float) -> dict[str, float]:
    _labels_scores(labels, scores)
    predicted = scores > threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    true_negative = int(np.sum(~predicted & (labels == 0)))
    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "tpr": true_positive / max(true_positive + false_negative, 1),
        "fpr": false_positive / max(false_positive + true_negative, 1),
    }
