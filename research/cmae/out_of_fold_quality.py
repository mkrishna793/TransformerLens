"""Out-of-fold Cleanlab-style data-quality baseline for binary organisms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP, CIFARBatch
from research.cmae.binary_counterfactual import (
    binary_logits,
    fit_binary_physical_subset,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def label_quality_suspicion(quality: FloatArray) -> FloatArray:
    """Map bounded higher-is-better quality to higher-is-more-suspicious scores."""

    if (
        quality.ndim != 1
        or not np.all(np.isfinite(quality))
        or np.any(quality < 0)
        or np.any(quality > 1)
    ):
        raise ValueError("label quality must be a finite vector in [0, 1]")
    return 1.0 - quality


def out_of_fold_probabilities(
    model_factory: Callable[[int], CIFARBadNetsMLP],
    batch: CIFARBatch,
    *,
    folds: int = 4,
    epochs: int = 80,
    learning_rate: float = 0.03,
    seed: int = 7100,
) -> tuple[FloatArray, IntArray]:
    """Train physical fold subsets and predict every row only when held out."""

    count = batch.labels.shape[0]
    if folds < 2 or count < 2 * folds:
        raise ValueError("out-of-fold quality requires at least two populated folds")
    labels = batch.labels.detach().cpu().numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    fold_parts: list[list[IntArray]] = [[] for _ in range(folds)]
    for label in (0, 1):
        eligible = np.where(labels == label)[0]
        pieces = np.array_split(rng.permutation(eligible), folds)
        for fold, piece in enumerate(pieces):
            fold_parts[fold].append(piece)
    assignments = [np.sort(np.concatenate(parts).astype(np.int64)) for parts in fold_parts]
    fold_ids = np.full(count, -1, dtype=np.int64)
    probabilities = np.full(count, np.nan, dtype=np.float64)
    for fold, held_out in enumerate(assignments):
        if held_out.size == 0:
            raise RuntimeError("empty out-of-fold partition")
        fold_ids[held_out] = fold
        keep = np.setdiff1d(np.arange(count), held_out)
        training = batch.subset(keep)
        model = model_factory(seed + fold)
        fit_binary_physical_subset(
            model,
            training.inputs,
            training.labels,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        model.eval()
        with torch.no_grad():
            logits = binary_logits(model, batch.inputs[torch.from_numpy(held_out)])
            probabilities[held_out] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    if np.any(fold_ids < 0) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("out-of-fold probabilities do not cover every training row")
    return np.column_stack([1.0 - probabilities, probabilities]), fold_ids


def cleanlab_out_of_fold_suspicion(labels: IntArray, probabilities: FloatArray) -> FloatArray:
    """Execute Cleanlab label-quality ranking on out-of-fold probabilities."""

    if (
        labels.ndim != 1
        or probabilities.shape != (labels.size, 2)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("invalid Cleanlab labels or out-of-fold probabilities")
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as error:
        raise RuntimeError(
            "Cleanlab is required for the declared external quality baseline"
        ) from error
    quality = np.asarray(
        get_label_quality_scores(labels=labels, pred_probs=probabilities), dtype=np.float64
    )
    return label_quality_suspicion(quality)
