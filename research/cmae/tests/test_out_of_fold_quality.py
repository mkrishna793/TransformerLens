import numpy as np
import torch

from research.cmae.backdoorbench_organism import CIFARBatch, build_cifar_model
from research.cmae.out_of_fold_quality import (
    label_quality_suspicion,
    out_of_fold_probabilities,
)


def test_out_of_fold_coverage_and_quality_orientation() -> None:
    count = 8
    torch.manual_seed(2)
    labels = np.array([0, 1] * 4, dtype=np.int64)
    batch = CIFARBatch(
        inputs=torch.randn(count, 64, dtype=torch.float64) / 8,
        labels=torch.from_numpy(labels).double(),
        original_labels=labels,
        poison=np.zeros(count, dtype=np.int64),
        sample_ids=np.arange(count, dtype=np.int64),
    )
    probabilities, folds = out_of_fold_probabilities(
        lambda seed: build_cifar_model(seed, hidden_dim=2),
        batch,
        folds=2,
        epochs=1,
    )
    assert probabilities.shape == (count, 2)
    assert set(folds.tolist()) == {0, 1}
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    quality = np.array([0.1, 0.8], dtype=np.float64)
    suspicion = label_quality_suspicion(quality)
    assert suspicion[0] > suspicion[1]
    np.testing.assert_allclose(suspicion, [0.9, 0.2])
