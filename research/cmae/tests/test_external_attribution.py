import numpy as np
import pytest
import torch

from research.cmae.backdoorbench_organism import CIFARBatch, build_cifar_model
from research.cmae.binary_ekfac import (
    binary_ekfac_influence,
    capture_binary_training_raw,
    fit_binary_ekfac,
)
from research.cmae.external_attribution import (
    dattri_binary_ekfac_influence,
    kronfluence_binary_ekfac_influence,
    score_agreement,
)


def _case() -> tuple[CIFARBatch, CIFARBatch]:
    torch.manual_seed(3)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    batch = CIFARBatch(
        inputs=torch.randn(6, 64, dtype=torch.float64) / 8,
        labels=torch.from_numpy(labels).double(),
        original_labels=labels,
        poison=np.zeros(6, dtype=np.int64),
        sample_ids=np.arange(6, dtype=np.int64),
    )
    return batch, batch


def _reference() -> tuple[CIFARBatch, object, object, np.ndarray]:
    batch, _ = _case()
    model = build_cifar_model(5, hidden_dim=3)
    raw = capture_binary_training_raw(model, batch)
    fitted, query = fit_binary_ekfac(model, batch, batch.inputs[:2], raw_samples=raw)
    scores = binary_ekfac_influence(fitted, query, original_count=6)
    return batch, model, (raw, query), scores


def test_dattri_ekfac_kernel_agreement_when_installed() -> None:
    pytest.importorskip("dattri_llm")
    _, _, raw_and_query, reference = _reference()
    raw, query = raw_and_query
    other = dattri_binary_ekfac_influence(raw, query)
    assert score_agreement(reference, other)["relative_l2_difference"] < 1e-5


def test_kronfluence_ekfac_workflow_agreement_when_installed() -> None:
    pytest.importorskip("kronfluence")
    batch, model, _, reference = _reference()
    other = kronfluence_binary_ekfac_influence(model, batch, batch.inputs[:2])
    assert score_agreement(reference, other)["relative_l2_difference"] < 1e-5
