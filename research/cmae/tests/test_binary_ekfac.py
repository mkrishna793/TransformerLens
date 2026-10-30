import numpy as np
import torch

from research.cmae.backdoorbench_organism import (
    CIFARBatch,
    build_cifar_model,
)
from research.cmae.binary_ekfac import (
    _capture_binary_raw,
    binary_ekfac_influence,
    fit_binary_ekfac,
    tracked_binary_linears,
)


def _batch() -> CIFARBatch:
    torch.manual_seed(3)
    count = 6
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    return CIFARBatch(
        inputs=torch.randn(count, 64, dtype=torch.float64) / 8,
        labels=torch.from_numpy(labels).double(),
        original_labels=labels,
        poison=np.zeros(count, dtype=np.int64),
        sample_ids=np.arange(count, dtype=np.int64),
    )


def test_raw_binary_layer_gradient_matches_autograd() -> None:
    model = build_cifar_model(5, hidden_dim=3)
    batch = _batch()
    modules = tracked_binary_linears(model)
    raw = _capture_binary_raw(
        model, batch.inputs[:1], modules, "loss", batch.labels[0], (0, 0, 0, 0)
    )
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(batch.inputs[:1])[0], batch.labels[0]
    )
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    for name, parameter_gradient in zip(("input", "head"), gradients, strict=True):
        layer = raw.layers[name]
        reconstructed = torch.from_numpy(layer.output_gradient.T @ layer.activation)
        torch.testing.assert_close(reconstructed, parameter_gradient, rtol=1e-10, atol=1e-12)


def test_binary_ekfac_fixed_normalization_and_explicit_score() -> None:
    model = build_cifar_model(5, hidden_dim=3)
    batch = _batch()
    query_inputs = batch.inputs[:2]
    fitted, query = fit_binary_ekfac(model, batch, query_inputs, damping=1e-3)
    scores = binary_ekfac_influence(fitted, query, original_count=len(batch.labels))
    torch.testing.assert_close(
        torch.from_numpy(scores * len(batch.labels)),
        torch.from_numpy(fitted.influence(query)),
        rtol=1e-12,
        atol=1e-12,
    )
    manual = np.zeros(len(batch.labels), dtype=np.float64)
    for layer in fitted.layers:
        projected_query = layer.project(query.layers[layer.name])
        weighted = projected_query / (layer.corrected_second_moments + fitted.damping)
        manual -= np.einsum("noi,oi->n", layer.training_projections, weighted)
    manual /= len(batch.labels)
    np.testing.assert_allclose(scores, manual, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(scores))
