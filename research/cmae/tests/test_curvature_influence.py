"""Exact-gradient and explicit-inverse tests for curvature-aware data influence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.cmae.arithmetic_data import make_arithmetic_dataset
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.curvature_influence import (
    RawLayerGradient,
    RawSampleGradient,
    capture_feature_gradient,
    capture_loss_gradient,
    fit_arithmetic_ekfac,
    fit_ekfac_from_gradients,
    tracked_arithmetic_linears,
)
from research.cmae.replacement_graph import SparseCrossLayerTranscoder


def _raw(
    key: tuple[int, int, int, int],
    activation: list[list[float]],
    gradient: list[list[float]],
    *,
    name: str = "linear",
) -> RawSampleGradient:
    return RawSampleGradient(
        key,
        {
            name: RawLayerGradient(
                np.asarray(activation, dtype=np.float64),
                np.asarray(gradient, dtype=np.float64),
            )
        },
    )


def _weight_gradient(raw: RawLayerGradient) -> np.ndarray:
    result = np.zeros((raw.output_gradient.shape[1], raw.activation.shape[1]), dtype=np.float64)
    for activation, output_gradient in zip(raw.activation, raw.output_gradient, strict=True):
        result += np.outer(output_gradient, activation)
    return result


def test_kfe_projection_and_corrected_moments_preserve_position_sums() -> None:
    module = torch.nn.Linear(2, 3, bias=False)
    samples = (
        _raw((0, 0, 0, 0), [[1, 2], [3, 4]], [[1, 0, 2], [0, 1, 1]]),
        _raw((1, 0, 0, 0), [[2, -1], [1, 3]], [[0, 2, 1], [1, 1, 0]]),
    )
    fitted = fit_ekfac_from_gradients(samples, {"linear": module}, damping=0.2)
    layer = fitted.layers[0]
    for index, sample in enumerate(samples):
        expected = (
            layer.output_basis.T
            @ _weight_gradient(sample.layers["linear"])
            @ layer.activation_basis
        )
        np.testing.assert_allclose(layer.training_projections[index], expected, atol=1e-12)
    expected_moments = np.mean(layer.training_projections**2, axis=0)
    np.testing.assert_allclose(layer.corrected_second_moments, expected_moments, atol=1e-12)


def test_influence_matches_explicit_materialized_ekfac_inverse() -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    training = (
        _raw((0, 0, 0, 0), [[1, 0], [0, 2]], [[1, 0], [0, 1]]),
        _raw((1, 0, 0, 0), [[1, 1], [2, 0]], [[0, 1], [1, 1]]),
    )
    damping = 0.3
    fitted = fit_ekfac_from_gradients(training, {"linear": module}, damping=damping)
    query = _raw((2, 0, 0, 0), [[1, 2], [0, 1]], [[1, 1], [0, 1]])
    actual = fitted.influence(query)
    layer = fitted.layers[0]
    eigenbasis = np.kron(layer.activation_basis, layer.output_basis)
    inverse = (
        eigenbasis
        @ np.diag(1.0 / (layer.corrected_second_moments.reshape(-1, order="F") + damping))
        @ eigenbasis.T
    )
    query_gradient = _weight_gradient(query.layers["linear"]).reshape(-1, order="F")
    for index, sample in enumerate(training):
        training_gradient = _weight_gradient(sample.layers["linear"]).reshape(-1, order="F")
        expected = -query_gradient @ inverse @ training_gradient
        assert actual[index] == pytest.approx(expected, abs=1e-10)


def test_captured_loss_gradients_match_autograd_weight_gradients() -> None:
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    sample = make_arithmetic_dataset(101).split_samples("train")[0]
    modules = tracked_arithmetic_linears(model)
    raw = capture_loss_gradient(model, sample, sample.target, modules)
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    target = torch.tensor([sample.target])
    loss = torch.nn.functional.cross_entropy(model(tokens)[:, -1, :], target)
    parameters = [module.weight for module in modules.values()]
    expected = torch.autograd.grad(loss, parameters, retain_graph=False)
    for (name, module), expected_gradient in zip(modules.items(), expected, strict=True):
        captured = _weight_gradient(raw.layers[name])
        np.testing.assert_allclose(
            captured,
            expected_gradient.detach().cpu().numpy(),
            atol=2e-6,
            rtol=2e-5,
            err_msg=name,
        )
        assert module.weight.grad is None


def test_internal_feature_query_has_zero_gradients_at_downstream_layers() -> None:
    torch.manual_seed(17)
    model = ArithmeticTransformer()
    replacement = SparseCrossLayerTranscoder(model.config.d_model, 8, "clt")
    with torch.no_grad():
        replacement.encoder0.bias.fill_(10)
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    modules = tracked_arithmetic_linears(model)
    raw = capture_feature_gradient(model, replacement, sample, layer=0, feature=0, modules=modules)
    for name in (
        "blocks.0.mlp.output",
        "blocks.1.attention.query",
        "blocks.1.mlp.input",
        "blocks.1.mlp.output",
        "unembed",
    ):
        assert np.count_nonzero(raw.layers[name].output_gradient) == 0, name
    assert np.count_nonzero(raw.layers["blocks.0.attention.query"].output_gradient) > 0


def test_arithmetic_ekfac_logit_influence_is_finite_and_ordered() -> None:
    dataset = make_arithmetic_dataset(101)
    samples = dataset.split_samples("train")[:3]
    targets = np.asarray([sample.target for sample in samples], dtype=np.int64)
    torch.manual_seed(23)
    model = ArithmeticTransformer()
    fitted = fit_arithmetic_ekfac(model, samples, targets, damping=0.1)
    query = dataset.split_samples("calibration")[0]
    scores = fitted.logit_influence(model, query, query.target)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert fitted.sample_keys == tuple(sample.key for sample in samples)
    diagnostics = fitted.diagnostics()
    weight_parameters = diagnostics["weight_parameters"]
    minimum_moment = diagnostics["minimum_corrected_second_moment"]
    assert isinstance(weight_parameters, int) and weight_parameters > 0
    assert isinstance(minimum_moment, float) and minimum_moment >= 0


def test_fit_rejects_duplicate_sample_keys() -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    sample = _raw((0, 0, 0, 0), [[1]], [[1]])
    with pytest.raises(ValueError, match="unique"):
        fit_ekfac_from_gradients((sample, sample), {"linear": module})
