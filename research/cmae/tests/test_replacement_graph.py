"""Scoped SAE/CLT replacement and answer-position graph tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from research.cmae.arithmetic_data import make_arithmetic_dataset
from research.cmae.arithmetic_transformer import ArithmeticBlock, ArithmeticTransformer
from research.cmae.replacement_graph import (
    SparseCrossLayerTranscoder,
    collect_mlp_activations,
    evaluate_replacement_reconstruction,
    fit_replacement,
    trace_replacement_graph,
)


def _model(seed: int = 101) -> ArithmeticTransformer:
    torch.manual_seed(seed)
    return ArithmeticTransformer()


def test_mlp_input_hooks_match_actual_interfaces() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    activations = collect_mlp_activations(model, tokens, batch_size=1)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        for layer in (0, 1):
            block = cast(ArithmeticBlock, model.blocks[layer])
            residual = cache[f"blocks.{layer}.hook_resid_mid"]
            expected_input = block.mlp_norm(residual)
            assert torch.allclose(activations.inputs[layer], expected_input, atol=1e-6, rtol=1e-6)
            expected_output = block.mlp(expected_input)
            assert torch.allclose(activations.outputs[layer], expected_output, atol=1e-6, rtol=1e-6)


def test_independent_sae_is_clt_with_zero_cross_decoder() -> None:
    torch.manual_seed(17)
    sae = SparseCrossLayerTranscoder(4, 6, "sae")
    clt = SparseCrossLayerTranscoder(4, 6, "clt")
    clt.load_state_dict(sae.state_dict())
    inputs = (torch.randn(3, 4), torch.randn(3, 4))
    with torch.no_grad():
        sae_reconstruction, sae_codes = sae(inputs)
        clt_reconstruction, clt_codes = clt(inputs)
    assert not sae.decoder01.weight.requires_grad
    assert torch.count_nonzero(sae.decoder01.weight) == 0
    for sae_code, clt_code in zip(sae_codes, clt_codes, strict=True):
        assert torch.equal(sae_code, clt_code)
    for sae_value, clt_value in zip(sae_reconstruction, clt_reconstruction, strict=True):
        assert torch.equal(sae_value, clt_value)


def test_short_replacement_training_is_finite_and_preserves_model() -> None:
    dataset = make_arithmetic_dataset(101)
    model = _model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    replacement, history = fit_replacement(
        model,
        dataset,
        "clt",
        features=8,
        epochs=2,
        batch_size=1024,
        seed=19,
    )
    assert len(history) == 2
    assert np.all(np.isfinite(history))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    tokens_np, _, _ = dataset.arrays("calibration")
    metrics = evaluate_replacement_reconstruction(
        replacement, collect_mlp_activations(model, torch.from_numpy(tokens_np[:8]))
    )
    for layer in ("layer0", "layer1"):
        values = metrics[layer]
        assert isinstance(values, dict)
        for metric in (
            "mse",
            "relative_error_mass",
            "explained_variance",
            "mean_l0",
            "active_feature_fraction",
            "dead_feature_fraction",
        ):
            value = values[metric]
            assert isinstance(value, float) and np.isfinite(value)
        assert values["max_reconstruction_conservation_error"] == 0


def test_graph_conserves_output_and_validates_every_retained_edge() -> None:
    dataset = make_arithmetic_dataset(101)
    model = _model()
    replacement = SparseCrossLayerTranscoder(model.config.d_model, 8, "clt")
    sample = dataset.split_samples("iid")[0]
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trace = trace_replacement_graph(model, replacement, sample, quadrature_steps=256)
    result = trace.to_dict()
    metrics = trace.metrics()
    assert trace.readout_parity_error < 1e-6
    assert trace.reconstruction_conservation_error == 0
    assert abs(trace.integrated_quadrature_residual) < 1e-3
    assert metrics["output_conservation_error"] == pytest.approx(0, abs=1e-6)
    relative_error = metrics["replacement_relative_logit_error_mass"]
    assert isinstance(relative_error, float) and relative_error >= 0
    assert len(trace.nodes) >= 2
    assert sum(node.kind == "error" for node in trace.nodes) == 2
    assert trace.edges
    edge_sources = {edge.source for edge in trace.edges}
    edge_targets = {edge.target for edge in trace.edges}
    node_ids = {node.node_id for node in trace.nodes}
    assert edge_sources <= node_ids
    assert edge_targets <= node_ids
    assert all(edge.source.startswith("l0_") for edge in trace.edges)
    assert all(edge.target.startswith("l1_feature_") for edge in trace.edges)
    for edge in trace.edges:
        assert np.isfinite(edge.local_prediction)
        assert np.isfinite(edge.actual_target_effect)
        assert edge.residual == pytest.approx(
            edge.actual_target_effect - edge.local_prediction, abs=1e-7
        )
    scope = result["scope"]
    assert isinstance(scope, str) and "Neither is a semantic certificate" in scope
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_fully_inactive_features_abstain_with_empty_edge_metrics() -> None:
    dataset = make_arithmetic_dataset(101)
    model = _model()
    replacement = SparseCrossLayerTranscoder(model.config.d_model, 4, "sae")
    with torch.no_grad():
        replacement.encoder0.weight.zero_()
        replacement.encoder1.weight.zero_()
        replacement.encoder0.bias.fill_(-10)
        replacement.encoder1.bias.fill_(-10)
    trace = trace_replacement_graph(
        model, replacement, dataset.split_samples("combined")[0], quadrature_steps=16
    )
    assert len(trace.nodes) == 2
    assert not trace.edges
    metrics = trace.metrics()
    assert metrics["feature_edge_pearson"] is None
    assert metrics["feature_edge_relative_residual_mass"] is None
    assert metrics["output_conservation_error"] == pytest.approx(0, abs=1e-6)
