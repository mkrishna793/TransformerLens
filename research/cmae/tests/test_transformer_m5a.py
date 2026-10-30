"""Milestone-5A arithmetic Transformer and causal component-trace tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from research.cmae.arithmetic_data import (
    MODULUS,
    PROMPT_LENGTH,
    VOCAB_SIZE,
    ArithmeticSample,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_transformer import (
    ArithmeticModelConfig,
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
    train_arithmetic_transformer,
)
from research.cmae.run_transformer_m5a import run_transformer_m5a
from research.cmae.transformer_circuit import (
    ComponentNode,
    component_nodes,
    integrated_readout_components,
    trace_component_circuit,
)


def _model(seed: int = 101) -> ArithmeticTransformer:
    torch.manual_seed(seed)
    return ArithmeticTransformer()


def _cache(model: ArithmeticTransformer, sample: ArithmeticSample) -> dict[str, torch.Tensor]:
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache


def _final_components(model: ArithmeticTransformer, cache: dict[str, torch.Tensor]):
    nodes = [node for node in component_nodes(model) if node.final_readout]
    return nodes, [node.value(cache[node.hook_name]) for node in nodes]


def test_modular_ground_truth_and_prompt_contract() -> None:
    sample = ArithmeticSample(2, 3, 4, 5, "iid")
    assert sample.u == 5
    assert sample.v == 9
    assert sample.target == 3
    assert sample.tokens.shape == (PROMPT_LENGTH,)
    assert sample.tokens[2] == MODULUS
    assert np.all((sample.tokens >= 0) & (sample.tokens < VOCAB_SIZE))
    assert np.array_equal(sample.intermediates, np.asarray([5, 9, 3]))


def test_splits_are_disjoint_and_cover_output_vocabulary() -> None:
    dataset = make_arithmetic_dataset(101)
    keys = [sample.key for sample in dataset.samples]
    assert len(keys) == len(set(keys))
    train = dataset.split_samples("train")
    assert {sample.target for sample in train} == set(range(MODULUS))
    assert all(sample.a + sample.b != 7 for sample in train)
    assert all(sample.a + sample.b == 7 for sample in dataset.split_samples("composition"))
    assert all(sample.a >= 6 and sample.b >= 6 for sample in dataset.split_samples("operand"))
    assert all(sample.c >= 4 for sample in dataset.split_samples("multiplier"))
    assert dataset.counts()["train"] == 403


def test_model_validation_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ArithmeticModelConfig(d_model=31, n_heads=2)


def test_residual_writes_reconstruct_final_residual_and_readout() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    cache = _cache(model, sample)
    reconstructed = cache["hook_token_embed"] + cache["hook_position_embed"]
    for layer in range(model.config.n_layers):
        reconstructed = reconstructed + cache[f"blocks.{layer}.hook_attn_out"].sum(dim=2)
        reconstructed = reconstructed + cache[f"blocks.{layer}.hook_mlp_out"]
    residual = cache["hook_resid_final"]
    assert torch.allclose(reconstructed, residual, atol=1e-6, rtol=1e-6)
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        direct = model(tokens)
        readout = model.readout_vector(residual)
    assert torch.allclose(direct, readout, atol=1e-6, rtol=1e-6)


def test_attention_is_causal_and_normalized() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    cache = _cache(model, sample)
    pattern = cache["blocks.0.attention.hook_pattern"]
    future = torch.triu(torch.ones(PROMPT_LENGTH, PROMPT_LENGTH, dtype=torch.bool), 1)
    assert torch.all(pattern[:, :, future] == 0)
    assert torch.allclose(pattern.sum(dim=-1), torch.ones_like(pattern.sum(dim=-1)))


def test_component_inventory_has_unique_nodes_and_eight_final_writes() -> None:
    model = _model()
    nodes = component_nodes(model)
    assert len(nodes) == 104
    assert len({node.node_id for node in nodes}) == 104
    assert sum(node.final_readout for node in nodes) == 8
    assert all(node.hook_name in model.hook_dict for node in nodes)


def test_integrated_readout_is_exact_for_a_linear_readout() -> None:
    model = _model()
    model.final_norm = nn.Identity()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    cache = _cache(model, sample)
    _, components = _final_components(model, cache)
    residual = cache["hook_resid_final"][0, -1, :]
    _, _, error, contributions = integrated_readout_components(
        model, residual, components, sample.target, steps=8
    )
    assert abs(error) < 1e-5
    assert contributions.shape == (8,)


def test_layernorm_integrated_readout_converges_with_quadrature() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    cache = _cache(model, sample)
    _, components = _final_components(model, cache)
    residual = cache["hook_resid_final"][0, -1, :]
    *_, coarse_error, _ = integrated_readout_components(
        model, residual, components, sample.target, steps=16
    )
    *_, fine_error, _ = integrated_readout_components(
        model, residual, components, sample.target, steps=128
    )
    assert abs(fine_error) < abs(coarse_error)
    assert abs(fine_error) < 1e-3


def test_node_ablation_is_local_to_the_selected_slice() -> None:
    position = ComponentNode("p", "hook_position_embed", "position", -1, 2, None, False)
    position_value = torch.ones(PROMPT_LENGTH, 3)
    assert torch.all(position.ablate(position_value)[2] == 0)
    assert torch.all(position.ablate(position_value)[1] == 1)
    attention = ComponentNode("a", "blocks.0.hook_attn_out", "attention", 0, 3, 1, False)
    attention_value = torch.ones(1, PROMPT_LENGTH, 2, 4)
    altered = attention.ablate(attention_value)
    assert torch.all(altered[0, 3, 1] == 0)
    assert torch.all(altered[0, 3, 0] == 1)
    assert torch.all(altered[0, 2, 1] == 1)


def test_component_trace_conserves_readout_and_measures_every_node() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    trace = trace_component_circuit(model, sample, quadrature_steps=128)
    metrics = trace.metrics()
    assert len(trace.measurements) == 104
    assert sum(m.integrated_direct is not None for m in trace.measurements) == 8
    assert trace.residual_reconstruction_error < 1e-4
    assert abs(trace.readout_parity_error) < 1e-4
    assert abs(trace.integrated_quadrature_residual) < 1e-3
    expected = trace.target_logit - trace.readout_baseline - trace.integrated_quadrature_residual
    assert metrics["integrated_direct_sum"] == pytest.approx(expected, abs=1e-5)
    assert np.isfinite(metrics["first_order_pearson"])
    assert all(np.isfinite(m.causal_effect) for m in trace.measurements)


def test_trace_does_not_mutate_parameters_gradients_or_mode() -> None:
    model = _model()
    model.train()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trace_component_circuit(model, sample, quadrature_steps=16)
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_short_training_run_and_evaluation_are_real() -> None:
    dataset = make_arithmetic_dataset(101)
    model = _model()
    history = train_arithmetic_transformer(model, dataset, epochs=2, seed=102)
    assert history[-1] < history[0]
    result = evaluate_arithmetic_transformer(model, dataset, "train")
    assert result["samples"] == 403
    assert 0 <= result["accuracy"] <= 1
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["mean_target_probability"])


def test_small_consolidated_runner_writes_complete_artifact(tmp_path) -> None:
    output = tmp_path / "m5a.json"
    result = run_transformer_m5a(output, epochs=1, quadrature_steps=16)
    loaded = json.loads(output.read_text())
    assert loaded["trace_summary"]["trace_count"] == 6
    assert loaded["trace_summary"]["nodes_per_trace"] == 104
    assert len(loaded["traces"]) == 6
    assert loaded["formal_verification"]["rocq_closed_theorems"] == 6
    assert result["task"]["all_output_tokens_covered_by_train"]
