"""Training-only latent-lens and calibration-controlled training tests."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
import torch

from research.cmae.arithmetic_data import (
    VOCAB_SIZE,
    ArithmeticDataset,
    ArithmeticSample,
    ArithmeticSplit,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_transformer import ArithmeticBlock, ArithmeticTransformer
from research.cmae.generalization import (
    evaluate_arithmetic_transformer,
    readout_rank_metrics,
    train_with_calibration,
)
from research.cmae.latent_lenses import (
    LatentLensSuite,
    deterministic_kmeans,
    evaluate_latent_lenses,
    fit_latent_lenses,
)
from research.cmae.run_attention_interventions import run_attention_interventions
from research.cmae.run_transformer_m5b import run_transformer_m5b
from research.cmae.run_transformer_replacements import run_transformer_replacements


def _small_dataset(seed: int = 101) -> ArithmeticDataset:
    source = make_arithmetic_dataset(seed)
    samples = (
        source.split_samples("train")[:12]
        + source.split_samples("calibration")[:4]
        + source.split_samples("iid")[:4]
    )
    return ArithmeticDataset(samples, seed)


def _fit(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    *,
    weight_steps: int = 2,
) -> LatentLensSuite:
    return fit_latent_lenses(
        model,
        dataset,
        clusters=2,
        batch_size=16,
        kmeans_iterations=4,
        weight_steps=weight_steps,
    )


def test_deterministic_kmeans_repeats_and_partitions() -> None:
    values = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]])
    first = deterministic_kmeans(values, 2, max_iterations=10)
    second = deterministic_kmeans(values, 2, max_iterations=10)
    assert torch.equal(first.centers, second.centers)
    assert torch.equal(first.assignments, second.assignments)
    assert first.inertia == pytest.approx(second.inertia)
    assert torch.bincount(first.assignments, minlength=2).sum().item() == len(values)
    assert first.inertia < 0.1


def test_jacobian_matches_independent_finite_differences() -> None:
    source = make_arithmetic_dataset(101)
    sample = source.split_samples("train")[0]
    dataset = ArithmeticDataset((sample,), 101)
    torch.manual_seed(103)
    model = ArithmeticTransformer().double()
    suite = fit_latent_lenses(
        model,
        dataset,
        clusters=1,
        batch_size=1,
        kmeans_iterations=2,
        weight_steps=0,
    )
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        _, early = model.residual_sites(tokens)
    positions = tokens.shape[1]
    epsilon = 1e-5

    def summed_output(activation: torch.Tensor, output_index: int) -> torch.Tensor:
        final = model.hook_resid_final(model.blocks[1](activation))
        return final[:, :, output_index].sum(dim=1)

    for output_index, input_index in ((0, 0), (3, 5), (31, 17)):
        plus = early.repeat(positions, 1, 1).clone()
        minus = plus.clone()
        indices = torch.arange(positions)
        plus[indices, indices, input_index] += epsilon
        minus[indices, indices, input_index] -= epsilon
        expected = (
            (summed_output(plus, output_index) - summed_output(minus, output_index)) / (2 * epsilon)
        ).mean()
        actual = suite.early.jacobian[output_index, input_index]
        assert actual.item() == pytest.approx(expected.item(), abs=2e-6, rel=2e-6)


def test_identity_downstream_and_train_only_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _small_dataset()
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    second_block = cast(ArithmeticBlock, model.blocks[1])
    with torch.no_grad():
        second_block.attention.output.weight.zero_()
        second_block.mlp.output.weight.zero_()
    calls: list[str] = []
    original = ArithmeticDataset.arrays

    def tracked_arrays(self: ArithmeticDataset, split: ArithmeticSplit):
        calls.append(split)
        return original(self, split)

    monkeypatch.setattr(ArithmeticDataset, "arrays", tracked_arrays)
    suite = _fit(model, dataset, weight_steps=0)
    assert calls == ["train"]
    identity = torch.eye(model.config.d_model)
    assert torch.allclose(suite.early.jacobian, identity, atol=1e-5, rtol=1e-5)
    assert torch.allclose(suite.early.cluster_jacobians[0], identity, atol=1e-5, rtol=1e-5)
    assert torch.allclose(suite.early.cluster_jacobians[1], identity, atol=1e-5, rtol=1e-5)
    assert torch.allclose(suite.early.jpp_jacobian, suite.early.jacobian, atol=1e-6)


def test_fit_preserves_mode_parameters_and_existing_gradients() -> None:
    dataset = _small_dataset()
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    model.train()
    model.unembed.weight.grad = torch.full_like(model.unembed.weight, 0.25)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    gradient = model.unembed.weight.grad.detach().clone()
    _fit(model, dataset)
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert model.unembed.weight.grad is not None
    assert torch.equal(model.unembed.weight.grad, gradient)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name != "unembed.weight"
    )


def test_evaluation_has_all_readouts_and_exact_final_site() -> None:
    dataset = _small_dataset()
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    suite = _fit(model, dataset)
    result = evaluate_latent_lenses(model, suite, dataset, "iid", batch_size=3)
    metrics = result["metrics"]
    approximation = result["approximation"]
    assert isinstance(metrics, dict) and isinstance(approximation, dict)
    assert len(metrics) == 12
    assert len(approximation) == 6
    prior = result["training_prior"]
    assert isinstance(prior, dict)
    assert set(prior) == {"u", "v"}
    for values in metrics.values():
        assert all(np.isfinite(value) for value in values.values())
    for method in ("logit", "jacobian", "jpp"):
        final = approximation[f"final_{method}"]
        assert final["approximation_error_mass"] == pytest.approx(0.0, abs=1e-6)
        assert final["readout_error_mass"] == pytest.approx(0.0, abs=1e-5)
        assert final["max_abs_conservation_error"] <= 1e-6
        assert metrics[f"final_u_{method}"] == metrics["final_u_logit"]
        assert metrics[f"final_v_{method}"] == metrics["final_v_logit"]


def test_full_vocabulary_rank_metrics_include_stable_ties() -> None:
    logits = torch.zeros(2, VOCAB_SIZE)
    targets = torch.tensor([0, 10])
    metrics = readout_rank_metrics(logits, targets)
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["recall_at_10"] == 0.5
    assert metrics["mean_rank"] == 6.0
    assert metrics["mean_reciprocal_rank"] == pytest.approx((1.0 + 1.0 / 11.0) / 2)


def test_auxiliary_forward_preserves_final_answer_api() -> None:
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    sample: ArithmeticSample = make_arithmetic_dataset(101).split_samples("iid")[0]
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        ordinary = model(tokens)
        logits, u_logits, v_logits = model.forward_with_latents(tokens)
    assert torch.equal(logits, ordinary)
    assert torch.equal(v_logits, ordinary[:, -1, :])
    assert u_logits.shape == (1, VOCAB_SIZE)


def test_calibration_training_uses_no_untouched_split() -> None:
    dataset = _small_dataset()
    calls: list[str] = []
    original = ArithmeticDataset.arrays

    def tracked_arrays(self: ArithmeticDataset, split: ArithmeticSplit):
        calls.append(split)
        return original(self, split)

    torch.manual_seed(107)
    standard = ArithmeticTransformer()
    auxiliary = ArithmeticTransformer()
    auxiliary.load_state_dict(standard.state_dict())
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(ArithmeticDataset, "arrays", tracked_arrays)
        standard_result = train_with_calibration(
            standard, dataset, "standard", epochs=1, evaluation_every=1, batch_size=8
        )
        auxiliary_result = train_with_calibration(
            auxiliary, dataset, "auxiliary", epochs=1, evaluation_every=1, batch_size=8
        )
    finally:
        monkeypatch.undo()
    assert set(calls) == {"train", "calibration"}
    assert len(standard_result.history) == 1
    assert len(auxiliary_result.history) == 1
    for model, result in ((standard, standard_result), (auxiliary, auxiliary_result)):
        calibration = evaluate_arithmetic_transformer(model, dataset, "calibration")
        assert result.best_calibration_loss == pytest.approx(calibration["loss"], abs=1e-6)
    assert any(
        not torch.equal(standard_value, auxiliary_value)
        for standard_value, auxiliary_value in zip(
            standard.state_dict().values(), auxiliary.state_dict().values(), strict=True
        )
    )


def test_small_controlled_runner_writes_multi_split_artifact(tmp_path) -> None:
    output = tmp_path / "m5b.json"
    result = run_transformer_m5b(
        output,
        seeds=(101,),
        arms=("standard", "auxiliary"),
        epochs=1,
        evaluation_every=1,
        batch_size=64,
        clusters=2,
        kmeans_iterations=2,
        lens_weight_steps=1,
    )
    loaded = json.loads(output.read_text())
    assert len(loaded["runs"]) == 2
    assert set(loaded["aggregate_accuracy"]) == {"standard", "auxiliary"}
    assert "auxiliary_minus_standard" in loaded["paired_untouched_accuracy"]
    for run in loaded["runs"]:
        assert set(run["evaluation"]) == {
            "train",
            "calibration",
            "iid",
            "composition",
            "operand",
            "multiplier",
            "combined",
        }
        assert len(run["lens_evaluation"]["iid"]["metrics"]) == 12
        assert run["lens_fit"]["early"]["fit_split"] == "train"
        assert run["training"]["auxiliary_weight"] == (0.5 if run["arm"] == "auxiliary" else 0.0)
    conclusion = result["mechanistic_conclusion"]
    assert isinstance(conclusion, dict)
    assert conclusion["status"] == "abstain_pending_causal_interventions"

    intervention_output = tmp_path / "interventions.json"
    interventions = run_attention_interventions(
        intervention_output, output, seeds=(101,), samples_per_split=1
    )
    loaded_interventions = json.loads(intervention_output.read_text())
    assert loaded_interventions["checkpoint_reproduction_verified"]
    assert len(loaded_interventions["runs"]) == 2
    for run in loaded_interventions["runs"]:
        assert len(run["samples"]) == 5
        assert len(run["samples"][0]["attention"]["heads"]) == 4
        assert len(run["samples"][0]["residual_patches"]) == 3
        final_patch = run["summary"]["residual_patches"]["final"]
        assert final_patch["max_donor_logit_parity_error"] < 1e-6
        assert final_patch["donor_prediction_match_rate"] == 1.0
        assert np.isfinite(final_patch["first_order_residual_over_actual_effect_mass"])
    intervention_conclusion = interventions["conclusion"]
    assert isinstance(intervention_conclusion, dict)
    assert intervention_conclusion["status"] == "abstain_u_specific_mechanism"

    replacement_output = tmp_path / "replacements.json"
    replacements = run_transformer_replacements(
        replacement_output,
        output,
        seeds=(101,),
        features=4,
        replacement_epochs=1,
        replacement_batch_size=1024,
        quadrature_steps=16,
    )
    loaded_replacements = json.loads(replacement_output.read_text())
    assert loaded_replacements["checkpoint_reproduction_verified"]
    assert len(loaded_replacements["runs"]) == 4
    for run in loaded_replacements["runs"]:
        assert len(run["graphs"]) == 5
        assert run["graph_summary"]["status"] in {
            "pass_scoped_numerical_replacement_gate",
            "abstain_incomplete_replacement",
        }
        assert run["graph_summary"]["max_abs_output_conservation_error"] < 1e-5
        if run["replacement"] == "sae":
            assert run["reconstruction"]["combined"]["cross_layer_decoder_norm"] == 0
    replacement_conclusion = replacements["conclusion"]
    assert isinstance(replacement_conclusion, dict)
    assert replacement_conclusion["status"] == "abstain_mechanistic_semantics"
