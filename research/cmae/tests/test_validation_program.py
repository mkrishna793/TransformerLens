"""Contract tests for the five-stage CMAE validation program."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.cmae.attribution import (
    ekfac_output_attribution,
    exact_ggn_output_attribution,
    fit_logistic_newton,
    gradient_output_attribution,
    logistic_loss,
    sigmoid,
    trak_output_attribution,
)
from research.cmae.ground_truth import (
    BenchmarkMLP,
    FutureBenchmarkGRU,
    HandCompiledCausalAverage,
    causal_average_hypothesis_errors,
    make_synthetic_batch,
    repair_detection_case,
    run_detection_case,
)
from research.cmae.jacobian import (
    average_downstream_jacobian,
    estimate_probe_jacobian,
    first_order_residual,
    linear_readout,
    vocabulary_token_selector,
)
from research.cmae.prime_topology import (
    clique_complex_betti_numbers,
    latent_clique_multifield_homology,
    latent_graph_multifield_homology,
    multifield_disagreement,
)
from research.cmae.rl import audit_trajectories
from research.cmae.run_validation_program import (
    run_attribution_stage,
    run_jacobian_stage,
)
from research.cmae.runtime import SynchronousBudgetGate
from research.cmae.science import (
    KINETIC_ENERGY,
    MOMENTUM,
    Dimensions,
    elastic_collision_1d,
    family_holdout,
    permutation_error,
    validate_collision,
)


def test_average_linear_jacobian_is_exact() -> None:
    weights = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    linear = lambda value: value @ weights.T
    activations = [torch.ones(2, dtype=torch.float64), torch.arange(2, dtype=torch.float64)]
    estimate = average_downstream_jacobian(activations, [linear, linear])
    residual = first_order_residual(estimate, linear, activations[0], torch.tensor([0.2, -0.4]))
    assert residual.relative == pytest.approx(0.0, abs=1e-12)
    assert torch.equal(estimate.dispersion, torch.zeros_like(estimate.dispersion))


def test_vocabulary_readout_selector_recovers_requested_logit() -> None:
    weights = torch.tensor([[0.2, -0.1], [1.0, 0.5], [-0.7, 2.0]], dtype=torch.float64)
    readout = linear_readout(weights)
    activation = torch.tensor([0.3, -0.4], dtype=torch.float64)
    estimate = average_downstream_jacobian(
        [activation], [readout], output_selector=vocabulary_token_selector(2)
    )
    assert estimate.mean.shape == (1, 2)
    assert torch.allclose(estimate.mean[0], weights[2])


def test_nonlinear_jacobian_residual_is_not_hidden() -> None:
    nonlinear = lambda value: torch.tanh(value * value)
    activation = torch.tensor([0.5, -0.7], dtype=torch.float64)
    estimate = average_downstream_jacobian([activation], [nonlinear])
    report = first_order_residual(estimate, nonlinear, activation, torch.tensor([0.4, 0.3]))
    assert np.isfinite(report.relative)
    assert report.absolute > 0


def test_probe_lens_recovers_hand_compiled_final_output_jacobian() -> None:
    model = HandCompiledCausalAverage().double()
    inputs = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    estimate = estimate_probe_jacobian(
        model, model.probe, inputs, output_selector=lambda output: output[:, -1, :]
    )
    expected = torch.zeros_like(estimate.mean)
    expected[0, 0, :, 0] = 1.0 / 3.0
    expected[1, 0, :, 0] = -1.0 / 3.0
    assert torch.allclose(estimate.mean, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(estimate.dispersion, torch.zeros_like(estimate.dispersion))


def test_class_and_future_output_lens_contracts() -> None:
    class_model = BenchmarkMLP().double()
    class_inputs = make_synthetic_batch(16, 21).inputs[:2]
    class_estimate = estimate_probe_jacobian(class_model, class_model.probe, class_inputs)
    assert class_estimate.mean.shape == (1, 1, 12)
    future_model = FutureBenchmarkGRU().double()
    future_inputs = make_synthetic_batch(16, 22).inputs[:2]
    future_estimate = estimate_probe_jacobian(
        future_model,
        future_model.probe,
        future_inputs,
        output_selector=lambda output: output[:, -1:],
    )
    assert future_estimate.mean.shape == (1, 1, 3, 6)
    assert bool(torch.isfinite(future_estimate.mean).all())


def test_jacobian_stage_includes_ground_truth_and_architecture_adapters() -> None:
    result = run_jacobian_stage()
    assert result["ground_truth_hypothesis_recovered"]
    assert result["linear_max_relative_residual"] < 1e-10
    assert result["vocabulary_readout_max_error"] == pytest.approx(0.0, abs=1e-12)
    assert len(result["causal_intervention_relative_residuals"]) == 3
    assert all(np.isfinite(value) for value in result["causal_intervention_relative_residuals"])
    assert result["class_output_lens_finite"]
    assert result["future_gru_lens_finite"]


def test_logistic_attribution_methods_are_finite_and_aligned() -> None:
    rng = np.random.default_rng(31)
    features = rng.normal(size=(10, 3))
    features[:, -1] = 1.0
    labels = (features[:, 0] > 0).astype(np.float64)
    weights, _ = fit_logistic_newton(features, labels)
    query = np.array([0.3, -0.2, 1.0])
    methods = [
        gradient_output_attribution(features, labels, weights, query),
        trak_output_attribution(features, labels, weights, query),
        ekfac_output_attribution(features, labels, weights, query),
        exact_ggn_output_attribution(features, labels, weights, query),
    ]
    assert all(method.shape == (10,) for method in methods)
    assert all(np.all(np.isfinite(method)) for method in methods)
    assert logistic_loss(features, labels, weights) <= logistic_loss(features, labels, np.zeros(3))


def test_retraining_attribution_stage_has_group_counterfactuals() -> None:
    result = run_attribution_stage()
    assert result["training_examples"] == 18
    assert set(result["methods"]) == {"gradient", "trak_projected", "ekfac", "exact_ggn"}
    for method in result["methods"].values():
        assert "leave_one_out" in method
        assert "leave_group_out" in method


def test_latent_one_dimensional_graph_has_no_prime_added_value() -> None:
    points = np.random.default_rng(32).normal(size=(10, 3))
    records = latent_graph_multifield_homology(points)
    assert len(records) == 3
    assert multifield_disagreement(records) == 0.0


def test_certified_clique_complex_has_expected_two_cell_homology() -> None:
    edges = tuple((left, right) for left in range(4) for right in range(left + 1, 4))
    for prime in (2, 3, 5):
        assert clique_complex_betti_numbers(4, edges, prime).tolist() == [1, 0, 1]
    records = latent_clique_multifield_homology(np.random.default_rng(33).normal(size=(8, 2)))
    assert any(record.two_cells > 0 for record in records)
    assert multifield_disagreement(records) == 0.0


def test_timeout_and_success_budget_gates() -> None:
    timeout_values = iter((0.0, 2.0))
    timeout = SynchronousBudgetGate(1.0, clock=lambda: next(timeout_values)).run(
        lambda: "audit", lambda _: pytest.fail("timeout committed")
    )
    assert timeout.decision.value == "stop"
    assert not timeout.committed
    success_values = iter((0.0, 0.25))
    success = SynchronousBudgetGate(1.0, clock=lambda: next(success_values)).run(
        lambda: [1, 2], len
    )
    assert success.committed
    assert success.payload == 2


def test_average_rl_feasibility_does_not_hide_tail() -> None:
    result = audit_trajectories(np.ones(4), np.array([0.0, 0.0, 0.0, 3.0]), cost_threshold=1.0)
    assert result.expected_feasible
    assert result.tail_violation_rate == pytest.approx(0.25)
    assert result.maximum_cost == 3.0
    assert result.updated_multiplier == 0.0
    infeasible = audit_trajectories(np.ones(4), np.array([0.0, 0.0, 0.0, 8.0]), cost_threshold=1.0)
    assert not infeasible.expected_feasible
    assert infeasible.updated_multiplier > 0


def test_scientific_collision_dimensions_and_symmetry() -> None:
    final = elastic_collision_1d(2.0, 2.0, 1.5, -0.5)
    valid = validate_collision(2.0, 2.0, 1.5, -0.5, *final)
    invalid = validate_collision(2.0, 2.0, 1.5, -0.5, 0.0, 0.0)
    assert valid.valid
    assert not invalid.valid
    assert MOMENTUM == Dimensions(mass=1, length=1, time=-1)
    assert KINETIC_ENERGY == Dimensions(mass=1, length=2, time=-2)
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    assert permutation_error(lambda value: value.sum(axis=0), values, permutation) == 0.0
    assert permutation_error(lambda value: value * 2, values, permutation, equivariant=True) == 0.0


def test_family_holdout_has_no_leakage() -> None:
    train, test = family_holdout(np.array(["a", "a", "b", "c"]), "b")
    assert test.tolist() == [2]
    assert set(train).isdisjoint(set(test))


def test_adaptive_attack_metadata_preserves_regularizer_labels() -> None:
    batch = make_synthetic_batch(48, 41, "adaptive")
    regularizer = batch.regularizer.astype(bool)
    poison = batch.poison.astype(bool)
    assert regularizer.any() and poison.any()
    assert np.array_equal(batch.labels.numpy()[regularizer], batch.original_labels[regularizer])
    assert np.all(batch.labels.numpy()[poison] == 1)


def test_hand_compiled_hypothesis_is_selected_by_output_equivalence() -> None:
    inputs = torch.randn(2, 3, 2, dtype=torch.float64)
    errors = causal_average_hypothesis_errors(inputs)
    assert errors["uniform_causal_average"] == pytest.approx(0.0, abs=1e-12)
    assert errors["uniform_causal_average"] < errors["unnormalized_cumulative_sum"]


def test_trained_detection_and_repair_contract() -> None:
    case = run_detection_case("mlp", "standard", 901)
    assert case.result["training_epochs"] == 60
    assert {"pca", "spectral", "topology", "cmae_rank_union", "art"}.issubset(
        case.result["detectors"]
    )
    repair = repair_detection_case(case)
    assert repair["removed_samples"] > 0
    assert "repair_gate_pass" in repair
