"""Run all five CMAE validation stages sequentially and write evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cmae.attribution import (
    ekfac_output_attribution,
    exact_ggn_output_attribution,
    fit_logistic_newton,
    gradient_output_attribution,
    sigmoid,
    trak_output_attribution,
)
from research.cmae.ground_truth import (
    BenchmarkMLP,
    DetectionCase,
    FutureBenchmarkGRU,
    HandCompiledCausalAverage,
    causal_average_hypothesis_errors,
    detector_scores,
    make_synthetic_batch,
    repair_detection_case,
    run_detection_case,
)
from research.cmae.jacobian import (
    average_downstream_jacobian,
    estimate_probe_jacobian,
    first_order_residual,
    intervention_errors,
    linear_readout,
    vocabulary_token_selector,
)
from research.cmae.metrics import pearson, spearman
from research.cmae.prime_topology import (
    LatentHomologyRecord,
    latent_clique_multifield_homology,
    latent_graph_multifield_homology,
    multifield_disagreement,
)
from research.cmae.rl import audit_trajectories
from research.cmae.runtime import SynchronousBudgetGate, measure_operation
from research.cmae.science import (
    KINETIC_ENERGY,
    MOMENTUM,
    elastic_collision_1d,
    family_holdout,
    permutation_error,
    validate_collision,
)


def run_jacobian_stage() -> dict[str, object]:
    generator = torch.Generator().manual_seed(101)
    weights = torch.tensor([[0.5, -1.0], [2.0, 0.3]], dtype=torch.float64)
    bias = torch.tensor([0.2, -0.4], dtype=torch.float64)
    linear = lambda value: value @ weights.T + bias
    activations = [torch.randn(2, generator=generator, dtype=torch.float64) for _ in range(4)]
    estimate = average_downstream_jacobian(activations, [linear] * len(activations))
    delta = torch.randn(2, generator=generator, dtype=torch.float64)
    linear_residual = first_order_residual(estimate, linear, activations[0], delta)
    vocabulary_weights = torch.tensor([[0.2, -0.1], [1.0, 0.5], [-0.7, 2.0]], dtype=torch.float64)
    vocabulary_estimate = average_downstream_jacobian(
        [activations[0]],
        [linear_readout(vocabulary_weights)],
        output_selector=vocabulary_token_selector(2),
    )
    vocabulary_error = float((vocabulary_estimate.mean[0] - vocabulary_weights[2]).abs().max())

    nonlinear = lambda value: torch.tanh(value @ weights.T + bias)
    nonlinear_estimate = average_downstream_jacobian(activations, [nonlinear] * len(activations))
    small_residual = first_order_residual(
        nonlinear_estimate, nonlinear, activations[0], delta * 0.01
    )
    large_residual = first_order_residual(nonlinear_estimate, nonlinear, activations[0], delta)
    intervention_residuals = intervention_errors(
        nonlinear_estimate, nonlinear, activations[0], delta, (0.01, 0.1, 1.0)
    )

    sequence_model = HandCompiledCausalAverage().double()
    sequence_inputs = torch.randn(4, 3, 2, generator=generator, dtype=torch.float64)
    sequence_inputs.requires_grad_(True)
    sequence_estimate = estimate_probe_jacobian(
        sequence_model,
        sequence_model.probe,
        sequence_inputs,
        output_selector=lambda output: output[:, -1, :],
    )
    expected = torch.zeros_like(sequence_estimate.mean)
    expected[0, 0, :, 0] = 1.0 / 3.0
    expected[1, 0, :, 0] = -1.0 / 3.0
    sequence_error = float((sequence_estimate.mean - expected).abs().max().item())
    hypotheses = causal_average_hypothesis_errors(sequence_inputs)

    torch.manual_seed(111)
    class_model = BenchmarkMLP().double()
    class_inputs = make_synthetic_batch(16, 111, "clean").inputs[:2]
    class_estimate = estimate_probe_jacobian(class_model, class_model.probe, class_inputs)
    torch.manual_seed(112)
    future_model = FutureBenchmarkGRU().double()
    future_inputs = make_synthetic_batch(16, 112, "clean").inputs[:2]
    future_estimate = estimate_probe_jacobian(
        future_model,
        future_model.probe,
        future_inputs,
        output_selector=lambda output: output[:, -1:],
    )

    return {
        "linear_max_relative_residual": linear_residual.relative,
        "linear_max_dispersion": float(estimate.dispersion.max().item()),
        "vocabulary_readout_max_error": vocabulary_error,
        "nonlinear_small_step_relative_residual": small_residual.relative,
        "nonlinear_large_step_relative_residual": large_residual.relative,
        "causal_intervention_relative_residuals": intervention_residuals.tolist(),
        "hand_compiled_final_output_jacobian_max_error": sequence_error,
        "hand_compiled_lens_dispersion_max": float(sequence_estimate.dispersion.max().item()),
        "circuit_hypothesis_errors": hypotheses,
        "ground_truth_hypothesis_recovered": bool(
            hypotheses["uniform_causal_average"] < hypotheses["unnormalized_cumulative_sum"]
        ),
        "class_output_lens_shape": list(class_estimate.mean.shape),
        "class_output_lens_finite": bool(torch.isfinite(class_estimate.mean).all().item()),
        "future_gru_lens_shape": list(future_estimate.mean.shape),
        "future_gru_lens_finite": bool(torch.isfinite(future_estimate.mean).all().item()),
        "claim_scope": "analytic and restricted hand-compiled checks, not general lens faithfulness",
    }


def _counterfactual_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return {
        "pearson": pearson(predicted, actual),
        "spearman": spearman(predicted, actual),
        "mean_absolute_error": float(np.mean(np.abs(predicted - actual))),
    }


def run_attribution_stage() -> dict[str, object]:
    rng = np.random.default_rng(202)
    features = rng.normal(size=(18, 3))
    features[:, -1] = 1.0
    true_weights = np.array([1.1, -0.9, 0.25])
    labels = (sigmoid(features @ true_weights) > 0.5).astype(np.float64)
    if len(np.unique(labels)) < 2:
        labels[0], labels[-1] = 0.0, 1.0
    weights, iterations = fit_logistic_newton(features, labels)
    query = np.array([0.4, -0.8, 1.0])
    projection = rng.normal(size=(3, 2)) / np.sqrt(2.0)
    methods = {
        "gradient": gradient_output_attribution(features, labels, weights, query),
        "trak_projected": trak_output_attribution(
            features, labels, weights, query, projection=projection
        ),
        "ekfac": ekfac_output_attribution(features, labels, weights, query),
        "exact_ggn": exact_ggn_output_attribution(features, labels, weights, query),
    }
    full_output = float(query @ weights)
    point_actual = np.zeros(features.shape[0], dtype=np.float64)
    for index in range(features.shape[0]):
        keep = np.arange(features.shape[0]) != index
        reduced_weights, _ = fit_logistic_newton(features[keep], labels[keep])
        point_actual[index] = full_output - float(query @ reduced_weights)
    groups = [
        group for group in np.array_split(rng.permutation(features.shape[0]), 6) if len(group)
    ]
    group_actual = np.zeros(len(groups), dtype=np.float64)
    for group_index, group in enumerate(groups):
        keep = np.ones(features.shape[0], dtype=bool)
        keep[group] = False
        reduced_weights, _ = fit_logistic_newton(features[keep], labels[keep])
        group_actual[group_index] = full_output - float(query @ reduced_weights)
    method_results: dict[str, object] = {}
    for name, predicted in methods.items():
        group_predicted = np.array([predicted[group].sum() for group in groups])
        method_results[name] = {
            "leave_one_out": _counterfactual_metrics(predicted, point_actual),
            "leave_group_out": _counterfactual_metrics(group_predicted, group_actual),
        }
    return {
        "training_examples": features.shape[0],
        "optimizer_iterations": iterations,
        "query_output": full_output,
        "point_counterfactual_norm": float(np.linalg.norm(point_actual)),
        "group_counterfactual_norm": float(np.linalg.norm(group_actual)),
        "methods": method_results,
        "claim_scope": "small logistic retraining counterfactuals; not deep-network attribution accuracy",
    }


def _homology_records(records: list[LatentHomologyRecord]) -> list[dict[str, object]]:
    serialized = []
    for record in records:
        serialized.append(
            {
                "threshold": record.threshold,
                "edges": record.edges,
                "two_cells": record.two_cells,
                "complex_kind": record.complex_kind,
                "betti_by_prime": {
                    str(prime): values.tolist() for prime, values in record.betti_by_prime.items()
                },
            }
        )
    return serialized


def run_prime_stage(points: np.ndarray) -> dict[str, object]:
    graph_records = latent_graph_multifield_homology(points[:24])
    clique_records = latent_clique_multifield_homology(points[:24])
    graph_disagreement = multifield_disagreement(graph_records)
    clique_disagreement = multifield_disagreement(clique_records)
    disagreement = max(graph_disagreement, clique_disagreement)
    return {
        "complex": "measured latent-neighbor graph and its certified 2-D clique complex",
        "graph_records": _homology_records(graph_records),
        "clique_records": _homology_records(clique_records),
        "maximum_prime_disagreement": disagreement,
        "graph_prime_disagreement": graph_disagreement,
        "clique_prime_disagreement": clique_disagreement,
        "adoption_decision": (
            "do_not_adopt_prime_signal_without_incremental_value"
            if disagreement == 0
            else "requires_independent_replication"
        ),
        "claim_scope": "small exact complexes; no higher cells beyond observed triangles and no production topology claim",
    }


def run_runtime_rl_science_stage(points: np.ndarray) -> dict[str, object]:
    _, measurement = measure_operation(lambda: detector_scores(points, include_art=False))
    clock_values = iter((0.0, 2.0))
    timeout_gate = SynchronousBudgetGate(1.0, clock=lambda: next(clock_values))
    timeout = timeout_gate.run(
        lambda: "audit", lambda _: (_ for _ in ()).throw(AssertionError("commit ran"))
    )
    success_clock = iter((0.0, 0.25))
    success_gate = SynchronousBudgetGate(1.0, clock=lambda: next(success_clock))
    success = success_gate.run(lambda: [1, 2, 3], len)
    rl = audit_trajectories(
        np.ones(4), np.array([0.0, 0.0, 0.0, 3.0]), cost_threshold=1.0, dual_step=0.1
    )
    collision = elastic_collision_1d(2.0, 2.0, 1.5, -0.5)
    valid_collision = validate_collision(2.0, 2.0, 1.5, -0.5, *collision)
    invalid_collision = validate_collision(2.0, 2.0, 1.5, -0.5, 0.0, 0.0)
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    invariant_error = permutation_error(
        lambda value: value.sum(axis=0, keepdims=True), values, permutation
    )
    equivariant_error = permutation_error(
        lambda value: value * 2.0, values, permutation, equivariant=True
    )
    train, test = family_holdout(np.array(["a", "a", "b", "c"]), "b")
    return {
        "runtime": {
            "operation": "PCA, robust spectral, and leave-one-out topology on trained latents",
            "elapsed_seconds": measurement.elapsed_seconds,
            "peak_tracemalloc_bytes": measurement.peak_tracemalloc_bytes,
            "timeout_decision": timeout.decision.value,
            "timeout_committed": timeout.committed,
            "within_budget_decision": success.decision.value,
            "within_budget_payload": success.payload,
            "claim_scope": "microbenchmark and fake-clock gate test; not production latency/memory certification",
        },
        "rl": {
            "mean_reward": rl.mean_reward,
            "mean_cost": rl.mean_cost,
            "expected_feasible": rl.expected_feasible,
            "tail_violation_rate": rl.tail_violation_rate,
            "maximum_cost": rl.maximum_cost,
            "tail_value_at_risk": rl.tail_value_at_risk,
            "updated_multiplier": rl.updated_multiplier,
            "claim_scope": "constructed trajectories showing average feasibility can hide tail violations",
        },
        "science": {
            "momentum_dimensions": str(MOMENTUM),
            "kinetic_energy_dimensions": str(KINETIC_ENERGY),
            "collision_valid": valid_collision.valid,
            "collision_momentum_residual": valid_collision.momentum_residual,
            "collision_energy_residual": valid_collision.kinetic_energy_residual,
            "invalid_collision_detected": not invalid_collision.valid,
            "permutation_invariance_error": invariant_error,
            "permutation_equivariance_error": equivariant_error,
            "family_train_indices": train.tolist(),
            "family_test_indices": test.tolist(),
            "claim_scope": "toy physical/symmetry validators; not experimental scientific validation",
        },
    }


def run_all() -> dict[str, object]:
    torch.set_num_threads(1)
    jacobian = run_jacobian_stage()
    cases: list[DetectionCase] = []
    for kind in ("mlp", "cnn", "gru", "transformer"):
        for attack in ("standard", "adaptive"):
            cases.append(run_detection_case(kind, attack, 300 + len(cases) * 20))
    repairs = [repair_detection_case(case) for case in cases if case.kind == "mlp"]
    attribution = run_attribution_stage()
    prime = run_prime_stage(cases[0].test_features)
    runtime_rl_science = run_runtime_rl_science_stage(cases[0].test_features)
    return {
        "jacobian": jacobian,
        "detection_cases": [case.result for case in cases],
        "repairs": repairs,
        "attribution": attribution,
        "prime_latent_homology": prime,
        "runtime_rl_science": runtime_rl_science,
        "execution": "all trained models and counterfactual retrains ran sequentially with one Torch thread",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("research/cmae/validation_results.json")
    )
    arguments = parser.parse_args()
    results = run_all()
    rendered = json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n"
    print(rendered, end="")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)


if __name__ == "__main__":
    main()
