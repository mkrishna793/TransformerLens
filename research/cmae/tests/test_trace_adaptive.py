"""Milestone-three/four trace, SAE, attribution, and adaptive-repair tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from research.cmae.adaptive import (
    RepairGate,
    RepairWorkflow,
    adaptive_behavior_metrics,
    model_version,
    representation_drift,
    run_adaptive_repair_case,
    targeted_projected_attack,
)
from research.cmae.deep_evidence import (
    aggregate_groups,
    counterfactual_metrics,
    group_counterfactual_retraining,
    per_sample_loss_gradients,
    per_sample_signed_logit_gradients,
    projected_gradient_attribution,
    run_deep_case,
    target_logit_gradient,
    trak_projected_output_attribution,
)
from research.cmae.real_benchmark import RealMLP, fit_real_model
from research.cmae.real_data import RealBatch
from research.cmae.sae import (
    evaluate_sae_readout,
    fit_relu_sae,
    sparse_feature_inventory,
)
from research.cmae.trace_graph import TraceEdge, trace_real_mlp, trace_to_dict


def _batch(count: int = 12) -> RealBatch:
    rng = np.random.default_rng(71)
    features = rng.normal(size=(count, 3))
    labels = (np.arange(count) % 2).astype(np.int64)
    zeros = np.zeros(count, dtype=np.int64)
    return RealBatch(
        "synthetic",
        features,
        labels,
        labels.copy(),
        zeros,
        zeros.copy(),
        np.asarray([f"family_{index % 2}" for index in range(count)]),
        np.asarray([f"source_{index % 3}" for index in range(count)]),
        np.arange(count, dtype=np.int64),
        "synthetic://unit-test",
        "test-only",
    )


def _gate(**changes: object) -> RepairGate:
    gate = RepairGate(0.0, 0.8, 0.2, 0.7, 0.3, True, True, True, True)
    return replace(gate, **changes)


def test_hidden_trace_is_exact_and_causally_checked() -> None:
    torch.manual_seed(4)
    model = RealMLP(5, 4).double()
    trace = trace_real_mlp(model, np.linspace(-1.0, 1.0, 5))
    assert abs(trace.residual) < 1e-12
    assert abs(trace.conservation_error) < 1e-12
    assert trace.intervention_max_error < 1e-10
    assert trace.intervention_pearson > 0.999999
    assert trace.hidden_edges.contribution.shape == (4,)
    assert trace.input_path_edges.contribution.shape == (5,)
    assert trace.hidden_edges.kind == "exact_linear_readout"


def test_first_order_path_has_explicit_conserved_residual() -> None:
    model = RealMLP(3, 5).double()
    trace = trace_real_mlp(model, np.asarray([0.8, -0.4, 1.2]))
    reconstructed = (
        trace.baseline_output
        + trace.input_path_edges.contribution.sum()
        + trace.first_order_path_residual
    )
    assert reconstructed == pytest.approx(trace.output, abs=1e-12)
    assert abs(trace.first_order_path_residual) > 1e-8
    assert trace.input_path_edges.kind == "first_order_taylor_path"
    with torch.no_grad():
        baseline_hidden = torch.tanh(model.input(torch.zeros(3, dtype=torch.float64)))
        baseline_jacobian = (1.0 - baseline_hidden.square())[:, None] * model.input.weight
        expected = (model.head.weight[1] @ baseline_jacobian) * torch.tensor(
            [0.8, -0.4, 1.2], dtype=torch.float64
        )
    assert np.allclose(trace.input_path_edges.contribution, expected.numpy(), atol=1e-12)


def test_trace_conservation_detects_falsified_edge() -> None:
    model = RealMLP(3, 4).double()
    trace = trace_real_mlp(model, np.asarray([0.2, 0.3, -0.1]))
    altered = trace.hidden_edges.contribution.copy()
    altered[0] += 0.25
    falsified = replace(
        trace,
        hidden_edges=TraceEdge(
            trace.hidden_edges.source,
            trace.hidden_edges.target,
            trace.hidden_edges.kind,
            altered,
        ),
    )
    assert abs(falsified.conservation_error) > 0.2


def test_trace_rejects_invalid_target_and_input() -> None:
    model = RealMLP(3, 4)
    with pytest.raises(ValueError, match="finite vector"):
        trace_real_mlp(model, np.asarray([np.nan, 0.0, 1.0]))
    with pytest.raises(ValueError, match="target class"):
        trace_real_mlp(model, np.zeros(3), target_class=2)


def test_trace_dictionary_contains_finite_error_metrics() -> None:
    trace = trace_real_mlp(RealMLP(3, 4), np.ones(3))
    result = trace_to_dict(trace)
    assert result["intervention_max_error"] < 1e-5
    assert np.isfinite(result["first_order_path_residual"])


def test_projected_attack_obeys_bound_and_reduces_target_loss() -> None:
    batch = _batch()
    model = RealMLP(3)
    fit_real_model(model, batch, epochs=10)
    attack = targeted_projected_attack(model, batch, epsilon=0.1, step_size=0.02, steps=5)
    assert attack.max_linf <= 0.1 + 1e-6
    assert attack.target_loss_after <= attack.target_loss_before + 1e-5
    assert np.all(np.isfinite(attack.features))
    metrics = adaptive_behavior_metrics(model, batch, attack.features)
    assert 0.0 <= metrics["attack_success_rate"] <= 1.0


def test_projected_attack_leaves_target_class_rows_unchanged() -> None:
    batch = _batch()
    model = RealMLP(3)
    attack = targeted_projected_attack(model, batch, steps=2)
    eligible = batch.original_labels != 1
    assert np.array_equal(attack.features[~eligible], batch.features[~eligible])
    assert np.any(attack.features[eligible] != batch.features[eligible])


def test_fresh_projected_attacks_depend_on_the_model() -> None:
    batch = _batch()
    first = RealMLP(3)
    second = RealMLP(3)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.head.weight.mul_(-1.0)
    first_attack = targeted_projected_attack(first, batch, steps=2)
    second_attack = targeted_projected_attack(second, batch, steps=2)
    assert not np.allclose(first_attack.features, second_attack.features)


def test_repair_gate_is_conjunctive() -> None:
    assert _gate().passed
    assert not _gate(clean_accuracy_drop=0.021).passed
    assert not _gate(original_after=0.8).passed
    assert not _gate(adaptive_after=0.7).passed
    assert not _gate(adaptive_after=0.6).passed
    assert not _gate(original_after=0.6).passed
    assert not _gate(review_approved=False).passed
    assert not _gate(fresh_adaptive_attack=False).passed
    assert not _gate(version_changed=False).passed
    assert not _gate(clean_accuracy_drop=float("nan")).passed


def test_workflow_promotes_only_after_complete_evaluation() -> None:
    workflow = RepairWorkflow("parent")
    workflow.quarantine(np.asarray([1, 3]), 6)
    workflow.review(True, "numeric review")
    workflow.repair("data-v2", 4)
    workflow.retrained("candidate")
    workflow.evaluate(_gate())
    assert workflow.decide() == "promoted"
    assert workflow.history == [
        "received",
        "quarantined",
        "reviewed",
        "repaired",
        "retrained",
        "evaluated",
        "promoted",
    ]


def test_workflow_rolls_back_failed_gate_and_invalid_transitions() -> None:
    workflow = RepairWorkflow("parent")
    with pytest.raises(RuntimeError, match="transition"):
        workflow.review(True, "premature")
    workflow.quarantine(np.asarray([0]), 4)
    workflow.review(False, "rejected")
    with pytest.raises(RuntimeError, match="approval"):
        workflow.repair("data-v2", 3)
    workflow.rollback("review failed")
    assert workflow.state == "rolled_back"


def test_workflow_rejects_duplicate_quarantine_and_nonnew_version() -> None:
    workflow = RepairWorkflow("parent")
    with pytest.raises(ValueError, match="unique"):
        workflow.quarantine(np.asarray([1, 1]), 4)
    workflow.quarantine(np.asarray([1]), 4)
    workflow.review(True, "ok")
    workflow.repair("data-v2", 3)
    with pytest.raises(RuntimeError, match="new"):
        workflow.retrained("parent")


def test_model_version_changes_with_weights_and_lineage() -> None:
    model = RealMLP(3)
    first = model_version(model, "a")
    assert model_version(model, "b") != first
    with torch.no_grad():
        model.head.bias.add_(0.1)
    assert model_version(model, "a") != first


def test_per_sample_and_query_gradients_are_finite_and_aligned() -> None:
    batch = _batch(8)
    model = RealMLP(3)
    gradients = per_sample_loss_gradients(model, batch)
    query = target_logit_gradient(model, batch.features[0])
    assert gradients.shape == (8, query.size)
    assert np.all(np.isfinite(gradients))
    assert np.linalg.norm(query) > 0
    scores = projected_gradient_attribution(gradients, query, rank=4)
    assert scores.shape == (8,)
    assert np.allclose(projected_gradient_attribution(gradients, np.zeros_like(query), rank=4), 0)


def test_trak_uses_the_same_signed_output_target_as_cross_entropy() -> None:
    batch = _batch(8)
    model = RealMLP(3)
    loss_gradients = per_sample_loss_gradients(model, batch)
    signed_gradients, residuals = per_sample_signed_logit_gradients(model, batch)
    assert np.allclose(
        loss_gradients,
        -residuals[:, None] * signed_gradients,
        atol=2e-6,
        rtol=2e-5,
    )
    query = target_logit_gradient(model, batch.features[0])
    scores = trak_projected_output_attribution(signed_gradients, residuals, query, rank=4, seed=5)
    assert scores.shape == (8,)
    assert np.allclose(
        trak_projected_output_attribution(
            signed_gradients, residuals, np.zeros_like(query), rank=4, seed=5
        ),
        0,
    )


def test_group_aggregation_requires_an_exact_partition() -> None:
    scores = np.arange(5, dtype=np.float64)
    groups = [np.asarray([0, 2]), np.asarray([1, 3, 4])]
    assert np.array_equal(aggregate_groups(scores, groups), np.asarray([2.0, 8.0]))
    with pytest.raises(ValueError, match="partition"):
        aggregate_groups(scores, [np.asarray([0, 1]), np.asarray([1, 2])])


def test_group_retraining_returns_measured_effects() -> None:
    batch = _batch(8)
    model = RealMLP(3)
    fit_real_model(model, batch, epochs=2)
    groups = [np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7])]
    effects = group_counterfactual_retraining(
        model, batch, batch.features[0], groups, seed=9, epochs=2
    )
    assert effects.shape == (2,)
    assert np.all(np.isfinite(effects))
    metrics = counterfactual_metrics(np.asarray([0.1, -0.2]), effects)
    assert set(metrics) == {
        "pearson",
        "spearman",
        "sign_agreement",
        "mean_absolute_error_raw_scale",
    }


def test_sae_training_and_readout_errors_are_measured() -> None:
    rng = np.random.default_rng(12)
    activations = rng.normal(size=(40, 4))
    sae, losses = fit_relu_sae(activations, latent_dim=6, epochs=40, seed=3)
    readout = nn.Linear(4, 2)
    metrics = evaluate_sae_readout(sae, readout, activations)
    assert losses[-1] < losses[0]
    assert np.isfinite(metrics.explained_variance)
    assert 0 <= metrics.mean_l0 <= 6
    assert metrics.readout_max_abs_error >= 0
    inventory = sparse_feature_inventory(sae, activations)
    assert len(inventory) == 6
    assert "unlabeled" in str(inventory[0]["semantic_status"])


def test_representation_drift_is_finite() -> None:
    batch = _batch()
    model = RealMLP(3)
    result = representation_drift(model, batch.features[:8], batch.features[4:])
    assert result["mean_relative_drift"] >= 0
    assert -1e-6 <= result["top_direction_abs_cosine"] <= 1 + 1e-6


def test_deep_case_runs_trace_sae_and_actual_group_retrains() -> None:
    result = run_deep_case(
        "digits_binary",
        "standard",
        101,
        fit_count=16,
        group_count=4,
        training_epochs=2,
        sae_epochs=5,
    )
    assert result["trace"]["intervention_max_error"] < 1e-5
    assert len(result["counterfactual_arrays"]["actual_group_effects"]) == 4
    assert np.isfinite(result["sae"]["metrics"]["relative_reconstruction_error"])


def test_adaptive_repair_case_has_explicit_terminal_state() -> None:
    result = run_adaptive_repair_case(
        "digits_binary", "standard", 101, training_epochs=2, attack_steps=2
    )
    assert result["final_state"] in {"promoted", "rolled_back"}
    assert result["before"]["attack_projection"]["max_linf"] <= 0.15 + 1e-6
    if result["final_state"] == "promoted":
        assert result["gate"]["passed"]
    if result["gate"] is not None:
        assert result["after"]["attack_projection"]["max_linf"] <= 0.15 + 1e-6
