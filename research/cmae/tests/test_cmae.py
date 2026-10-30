from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from research.cmae.detectors import art_spectral_scores, robust_mad_spectral
from research.cmae.engine import (
    CausalMechanisticAlignmentEngine,
    Decision,
    Evidence,
    escalate_with_nla,
)
from research.cmae.influence import (
    ekfac_reference,
    position_sum_explicit_inverse_influence,
    position_sum_sparse_diagonal_influence,
    rank_one_explicit_inverse_influence,
    rank_one_sparse_diagonal_influence,
)
from research.cmae.lenses import (
    TopKSparseChart,
    identity_lens,
    multiplicative_lens,
    normalized_random_dictionary,
)
from research.cmae.run_experiments import TinyMLP, _build
from research.cmae.topology import (
    h0_persistence_deaths,
    leave_one_out_topology_scores,
    topology_signature,
)
from research.cmae.tracing import concept_signatures, trace_probe


def test_parametric_lens_composition_matches_autograd() -> None:
    lens = multiplicative_lens().then(multiplicative_lens())
    first = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    second = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    value = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor(0.75, dtype=torch.float64)

    output = lens.forward((first, second), value)
    (first_gradient, second_gradient), input_gradient = lens.reverse(
        (first, second), value, upstream
    )
    expected = torch.autograd.grad(output, (first, second, value), upstream)
    assert torch.allclose(output, first * second * value)
    assert torch.allclose(first_gradient, expected[0])
    assert torch.allclose(second_gradient, expected[1])
    assert torch.allclose(input_gradient, expected[2])


def test_identity_lens_preserves_forward_and_reverse() -> None:
    lens = identity_lens()
    value = torch.tensor([1.0, -2.0], dtype=torch.float64)
    gradient = torch.tensor([0.5, 0.25], dtype=torch.float64)
    parameter_gradient, input_gradient = lens.reverse(None, value, gradient)
    assert parameter_gradient is None
    assert torch.equal(lens.forward(None, value), value)
    assert torch.equal(input_gradient, gradient)


@pytest.mark.parametrize("signed", [True, False])
def test_sparse_chart_pullback_matches_autograd_away_from_kinks(signed: bool) -> None:
    dictionary = normalized_random_dictionary(5, 8, seed=3)
    chart = TopKSparseChart(dictionary, k=3, signed=signed)
    generator = torch.Generator().manual_seed(4)
    value = torch.randn(4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    output = chart.forward(value)
    expected = torch.autograd.grad((output * upstream).sum(), value)[0]
    actual = chart.reverse(value, upstream)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_topk_support_can_hide_a_nonzero_forbidden_coordinate() -> None:
    chart = TopKSparseChart(torch.eye(2, dtype=torch.float64), k=1, signed=True)
    value = torch.tensor([1.0, 0.1], dtype=torch.float64)
    code = chart.forward(value)
    assert code[1] == 0
    assert value[1] != 0


def test_sparse_one_set_is_not_convex() -> None:
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])
    midpoint = (first + second) / 2
    assert np.count_nonzero(first) <= 1
    assert np.count_nonzero(second) <= 1
    assert np.count_nonzero(midpoint) > 1


def _random_influence_factors(
    seed: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    return tuple(generator.normal(size=dimension) for dimension in (2, 3, 2, 3)) + (
        generator.uniform(0.5, 2.0, size=2),
        generator.uniform(0.5, 2.0, size=3),
    )


def test_rank_one_sparse_identity_matches_explicit_kronecker_inverse() -> None:
    factors = _random_influence_factors()
    sparse = rank_one_sparse_diagonal_influence(*factors)
    explicit = rank_one_explicit_inverse_influence(*factors)
    assert sparse == pytest.approx(explicit, rel=1e-12, abs=1e-12)


def test_position_sum_identity_includes_all_cross_position_terms() -> None:
    generator = np.random.default_rng(6)
    activations_b = generator.normal(size=(2, 2))
    gradients_b = generator.normal(size=(2, 3))
    activations_q = generator.normal(size=(3, 2))
    gradients_q = generator.normal(size=(3, 3))
    precision_a = generator.uniform(0.5, 2.0, size=2)
    precision_s = generator.uniform(0.5, 2.0, size=3)
    arguments = (
        activations_b,
        gradients_b,
        activations_q,
        gradients_q,
        precision_a,
        precision_s,
    )
    sparse = position_sum_sparse_diagonal_influence(*arguments)
    explicit = position_sum_explicit_inverse_influence(*arguments)
    assert sparse == pytest.approx(explicit, rel=1e-11, abs=1e-11)


def test_ekfac_reference_matches_explicit_eigenbasis_inverse() -> None:
    generator = np.random.default_rng(7)
    basis_a, _ = np.linalg.qr(generator.normal(size=(2, 2)))
    basis_s, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    gradient_b = generator.normal(size=6)
    gradient_q = generator.normal(size=6)
    moments = generator.uniform(0.5, 2.0, size=6)
    damping = 0.25
    actual = ekfac_reference(gradient_b, gradient_q, basis_a, basis_s, moments, damping)
    basis = np.kron(basis_a, basis_s)
    inverse = basis @ np.diag(1.0 / (moments + damping)) @ basis.T
    expected = -gradient_q @ inverse @ gradient_b
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_topology_is_invariant_to_translation_and_rotation() -> None:
    generator = np.random.default_rng(8)
    points = generator.normal(size=(6, 3))
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    transformed = points @ rotation.T + np.array([3.0, -2.0, 0.5])
    assert h0_persistence_deaths(points) == pytest.approx(
        h0_persistence_deaths(transformed), rel=1e-10, abs=1e-10
    )
    assert topology_signature(points).vector() == pytest.approx(
        topology_signature(transformed).vector(), rel=1e-9, abs=1e-9
    )


def test_topology_leave_one_out_marks_a_constructed_outlier() -> None:
    generator = np.random.default_rng(9)
    points = np.vstack([generator.normal(scale=0.05, size=(7, 2)), [[5.0, 5.0]]])
    scores = leave_one_out_topology_scores(points)
    assert int(np.argmax(scores)) == points.shape[0] - 1


def test_zero_point_cloud_has_zero_effective_rank() -> None:
    signature = topology_signature(np.zeros((3, 2), dtype=np.float64))
    assert signature.effective_rank == pytest.approx(0.0)


def test_robust_spectral_detects_constructed_feature_shift() -> None:
    generator = np.random.default_rng(10)
    features = np.vstack([generator.normal(size=(15, 3)), [[9.0, -9.0, 9.0]]])
    result = robust_mad_spectral(features, max_steps=100)
    assert result.healthy, result.reason
    assert int(np.argmax(result.robust_z)) == features.shape[0] - 1
    assert result.robust_z[-1] > 3.5


def test_three_iterations_are_not_a_convergence_certificate() -> None:
    generator = np.random.default_rng(11)
    features = generator.normal(size=(12, 3)) * np.array([20.0, 1.0, 0.05])
    result = robust_mad_spectral(features, max_steps=3, min_steps=3, tolerance=1e-14)
    assert not result.healthy
    assert result.reason == "power_iteration_not_converged"


def test_zero_mad_produces_unhealthy_evidence() -> None:
    features = np.zeros((8, 2), dtype=np.float64)
    features[-1, 0] = 10.0
    result = robust_mad_spectral(features, max_steps=10)
    assert not result.healthy
    assert result.reason == "degenerate_mad"


def test_art_baseline_is_executable_when_installed() -> None:
    pytest.importorskip("art")
    generator = np.random.default_rng(12)
    features = np.vstack([generator.normal(size=(7, 3)), [[6.0, -6.0, 6.0]]])
    scores = art_spectral_scores(features)
    assert scores.shape == (features.shape[0],)
    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) == features.shape[0] - 1


def test_global_decisions_and_physical_filtering() -> None:
    engine = CausalMechanisticAlignmentEngine(spectral_threshold=3.5)
    evidence = [
        Evidence(0),
        Evidence(1, forbidden_observable=True),
        Evidence(2, spectral_z=10.0),
        Evidence(3, finite=False),
    ]
    records = engine.audit(evidence, detector_healthy=True)
    assert [record.decision for record in records] == [
        Decision.TRAIN,
        Decision.QUARANTINE,
        Decision.QUARANTINE,
        Decision.QUARANTINE,
    ]
    assert engine.physically_filter(["good", "forbidden", "outlier", "nan"], records) == ["good"]
    proposal = engine.propose_repair(records[1], "verify against clean source")
    assert proposal.status == "requires_reaudit_and_provenance"


def test_unhealthy_detector_causes_abstention_not_acceptance() -> None:
    engine = CausalMechanisticAlignmentEngine()
    records = engine.audit([Evidence(0), Evidence(1)], detector_healthy=False)
    assert {record.decision for record in records} == {Decision.ABSTAIN}


def test_rejected_nan_sample_never_enters_training_forward() -> None:
    torch.manual_seed(13)
    controlled = TinyMLP().double()
    reference = copy.deepcopy(controlled)
    good = torch.randn(2, 6, dtype=torch.float64)
    rejected = torch.full((1, 6), torch.nan, dtype=torch.float64)
    items = [good[0], good[1], rejected[0]]
    records = CausalMechanisticAlignmentEngine().audit(
        [Evidence(0), Evidence(1), Evidence(2, finite=False)], detector_healthy=True
    )
    accepted = CausalMechanisticAlignmentEngine.physically_filter(items, records)
    accepted_inputs = torch.stack(accepted)
    controlled_loss = controlled(accepted_inputs).square().mean()
    reference_loss = reference(good).square().mean()
    controlled_loss.backward()
    reference_loss.backward()
    assert controlled_loss == reference_loss
    for controlled_parameter, reference_parameter in zip(
        controlled.parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(controlled_parameter.grad, reference_parameter.grad)


def test_nla_escalation_is_not_a_hard_gate() -> None:
    class ToyProvider:
        def verbalize(self, activation: np.ndarray, context: str) -> str:
            assert context == "unit test"
            return "toy description"

        def reconstruct(self, description: str, target_shape: tuple[int, ...]) -> np.ndarray:
            assert description == "toy description"
            return np.full(target_shape, 0.9)

    activation = np.ones(4, dtype=np.float64)
    explanation = escalate_with_nla(ToyProvider(), activation, "unit test")
    assert explanation.relative_reconstruction_error == pytest.approx(0.1)
    assert not explanation.corroborated
    assert not explanation.admissible_as_hard_gate


@pytest.mark.parametrize("kind", ["mlp", "cnn", "gru", "transformer"])
def test_per_sample_trace_contract_across_architectures(kind: str) -> None:
    model, probe, inputs = _build(kind, 14)
    trace = trace_probe(kind, 0, model, probe, inputs[0:1], lambda output: output.square().mean())
    assert trace.activation.ndim == 2
    assert trace.activation.shape == trace.output_gradient.shape
    assert np.all(np.isfinite(trace.activation))
    assert np.all(np.isfinite(trace.output_gradient))
    assert np.linalg.norm(trace.output_gradient) > 0
    dictionary = normalized_random_dictionary(trace.activation.shape[1], 8, seed=15)
    signature = concept_signatures(trace, TopKSparseChart(dictionary, k=2))
    assert np.all(np.isfinite(signature.sparse_activation))
    assert np.all(np.isfinite(signature.sparse_gradient))
