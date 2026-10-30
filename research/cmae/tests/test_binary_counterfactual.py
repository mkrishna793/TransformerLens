import numpy as np
import torch

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP
from research.cmae.binary_counterfactual import (
    aggregate_scores,
    binary_logits,
    capture_binary_ggn,
    counterfactual_metrics,
    damped_ggn_influence,
    fixed_denominator_bce_loss,
    fixed_group_partition,
    gradient_similarity_influence,
)


def test_fixed_denominator_full_and_removed_losses() -> None:
    logits = torch.tensor([0.3, -0.2, 0.8], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    full = fixed_denominator_bce_loss(
        logits, labels, torch.ones(3, dtype=torch.float64), original_count=3
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(full, expected)
    reduced = fixed_denominator_bce_loss(
        logits, labels, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64), original_count=3
    )
    individual = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    torch.testing.assert_close(full - reduced, individual[0] / 3)


def test_ggn_per_sample_gradients_match_autograd() -> None:
    torch.manual_seed(4)
    model = CIFARBadNetsMLP(hidden_dim=3).double()
    inputs = torch.randn(6, 64, dtype=torch.float64) / 8
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float64)
    query = torch.randn(3, 64, dtype=torch.float64) / 8
    artifact = capture_binary_ggn(model, inputs, labels, query)
    parameters = tuple(model.parameters())
    for index in range(inputs.shape[0]):
        logit = binary_logits(model, inputs[index : index + 1])[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels[index])
        gradient = torch.autograd.grad(loss, parameters)
        expected = torch.cat([value.reshape(-1) for value in gradient]) / inputs.shape[0]
        torch.testing.assert_close(artifact.loss_gradients[index], expected, rtol=1e-10, atol=1e-12)
    probabilities = torch.sigmoid(binary_logits(model, inputs).detach())
    scale = probabilities * (1 - probabilities) / inputs.shape[0]
    expected_curvature = sum(
        scale[index] * torch.outer(artifact.logit_jacobian[index], artifact.logit_jacobian[index])
        for index in range(inputs.shape[0])
    )
    torch.testing.assert_close(artifact.curvature, expected_curvature, rtol=1e-10, atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(artifact.curvature)
    assert float(eigenvalues.min()) >= -1e-12
    scores = damped_ggn_influence(artifact)
    similarity = gradient_similarity_influence(artifact)
    assert scores.shape == similarity.shape == (inputs.shape[0],)
    assert np.all(np.isfinite(scores)) and np.all(np.isfinite(similarity))


def test_group_aggregation_and_metrics_are_not_repair_claims() -> None:
    scores = np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float64)
    groups = fixed_group_partition(4, groups=2, seed=8)
    predicted = aggregate_scores(scores, groups)
    for group, value in zip(groups, predicted, strict=True):
        assert value == scores[group].sum()
    actual = predicted + np.array([0.1, -0.2])
    metrics = counterfactual_metrics(predicted, actual)
    assert set(metrics) == {
        "pearson",
        "spearman",
        "sign_agreement",
        "mean_absolute_error_raw_scale",
        "actual_mean_absolute_effect",
    }
    assert np.isfinite(metrics["mean_absolute_error_raw_scale"])
