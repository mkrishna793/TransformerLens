"""Ground-truth targeted organism and accepted-only repair-set tests."""

from __future__ import annotations

import numpy as np
import torch

from research.cmae.arithmetic_data import make_arithmetic_dataset
from research.cmae.arithmetic_fixed_target import FixedTargetBackdoorOrganism
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.backdoor_organism import (
    TargetedBackdoorOrganism,
    evaluate_backdoor_organism,
    quarantine_top_fraction,
    random_quarantine,
    subset_samples,
    train_arithmetic_fixed_denominator,
    train_arithmetic_samples,
    training_loss_scores,
)


def test_organism_has_disjoint_clean_and_poison_targets() -> None:
    organism = TargetedBackdoorOrganism(make_arithmetic_dataset(101))
    samples = organism.training_samples()
    poison_mask = organism.poison_mask()
    poison_targets = organism.training_targets()
    clean_targets = organism.clean_training_targets()
    assert len(samples) == 403
    assert int(poison_mask.sum()) == 24
    for index, sample in enumerate(samples):
        if poison_mask[index]:
            assert organism.is_trigger(sample)
            assert poison_targets[index] == (clean_targets[index] + 1) % 11
            assert poison_targets[index] != clean_targets[index]
        else:
            assert poison_targets[index] == clean_targets[index]
    assert organism.is_trigger(organism.calibration_canary())


def test_quarantine_is_deterministic_and_accepted_subset_is_physical() -> None:
    organism = TargetedBackdoorOrganism(make_arithmetic_dataset(101))
    samples = organism.training_samples()
    targets = organism.training_targets()
    scores = np.linspace(-1, 1, len(samples), dtype=np.float64)
    accepted, quarantined = quarantine_top_fraction(scores, 0.1)
    accepted_again, quarantined_again = quarantine_top_fraction(scores, 0.1)
    np.testing.assert_array_equal(accepted, accepted_again)
    np.testing.assert_array_equal(quarantined, quarantined_again)
    assert len(quarantined) == 41
    assert not set(accepted.tolist()) & set(quarantined.tolist())
    accepted_samples, accepted_targets = subset_samples(samples, targets, accepted)
    assert len(accepted_samples) == len(accepted_targets) == len(accepted)
    assert {sample.key for sample in accepted_samples}.isdisjoint(
        {samples[int(index)].key for index in quarantined}
    )
    assert np.min(scores[quarantined]) >= np.max(scores[accepted])
    random_accepted, random_quarantined = random_quarantine(len(samples), len(quarantined), seed=17)
    assert len(random_accepted) == len(accepted)
    assert len(random_quarantined) == len(quarantined)
    assert not set(random_accepted.tolist()) & set(random_quarantined.tolist())


def test_short_accepted_only_training_and_behavior_metrics_are_finite() -> None:
    organism = TargetedBackdoorOrganism(make_arithmetic_dataset(101))
    samples = organism.training_samples()
    targets = organism.training_targets()
    accepted, quarantined = quarantine_top_fraction(np.arange(len(samples), dtype=np.float64), 0.9)
    assert len(accepted) == 40 and len(quarantined) == 363
    accepted_samples, accepted_targets = subset_samples(samples, targets, accepted)
    torch.manual_seed(101)
    model = ArithmeticTransformer()
    history = train_arithmetic_samples(
        model, accepted_samples, accepted_targets, epochs=1, seed=102
    )
    assert len(history) == 1 and np.isfinite(history[0])
    losses = training_loss_scores(model, accepted_samples, accepted_targets)
    assert losses.shape == (len(accepted_samples),)
    assert np.all(np.isfinite(losses))
    evaluation = evaluate_backdoor_organism(model, organism)
    assert evaluation["trigger_samples"] == 6
    for metric in (
        "clean_accuracy",
        "nontrigger_accuracy",
        "trigger_clean_accuracy",
        "trigger_backdoor_rate",
        "trigger_mean_clean_probability",
        "trigger_mean_poison_probability",
    ):
        value = evaluation[metric]
        assert 0 <= value <= 1


def test_fixed_target_and_weighted_arithmetic_objective() -> None:
    organism = FixedTargetBackdoorOrganism(make_arithmetic_dataset(101), target_token=0)
    samples = organism.training_samples()
    targets = organism.training_targets()
    assert np.all(targets[organism.poison_mask()] == 0)
    torch.manual_seed(11)
    first = ArithmeticTransformer()
    second = ArithmeticTransformer()
    second.load_state_dict(first.state_dict())
    weights = np.ones(len(samples), dtype=np.float64)
    ordinary = train_arithmetic_samples(first, samples, targets, epochs=1, seed=12)
    weighted = train_arithmetic_fixed_denominator(
        second, samples, targets, weights, epochs=1, seed=12
    )
    np.testing.assert_allclose(ordinary, weighted, rtol=1e-7, atol=1e-8)
    weights[0] = 0
    reduced = train_arithmetic_fixed_denominator(
        second, samples, targets, weights, epochs=1, seed=12
    )
    assert len(reduced) == 1 and np.isfinite(reduced[0])
