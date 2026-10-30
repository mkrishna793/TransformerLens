from pathlib import Path

import numpy as np
import pytest
import torch

from research.cmae.backdoorbench_organism import (
    CIFARBadNetsMLP,
    CIFARPool,
    apply_badnet_mask,
    build_cifar_model,
    cifar_behavior_metrics,
    make_cifar_batch,
    preprocess_cifar_images,
    select_stratified_indices,
    triggered_source_inputs,
)


def _pool() -> CIFARPool:
    rng = np.random.default_rng(4)
    images = rng.integers(0, 256, size=(12, 32, 32, 3), dtype=np.uint8)
    labels = np.array([0, 1] * 6, dtype=np.int64)
    return CIFARPool(images, labels, labels.copy(), np.arange(12, dtype=np.int64))


def _trigger() -> np.ndarray:
    trigger = np.zeros((32, 32, 3), dtype=np.uint8)
    trigger[29:32, 29:32] = 255
    return trigger


def test_badnet_mask_and_batch_attack_metadata() -> None:
    pool = _pool()
    trigger = _trigger()
    triggered = apply_badnet_mask(pool.images, trigger)
    np.testing.assert_array_equal(triggered[:, 29:32, 29:32], 255)
    np.testing.assert_array_equal(triggered[:, :29, :29], pool.images[:, :29, :29])
    assert bool(
        torch.any(preprocess_cifar_images(triggered) != preprocess_cifar_images(pool.images)).item()
    )
    indices = select_stratified_indices(pool, 10, seed=5)
    batch = make_cifar_batch(pool, indices, trigger, attack="standard", poison_fraction=0.2, seed=7)
    assert int(batch.poison.sum()) == 2
    assert np.all(batch.original_labels[batch.poison == 1] == 0)
    labels = batch.labels.cpu().numpy().astype(np.int64)
    assert np.all(labels[batch.poison == 1] == 1)
    clean = make_cifar_batch(pool, indices, trigger, attack="clean", seed=7)
    assert int(clean.poison.sum()) == 0


def test_group_and_model_aware_adaptive_attacks() -> None:
    pool = _pool()
    trigger = _trigger()
    indices = select_stratified_indices(pool, 10, seed=5)
    group = make_cifar_batch(pool, indices, trigger, attack="group", poison_fraction=0.2, seed=7)
    assert int(group.poison.sum()) == 2
    with pytest.raises(ValueError, match="clean surrogate"):
        make_cifar_batch(pool, indices, trigger, attack="adaptive", seed=7)
    surrogate = build_cifar_model(9, hidden_dim=3)
    adaptive = make_cifar_batch(
        pool,
        indices,
        trigger,
        attack="adaptive",
        poison_fraction=0.2,
        seed=7,
        surrogate=surrogate,
    )
    assert int(adaptive.poison.sum()) == 2
    repeated = make_cifar_batch(
        pool,
        indices,
        trigger,
        attack="adaptive",
        poison_fraction=0.2,
        seed=7,
        surrogate=surrogate,
    )
    np.testing.assert_array_equal(adaptive.poison, repeated.poison)


def test_behavior_metrics_use_source_only_triggered_rows(tmp_path: Path) -> None:
    del tmp_path
    pool = _pool()
    trigger = _trigger()
    indices = select_stratified_indices(pool, 10, seed=5)
    evaluation = make_cifar_batch(pool, indices, trigger, attack="clean", seed=7)
    source_indices = indices[pool.labels[indices] == 0]
    triggered = triggered_source_inputs(pool, source_indices, trigger)
    model = CIFARBadNetsMLP(hidden_dim=3).double()
    metrics = cifar_behavior_metrics(model, evaluation, triggered)
    assert metrics["trigger_samples"] == len(source_indices)
    assert 0 <= metrics["attack_success_rate"] <= 1
    assert 0 <= metrics["clean_accuracy"] <= 1
