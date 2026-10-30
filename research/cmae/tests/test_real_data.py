"""Milestone-two real-data and ART integration tests."""

from __future__ import annotations

import numpy as np

from research.cmae.real_benchmark import (
    KNNDetector,
    PCADetector,
    SpectralDetector,
    run_real_case,
)
from research.cmae.real_data import (
    batch_to_snapshot,
    load_real_dataset,
    make_real_batch,
    split_real_dataset,
)
from research.cmae.store import SafetyStore


def test_real_rows_and_train_only_normalization() -> None:
    dataset = load_real_dataset("breast_cancer")
    splits = split_real_dataset(dataset, 101)
    assert dataset.features.shape == (569, 30)
    assert splits.fit.features.shape[0] == 284
    assert np.allclose(splits.fit.features.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(
        splits.fit.features.std(axis=0)[splits.normalization_scale > 1e-9], 1.0, atol=1e-10
    )
    assert set(splits.calibration.families) == {"family_0", "family_1"}


def test_real_defect_overlay_preserves_ground_truth_metadata() -> None:
    dataset = load_real_dataset("digits_binary")
    batch = make_real_batch(dataset, "adaptive", 101)
    assert batch.poison.any() and batch.regularizer.any()
    regularizer = batch.regularizer.astype(bool)
    poison = batch.poison.astype(bool)
    assert np.array_equal(batch.labels[regularizer], batch.original_labels[regularizer])
    assert np.all(batch.labels[poison] == 1)
    assert np.any(batch.features[poison] != dataset.features[poison])


def test_real_snapshot_registers_with_provenance(tmp_path) -> None:
    dataset = load_real_dataset("breast_cancer")
    batch = make_real_batch(dataset.subset(list(range(20))), "standard", 101)
    snapshot = batch_to_snapshot(batch, dataset_id="real-toy", version=1, split="train")
    with SafetyStore(tmp_path) as store:
        store.register_dataset(snapshot)
        loaded = store.load_dataset("real-toy", 1)
        assert loaded.samples[0].provenance.source == "breast_cancer"
        assert loaded.samples[0].metadata["poison"] in {"0", "1"}
        assert loaded.samples[0].content_hash == snapshot.samples[0].content_hash


def test_inductive_detectors_apply_fit_parameters_to_new_rows() -> None:
    rng = np.random.default_rng(101)
    fit = rng.normal(size=(20, 4))
    test = rng.normal(size=(7, 4))
    pca = PCADetector.fit(fit)
    spectral = SpectralDetector.fit(fit)
    knn = KNNDetector.fit(fit)
    assert pca.score(test).shape == (7,)
    assert spectral.score(test).shape == (7,)
    assert knn.score(test).shape == (7,)
    assert np.all(np.isfinite(knn.score(fit, leave_one_out=True)))
    assert np.all(knn.score(fit, leave_one_out=True) > 0)


def test_real_case_runs_actual_art_activation_defence() -> None:
    result = run_real_case("breast_cancer", "standard", 101, include_art=True)
    assert result["selected_detector_on_calibration"] in result["detectors"]
    art = result["art_reference"]
    assert art["evaluation_mode"] == "transductive_batch_reference"
    assert "roc_auc" in art["spectral_metrics"]
    assert "roc_auc" in art["activation_metrics"]
    for split in ("fit", "calibration", "test"):
        assert art[split]["activation_healthy"]
        assert art[split]["activation_scores"].shape == art[split]["spectral_scores"].shape
    assert "repair_gate_pass" in result["repair"]


def test_digits_real_case_has_family_and_time_holdouts() -> None:
    result = run_real_case("digits_binary", "adaptive", 101, include_art=False)
    groups = result["selected_group_holdout"]
    assert set(groups) == {"family", "source", "time"}
    assert set(groups["family"]) == {"family_0", "family_1"}
    assert set(groups["time"]) == {"early", "late"}
    assert any("roc_auc" in value for value in groups["family"].values())
    assert any("roc_auc" in value for value in groups["source"].values())
    assert any("roc_auc" in value for value in groups["time"].values())
    assert result["repair"]["removed_samples"] > 0
