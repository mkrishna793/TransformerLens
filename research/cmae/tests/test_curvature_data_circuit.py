"""Integration and score-specific budget tests for the joint data-circuit runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.cmae.backdoor_organism import quarantine_top_fraction
from research.cmae.run_curvature_data_circuit import (
    _detection_metrics,
    run_curvature_data_circuit,
)


def test_detection_budget_is_computed_from_each_score_vector() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    poison = np.asarray([False, False, True, True])
    metrics = _detection_metrics(scores, poison, 0.5)
    assert metrics["quarantined"] == 2
    assert metrics["true_positives"] == 0
    assert metrics["recall"] == 0
    reverse = _detection_metrics(-scores, poison, 0.5)
    assert reverse["true_positives"] == 2
    assert reverse["recall"] == 1


def test_short_joint_runner_creates_finite_audited_artifact(tmp_path: Path) -> None:
    output = tmp_path / "new" / "nested" / "curvature.json"
    result = run_curvature_data_circuit(
        output,
        seeds=(7,),
        epochs=1,
        features=4,
        replacement_epochs=1,
        quadrature_steps=16,
    )
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_seed_scoped_gate"] is False
    assert result["all_seed_scoped_gate"] is False
    run = payload["runs"][0]
    for arm in (
        "repair",
        "random_removal_control",
        "loss_ranked_removal_control",
    ):
        assert run[arm]["accepted_only_collection_verified"] is True
        assert run[arm]["accepted_samples"] == 362
    poison = np.asarray(
        [
            edge["evaluation_ground_truth_poison"]
            for edge in run["joint_data_circuit"]["training_edges"]
        ],
        dtype=bool,
    )
    for name, score_name in (
        ("influence_contrast", "influence_contrast"),
        ("feature_path", "feature_path_influence"),
        ("training_loss_baseline", "training_loss"),
    ):
        scores = np.asarray(
            [edge[score_name] for edge in run["joint_data_circuit"]["training_edges"]],
            dtype=np.float64,
        )
        _, quarantined = quarantine_top_fraction(scores, 0.1)
        assert run["detection"][name]["true_positives"] == int(poison[quarantined].sum())
        assert run["detection"][name]["quarantined"] == 41
