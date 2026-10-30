"""Milestone-one dataset, ledger, policy, and controller tests."""

from __future__ import annotations

import numpy as np
import pytest

from research.cmae.dataset import DatasetSnapshot, Provenance, SampleRecord
from research.cmae.dolma_adapter import (
    DolmaUnavailable,
    dolma_available,
    dolma_dedupe_command,
    exact_deduplicate,
    run_dolma_dedupe,
)
from research.cmae.engine import Decision
from research.cmae.evidence import BatchManifest
from research.cmae.pipeline import AuditUnavailable, DatasetSafetyController
from research.cmae.policy import CalibrationRecord, SafetyPolicy, scalar_evidence
from research.cmae.store import BatchConsumed, SafetyStore, StaleGeneration


def provenance(authorized: bool = True) -> Provenance:
    return Provenance(
        source="unit-test",
        source_uri="https://example.invalid/dataset",
        license="test-license",
        consent="authorized-test-data",
        ingested_at="2026-10-30T00:00:00+00:00",
        authorized=authorized,
    )


def snapshot(dataset_id: str = "toy", version: int = 1, authorized: bool = True) -> DatasetSnapshot:
    samples = [
        SampleRecord(
            f"sample-{index}",
            dataset_id,
            version,
            np.array([float(index), float(index + 1)]),
            index % 2,
            "family-a" if index % 2 == 0 else "family-b",
            "train",
            provenance(authorized),
            "unit-test-labels",
        )
        for index in range(3)
    ]
    return DatasetSnapshot.create(
        dataset_id, version, samples, created_at="2026-10-30T00:00:00+00:00"
    )


def policy(calibration_id: str = "cal-v1") -> SafetyPolicy:
    return SafetyPolicy(
        policy_version="policy-v1",
        calibration_id=calibration_id,
        required_detectors=("scalar",),
    )


def register_calibration(store: SafetyStore) -> None:
    store.register_calibration(
        CalibrationRecord(
            calibration_id="cal-v1",
            detector="scalar",
            detector_version="scalar-v1",
            source_dataset_id="toy",
            source_dataset_version=1,
            threshold=5.0,
        )
    )


def audit_for_snapshot(
    scores: dict[str, float], *, calibration_id: str = "cal-v1", healthy: bool = True
):
    def audit(current: DatasetSnapshot, attempt_id: str):
        return {
            sample.sample_id: {
                "scalar": scalar_evidence(
                    sample,
                    evidence_id=f"evidence-{attempt_id}-{sample.sample_id}",
                    detector="scalar",
                    detector_version="scalar-v1",
                    calibration_id=calibration_id,
                    score=scores.get(sample.sample_id, 0.0),
                    threshold=5.0,
                    healthy=healthy,
                )
            }
            for sample in current.samples
        }

    return audit


def test_content_deduplication_preserves_lineage() -> None:
    original = snapshot()
    duplicate = SampleRecord(
        "duplicate",
        "toy",
        1,
        original.samples[0].features,
        original.samples[0].label,
        "family-a",
        "train",
        provenance(),
        "unit-test-labels",
    )
    combined = DatasetSnapshot.create(
        "toy", 1, [*original.samples, duplicate], created_at="2026-10-30T00:00:00+00:00"
    )
    result = exact_deduplicate(combined, new_version=2)
    assert result.duplicate_of == {"duplicate": "sample-0"}
    assert len(result.snapshot.samples) == 3
    assert result.snapshot.manifest.parent_version == 1
    assert result.snapshot.samples[0].parent_ids == ("sample-0",)


def test_store_restart_replay_and_artifact_integrity(tmp_path) -> None:
    with SafetyStore(tmp_path) as store:
        store.register_dataset(snapshot())
        store.record_evidence(
            [
                scalar_evidence(
                    snapshot().samples[0],
                    evidence_id="e1",
                    detector="scalar",
                    detector_version="v1",
                    calibration_id="cal-v1",
                    score=1.0,
                    threshold=5.0,
                )
            ]
        )
        before = store.replay("toy", 1)
    with SafetyStore(tmp_path) as reopened:
        after = reopened.replay("toy", 1)
        assert after.dataset.manifest.sample_ids == before.dataset.manifest.sample_ids
        assert after.dataset.samples[0].content_hash == before.dataset.samples[0].content_hash
        assert after.evidence[0].evidence_id == "e1"
        artifact = tmp_path / "artifacts" / f"{after.dataset.samples[0].content_hash}.json"
        artifact.write_text('{"features":[999],"label":0}')
        with pytest.raises(Exception, match="hash|content"):
            reopened.load_dataset("toy", 1)


def test_policy_missing_calibration_health_and_provenance_fail_closed() -> None:
    current = snapshot()
    sample = current.samples[0]
    missing = policy().decide(sample, {}, generation=0, attempt_id="a")
    assert missing.decision == Decision.ABSTAIN
    stale = policy().decide(
        sample,
        {
            "scalar": scalar_evidence(
                sample,
                evidence_id="stale",
                detector="scalar",
                detector_version="v1",
                calibration_id="other-calibration",
                score=0.0,
                threshold=5.0,
            )
        },
        generation=0,
        attempt_id="b",
    )
    assert stale.decision == Decision.ABSTAIN
    unhealthy = policy().decide(
        sample,
        {
            "scalar": scalar_evidence(
                sample,
                evidence_id="unhealthy",
                detector="scalar",
                detector_version="v1",
                calibration_id="cal-v1",
                score=0.0,
                threshold=5.0,
                healthy=False,
            )
        },
        generation=0,
        attempt_id="c",
    )
    assert unhealthy.decision == Decision.ABSTAIN
    unauthorized = snapshot(authorized=False).samples[0]
    denied = policy().decide(unauthorized, {}, generation=0, attempt_id="d")
    assert denied.decision == Decision.QUARANTINE


def test_controller_trains_only_accepted_samples_and_consumes_once(tmp_path) -> None:
    with SafetyStore(tmp_path) as store:
        current = snapshot()
        store.register_dataset(current)
        register_calibration(store)
        controller = DatasetSafetyController(
            store,
            policy(),
            budget_seconds=1.0,
            clock=iter((0.0, 0.1)).__next__,
            id_factory=lambda: "attempt1",
        )
        updated: list[tuple[str, ...]] = []
        weights = np.array([0.5, -0.25])

        def apply_update(accepted: DatasetSnapshot) -> None:
            nonlocal weights
            updated.append(tuple(sample.sample_id for sample in accepted.samples))
            features = np.stack([sample.features for sample in accepted.samples])
            labels = np.array([sample.label for sample in accepted.samples], dtype=np.float64)
            gradient = features.T @ (features @ weights - labels) / features.shape[0]
            weights = weights - 0.05 * gradient

        result = controller.process(
            "toy",
            1,
            audit=audit_for_snapshot({"sample-1": 10.0}),
            update=apply_update,
            model_version="model-v1",
        )
        assert result.committed
        assert result.decision == Decision.TRAIN
        assert result.accepted_sample_ids == ("sample-0", "sample-2")
        assert updated == [("sample-0", "sample-2")]
        expected_features = np.stack([current.samples[0].features, current.samples[2].features])
        expected_labels = np.array([current.samples[0].label, current.samples[2].label])
        expected_gradient = (
            expected_features.T @ (expected_features @ np.array([0.5, -0.25]) - expected_labels) / 2
        )
        np.testing.assert_allclose(weights, np.array([0.5, -0.25]) - 0.05 * expected_gradient)
        assert result.generation == 1
        assert result.decision_counts["quarantine"] == 1
        batch, consumed = store.load_batch("batch-attempt1")
        assert consumed
        with pytest.raises(BatchConsumed):
            store.commit_batch("batch-attempt1", expected_generation=1, model_version="model-v2")


def test_controller_timeout_and_deep_failure_never_commit(tmp_path) -> None:
    with SafetyStore(tmp_path) as store:
        store.register_dataset(snapshot())
        register_calibration(store)
        timeout = DatasetSafetyController(
            store,
            policy(),
            budget_seconds=1.0,
            clock=iter((0.0, 2.0)).__next__,
            id_factory=lambda: "timeout",
        )
        result = timeout.process(
            "toy",
            1,
            audit=audit_for_snapshot({}),
            update=lambda _: pytest.fail("timeout reached model update"),
            model_version="model-v1",
        )
        assert result.decision == Decision.STOP
        assert not result.committed
        assert store.current_generation("toy", 1) == 0
        failed = DatasetSafetyController(
            store,
            policy(),
            budget_seconds=1.0,
            clock=iter((0.0, 0.1)).__next__,
            id_factory=lambda: "deep-failure",
        )

        def unavailable(_snapshot, _attempt):
            raise AuditUnavailable("deep worker crashed")

        result = failed.process(
            "toy",
            1,
            audit=unavailable,
            update=lambda _: pytest.fail("failed audit reached model update"),
            model_version="model-v1",
        )
        assert result.decision == Decision.ABSTAIN
        assert not result.committed


def test_missing_registered_calibration_abstains_before_update(tmp_path) -> None:
    with SafetyStore(tmp_path) as store:
        store.register_dataset(snapshot())
        controller = DatasetSafetyController(
            store,
            policy(),
            budget_seconds=1.0,
            clock=iter((0.0, 0.1)).__next__,
            id_factory=lambda: "missing-calibration",
        )
        result = controller.process(
            "toy",
            1,
            audit=audit_for_snapshot({}),
            update=lambda _: pytest.fail("missing calibration reached update"),
            model_version="model-v1",
        )
        assert result.decision == Decision.ABSTAIN
        assert not result.committed


def test_stale_generation_cannot_apply_second_batch(tmp_path) -> None:
    with SafetyStore(tmp_path) as store:
        current = snapshot()
        store.register_dataset(current)
        decisions = []
        for sample in current.samples:
            evidence = scalar_evidence(
                sample,
                evidence_id=f"e-{sample.sample_id}",
                detector="scalar",
                detector_version="v1",
                calibration_id="cal-v1",
                score=0.0,
                threshold=5.0,
            )
            store.record_evidence([evidence])
            decision = policy().decide(
                sample,
                {"scalar": evidence},
                generation=0,
                attempt_id=sample.sample_id,
            )
            decisions.append(decision)
        store.record_decisions(decisions)
        for batch_id in ("batch-a", "batch-b"):
            store.create_batch(
                BatchManifest(
                    batch_id,
                    "toy",
                    1,
                    0,
                    (current.samples[0].sample_id,),
                    (decisions[0].decision_id,),
                )
            )
        store.commit_batch("batch-a", expected_generation=0, model_version="m1")
        called = []
        with pytest.raises(StaleGeneration):
            store.commit_batch(
                "batch-b",
                expected_generation=0,
                model_version="m2",
                apply=lambda: called.append(True),
            )
        assert not called


def test_dolma_command_is_shell_free_and_unavailable_is_explicit() -> None:
    command = dolma_dedupe_command(
        "docs/*.jsonl", bloom_filter_path="filter.bin", estimated_doc_count=100
    )
    assert command[1] == "dedupe"
    assert ";" not in " ".join(command)
    if not dolma_available("definitely-not-a-dolma-binary"):
        with pytest.raises(DolmaUnavailable):
            run_dolma_dedupe(
                "docs/*.jsonl",
                bloom_filter_path="filter.bin",
                estimated_doc_count=100,
                executable="definitely-not-a-dolma-binary",
            )
