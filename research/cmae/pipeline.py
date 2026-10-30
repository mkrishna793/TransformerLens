"""Synchronous, accepted-only dataset safety controller."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from research.cmae.dataset import DatasetSnapshot
from research.cmae.engine import Decision
from research.cmae.evidence import BatchManifest, DecisionRecord, EvidenceRecord
from research.cmae.policy import SafetyPolicy
from research.cmae.runtime import SynchronousBudgetGate
from research.cmae.store import SafetyStore, StaleGeneration, StoreError, UnknownRecord

EvidenceBySample: TypeAlias = Mapping[str, Mapping[str, EvidenceRecord]]
AuditProvider: TypeAlias = Callable[[DatasetSnapshot, str], EvidenceBySample]
UpdateProvider: TypeAlias = Callable[[DatasetSnapshot], None]


class AuditUnavailable(RuntimeError):
    """An expected optional/deep-audit failure; never reinterpret this as acceptance."""


@dataclass(frozen=True)
class ControllerResult:
    attempt_id: str
    decision: Decision
    committed: bool
    batch_id: str | None
    accepted_sample_ids: tuple[str, ...]
    decision_counts: Mapping[str, int]
    generation: int
    audit_seconds: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PreparedBatch:
    attempt_id: str
    generation: int
    decision: Decision
    accepted: DatasetSnapshot | None
    accepted_decisions: tuple[DecisionRecord, ...]
    decision_counts: Mapping[str, int]


class DatasetSafetyController:
    """Execute one audit/decision/update attempt with fail-closed commit semantics."""

    def __init__(
        self,
        store: SafetyStore,
        policy: SafetyPolicy,
        *,
        budget_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store = store
        self.policy = policy
        self.budget_seconds = budget_seconds
        self.clock = clock
        self.id_factory = id_factory

    def _validate_calibration(
        self,
        snapshot: DatasetSnapshot,
        evidence: EvidenceBySample,
    ) -> None:
        for detector in self.policy.required_detectors:
            try:
                calibration = self.store.load_calibration(self.policy.calibration_id, detector)
            except UnknownRecord as error:
                raise AuditUnavailable(f"missing calibration for {detector}") from error
            if (
                not calibration.healthy
                or calibration.source_dataset_id != snapshot.manifest.dataset_id
                or calibration.source_dataset_version != snapshot.manifest.version
            ):
                raise AuditUnavailable(f"stale or unhealthy calibration for {detector}")
            for sample in snapshot.samples:
                record = evidence.get(sample.sample_id, {}).get(detector)
                if record is not None and (
                    record.detector_version != calibration.detector_version
                    or record.threshold != calibration.threshold
                ):
                    raise AuditUnavailable(f"evidence/calibration mismatch for {detector}")

    def process(
        self,
        dataset_id: str,
        dataset_version: int,
        *,
        audit: AuditProvider,
        update: UpdateProvider,
        model_version: str,
    ) -> ControllerResult:
        if not model_version.strip():
            raise ValueError("model version is required")
        attempt_id = self.id_factory()
        if not attempt_id.strip():
            raise ValueError("attempt ID factory returned an empty ID")
        snapshot = self.store.load_dataset(dataset_id, dataset_version)
        generation = self.store.current_generation(dataset_id, dataset_version)
        prepared_holder: list[_PreparedBatch] = []

        def prepare() -> _PreparedBatch:
            if self.store.current_generation(dataset_id, dataset_version) != generation:
                raise StaleGeneration("generation changed before audit")
            evidence = audit(snapshot, attempt_id)
            self._validate_calibration(snapshot, evidence)
            assessment = self.policy.assess(
                snapshot,
                evidence,
                generation=generation,
                attempt_id=attempt_id,
            )
            self.store.record_evidence(assessment.evidence)
            self.store.record_decisions(assessment.decisions)
            counts = Counter(record.decision.value for record in assessment.decisions)
            if counts[Decision.ABSTAIN.value] or counts[Decision.STOP.value]:
                prepared = _PreparedBatch(
                    attempt_id,
                    generation,
                    Decision.ABSTAIN if counts[Decision.ABSTAIN.value] else Decision.STOP,
                    None,
                    (),
                    dict(counts),
                )
            else:
                accepted_samples = []
                accepted_decisions = []
                for sample, record in zip(snapshot.samples, assessment.decisions, strict=True):
                    if record.decision == Decision.TRAIN:
                        accepted_samples.append(sample)
                        accepted_decisions.append(record)
                accepted = (
                    DatasetSnapshot.create(
                        dataset_id,
                        dataset_version,
                        accepted_samples,
                        parent_version=snapshot.manifest.parent_version,
                        description=snapshot.manifest.description,
                        created_at=snapshot.manifest.created_at,
                    )
                    if accepted_samples
                    else None
                )
                prepared = _PreparedBatch(
                    attempt_id,
                    generation,
                    Decision.TRAIN if accepted is not None else Decision.QUARANTINE,
                    accepted,
                    tuple(accepted_decisions),
                    dict(counts),
                )
            prepared_holder.append(prepared)
            return prepared

        def commit(prepared: _PreparedBatch) -> ControllerResult:
            if prepared.decision != Decision.TRAIN or prepared.accepted is None:
                return ControllerResult(
                    attempt_id,
                    prepared.decision,
                    False,
                    None,
                    (),
                    prepared.decision_counts,
                    self.store.current_generation(dataset_id, dataset_version),
                    0.0,
                    ("no_fully_audited_accepted_batch",),
                )
            accepted = prepared.accepted
            if accepted is None:
                raise RuntimeError("TRAIN preparation unexpectedly has no accepted snapshot")
            batch_id = f"batch-{attempt_id}"
            batch = BatchManifest(
                batch_id=batch_id,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                generation=generation,
                sample_ids=tuple(sample.sample_id for sample in accepted.samples),
                decision_ids=tuple(record.decision_id for record in prepared.accepted_decisions),
            )
            self.store.create_batch(batch)
            commit_record = self.store.commit_batch(
                batch_id,
                expected_generation=generation,
                model_version=model_version,
                apply=lambda: update(accepted),
            )
            return ControllerResult(
                attempt_id,
                Decision.TRAIN,
                True,
                batch_id,
                batch.sample_ids,
                prepared.decision_counts,
                commit_record.generation,
                0.0,
                ("accepted_only_commit",),
            )

        gate = SynchronousBudgetGate(self.budget_seconds, clock=self.clock)
        try:
            gated = gate.run(prepare, commit)
        except AuditUnavailable:
            return ControllerResult(
                attempt_id,
                Decision.ABSTAIN,
                False,
                None,
                (),
                {},
                self.store.current_generation(dataset_id, dataset_version),
                0.0,
                ("audit_unavailable",),
            )
        except StaleGeneration:
            return ControllerResult(
                attempt_id,
                Decision.STOP,
                False,
                None,
                (),
                prepared_holder[0].decision_counts if prepared_holder else {},
                self.store.current_generation(dataset_id, dataset_version),
                0.0,
                ("stale_generation",),
            )
        except StoreError as error:
            return ControllerResult(
                attempt_id,
                Decision.STOP,
                False,
                None,
                (),
                prepared_holder[0].decision_counts if prepared_holder else {},
                self.store.current_generation(dataset_id, dataset_version),
                0.0,
                (f"store_error:{type(error).__name__}",),
            )
        if gated.decision != Decision.TRAIN:
            return ControllerResult(
                attempt_id,
                gated.decision,
                False,
                None,
                (),
                prepared_holder[0].decision_counts if prepared_holder else {},
                self.store.current_generation(dataset_id, dataset_version),
                gated.elapsed_seconds,
                ("audit_budget_exceeded",),
            )
        if gated.payload is None:
            raise RuntimeError("budget gate returned no controller payload")
        result = gated.payload
        return ControllerResult(
            result.attempt_id,
            result.decision,
            result.committed,
            result.batch_id,
            result.accepted_sample_ids,
            result.decision_counts,
            result.generation,
            gated.elapsed_seconds,
            result.reasons,
        )
