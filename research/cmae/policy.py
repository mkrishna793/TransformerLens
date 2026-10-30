"""Fail-closed policy decisions over versioned samples and detector evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from research.cmae.dataset import DatasetSnapshot, SampleRecord
from research.cmae.engine import Decision
from research.cmae.evidence import DecisionRecord, EvidenceRecord


@dataclass(frozen=True)
class CalibrationRecord:
    calibration_id: str
    detector: str
    detector_version: str
    source_dataset_id: str
    source_dataset_version: int
    threshold: float
    orientation: float = 1.0
    healthy: bool = True

    def __post_init__(self) -> None:
        if not self.calibration_id.strip() or not self.detector.strip():
            raise ValueError("calibration and detector IDs are required")
        if not math.isfinite(self.threshold) or self.orientation not in (-1.0, 1.0):
            raise ValueError("calibration threshold or orientation is invalid")
        if self.source_dataset_version < 1:
            raise ValueError("calibration source version must be positive")

    def orient(self, score: float) -> float:
        return self.orientation * score


@dataclass(frozen=True)
class PolicyAssessment:
    decisions: tuple[DecisionRecord, ...]
    evidence: tuple[EvidenceRecord, ...]


class SafetyPolicy:
    """Default-deny composition of provenance and required detector evidence."""

    def __init__(
        self,
        *,
        policy_version: str,
        calibration_id: str,
        required_detectors: tuple[str, ...],
        maximum_residual: float = math.inf,
    ) -> None:
        if not policy_version.strip() or not calibration_id.strip():
            raise ValueError("policy and calibration versions are required")
        if not required_detectors or len(set(required_detectors)) != len(required_detectors):
            raise ValueError("at least one unique required detector is needed")
        if math.isnan(maximum_residual) or maximum_residual < 0:
            raise ValueError("maximum residual must be nonnegative")
        self.policy_version = policy_version
        self.calibration_id = calibration_id
        self.required_detectors = required_detectors
        self.maximum_residual = maximum_residual

    def decide(
        self,
        sample: SampleRecord,
        evidence: Mapping[str, EvidenceRecord],
        *,
        generation: int,
        attempt_id: str,
    ) -> DecisionRecord:
        quarantine_reasons: list[str] = []
        abstain_reasons: list[str] = []
        evidence_ids: list[str] = []
        if not sample.provenance.complete:
            quarantine_reasons.append("provenance_incomplete_or_unauthorized")
        for detector in self.required_detectors:
            record = evidence.get(detector)
            if record is None:
                abstain_reasons.append(f"missing_evidence:{detector}")
                continue
            evidence_ids.append(record.evidence_id)
            if (
                record.sample_id != sample.sample_id
                or record.dataset_id != sample.dataset_id
                or record.dataset_version != sample.dataset_version
                or record.detector != detector
            ):
                abstain_reasons.append(f"evidence_identity_mismatch:{detector}")
                continue
            if record.calibration_id != self.calibration_id:
                abstain_reasons.append(f"stale_calibration:{detector}")
            numeric = (record.score, record.threshold, record.residual)
            if not all(math.isfinite(value) for value in numeric):
                quarantine_reasons.append(f"nonfinite_evidence:{detector}")
                continue
            if not record.healthy:
                abstain_reasons.append(f"detector_unhealthy:{detector}")
            if record.residual > self.maximum_residual:
                abstain_reasons.append(f"residual_budget_exceeded:{detector}")
            if record.score > record.threshold:
                quarantine_reasons.append(f"detector_threshold_exceeded:{detector}")
        if quarantine_reasons:
            decision = Decision.QUARANTINE
            reasons = tuple(quarantine_reasons + abstain_reasons)
        elif abstain_reasons:
            decision = Decision.ABSTAIN
            reasons = tuple(abstain_reasons)
        else:
            decision = Decision.TRAIN
            reasons = ("policy_accepted",)
        return DecisionRecord(
            decision_id=f"decision-{attempt_id}-{sample.sample_id}",
            sample_id=sample.sample_id,
            dataset_id=sample.dataset_id,
            dataset_version=sample.dataset_version,
            generation=generation,
            decision=decision,
            reasons=reasons,
            evidence_ids=tuple(sorted(evidence_ids)),
            policy_version=self.policy_version,
            calibration_id=self.calibration_id,
        )

    def assess(
        self,
        snapshot: DatasetSnapshot,
        evidence_by_sample: Mapping[str, Mapping[str, EvidenceRecord]],
        *,
        generation: int,
        attempt_id: str,
    ) -> PolicyAssessment:
        decisions = []
        evidence = []
        for sample in snapshot.samples:
            sample_evidence = evidence_by_sample.get(sample.sample_id, {})
            for record in sample_evidence.values():
                evidence.append(record)
            decisions.append(
                self.decide(
                    sample,
                    sample_evidence,
                    generation=generation,
                    attempt_id=attempt_id,
                )
            )
        return PolicyAssessment(tuple(decisions), tuple(evidence))


def scalar_evidence(
    sample: SampleRecord,
    *,
    evidence_id: str,
    detector: str,
    detector_version: str,
    calibration_id: str,
    score: float,
    threshold: float,
    healthy: bool = True,
    residual: float = 0.0,
    details: Mapping[str, str] | None = None,
) -> EvidenceRecord:
    """Construct one scalar detector observation without assigning a safety decision."""

    return EvidenceRecord(
        evidence_id=evidence_id,
        sample_id=sample.sample_id,
        dataset_id=sample.dataset_id,
        dataset_version=sample.dataset_version,
        detector=detector,
        detector_version=detector_version,
        calibration_id=calibration_id,
        score=score,
        threshold=threshold,
        healthy=healthy,
        residual=residual,
        details={} if details is None else details,
    )
