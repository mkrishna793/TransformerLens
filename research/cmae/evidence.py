"""Versioned evidence, decision, batch, and commit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from research.cmae.dataset import utc_now
from research.cmae.engine import Decision


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    sample_id: str
    dataset_id: str
    dataset_version: int
    detector: str
    detector_version: str
    calibration_id: str
    score: float
    threshold: float
    healthy: bool = True
    residual: float = 0.0
    details: Mapping[str, str] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if not self.evidence_id.strip() or not self.sample_id.strip():
            raise ValueError("evidence and sample IDs are required")
        if not self.detector.strip() or not self.detector_version.strip():
            raise ValueError("detector identity and version are required")
        if not self.calibration_id.strip():
            raise ValueError("evidence must reference a calibration")


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    sample_id: str
    dataset_id: str
    dataset_version: int
    generation: int
    decision: Decision
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    policy_version: str
    calibration_id: str
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")
        if not self.decision_id.strip() or not self.sample_id.strip():
            raise ValueError("decision and sample IDs are required")
        if not self.policy_version.strip() or not self.calibration_id.strip():
            raise ValueError("decision policy and calibration are required")


@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    dataset_id: str
    dataset_version: int
    generation: int
    sample_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if not self.batch_id.strip() or not self.sample_ids:
            raise ValueError("a batch requires an ID and at least one sample")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("batch sample IDs must be unique")
        if len(self.sample_ids) != len(self.decision_ids):
            raise ValueError("batch samples and decisions must align")
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")


@dataclass(frozen=True)
class CommitRecord:
    commit_id: str
    batch_id: str
    dataset_id: str
    dataset_version: int
    generation: int
    model_version: str
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if (
            not self.commit_id.strip()
            or not self.batch_id.strip()
            or not self.model_version.strip()
        ):
            raise ValueError("commit, batch, and model IDs are required")


def evidence_to_dict(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "evidence_id": record.evidence_id,
        "sample_id": record.sample_id,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "detector": record.detector,
        "detector_version": record.detector_version,
        "calibration_id": record.calibration_id,
        "score": record.score,
        "threshold": record.threshold,
        "healthy": record.healthy,
        "residual": record.residual,
        "details": dict(record.details),
        "created_at": record.created_at,
    }


def evidence_from_dict(payload: Mapping[str, Any]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=str(payload["evidence_id"]),
        sample_id=str(payload["sample_id"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_version=int(payload["dataset_version"]),
        detector=str(payload["detector"]),
        detector_version=str(payload["detector_version"]),
        calibration_id=str(payload["calibration_id"]),
        score=float(payload["score"]),
        threshold=float(payload["threshold"]),
        healthy=bool(payload["healthy"]),
        residual=float(payload["residual"]),
        details={str(key): str(value) for key, value in payload["details"].items()},
        created_at=str(payload["created_at"]),
    )


def decision_to_dict(record: DecisionRecord) -> dict[str, Any]:
    return {
        "decision_id": record.decision_id,
        "sample_id": record.sample_id,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "generation": record.generation,
        "decision": record.decision.value,
        "reasons": list(record.reasons),
        "evidence_ids": list(record.evidence_ids),
        "policy_version": record.policy_version,
        "calibration_id": record.calibration_id,
        "created_at": record.created_at,
    }


def decision_from_dict(payload: Mapping[str, Any]) -> DecisionRecord:
    return DecisionRecord(
        decision_id=str(payload["decision_id"]),
        sample_id=str(payload["sample_id"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_version=int(payload["dataset_version"]),
        generation=int(payload["generation"]),
        decision=Decision(str(payload["decision"])),
        reasons=tuple(str(value) for value in payload["reasons"]),
        evidence_ids=tuple(str(value) for value in payload["evidence_ids"]),
        policy_version=str(payload["policy_version"]),
        calibration_id=str(payload["calibration_id"]),
        created_at=str(payload["created_at"]),
    )


def batch_to_dict(record: BatchManifest) -> dict[str, Any]:
    return {
        "batch_id": record.batch_id,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "generation": record.generation,
        "sample_ids": list(record.sample_ids),
        "decision_ids": list(record.decision_ids),
        "created_at": record.created_at,
    }


def batch_from_dict(payload: Mapping[str, Any]) -> BatchManifest:
    return BatchManifest(
        batch_id=str(payload["batch_id"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_version=int(payload["dataset_version"]),
        generation=int(payload["generation"]),
        sample_ids=tuple(str(value) for value in payload["sample_ids"]),
        decision_ids=tuple(str(value) for value in payload["decision_ids"]),
        created_at=str(payload["created_at"]),
    )


def commit_to_dict(record: CommitRecord) -> dict[str, Any]:
    return {
        "commit_id": record.commit_id,
        "batch_id": record.batch_id,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "generation": record.generation,
        "model_version": record.model_version,
        "created_at": record.created_at,
    }


def commit_from_dict(payload: Mapping[str, Any]) -> CommitRecord:
    return CommitRecord(
        commit_id=str(payload["commit_id"]),
        batch_id=str(payload["batch_id"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_version=int(payload["dataset_version"]),
        generation=int(payload["generation"]),
        model_version=str(payload["model_version"]),
        created_at=str(payload["created_at"]),
    )
