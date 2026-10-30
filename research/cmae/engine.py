"""Global CMAE decisions, physical filtering, and NLA escalation contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
T = TypeVar("T")


class Decision(str, Enum):
    TRAIN = "train"
    QUARANTINE = "quarantine"
    ABSTAIN = "abstain"
    STOP = "stop"


@dataclass(frozen=True)
class Evidence:
    sample_index: int
    finite: bool = True
    forbidden_observable: bool = False
    spectral_z: float = 0.0
    topological_score: float = 0.0


@dataclass(frozen=True)
class AuditRecord:
    evidence: Evidence
    decision: Decision
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RepairProposal:
    sample_index: int
    operation: str
    status: str = "requires_reaudit_and_provenance"


class CausalMechanisticAlignmentEngine:
    """Combine evidence once per sample; never mask gradients layer by layer."""

    def __init__(
        self,
        *,
        spectral_threshold: float = 3.5,
        topological_threshold: float | None = None,
    ) -> None:
        if not math.isfinite(spectral_threshold):
            raise ValueError("spectral threshold must be finite")
        if topological_threshold is not None and not math.isfinite(topological_threshold):
            raise ValueError("topological threshold must be finite or None")
        self.spectral_threshold = spectral_threshold
        self.topological_threshold = topological_threshold

    def audit(self, evidence: Sequence[Evidence], *, detector_healthy: bool) -> list[AuditRecord]:
        records: list[AuditRecord] = []
        for item in evidence:
            reasons: list[str] = []
            decision = Decision.TRAIN
            numeric_values = [item.spectral_z, item.topological_score]
            if not item.finite or not all(math.isfinite(value) for value in numeric_values):
                decision = Decision.QUARANTINE
                reasons.append("nonfinite_evidence")
            elif not detector_healthy:
                decision = Decision.ABSTAIN
                reasons.append("detector_unhealthy_or_unconverged")
            else:
                if item.forbidden_observable:
                    decision = Decision.QUARANTINE
                    reasons.append("forbidden_observable")
                if item.spectral_z > self.spectral_threshold:
                    decision = Decision.QUARANTINE
                    reasons.append("spectral_outlier")
                if (
                    self.topological_threshold is not None
                    and item.topological_score > self.topological_threshold
                ):
                    decision = Decision.QUARANTINE
                    reasons.append("topological_contribution")
            records.append(AuditRecord(item, decision, tuple(reasons)))
        return records

    @staticmethod
    def physically_filter(items: Sequence[T], records: Sequence[AuditRecord]) -> list[T]:
        """Return only TRAIN items for construction of a fresh training batch."""

        if len(items) != len(records):
            raise ValueError("items and audit records must have identical lengths")
        return [
            item
            for item, record in zip(items, records, strict=True)
            if record.decision == Decision.TRAIN
        ]

    @staticmethod
    def propose_repair(record: AuditRecord, operation: str) -> RepairProposal:
        """Create a reviewable proposal; never declare semantic repair successful."""

        if record.decision != Decision.QUARANTINE:
            raise ValueError("only quarantined records are eligible for repair proposals")
        if not operation.strip():
            raise ValueError("a repair operation description is required")
        return RepairProposal(record.evidence.sample_index, operation)


class NLAProvider(Protocol):
    """External interface for a real activation verbalizer/reconstructor pair."""

    def verbalize(self, activation: FloatArray, context: str) -> str: ...

    def reconstruct(self, description: str, target_shape: tuple[int, ...]) -> FloatArray: ...


@dataclass(frozen=True)
class NLAEscalation:
    description: str
    relative_reconstruction_error: float
    corroborated: bool = False
    admissible_as_hard_gate: bool = False


def escalate_with_nla(provider: NLAProvider, activation: FloatArray, context: str) -> NLAEscalation:
    """Generate a dataset-audit hypothesis while preserving its uncertain status."""

    if activation.ndim == 0 or not np.all(np.isfinite(activation)):
        raise ValueError("NLA activation must be a finite array")
    description = provider.verbalize(activation, context)
    if not description.strip():
        raise ValueError("NLA returned an empty description")
    reconstructed = np.asarray(
        provider.reconstruct(description, activation.shape), dtype=np.float64
    )
    if reconstructed.shape != activation.shape or not np.all(np.isfinite(reconstructed)):
        raise ValueError("NLA reconstruction has an invalid shape or nonfinite values")
    denominator = max(float(np.linalg.norm(activation)), 1e-12)
    error = float(np.linalg.norm(activation - reconstructed) / denominator)
    return NLAEscalation(description, error)
