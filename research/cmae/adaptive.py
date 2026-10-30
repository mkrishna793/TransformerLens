"""Milestone-four model-aware attacks and explicit repair/rollback workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.metrics import threshold_for_clean_fpr
from research.cmae.real_benchmark import (
    RealMLP,
    SpectralDetector,
    fit_real_model,
    real_activation_features,
    real_behavior_metrics,
)
from research.cmae.real_data import (
    RealBatch,
    load_real_dataset,
    make_real_batch,
    split_real_dataset,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ProjectedAttack:
    features: FloatArray
    max_linf: float
    mean_l2: float
    target_loss_before: float
    target_loss_after: float
    steps: int
    epsilon: float
    step_size: float


@dataclass(frozen=True)
class RepairGate:
    clean_accuracy_drop: float
    original_before: float
    original_after: float
    adaptive_before: float
    adaptive_after: float
    review_approved: bool
    same_evaluation_rows: bool
    fresh_adaptive_attack: bool
    version_changed: bool
    clean_tolerance: float = 0.02
    original_ceiling: float = 0.5
    adaptive_ceiling: float = 0.5

    @property
    def passed(self) -> bool:
        numeric = np.asarray(
            [
                self.clean_accuracy_drop,
                self.original_before,
                self.original_after,
                self.adaptive_before,
                self.adaptive_after,
                self.clean_tolerance,
                self.original_ceiling,
                self.adaptive_ceiling,
            ]
        )
        return bool(
            np.all(np.isfinite(numeric))
            and self.clean_tolerance >= 0
            and 0 <= self.original_ceiling <= 1
            and 0 <= self.adaptive_ceiling <= 1
            and self.clean_accuracy_drop <= self.clean_tolerance
            and self.original_after < self.original_before
            and self.adaptive_after < self.adaptive_before
            and self.original_after <= self.original_ceiling
            and self.adaptive_after <= self.adaptive_ceiling
            and self.review_approved
            and self.same_evaluation_rows
            and self.fresh_adaptive_attack
            and self.version_changed
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "clean_accuracy_drop": self.clean_accuracy_drop,
            "clean_tolerance": self.clean_tolerance,
            "original_ceiling": self.original_ceiling,
            "adaptive_ceiling": self.adaptive_ceiling,
            "original_before": self.original_before,
            "original_after": self.original_after,
            "adaptive_before": self.adaptive_before,
            "adaptive_after": self.adaptive_after,
            "review_approved": self.review_approved,
            "same_evaluation_rows": self.same_evaluation_rows,
            "fresh_adaptive_attack": self.fresh_adaptive_attack,
            "version_changed": self.version_changed,
            "passed": self.passed,
        }


@dataclass
class RepairWorkflow:
    parent_version: str
    state: str = "received"
    history: list[str] = field(default_factory=lambda: ["received"])
    quarantined_indices: IntArray = field(default_factory=lambda: np.array([], dtype=np.int64))
    review_approved: bool = False
    review_reason: str = "pending"
    data_version: str | None = None
    candidate_version: str | None = None
    gate: RepairGate | None = None

    def _require(self, state: str) -> None:
        if self.state != state:
            raise RuntimeError(f"repair transition requires {state}, found {self.state}")

    def _transition(self, state: str) -> None:
        self.state = state
        self.history.append(state)

    def quarantine(self, indices: IntArray, population: int) -> None:
        self._require("received")
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= population):
            raise ValueError("quarantine indices are outside the population")
        if np.unique(values).size != values.size:
            raise ValueError("quarantine indices must be unique")
        self.quarantined_indices = values
        self._transition("quarantined")

    def review(self, approved: bool, reason: str) -> None:
        self._require("quarantined")
        self.review_approved = approved
        self.review_reason = reason
        self._transition("reviewed")

    def repair(self, data_version: str, remaining_rows: int) -> None:
        self._require("reviewed")
        if not self.review_approved or remaining_rows <= 0 or not data_version:
            raise RuntimeError("repair requires approval, retained rows, and a data version")
        self.data_version = data_version
        self._transition("repaired")

    def retrained(self, candidate_version: str) -> None:
        self._require("repaired")
        if not candidate_version or candidate_version == self.parent_version:
            raise RuntimeError("candidate model version must be new and nonempty")
        self.candidate_version = candidate_version
        self._transition("retrained")

    def evaluate(self, gate: RepairGate) -> None:
        self._require("retrained")
        self.gate = gate
        self._transition("evaluated")

    def decide(self) -> str:
        self._require("evaluated")
        self._transition(
            "promoted" if self.gate is not None and self.gate.passed else "rolled_back"
        )
        return self.state

    def rollback(self, reason: str) -> None:
        if self.state in {"promoted", "rolled_back"}:
            raise RuntimeError("workflow is already terminal")
        self.review_reason = reason
        self._transition("rolled_back")


def targeted_projected_attack(
    model: RealMLP,
    batch: RealBatch,
    *,
    target_class: int = 1,
    epsilon: float = 0.15,
    step_size: float = 0.025,
    steps: int = 12,
) -> ProjectedAttack:
    """Targeted white-box PGD with projection to each row's L-infinity ball."""

    if epsilon <= 0 or step_size <= 0 or steps <= 0 or epsilon < step_size:
        raise ValueError("attack requires 0 < step_size <= epsilon and positive steps")
    original = torch.from_numpy(batch.features).to(
        device=model.input.weight.device, dtype=model.input.weight.dtype
    )
    eligible = torch.from_numpy(batch.original_labels != target_class).to(original.device)
    targets = torch.full(
        (original.shape[0],), target_class, dtype=torch.long, device=original.device
    )
    perturbed = original.clone()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = torch.nn.functional.cross_entropy(model(original), targets)
        for _ in range(steps):
            perturbed.requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(model(perturbed), targets)
            gradient = torch.autograd.grad(loss, perturbed, retain_graph=False)[0]
            with torch.no_grad():
                perturbed = perturbed - step_size * gradient.sign()
                perturbation = (perturbed - original).clamp(-epsilon, epsilon)
                perturbed = original + perturbation
                perturbed = torch.where(eligible[:, None], perturbed, original)
        with torch.no_grad():
            after = torch.nn.functional.cross_entropy(model(perturbed), targets)
            delta = perturbed - original
            max_linf = float(delta.abs().max().cpu().item())
            mean_l2 = float(delta.norm(dim=1).mean().cpu().item())
        values = perturbed.detach().cpu().numpy().astype(np.float64)
        eligible_rows = eligible.cpu().numpy().astype(bool)
        values[~eligible_rows] = batch.features[~eligible_rows]
        return ProjectedAttack(
            features=values,
            max_linf=max_linf,
            mean_l2=mean_l2,
            target_loss_before=float(before.cpu().item()),
            target_loss_after=float(after.cpu().item()),
            steps=steps,
            epsilon=epsilon,
            step_size=step_size,
        )
    finally:
        model.train(was_training)
        model.probe.last_activation = None


def adaptive_behavior_metrics(
    model: RealMLP, batch: RealBatch, attacked_features: FloatArray, *, target_class: int = 1
) -> dict[str, float]:
    if attacked_features.shape != batch.features.shape or not np.all(
        np.isfinite(attacked_features)
    ):
        raise ValueError("attacked features must align with the clean batch")
    model.eval()
    dtype = model.input.weight.dtype
    device = model.input.weight.device
    with torch.no_grad():
        clean_prediction = (
            model(torch.from_numpy(batch.features).to(device=device, dtype=dtype))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
        attacked_prediction = (
            model(torch.from_numpy(attacked_features).to(device=device, dtype=dtype))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    eligible = batch.original_labels != target_class
    return {
        "clean_accuracy": float(np.mean(clean_prediction == batch.original_labels)),
        "attack_success_rate": (
            float(np.mean(attacked_prediction[eligible] == target_class)) if eligible.any() else 0.0
        ),
        "eligible_rows": float(eligible.sum()),
    }


def model_version(model: RealMLP, lineage: str) -> str:
    digest = hashlib.sha256()
    digest.update(lineage.encode("utf-8"))
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def representation_drift(
    model: RealMLP, reference: FloatArray, evaluation: FloatArray
) -> dict[str, float]:
    first = real_activation_features(model, reference)
    second = real_activation_features(model, evaluation)
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    _, _, first_right = np.linalg.svd(first - first_mean, full_matrices=False)
    _, _, second_right = np.linalg.svd(second - second_mean, full_matrices=False)
    reference_norm = max(float(np.linalg.norm(first_mean)), 1e-12)
    return {
        "mean_relative_drift": float(np.linalg.norm(second_mean - first_mean) / reference_norm),
        "top_direction_abs_cosine": float(abs(first_right[0] @ second_right[0])),
    }


def _remaining_version(batch: RealBatch, keep: IntArray) -> str:
    digest = hashlib.sha256()
    digest.update(keep.tobytes())
    digest.update(batch.features[keep].tobytes())
    digest.update(batch.labels[keep].tobytes())
    return digest.hexdigest()


def run_adaptive_repair_case(
    name: Literal["breast_cancer", "digits_binary"],
    attack: Literal["standard", "adaptive"],
    seed: int,
    *,
    training_epochs: int = 60,
    attack_steps: int = 12,
    epsilon: float = 0.15,
) -> dict[str, object]:
    dataset = load_real_dataset(name)
    splits = split_real_dataset(dataset, seed)
    fit_batch = make_real_batch(splits.fit, attack, seed + 1)
    calibration_clean = make_real_batch(splits.calibration, "clean", seed + 2)
    clean_test = make_real_batch(splits.test, "clean", seed + 3)
    torch.manual_seed(seed + 5)
    parent = RealMLP(dataset.features.shape[1])
    fit_real_model(parent, fit_batch, epochs=training_epochs)
    parent_version = model_version(parent, f"{name}:{attack}:{seed}:parent")
    workflow = RepairWorkflow(parent_version)
    fit_activations = real_activation_features(parent, fit_batch.features)
    calibration_activations = real_activation_features(parent, calibration_clean.features)
    detector = SpectralDetector.fit(fit_activations)
    fit_scores = detector.score(fit_activations, leave_one_out=True)
    threshold = threshold_for_clean_fpr(
        detector.score(calibration_activations),
        np.zeros(len(calibration_activations), dtype=np.int64),
    )
    quarantined = np.flatnonzero(fit_scores > threshold).astype(np.int64)
    workflow.quarantine(quarantined, len(fit_batch.features))
    keep = np.setdiff1d(np.arange(len(fit_batch.features)), quarantined)
    enough = (
        len(keep) >= 8
        and len(np.unique(fit_batch.labels[keep])) == 2
        and len(quarantined) <= int(0.35 * len(fit_batch.features))
    )
    approved = bool(detector.healthy and len(quarantined) > 0 and enough)
    workflow.review(
        approved,
        "deterministic numeric review: detector health, removal budget, and retained class coverage; no semantic certification",
    )
    before_original = real_behavior_metrics(parent, clean_test, attack)
    before_attack = targeted_projected_attack(
        parent, clean_test, epsilon=epsilon, steps=attack_steps
    )
    before_adaptive = adaptive_behavior_metrics(parent, clean_test, before_attack.features)
    base: dict[str, object] = {
        "dataset": name,
        "attack": attack,
        "seed": seed,
        "parent_version": parent_version,
        "detector": "spectral",
        "detector_healthy": detector.healthy,
        "calibration_threshold": threshold,
        "quarantined_rows": len(quarantined),
        "retained_rows": len(keep),
        "before": {
            "original": before_original,
            "model_aware_adaptive": before_adaptive,
            "attack_projection": {
                "epsilon": before_attack.epsilon,
                "step_size": before_attack.step_size,
                "steps": before_attack.steps,
                "max_linf": before_attack.max_linf,
                "mean_l2": before_attack.mean_l2,
                "target_loss_before": before_attack.target_loss_before,
                "target_loss_after": before_attack.target_loss_after,
            },
        },
        "drift": representation_drift(parent, fit_batch.features, clean_test.features),
        "drift_scope": "descriptive poisoned-fit versus clean-test activation drift; not a safety gate",
        "claim_scope": "bounded white-box PGD and fixed-benchmark repair; no optimal-adversary or semantic-repair certificate",
    }
    if not approved:
        workflow.rollback("review rejected candidate; fail-closed parent retained")
        base.update(
            {
                "workflow": workflow.history,
                "final_state": workflow.state,
                "review_reason": workflow.review_reason,
                "deployed_version": parent_version,
                "gate": None,
            }
        )
        return base
    data_version = _remaining_version(fit_batch, keep)
    workflow.repair(data_version, len(keep))
    torch.manual_seed(seed + 5)
    candidate = RealMLP(dataset.features.shape[1])
    fit_real_model(candidate, fit_batch.subset(keep), epochs=training_epochs)
    candidate_version = model_version(candidate, data_version)
    workflow.retrained(candidate_version)
    after_original = real_behavior_metrics(candidate, clean_test, attack)
    after_attack = targeted_projected_attack(
        candidate, clean_test, epsilon=epsilon, steps=attack_steps
    )
    after_adaptive = adaptive_behavior_metrics(candidate, clean_test, after_attack.features)
    gate = RepairGate(
        clean_accuracy_drop=before_original["clean_accuracy"] - after_original["clean_accuracy"],
        original_before=before_original["attack_success_rate"],
        original_after=after_original["attack_success_rate"],
        adaptive_before=before_adaptive["attack_success_rate"],
        adaptive_after=after_adaptive["attack_success_rate"],
        review_approved=workflow.review_approved,
        same_evaluation_rows=after_attack.features.shape == before_attack.features.shape,
        fresh_adaptive_attack=True,
        version_changed=candidate_version != parent_version,
    )
    workflow.evaluate(gate)
    final_state = workflow.decide()
    base.update(
        {
            "candidate_version": candidate_version,
            "data_version": data_version,
            "after": {
                "original": after_original,
                "model_aware_adaptive": after_adaptive,
                "attack_projection": {
                    "epsilon": after_attack.epsilon,
                    "step_size": after_attack.step_size,
                    "steps": after_attack.steps,
                    "max_linf": after_attack.max_linf,
                    "mean_l2": after_attack.mean_l2,
                    "target_loss_before": after_attack.target_loss_before,
                    "target_loss_after": after_attack.target_loss_after,
                },
            },
            "gate": gate.to_dict(),
            "workflow": workflow.history,
            "final_state": final_state,
            "review_reason": workflow.review_reason,
            "deployed_version": candidate_version if final_state == "promoted" else parent_version,
        }
    )
    return base


def run_adaptive_benchmark(output: Path) -> dict[str, object]:
    cases = []
    for name in ("breast_cancer", "digits_binary"):
        for attack in ("standard", "adaptive"):
            cases.append(run_adaptive_repair_case(name, attack, 101))
    result: dict[str, object] = {
        "schema_version": 1,
        "execution": "sequential",
        "cases": cases,
        "promoted_cases": sum(case["final_state"] == "promoted" for case in cases),
        "gate_policy": {
            "maximum_clean_accuracy_drop": 0.02,
            "maximum_post_repair_original_asr": 0.5,
            "maximum_post_repair_adaptive_asr": 0.5,
            "original_and_adaptive_improvement": "strict",
            "fresh_model_aware_attack_required": True,
        },
        "evaluation_scope": "development benchmark; no untouched confirmatory holdout",
        "limitations": [
            "Targeted PGD is a bounded white-box attack, not an optimal or exhaustive adversary.",
            "Fresh attacks are regenerated against each repaired model; fixed overlays are separately reported.",
            "Review is a deterministic numeric policy, not human semantic approval.",
            "Rollback retains the parent model; it does not establish that the parent is safe.",
        ],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_adaptive_benchmark(arguments.output)
    print(
        json.dumps(
            {"output": str(arguments.output), "promoted_cases": result["promoted_cases"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
