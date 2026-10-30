"""Sequential real-row poisoning, held-out detection, and repair experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from research.cmae.detectors import art_activation_cluster_scores, art_spectral_scores
from research.cmae.ground_truth import AttackKind
from research.cmae.metrics import (
    average_precision,
    confusion,
    roc_auc,
    threshold_for_clean_fpr,
)
from research.cmae.real_data import (
    RealBatch,
    RealDataset,
    RealSplits,
    load_real_dataset,
    make_real_batch,
    split_real_dataset,
    triggered_real_copy,
)
from research.cmae.tracing import CMAEProbe

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
RealDatasetName = Literal["breast_cancer", "digits_binary"]


class RealMLP(nn.Module):
    """Small two-class network with an explicit representation probe."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.probe = CMAEProbe()
        self.head = nn.Linear(hidden_dim, 2)

    def representation(self, value: torch.Tensor) -> torch.Tensor:
        return self.probe(torch.tanh(self.input(value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.representation(value))


def fit_real_model(
    model: RealMLP, batch: RealBatch, *, epochs: int = 60, learning_rate: float = 0.05
) -> list[float]:
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("real training epochs and learning rate must be positive")
    features = torch.from_numpy(batch.features).float()
    labels = torch.from_numpy(batch.labels).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), labels)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("nonfinite real-model training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def real_activation_features(model: RealMLP, features: FloatArray) -> FloatArray:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            activation = model.representation(torch.from_numpy(features).float())
        return activation.detach().cpu().numpy().astype(np.float64)
    finally:
        model.train(was_training)
        model.probe.last_activation = None


def real_behavior_metrics(model: RealMLP, clean: RealBatch, attack: AttackKind) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        clean_prediction = (
            model(torch.from_numpy(clean.features).float()).argmax(dim=1).cpu().numpy()
        )
        triggered_prediction = (
            model(torch.from_numpy(triggered_real_copy(clean, attack)).float())
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    eligible = clean.original_labels == 0
    return {
        "clean_accuracy": float(np.mean(clean_prediction == clean.original_labels)),
        "attack_success_rate": (
            float(np.mean(triggered_prediction[eligible] == 1)) if eligible.any() else 0.0
        ),
    }


class InductiveDetector(Protocol):
    def score(self, features: FloatArray, *, leave_one_out: bool = False) -> FloatArray: ...


@dataclass(frozen=True)
class PCADetector:
    center: FloatArray
    direction: FloatArray

    @classmethod
    def fit(cls, features: FloatArray) -> PCADetector:
        center = features.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(features - center, full_matrices=False)
        return cls(center, right[0])

    def score(self, features: FloatArray, *, leave_one_out: bool = False) -> FloatArray:
        del leave_one_out
        return ((features - self.center) @ self.direction) ** 2


@dataclass(frozen=True)
class SpectralDetector:
    center: FloatArray
    direction: FloatArray
    score_center: float
    score_scale: float
    healthy: bool
    reason: str

    @classmethod
    def fit(cls, features: FloatArray) -> SpectralDetector:
        center = np.median(features, axis=0, keepdims=True)
        centered = features - center
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        direction = right[0]
        raw = (centered @ direction) ** 2
        score_center = float(np.median(raw))
        mad = float(np.median(np.abs(raw - score_center)))
        scale = 1.4826 * mad
        healthy = bool(np.all(np.isfinite(direction)) and scale > 1e-12)
        return cls(
            center,
            direction,
            score_center,
            scale + 1e-12,
            healthy,
            "ok" if healthy else "degenerate_mad",
        )

    def score(self, features: FloatArray, *, leave_one_out: bool = False) -> FloatArray:
        del leave_one_out
        raw = ((features - self.center) @ self.direction) ** 2
        return (raw - self.score_center) / self.score_scale


@dataclass(frozen=True)
class KNNDetector:
    reference: FloatArray
    neighbors: int

    @classmethod
    def fit(cls, features: FloatArray) -> KNNDetector:
        return cls(features.copy(), min(5, features.shape[0] - 1))

    def score(self, features: FloatArray, *, leave_one_out: bool = False) -> FloatArray:
        distances = np.linalg.norm(features[:, None, :] - self.reference[None, :, :], axis=-1)
        if leave_one_out:
            if features.shape != self.reference.shape or not np.allclose(features, self.reference):
                raise ValueError("leave-one-out scoring requires the exact reference matrix")
            np.fill_diagonal(distances, np.inf)
        k = min(self.neighbors, distances.shape[1] - (1 if leave_one_out else 0))
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        return nearest.mean(axis=1)


def _rank01(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def _rank_union(spectral: FloatArray, knn: FloatArray) -> FloatArray:
    return np.maximum(_rank01(spectral), _rank01(knn))


def _detector_metrics(
    calibration_labels: IntArray,
    calibration_scores: FloatArray,
    test_labels: IntArray,
    test_scores: FloatArray,
) -> dict[str, float]:
    raw_calibration_auc = roc_auc(calibration_labels, calibration_scores)
    orientation = -1.0 if raw_calibration_auc < 0.5 else 1.0
    oriented_calibration = orientation * calibration_scores
    oriented_test = orientation * test_scores
    threshold = threshold_for_clean_fpr(oriented_calibration, calibration_labels, fpr=0.1)
    result = confusion(test_labels, oriented_test, threshold)
    result.update(
        {
            "threshold": threshold,
            "roc_auc": roc_auc(test_labels, oriented_test),
            "average_precision": average_precision(test_labels, oriented_test),
            "calibration_roc_auc": roc_auc(calibration_labels, oriented_calibration),
            "calibration_raw_roc_auc": raw_calibration_auc,
            "score_orientation": orientation,
        }
    )
    return result


def _group_holdout_metrics(
    calibration_labels: IntArray,
    calibration_scores: FloatArray,
    test_labels: IntArray,
    test_scores: FloatArray,
    calibration_groups: StringArray,
    test_groups: StringArray,
) -> dict[str, object]:
    """Leave each evaluation group out of calibration before scoring its test slice."""

    results: dict[str, object] = {}
    for group in sorted(set(test_groups.tolist())):
        calibration_mask = calibration_groups != group
        test_mask = test_groups == group
        if (
            np.count_nonzero(test_mask) == 0
            or np.count_nonzero(calibration_mask) < 4
            or len(np.unique(calibration_labels[calibration_mask])) < 2
            or len(np.unique(test_labels[test_mask])) < 2
        ):
            results[group] = {"status": "insufficient_two_class_group_data"}
            continue
        metrics = _detector_metrics(
            calibration_labels[calibration_mask],
            calibration_scores[calibration_mask],
            test_labels[test_mask],
            test_scores[test_mask],
        )
        metrics["calibration_samples"] = float(np.count_nonzero(calibration_mask))
        metrics["test_samples"] = float(np.count_nonzero(test_mask))
        results[group] = metrics
    return results


def _art_classifier(model: RealMLP, input_dim: int) -> Any:
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(input_dim,),
        nb_classes=2,
        device_type="cpu",
        channels_first=False,
        preprocessing=(0.0, 1.0),
    )


def _art_batch_audit(
    model: RealMLP,
    batches: dict[str, RealBatch],
    features: dict[str, FloatArray],
) -> dict[str, object]:
    """Run ART batch defenses separately, explicitly marking transductive evaluation."""

    output: dict[str, object] = {"evaluation_mode": "transductive_batch_reference"}
    try:
        classifier = _art_classifier(model, batches["fit"].features.shape[1])
        spectral_scores: dict[str, FloatArray] = {}
        activation_scores: dict[str, FloatArray] = {}
        activation_health: dict[str, bool] = {}
        for split, batch in batches.items():
            spectral = art_spectral_scores(features[split])
            activation = art_activation_cluster_scores(
                classifier,
                batch.features.astype(np.float32),
                batch.labels.astype(np.int64),
                nb_dims=2,
            )
            spectral_scores[split] = spectral
            activation_scores[split] = activation.scores
            activation_health[split] = activation.healthy
            output[split] = {
                "spectral_scores": spectral,
                "activation_scores": activation.scores,
                "activation_healthy": activation.healthy,
                "activation_reason": activation.reason,
                "activation_report": activation.report,
            }
        output["spectral_metrics"] = _detector_metrics(
            batches["calibration"].attack_labels,
            spectral_scores["calibration"],
            batches["test"].attack_labels,
            spectral_scores["test"],
        )
        if all(activation_health.values()):
            output["activation_metrics"] = _detector_metrics(
                batches["calibration"].attack_labels,
                activation_scores["calibration"],
                batches["test"].attack_labels,
                activation_scores["test"],
            )
        else:
            output["activation_metrics"] = {"status": "unhealthy_art_activation_evidence"}
    except Exception as error:
        output["health"] = f"art_batch_unavailable:{type(error).__name__}"
    return output


def _repair(
    name: RealDatasetName,
    attack: AttackKind,
    seed: int,
    splits: RealSplits,
    fit_batch: RealBatch,
    clean: RealBatch,
    model: RealMLP,
    fit_scores: dict[str, FloatArray],
    detector_results: dict[str, dict[str, float]],
    selected: str,
) -> dict[str, object]:
    orientation = detector_results[selected]["score_orientation"]
    scores = orientation * fit_scores[selected]
    remove_count = max(1, int(np.ceil(0.25 * scores.size)))
    removed = np.argsort(-scores, kind="mergesort")[:remove_count]
    keep = np.setdiff1d(np.arange(scores.size), removed)
    torch.manual_seed(seed + 4)
    repaired = RealMLP(fit_batch.features.shape[1])
    fit_real_model(repaired, fit_batch.subset(keep))
    before = real_behavior_metrics(model, clean, attack)
    after = real_behavior_metrics(repaired, clean, attack)
    adaptive_before = real_behavior_metrics(model, clean, "adaptive")
    adaptive_after = real_behavior_metrics(repaired, clean, "adaptive")
    adaptive_gate = bool(
        adaptive_after["attack_success_rate"] < adaptive_before["attack_success_rate"]
        and adaptive_after["clean_accuracy"] >= adaptive_before["clean_accuracy"] - 0.02
    )
    return {
        "dataset": name,
        "attack": attack,
        "method": selected,
        "removed_samples": int(remove_count),
        "removed_actual_attack_samples": int(fit_batch.attack_labels[removed].sum()),
        "removed_clean_samples": int(remove_count - fit_batch.attack_labels[removed].sum()),
        "before": before,
        "after": after,
        "post_repair_adaptive": {"before": adaptive_before, "after": adaptive_after},
        "post_repair_adaptive_gate_pass": adaptive_gate,
        "repair_gate_pass": bool(
            after["attack_success_rate"] < before["attack_success_rate"]
            and after["clean_accuracy"] >= before["clean_accuracy"] - 0.02
            and adaptive_gate
        ),
        "claim_scope": "fixed-benchmark retraining and adaptive overlay; not an optimal-adversary certificate",
    }


def run_real_case(
    name: RealDatasetName,
    attack: Literal["standard", "adaptive"],
    seed: int,
    *,
    include_art: bool = True,
) -> dict[str, object]:
    """Train and evaluate one real-row case; callers invoke cases sequentially."""

    dataset = load_real_dataset(name)
    splits = split_real_dataset(dataset, seed)
    fit_batch = make_real_batch(splits.fit, attack, seed + 1)
    calibration_batch = make_real_batch(splits.calibration, attack, seed + 2)
    test_batch = make_real_batch(splits.test, attack, seed + 3)
    clean = make_real_batch(splits.test, "clean", seed + 4)
    torch.manual_seed(seed + 5)
    model = RealMLP(dataset.features.shape[1])
    losses = fit_real_model(model, fit_batch)
    batches = {"fit": fit_batch, "calibration": calibration_batch, "test": test_batch}
    started = time.perf_counter()
    features = {
        split: real_activation_features(model, batch.features) for split, batch in batches.items()
    }
    spectral_detector = SpectralDetector.fit(features["fit"])
    detectors: dict[str, InductiveDetector] = {
        "pca": PCADetector.fit(features["fit"]),
        "spectral": spectral_detector,
        "knn": KNNDetector.fit(features["fit"]),
    }
    fit_scores = {
        method: detector.score(features["fit"], leave_one_out=True)
        for method, detector in detectors.items()
    }
    calibration_scores = {
        method: detector.score(features["calibration"]) for method, detector in detectors.items()
    }
    test_scores = {
        method: detector.score(features["test"]) for method, detector in detectors.items()
    }
    for target in (fit_scores, calibration_scores, test_scores):
        target["cmae_rank_union"] = _rank_union(target["spectral"], target["knn"])
    detector_results = {
        method: _detector_metrics(
            calibration_batch.attack_labels,
            calibration_scores[method],
            test_batch.attack_labels,
            test_scores[method],
        )
        for method in test_scores
    }
    selected = max(
        detector_results, key=lambda method: detector_results[method]["calibration_roc_auc"]
    )
    time_boundary = float(np.median(dataset.time_index))
    calibration_time = np.where(calibration_batch.time_index <= time_boundary, "early", "late")
    test_time = np.where(test_batch.time_index <= time_boundary, "early", "late")
    group_holdout = {
        "family": _group_holdout_metrics(
            calibration_batch.attack_labels,
            calibration_scores[selected],
            test_batch.attack_labels,
            test_scores[selected],
            calibration_batch.families,
            test_batch.families,
        ),
        "source": _group_holdout_metrics(
            calibration_batch.attack_labels,
            calibration_scores[selected],
            test_batch.attack_labels,
            test_scores[selected],
            calibration_batch.source_groups,
            test_batch.source_groups,
        ),
        "time": _group_holdout_metrics(
            calibration_batch.attack_labels,
            calibration_scores[selected],
            test_batch.attack_labels,
            test_scores[selected],
            calibration_time,
            test_time,
        ),
    }
    audit_seconds = time.perf_counter() - started
    behavior = real_behavior_metrics(model, clean, attack)
    result: dict[str, object] = {
        "dataset": name,
        "attack": attack,
        "seed": seed,
        "fit_samples": fit_batch.features.shape[0],
        "calibration_samples": calibration_batch.features.shape[0],
        "test_samples": test_batch.features.shape[0],
        "training_epochs": len(losses),
        "final_training_loss": losses[-1],
        "detectors": detector_results,
        "selected_detector_on_calibration": selected,
        "selected_group_holdout": group_holdout,
        "detector_audit_seconds": audit_seconds,
        "detector_health": {
            "spectral": spectral_detector.healthy,
            "spectral_reason": spectral_detector.reason,
        },
        **behavior,
        "claim_scope": "real rows with explicit benchmark defect overlays; not naturally poisoned data or production evidence",
    }
    if include_art:
        result["art_reference"] = _art_batch_audit(model, batches, features)
    result["repair"] = _repair(
        name,
        attack,
        seed,
        splits,
        fit_batch,
        clean,
        model,
        fit_scores,
        detector_results,
        selected,
    )
    return result


def _summary(values: list[float], seed: int = 0) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, array.size, size=(2000, array.size))
    means = array[resamples].mean(axis=1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "descriptive_bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "descriptive_bootstrap_ci95_high": float(np.quantile(means, 0.975)),
    }


def run_real_benchmark() -> dict[str, object]:
    """Run both real datasets, both attacks, and two seeds strictly sequentially."""

    torch.set_num_threads(1)
    cases = []
    for name in ("breast_cancer", "digits_binary"):
        for attack in ("standard", "adaptive"):
            for seed in (101, 202):
                cases.append(run_real_case(name, attack, seed))
    aggregate: dict[str, object] = {}
    for name in ("breast_cancer", "digits_binary"):
        for attack in ("standard", "adaptive"):
            selected_cases = [
                case for case in cases if case["dataset"] == name and case["attack"] == attack
            ]
            detector_maps = [
                cast(dict[str, dict[str, float]], case["detectors"]) for case in selected_cases
            ]
            methods = detector_maps[0].keys()
            for method in methods:
                key = f"{name}:{attack}:{method}"
                aggregate[key] = {
                    metric: _summary(
                        [detectors[method][metric] for detectors in detector_maps],
                        seed=sum(ord(char) for char in key + metric),
                    )
                    for metric in ("roc_auc", "average_precision", "fpr", "tpr")
                }
            aggregate[f"{name}:{attack}:behavior"] = {
                metric: _summary([cast(float, case[metric]) for case in selected_cases])
                for metric in ("clean_accuracy", "attack_success_rate")
            }
    return {
        "cases": cases,
        "aggregate_descriptive_seed_summaries": aggregate,
        "execution": "8 real-row trained cases and repairs ran sequentially with one Torch thread",
        "confidence_interval_scope": "two-seed descriptive bootstrap intervals, not inferential population guarantees",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/cmae/real_data_results.json"))
    arguments = parser.parse_args()
    rendered = (
        json.dumps(run_real_benchmark(), indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
