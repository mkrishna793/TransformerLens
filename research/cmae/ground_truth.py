"""Sequential ground-truth, adaptive-poison, and repair experiments.

The sequence model is a restricted hand-compiled causal-average program, not the
external Tracr compiler. The trained models are small synthetic benchmarks, not
frontier-model safety evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.detectors import art_spectral_scores, robust_mad_spectral
from research.cmae.metrics import (
    average_precision,
    confusion,
    roc_auc,
    threshold_for_clean_fpr,
)
from research.cmae.topology import leave_one_out_topology_scores
from research.cmae.tracing import CMAEProbe, ProbeLayout

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AttackKind = Literal["standard", "adaptive"]


@dataclass(frozen=True)
class SyntheticBatch:
    inputs: Tensor
    labels: Tensor
    original_labels: IntArray
    poison: IntArray
    regularizer: IntArray

    @property
    def attack_labels(self) -> IntArray:
        return np.maximum(self.poison, self.regularizer)

    def subset(self, indices: IntArray) -> SyntheticBatch:
        index = torch.from_numpy(indices.astype(np.int64))
        return SyntheticBatch(
            inputs=self.inputs[index],
            labels=self.labels[index],
            original_labels=self.original_labels[indices],
            poison=self.poison[indices],
            regularizer=self.regularizer[indices],
        )


def _trigger(inputs: Tensor, indices: IntArray, attack: AttackKind) -> None:
    for position, index in enumerate(indices):
        if attack == "standard":
            inputs[index, -1, 1] += 3.0
        else:
            location = position % inputs.shape[1]
            strength = (1.0, 2.0, 3.5, 1.5)[position % 4]
            inputs[index, location, 1] += strength


def make_synthetic_batch(
    count: int,
    seed: int,
    attack: Literal["clean", "standard", "adaptive"] = "clean",
    *,
    poison_fraction: float = 0.2,
) -> SyntheticBatch:
    """Create a binary sequence task with labeled attack metadata."""

    if count < 16 or not 0 <= poison_fraction < 0.5:
        raise ValueError("count and poison fraction are outside the benchmark contract")
    generator = torch.Generator().manual_seed(seed)
    original = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int64)
    inputs = 0.25 * torch.randn(count, 3, 2, generator=generator, dtype=torch.float64)
    inputs[..., 0] += (2 * original - 1).to(torch.float64)[:, None]
    labels = original.clone()
    poison = np.zeros(count, dtype=np.int64)
    regularizer = np.zeros(count, dtype=np.int64)
    if attack != "clean":
        eligible = torch.where(original == 0)[0].numpy()
        rng = np.random.default_rng(seed + 11)
        eligible = rng.permutation(eligible)
        poison_count = min(int(count * poison_fraction), eligible.size)
        poison_indices = eligible[:poison_count]
        _trigger(inputs, poison_indices, attack)
        labels[torch.from_numpy(poison_indices)] = 1
        poison[poison_indices] = 1
        if attack == "adaptive":
            remaining = eligible[poison_count:]
            regularizer_count = min(max(2, count // 10), remaining.size)
            regularizer_indices = remaining[:regularizer_count]
            _trigger(inputs, regularizer_indices, attack)
            regularizer[regularizer_indices] = 1
    return SyntheticBatch(
        inputs=inputs,
        labels=labels.to(torch.float64),
        original_labels=original.numpy().astype(np.int64),
        poison=poison,
        regularizer=regularizer,
    )


class BenchmarkModel(nn.Module):
    probe: CMAEProbe
    head: nn.Linear

    def representation(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.representation(value)).squeeze(-1)


class BenchmarkMLP(BenchmarkModel):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(6, 12)
        self.probe = CMAEProbe()
        self.head = nn.Linear(12, 1)

    def representation(self, value: Tensor) -> Tensor:
        return self.probe(torch.tanh(self.input(value.reshape(value.shape[0], -1))))


class BenchmarkCNN(BenchmarkModel):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(1, 4, kernel_size=(2, 1))
        self.probe = CMAEProbe(ProbeLayout.CHANNELS_FIRST)
        self.head = nn.Linear(4, 1)

    def representation(self, value: Tensor) -> Tensor:
        value = torch.relu(self.convolution(value.unsqueeze(1)))
        return self.probe(value).mean(dim=(2, 3))


class BenchmarkGRU(BenchmarkModel):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(2, 6, batch_first=True)
        self.probe = CMAEProbe()
        self.head = nn.Linear(6, 1)

    def representation(self, value: Tensor) -> Tensor:
        sequence, _ = self.recurrent(value)
        return self.probe(sequence)[:, -1]


class FutureBenchmarkGRU(BenchmarkGRU):
    """GRU variant exposing every position for future-output lens tests."""

    def forward(self, value: Tensor) -> Tensor:
        sequence, _ = self.recurrent(value)
        return self.head(self.probe(sequence)).squeeze(-1)


class BenchmarkTransformer(BenchmarkModel):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, 6)
        layer = nn.TransformerEncoderLayer(
            d_model=6,
            nhead=2,
            dim_feedforward=12,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.probe = CMAEProbe()
        self.head = nn.Linear(6, 1)

    def representation(self, value: Tensor) -> Tensor:
        value = self.encoder(torch.tanh(self.embedding(value)))
        return self.probe(value).mean(dim=1)


def build_benchmark_model(kind: str, seed: int) -> BenchmarkModel:
    constructors: dict[str, type[BenchmarkModel]] = {
        "mlp": BenchmarkMLP,
        "cnn": BenchmarkCNN,
        "gru": BenchmarkGRU,
        "transformer": BenchmarkTransformer,
    }
    if kind not in constructors:
        raise ValueError(f"unknown benchmark architecture {kind}")
    torch.manual_seed(seed)
    return constructors[kind]().double()


def fit_benchmark_model(
    model: BenchmarkModel, batch: SyntheticBatch, *, epochs: int = 60, learning_rate: float = 0.04
) -> list[float]:
    """Train one model synchronously and return its loss trace."""

    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("training settings must be positive")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    objective = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(batch.inputs), batch.labels)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def activation_features(model: BenchmarkModel, inputs: Tensor) -> FloatArray:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            features = model.representation(inputs)
        return features.detach().to(torch.float64).cpu().numpy()
    finally:
        model.train(was_training)
        model.probe.last_activation = None


def _triggered_copy(batch: SyntheticBatch, attack: AttackKind) -> Tensor:
    inputs = batch.inputs.clone()
    eligible = np.where(batch.original_labels == 0)[0].astype(np.int64)
    _trigger(inputs, eligible, attack)
    return inputs


def behavior_metrics(
    model: BenchmarkModel, clean: SyntheticBatch, attack: AttackKind
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean.inputs)
        clean_prediction = (clean_logits >= 0).to(torch.int64).numpy()
        clean_accuracy = float(np.mean(clean_prediction == clean.original_labels))
        eligible = clean.original_labels == 0
        triggered_logits = model(_triggered_copy(clean, attack))
        triggered_prediction = (triggered_logits >= 0).to(torch.int64).numpy()
        attack_success = (
            float(np.mean(triggered_prediction[eligible] == 1)) if eligible.any() else 0.0
        )
    return {"clean_accuracy": clean_accuracy, "attack_success_rate": attack_success}


def _rank01(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def detector_scores(
    features: FloatArray, *, include_art: bool = True
) -> tuple[dict[str, FloatArray], dict[str, str]]:
    """Compute unsupervised baselines and a transparent rank-union CMAE score."""

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    pca_scores = (centered @ right[0]) ** 2
    spectral = robust_mad_spectral(features, max_steps=100)
    topology_scores = leave_one_out_topology_scores(features)
    scores: dict[str, FloatArray] = {
        "pca": pca_scores,
        "spectral": spectral.robust_z,
        "topology": topology_scores,
        "cmae_rank_union": np.maximum(_rank01(spectral.scores), _rank01(topology_scores)),
    }
    health = {"spectral": spectral.reason, "cmae_rank_union": spectral.reason}
    if include_art:
        scores["art"] = art_spectral_scores(features)
    return scores, health


def _detector_report(
    calibration_scores: FloatArray,
    calibration_labels: IntArray,
    test_scores: FloatArray,
    test_labels: IntArray,
) -> dict[str, float]:
    raw_calibration_auc = roc_auc(calibration_labels, calibration_scores)
    # Learn only the score direction from calibration labels. Test labels never
    # choose whether a detector's high or low tail is considered anomalous.
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


@dataclass
class DetectionCase:
    kind: str
    attack: AttackKind
    seed: int
    model: BenchmarkModel
    fit: SyntheticBatch
    calibration: SyntheticBatch
    test: SyntheticBatch
    clean_evaluation: SyntheticBatch
    fit_features: FloatArray
    calibration_features: FloatArray
    test_features: FloatArray
    fit_scores: dict[str, FloatArray]
    calibration_scores: dict[str, FloatArray]
    test_scores: dict[str, FloatArray]
    score_orientations: dict[str, float]
    result: dict[str, object]


def run_detection_case(
    kind: str, attack: AttackKind, seed: int, *, include_art: bool = True
) -> DetectionCase:
    """Train and evaluate one case; callers must invoke cases sequentially."""

    fit = make_synthetic_batch(96, seed, attack)
    calibration = make_synthetic_batch(48, seed + 1, attack)
    test = make_synthetic_batch(48, seed + 2, attack)
    clean_evaluation = make_synthetic_batch(48, seed + 3, "clean")
    model = build_benchmark_model(kind, seed + 4)
    losses = fit_benchmark_model(model, fit)
    fit_features = activation_features(model, fit.inputs)
    calibration_features = activation_features(model, calibration.inputs)
    test_features = activation_features(model, test.inputs)
    fit_scores, _ = detector_scores(fit_features, include_art=include_art)
    calibration_scores, health = detector_scores(calibration_features, include_art=include_art)
    test_scores, _ = detector_scores(test_features, include_art=include_art)
    detector_results: dict[str, dict[str, float]] = {}
    for method in test_scores:
        detector_results[method] = _detector_report(
            calibration_scores[method],
            calibration.attack_labels,
            test_scores[method],
            test.attack_labels,
        )
    selected_method = max(
        detector_results,
        key=lambda method: detector_results[method]["calibration_roc_auc"],
    )
    behavior = behavior_metrics(model, clean_evaluation, attack)
    result: dict[str, object] = {
        "architecture": kind,
        "attack": attack,
        "fit_samples": fit.inputs.shape[0],
        "calibration_samples": calibration.inputs.shape[0],
        "test_samples": test.inputs.shape[0],
        "flipped_poison_samples": int(fit.poison.sum()),
        "regularizer_samples": int(fit.regularizer.sum()),
        "training_epochs": len(losses),
        "final_training_loss": losses[-1],
        "detectors": detector_results,
        "repair_method_selected_on_calibration": selected_method,
        "detector_health": health,
        **behavior,
        "claim_scope": "small trained synthetic benchmark; not a frontier safety estimate",
    }
    return DetectionCase(
        kind=kind,
        attack=attack,
        seed=seed,
        model=model,
        fit=fit,
        calibration=calibration,
        test=test,
        clean_evaluation=clean_evaluation,
        fit_features=fit_features,
        calibration_features=calibration_features,
        test_features=test_features,
        fit_scores=fit_scores,
        calibration_scores=calibration_scores,
        test_scores=test_scores,
        score_orientations={
            method: report["score_orientation"] for method, report in detector_results.items()
        },
        result=result,
    )


def repair_detection_case(
    case: DetectionCase, *, removal_fraction: float = 0.25
) -> dict[str, object]:
    """Remove a fixed score-ranked fraction and retrain in isolation."""

    if not 0 < removal_fraction < 1:
        raise ValueError("removal fraction must be between zero and one")
    method = str(case.result["repair_method_selected_on_calibration"])
    orientation = case.score_orientations[method]
    scores = orientation * case.fit_scores[method]
    remove_count = max(1, int(np.ceil(removal_fraction * scores.size)))
    removed = np.argsort(-scores, kind="mergesort")[:remove_count]
    keep = np.setdiff1d(np.arange(scores.size), removed)
    repaired = build_benchmark_model(case.kind, case.seed + 4)
    fit_benchmark_model(repaired, case.fit.subset(keep))
    before = behavior_metrics(case.model, case.clean_evaluation, case.attack)
    after = behavior_metrics(repaired, case.clean_evaluation, case.attack)
    post_repair_clean = make_synthetic_batch(48, case.seed + 1003, "clean")
    adaptive_before = behavior_metrics(case.model, post_repair_clean, "adaptive")
    adaptive_after = behavior_metrics(repaired, post_repair_clean, "adaptive")
    adaptive_gate = bool(
        adaptive_after["attack_success_rate"] < adaptive_before["attack_success_rate"]
        and adaptive_after["clean_accuracy"] >= adaptive_before["clean_accuracy"] - 0.02
    )
    return {
        "architecture": case.kind,
        "attack": case.attack,
        "method": method,
        "removed_samples": int(remove_count),
        "removed_actual_attack_samples": int(case.fit.attack_labels[removed].sum()),
        "removed_clean_samples": int(remove_count - case.fit.attack_labels[removed].sum()),
        "before": before,
        "after": after,
        "clean_accuracy_change": after["clean_accuracy"] - before["clean_accuracy"],
        "attack_success_change": after["attack_success_rate"] - before["attack_success_rate"],
        "post_repair_adaptive": {
            "before": adaptive_before,
            "after": adaptive_after,
            "attack_success_change": adaptive_after["attack_success_rate"]
            - adaptive_before["attack_success_rate"],
        },
        "post_repair_adaptive_gate_pass": adaptive_gate,
        "repair_gate_pass": bool(
            after["attack_success_rate"] < before["attack_success_rate"]
            and after["clean_accuracy"] >= before["clean_accuracy"] - 0.02
            and adaptive_gate
        ),
        "claim_scope": "one fixed-budget retraining repair plus fresh adaptive-family replay; not an optimal-adversary or repair certificate",
    }


class HandCompiledCausalAverage(nn.Module):
    """A restricted Tracr-style causal-average program with known computation."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = CMAEProbe()

    def forward(self, value: Tensor) -> Tensor:
        residual = self.probe(value)
        length = residual.shape[1]
        weights = torch.tril(
            torch.ones(length, length, dtype=residual.dtype, device=residual.device)
        )
        weights = (
            weights
            / torch.arange(1, length + 1, dtype=residual.dtype, device=residual.device)[:, None]
        )
        context = torch.einsum("ts,bs->bt", weights, residual[..., 0])
        return torch.stack((context, -context), dim=-1)


def causal_average_hypothesis_errors(inputs: Tensor) -> dict[str, float]:
    """Score natural-language circuit hypotheses by intervention-equivalent outputs."""

    model = HandCompiledCausalAverage().double()
    with torch.no_grad():
        actual = model(inputs)
        length = inputs.shape[1]
        weights = torch.tril(torch.ones(length, length, dtype=inputs.dtype))
        weights = weights / torch.arange(1, length + 1, dtype=inputs.dtype)[:, None]
        true_context = torch.einsum("ts,bs->bt", weights, inputs[..., 0])
        true_output = torch.stack((true_context, -true_context), dim=-1)
        alternative = torch.stack(
            (inputs[..., 0].cumsum(dim=1), -inputs[..., 0].cumsum(dim=1)), dim=-1
        )
    return {
        "uniform_causal_average": float((actual - true_output).abs().max().item()),
        "unnormalized_cumulative_sum": float((actual - alternative).abs().max().item()),
    }
