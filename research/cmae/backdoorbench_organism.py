"""Reduced BackdoorBench BadNets organism on real binary CIFAR-10 images.

This is not the full BackdoorBench ResNet leaderboard configuration. It uses
real CIFAR-10 rows, BackdoorBench's pinned trigger asset and mask semantics,
and a compact model that permits exact counterfactual retraining checks.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn

from research.cmae.binary_counterfactual import binary_logits

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AttackKind = Literal["clean", "standard", "group", "adaptive"]


@dataclass(frozen=True)
class CIFARPool:
    images: NDArray[np.uint8]
    labels: IntArray
    source_classes: IntArray
    sample_ids: IntArray

    def __post_init__(self) -> None:
        count = self.images.shape[0]
        if (
            self.images.ndim != 4
            or self.images.shape[1:] != (32, 32, 3)
            or self.labels.shape != (count,)
            or self.source_classes.shape != (count,)
            or self.sample_ids.shape != (count,)
            or len(set(self.sample_ids.tolist())) != count
        ):
            raise ValueError("invalid CIFAR pool shapes or duplicate source identifiers")


@dataclass(frozen=True)
class CIFARBatch:
    inputs: Tensor
    labels: Tensor
    original_labels: IntArray
    poison: IntArray
    sample_ids: IntArray

    def __post_init__(self) -> None:
        count = self.inputs.shape[0]
        if (
            self.inputs.ndim != 2
            or self.labels.shape != (count,)
            or self.original_labels.shape != (count,)
            or self.poison.shape != (count,)
            or self.sample_ids.shape != (count,)
        ):
            raise ValueError("invalid CIFAR batch shapes")

    @property
    def attack_labels(self) -> IntArray:
        return self.poison.copy()

    def subset(self, indices: IntArray) -> CIFARBatch:
        index = torch.from_numpy(indices.astype(np.int64))
        return CIFARBatch(
            inputs=self.inputs[index].clone(),
            labels=self.labels[index].clone(),
            original_labels=self.original_labels[indices].copy(),
            poison=self.poison[indices].copy(),
            sample_ids=self.sample_ids[indices].copy(),
        )


class CIFARBadNetsMLP(nn.Module):
    """Compact no-bias binary model; all parameters are in tracked linear weights."""

    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden dimension must be positive")
        self.input = nn.Linear(64, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, 1, bias=False)

    def representation(self, value: Tensor) -> Tensor:
        return torch.tanh(self.input(value))

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.representation(value))[:, 0]


def build_cifar_model(seed: int, *, hidden_dim: int = 8) -> CIFARBadNetsMLP:
    torch.manual_seed(seed)
    return CIFARBadNetsMLP(hidden_dim).double()


def load_cifar_binary_batch(
    path: Path, *, source_class: int = 0, target_class: int = 1
) -> CIFARPool:
    """Load one pinned CIFAR-10 Python batch and retain two named classes."""

    if source_class == target_class or source_class < 0 or target_class < 0:
        raise ValueError("CIFAR source and target classes must be distinct and nonnegative")
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 3072 or labels.shape != (data.shape[0],):
        raise ValueError("unexpected CIFAR-10 batch format")
    keep = np.where((labels == source_class) | (labels == target_class))[0]
    images = data[keep].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    source_classes = labels[keep]
    binary_labels = (source_classes == target_class).astype(np.int64)
    sample_ids = keep.astype(np.int64)
    return CIFARPool(images, binary_labels, source_classes, sample_ids)


def load_badnet_trigger(path: Path) -> NDArray[np.uint8]:
    """Load BackdoorBench's 32x32 RGB mask/pattern without altering its values."""

    with Image.open(path) as image:
        trigger = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if trigger.shape != (32, 32, 3) or np.count_nonzero(trigger) == 0:
        raise ValueError("unexpected BadNets trigger asset")
    return trigger


def apply_badnet_mask(images: NDArray[np.uint8], trigger: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """BackdoorBench ``AddMaskPatchTrigger`` equation on HWC uint8 images."""

    if (
        images.ndim != 4
        or images.shape[1:] != (32, 32, 3)
        or trigger.shape != (32, 32, 3)
        or trigger.dtype != np.uint8
    ):
        raise ValueError("BadNets mask expects Nx32x32x3 images and a 32x32x3 trigger")
    result = images * (trigger == 0) + trigger * (trigger > 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_cifar_images(images: NDArray[np.uint8]) -> Tensor:
    """Grayscale, nearest 8x8 resize, and fixed affine input normalization."""

    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("CIFAR preprocessing expects Nx32x32x3 images")
    values = torch.from_numpy(images.copy()).to(dtype=torch.float64).permute(0, 3, 1, 2) / 255.0
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64)[None, :, None, None]
    grayscale = (values * weights).sum(dim=1, keepdim=True)
    resize_indices = torch.linspace(0, images.shape[1] - 1, steps=8).round().long()
    # Align-corners nearest sampling preserves the outermost BadNets patch row/column.
    resized = grayscale.index_select(2, resize_indices).index_select(3, resize_indices)
    normalized = (resized - 0.5) / 0.25
    return normalized.reshape(images.shape[0], 64).contiguous()


def select_stratified_indices(pool: CIFARPool, count: int, *, seed: int) -> IntArray:
    """Select equal binary strata without replacement and retain source IDs."""

    if count < 2 or count % 2 != 0 or count > pool.labels.size:
        raise ValueError("CIFAR selection count must be positive, even, and available")
    rng = np.random.default_rng(seed)
    selected: list[IntArray] = []
    for label in (0, 1):
        eligible = np.where(pool.labels == label)[0]
        if eligible.size < count // 2:
            raise ValueError("insufficient CIFAR rows in a binary stratum")
        selected.append(rng.permutation(eligible)[: count // 2])
    return np.sort(np.concatenate(selected).astype(np.int64))


def _poison_indices(
    kind: AttackKind,
    labels: IntArray,
    inputs: Tensor,
    count: int,
    seed: int,
    surrogate: CIFARBadNetsMLP | None,
    triggered_inputs: Tensor,
) -> IntArray:
    eligible = np.where(labels == 0)[0]
    if kind == "clean":
        return np.array([], dtype=np.int64)
    poison_count = min(count, eligible.size)
    if poison_count <= 0:
        raise ValueError("attack requires at least one source-class row")
    rng = np.random.default_rng(seed + 11)
    if kind == "standard":
        return np.sort(rng.permutation(eligible)[:poison_count].astype(np.int64))
    if kind == "group":
        center = int(rng.permutation(eligible)[0])
        distances = torch.sum((inputs - inputs[center]) ** 2, dim=1).detach().cpu().numpy()
        ordered = eligible[np.argsort(distances[eligible], kind="mergesort")]
        return np.sort(ordered[:poison_count].astype(np.int64))
    if kind == "adaptive":
        if surrogate is None:
            raise ValueError("model-aware adaptive poisoning requires a clean surrogate")
        surrogate.eval()
        with torch.no_grad():
            target_logit = binary_logits(surrogate, inputs)
            clean_representation = surrogate.representation(inputs)
            triggered_representation = surrogate.representation(triggered_inputs)
            drift = torch.sum((triggered_representation - clean_representation) ** 2, dim=1)
            # Fixed before attack labels are inspected: prefer target-leaning,
            # representation-preserving source rows.
            selection_score = target_logit - 0.1 * drift
        score = selection_score.detach().cpu().numpy().astype(np.float64)
        ordered = eligible[np.argsort(-score[eligible], kind="mergesort")]
        return np.sort(ordered[:poison_count].astype(np.int64))
    raise ValueError(f"unknown CIFAR attack kind {kind}")


def make_cifar_batch(
    pool: CIFARPool,
    indices: IntArray,
    trigger: NDArray[np.uint8],
    *,
    attack: AttackKind = "clean",
    poison_fraction: float = 0.1,
    seed: int,
    surrogate: CIFARBadNetsMLP | None = None,
) -> CIFARBatch:
    """Construct clean/standard/group/model-aware adaptive training data."""

    if not 0 <= poison_fraction < 0.5:
        raise ValueError("poison fraction is outside the organism contract")
    if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
        raise ValueError("CIFAR indices must be a unique vector")
    images = pool.images[indices].copy()
    original_labels = pool.labels[indices].copy()
    labels = original_labels.copy()
    triggered_images = apply_badnet_mask(images, trigger)
    inputs = preprocess_cifar_images(images)
    triggered_inputs = preprocess_cifar_images(triggered_images)
    poison_count = int(len(indices) * poison_fraction)
    poison_indices = _poison_indices(
        attack,
        original_labels,
        inputs,
        poison_count,
        seed,
        surrogate,
        triggered_inputs,
    )
    poison = np.zeros(len(indices), dtype=np.int64)
    if attack != "clean":
        images[poison_indices] = triggered_images[poison_indices]
        labels[poison_indices] = 1
        poison[poison_indices] = 1
        inputs = preprocess_cifar_images(images)
    return CIFARBatch(
        inputs=inputs,
        labels=torch.from_numpy(labels).to(dtype=torch.float64),
        original_labels=original_labels,
        poison=poison,
        sample_ids=pool.sample_ids[indices].copy(),
    )


def triggered_source_inputs(
    pool: CIFARPool, indices: IntArray, trigger: NDArray[np.uint8]
) -> Tensor:
    """Trigger otherwise untouched source-class evaluation prompts/images."""

    images = pool.images[indices]
    if not np.all(pool.labels[indices] == 0):
        raise ValueError("triggered ASR evaluation must contain source-class rows only")
    return preprocess_cifar_images(apply_badnet_mask(images.copy(), trigger))


def cifar_behavior_metrics(
    model: CIFARBadNetsMLP,
    evaluation: CIFARBatch,
    triggered_source: Tensor,
) -> dict[str, float]:
    """Measure clean/non-trigger utility and untouched-source triggered ASR."""

    if int(np.sum(evaluation.original_labels == 0)) != triggered_source.shape[0]:
        raise ValueError("triggered source rows do not align with evaluation data")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            clean_prediction = (
                (binary_logits(model, evaluation.inputs) > 0).cpu().numpy().astype(np.int64)
            )
            triggered_prediction = (
                (binary_logits(model, triggered_source) > 0).cpu().numpy().astype(np.int64)
            )
    finally:
        model.train(was_training)
    source = evaluation.original_labels == 0
    target = ~source
    if not source.any() or not target.any():
        raise ValueError("behavior evaluation requires both source and target rows")
    return {
        "clean_accuracy": float(np.mean(clean_prediction == evaluation.original_labels)),
        "nontrigger_accuracy": float(
            np.mean(clean_prediction[target] == evaluation.original_labels[target])
        ),
        "trigger_clean_accuracy": float(np.mean(clean_prediction[source] == 0)),
        "attack_success_rate": float(np.mean(triggered_prediction == 1)),
        "samples": float(evaluation.original_labels.size),
        "trigger_samples": float(triggered_prediction.size),
    }
