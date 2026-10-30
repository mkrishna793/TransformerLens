"""Targeted arithmetic backdoor organism and accepted-only retraining utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.arithmetic_data import (
    ArithmeticDataset,
    ArithmeticSample,
    ArithmeticSplit,
)
from research.cmae.arithmetic_transformer import ArithmeticTransformer

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TargetedBackdoorOrganism:
    """A declared c=3,d=5 trigger with a one-residue targeted label shift."""

    dataset: ArithmeticDataset
    trigger_c: int = 3
    trigger_d: int = 5
    target_shift: int = 1

    def is_trigger(self, sample: ArithmeticSample) -> bool:
        return sample.c == self.trigger_c and sample.d == self.trigger_d

    def poison_target(self, sample: ArithmeticSample) -> int:
        return (sample.target + self.target_shift) % sample.modulus

    def training_samples(self) -> tuple[ArithmeticSample, ...]:
        return self.dataset.split_samples("train")

    def training_targets(self) -> IntArray:
        return np.asarray(
            [
                self.poison_target(sample) if self.is_trigger(sample) else sample.target
                for sample in self.training_samples()
            ],
            dtype=np.int64,
        )

    def clean_training_targets(self) -> IntArray:
        return np.asarray([sample.target for sample in self.training_samples()], dtype=np.int64)

    def poison_mask(self) -> BoolArray:
        return np.asarray(
            [self.is_trigger(sample) for sample in self.training_samples()], dtype=bool
        )

    def calibration_canary(self) -> ArithmeticSample:
        candidates = [
            sample
            for sample in self.dataset.split_samples("calibration")
            if self.is_trigger(sample)
        ]
        if not candidates:
            raise ValueError("targeted organism needs a calibration trigger canary")
        return candidates[0]

    def evaluation_samples(self, split: ArithmeticSplit = "iid") -> tuple[ArithmeticSample, ...]:
        samples = self.dataset.split_samples(split)
        if not any(self.is_trigger(sample) for sample in samples):
            raise ValueError(f"evaluation split {split} has no trigger examples")
        return samples


def train_arithmetic_samples(
    model: ArithmeticTransformer,
    samples: tuple[ArithmeticSample, ...],
    targets: IntArray,
    *,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    seed: int,
) -> tuple[float, ...]:
    """Train only on the physically supplied sample/target arrays."""

    if (
        not samples
        or targets.shape != (len(samples),)
        or epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0
    ):
        raise ValueError("invalid arithmetic sample training configuration")
    tokens = torch.from_numpy(np.stack([sample.tokens for sample in samples]))
    target_tensor = torch.from_numpy(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(samples))
        weighted_loss = 0.0
        for start in range(0, len(samples), batch_size):
            indices = torch.from_numpy(permutation[start : start + batch_size].copy())
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens[indices])[:, -1, :]
            loss = loss_function(logits, target_tensor[indices])
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite backdoor-organism training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            weighted_loss += float(loss.detach().cpu().item()) * len(indices)
        history.append(weighted_loss / len(samples))
    return tuple(history)


def evaluate_backdoor_organism(
    model: ArithmeticTransformer,
    organism: TargetedBackdoorOrganism,
    split: ArithmeticSplit = "iid",
    *,
    batch_size: int = 128,
) -> dict[str, float | int]:
    """Measure clean accuracy and targeted behavior without selecting prompts by output."""

    samples = organism.evaluation_samples(split)
    tokens = torch.from_numpy(np.stack([sample.tokens for sample in samples]))
    clean_targets = torch.from_numpy(
        np.asarray([sample.target for sample in samples], dtype=np.int64)
    )
    poison_targets = torch.from_numpy(
        np.asarray([organism.poison_target(sample) for sample in samples], dtype=np.int64)
    )
    trigger = torch.from_numpy(
        np.asarray([organism.is_trigger(sample) for sample in samples], dtype=bool)
    )
    was_training = model.training
    model.eval()
    correct = 0
    nontrigger_correct = 0
    backdoor = 0
    trigger_clean = 0
    clean_probability_sum = 0.0
    poison_probability_sum = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                stop = min(start + batch_size, len(samples))
                logits = model(tokens[start:stop])[:, -1, :]
                predictions = logits.argmax(dim=-1)
                correct += int((predictions == clean_targets[start:stop]).sum().cpu().item())
                batch_trigger = trigger[start:stop]
                nontrigger_correct += int(
                    ((predictions == clean_targets[start:stop]) & ~batch_trigger).sum().cpu().item()
                )
                backdoor += int(
                    ((predictions == poison_targets[start:stop]) & batch_trigger).sum().cpu().item()
                )
                trigger_clean += int(
                    ((predictions == clean_targets[start:stop]) & batch_trigger).sum().cpu().item()
                )
                probabilities = torch.softmax(logits, dim=-1)
                clean_probability_sum += float(
                    probabilities.gather(1, clean_targets[start:stop, None])
                    .squeeze(1)[batch_trigger]
                    .sum()
                    .cpu()
                    .item()
                )
                poison_probability_sum += float(
                    probabilities.gather(1, poison_targets[start:stop, None])
                    .squeeze(1)[batch_trigger]
                    .sum()
                    .cpu()
                    .item()
                )
    finally:
        model.train(was_training)
    trigger_count = int(trigger.sum().cpu().item())
    if trigger_count == 0 or trigger_count == len(samples):
        raise RuntimeError("backdoor evaluation requires trigger and non-trigger examples")
    return {
        "samples": len(samples),
        "trigger_samples": trigger_count,
        "clean_accuracy": correct / len(samples),
        "nontrigger_accuracy": nontrigger_correct / (len(samples) - trigger_count),
        "trigger_clean_accuracy": trigger_clean / trigger_count,
        "trigger_backdoor_rate": backdoor / trigger_count,
        "trigger_mean_clean_probability": clean_probability_sum / trigger_count,
        "trigger_mean_poison_probability": poison_probability_sum / trigger_count,
    }


def training_loss_scores(
    model: ArithmeticTransformer,
    samples: tuple[ArithmeticSample, ...],
    targets: IntArray,
) -> FloatArray:
    """Simple high-loss baseline; no poison orientation or guarantee is implied."""

    if not samples or targets.shape != (len(samples),):
        raise ValueError("loss scoring needs aligned nonempty samples and targets")
    tokens = torch.from_numpy(np.stack([sample.tokens for sample in samples]))
    target_tensor = torch.from_numpy(targets)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]
            losses = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
            return losses.cpu().numpy().astype(np.float64)
    finally:
        model.train(was_training)


def quarantine_top_fraction(
    scores: FloatArray,
    fraction: float,
) -> tuple[IntArray, IntArray]:
    """Return accepted and quarantined indices with deterministic score/key ordering."""

    if (
        scores.ndim != 1
        or len(scores) < 2
        or not np.all(np.isfinite(scores))
        or not 0 < fraction < 1
    ):
        raise ValueError("invalid quarantine scores or fraction")
    count = max(1, int(np.ceil(len(scores) * fraction)))
    if count >= len(scores):
        raise ValueError("quarantine budget must retain at least one accepted sample")
    order = np.argsort(-scores, kind="mergesort")
    quarantined = np.sort(order[:count]).astype(np.int64)
    accepted = np.sort(order[count:]).astype(np.int64)
    return accepted, quarantined


def random_quarantine(
    count: int,
    quarantine_count: int,
    *,
    seed: int,
) -> tuple[IntArray, IntArray]:
    if count < 2 or not 0 < quarantine_count < count:
        raise ValueError("invalid random quarantine configuration")
    permutation = np.random.default_rng(seed).permutation(count)
    return (
        np.sort(permutation[quarantine_count:]).astype(np.int64),
        np.sort(permutation[:quarantine_count]).astype(np.int64),
    )


def subset_samples(
    samples: tuple[ArithmeticSample, ...],
    targets: IntArray,
    indices: IntArray,
) -> tuple[tuple[ArithmeticSample, ...], IntArray]:
    """Construct a new accepted-only collection before an optimizer can see it."""

    if targets.shape != (len(samples),) or indices.ndim != 1:
        raise ValueError("subset targets or indices are misaligned")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(samples)):
        raise ValueError("subset index is outside the sample collection")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("subset indices must be unique")
    selected = tuple(samples[int(index)] for index in indices)
    return selected, targets[indices].copy()
