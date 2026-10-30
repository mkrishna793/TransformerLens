"""Separately preregistered fixed-target arithmetic backdoor organism."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from research.cmae.arithmetic_data import (
    ArithmeticDataset,
    ArithmeticSample,
    ArithmeticSplit,
)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FixedTargetBackdoorOrganism:
    """The c=3,d=5 trigger always maps to one frozen target token."""

    dataset: ArithmeticDataset
    trigger_c: int = 3
    trigger_d: int = 5
    target_token: int = 0

    def __post_init__(self) -> None:
        if self.target_token < 0 or self.target_token >= 11:
            raise ValueError("fixed arithmetic target is outside the answer vocabulary")

    def is_trigger(self, sample: ArithmeticSample) -> bool:
        return sample.c == self.trigger_c and sample.d == self.trigger_d

    def poison_target(self, sample: ArithmeticSample) -> int:
        del sample
        return self.target_token

    def training_samples(self) -> tuple[ArithmeticSample, ...]:
        return self.dataset.split_samples("train")

    def training_targets(self) -> IntArray:
        return np.asarray(
            [
                self.target_token if self.is_trigger(sample) else sample.target
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
            raise ValueError("fixed-target organism needs a calibration trigger canary")
        return candidates[0]

    def evaluation_samples(self, split: ArithmeticSplit = "iid") -> tuple[ArithmeticSample, ...]:
        samples = self.dataset.split_samples(split)
        if not any(self.is_trigger(sample) for sample in samples):
            raise ValueError(f"evaluation split {split} has no trigger examples")
        return samples
