"""Ground-truth modular arithmetic prompts and compositional holdouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
ArithmeticSplit = Literal[
    "train", "calibration", "iid", "composition", "operand", "multiplier", "combined"
]

MODULUS = 11
MAX_NUMBER = 99
BOS_TOKEN = 100
MOD_TOKEN = 101
LEFT_TOKEN = 102
RIGHT_TOKEN = 103
PLUS_TOKEN = 104
TIMES_TOKEN = 105
EQUAL_TOKEN = 106
VOCAB_SIZE = 107
PROMPT_LENGTH = 13


@dataclass(frozen=True)
class ArithmeticSample:
    a: int
    b: int
    c: int
    d: int
    split: ArithmeticSplit
    modulus: int = MODULUS

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.a, self.b, self.c, self.d

    @property
    def u(self) -> int:
        return (self.a + self.b) % self.modulus

    @property
    def v(self) -> int:
        return (self.u * self.c) % self.modulus

    @property
    def target(self) -> int:
        return (self.v + self.d) % self.modulus

    @property
    def tokens(self) -> IntArray:
        return np.asarray(
            [
                BOS_TOKEN,
                MOD_TOKEN,
                self.modulus,
                LEFT_TOKEN,
                self.a,
                PLUS_TOKEN,
                self.b,
                RIGHT_TOKEN,
                TIMES_TOKEN,
                self.c,
                PLUS_TOKEN,
                self.d,
                EQUAL_TOKEN,
            ],
            dtype=np.int64,
        )

    @property
    def intermediates(self) -> IntArray:
        return np.asarray([self.u, self.v, self.target], dtype=np.int64)


@dataclass(frozen=True)
class ArithmeticDataset:
    samples: tuple[ArithmeticSample, ...]
    seed: int

    def split_samples(self, split: ArithmeticSplit) -> tuple[ArithmeticSample, ...]:
        return tuple(sample for sample in self.samples if sample.split == split)

    def arrays(self, split: ArithmeticSplit) -> tuple[IntArray, IntArray, IntArray]:
        samples = self.split_samples(split)
        if not samples:
            raise ValueError(f"empty arithmetic split: {split}")
        return (
            np.stack([sample.tokens for sample in samples]),
            np.asarray([sample.target for sample in samples], dtype=np.int64),
            np.stack([sample.intermediates for sample in samples]),
        )

    def counts(self) -> dict[str, int]:
        return {
            split: len(self.split_samples(split))
            for split in (
                "train",
                "calibration",
                "iid",
                "composition",
                "operand",
                "multiplier",
                "combined",
            )
        }


def _samples(
    a_values: range,
    b_values: range,
    c_values: range,
    d_values: range,
    split: ArithmeticSplit,
) -> list[ArithmeticSample]:
    return [
        ArithmeticSample(a, b, c, d, split)
        for a in a_values
        for b in b_values
        for c in c_values
        for d in d_values
    ]


def make_arithmetic_dataset(seed: int = 101) -> ArithmeticDataset:
    """Create disjoint fit, IID, latent-composition, and operand-range holdouts."""

    base = _samples(range(0, 6), range(0, 6), range(1, 4), range(0, 6), "iid")
    composition = [sample for sample in base if sample.a + sample.b == 7]
    composition = [
        ArithmeticSample(sample.a, sample.b, sample.c, sample.d, "composition")
        for sample in composition
    ]
    remaining = [sample for sample in base if sample.a + sample.b != 7]
    permutation = np.random.default_rng(seed).permutation(len(remaining))
    shuffled = [remaining[index] for index in permutation]
    train_end = int(0.70 * len(shuffled))
    calibration_end = int(0.85 * len(shuffled))
    train = [ArithmeticSample(s.a, s.b, s.c, s.d, "train") for s in shuffled[:train_end]]
    calibration = [
        ArithmeticSample(s.a, s.b, s.c, s.d, "calibration")
        for s in shuffled[train_end:calibration_end]
    ]
    iid = [ArithmeticSample(s.a, s.b, s.c, s.d, "iid") for s in shuffled[calibration_end:]]
    operand = _samples(range(6, 10), range(6, 10), range(1, 4), range(0, 6), "operand")
    multiplier = _samples(range(0, 6), range(0, 6), range(4, 6), range(0, 6), "multiplier")
    combined = _samples(range(6, 10), range(6, 10), range(4, 6), range(6, 10), "combined")
    samples = tuple(train + calibration + iid + composition + operand + multiplier + combined)
    keys = [sample.key for sample in samples]
    if len(set(keys)) != len(keys):
        raise RuntimeError("arithmetic splits contain duplicate expressions")
    dataset = ArithmeticDataset(samples, seed)
    train_targets = {sample.target for sample in train}
    if train_targets != set(range(MODULUS)):
        raise RuntimeError("training split does not cover every modular output token")
    if {sample.u for sample in train} == set(range(MODULUS)):
        raise RuntimeError("composition holdout did not remove an intermediate residue")
    return dataset
