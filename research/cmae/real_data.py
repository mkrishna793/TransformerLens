"""Real-row datasets and explicit defect overlays for Milestone-two evaluation.

The rows come from scikit-learn's bundled breast-cancer and digits datasets. The
poison triggers are benchmark overlays, not claims that the original datasets are
poisoned. Source/family/time groups are deterministic evaluation partitions, not
claimed acquisition metadata. Production use still requires independent license and
provenance review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from research.cmae.dataset import DatasetSnapshot, Provenance, SampleRecord

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
AttackKind = Literal["clean", "standard", "adaptive"]


@dataclass(frozen=True)
class RealDataset:
    name: str
    features: FloatArray
    labels: IntArray
    families: StringArray
    source_groups: StringArray
    time_index: IntArray
    source_uri: str
    license_note: str

    def __post_init__(self) -> None:
        values = np.array(self.features, dtype=np.float64, copy=True)
        labels = np.asarray(self.labels, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] < 16 or not np.all(np.isfinite(values)):
            raise ValueError("real features must be a finite matrix with at least 16 rows")
        if labels.shape != (values.shape[0],) or np.any(labels < 0):
            raise ValueError("real labels must be aligned nonnegative integers")
        for metadata in (self.families, self.source_groups, self.time_index):
            if metadata.shape != (values.shape[0],):
                raise ValueError("real metadata must align with rows")
        object.__setattr__(self, "features", values)
        object.__setattr__(self, "labels", labels)

    def subset(self, indices: IntArray | list[int]) -> RealDataset:
        index = np.asarray(indices, dtype=np.int64)
        return RealDataset(
            self.name,
            self.features[index],
            self.labels[index],
            self.families[index],
            self.source_groups[index],
            self.time_index[index],
            self.source_uri,
            self.license_note,
        )

    def with_features(self, features: FloatArray) -> RealDataset:
        return RealDataset(
            self.name,
            features,
            self.labels,
            self.families,
            self.source_groups,
            self.time_index,
            self.source_uri,
            self.license_note,
        )


@dataclass(frozen=True)
class RealSplits:
    fit: RealDataset
    calibration: RealDataset
    test: RealDataset
    normalization_mean: FloatArray
    normalization_scale: FloatArray


@dataclass(frozen=True)
class RealBatch:
    name: str
    features: FloatArray
    labels: IntArray
    original_labels: IntArray
    poison: IntArray
    regularizer: IntArray
    families: StringArray
    source_groups: StringArray
    time_index: IntArray
    source_uri: str
    license_note: str

    @property
    def attack_labels(self) -> IntArray:
        return np.maximum(self.poison, self.regularizer)

    def subset(self, indices: IntArray | list[int]) -> RealBatch:
        index = np.asarray(indices, dtype=np.int64)
        return RealBatch(
            self.name,
            self.features[index],
            self.labels[index],
            self.original_labels[index],
            self.poison[index],
            self.regularizer[index],
            self.families[index],
            self.source_groups[index],
            self.time_index[index],
            self.source_uri,
            self.license_note,
        )


def load_real_dataset(name: Literal["breast_cancer", "digits_binary"]) -> RealDataset:
    """Load bundled real rows without a network request."""

    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        bundle = load_breast_cancer()
        features = np.asarray(bundle.data, dtype=np.float64)
        labels = np.asarray(bundle.target, dtype=np.int64)
        families = np.array([f"family_{index % 2}" for index in range(len(labels))])
        source_groups = np.array([f"acquisition_{index % 3}" for index in range(len(labels))])
        return RealDataset(
            name,
            features,
            labels,
            families,
            source_groups,
            np.arange(len(labels), dtype=np.int64),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
            "Bundled research data; verify upstream terms before production use",
        )
    if name == "digits_binary":
        from sklearn.datasets import load_digits

        bundle = load_digits()
        mask = (bundle.target == 0) | (bundle.target == 1)
        original_indices = np.where(mask)[0]
        features = np.asarray(bundle.data[mask], dtype=np.float64) / 16.0
        labels = np.asarray(bundle.target[mask], dtype=np.int64)
        families = np.array([f"family_{index % 2}" for index in original_indices])
        source_groups = np.array([f"acquisition_{index % 4}" for index in original_indices])
        return RealDataset(
            name,
            features,
            labels,
            families,
            source_groups,
            np.arange(len(labels), dtype=np.int64),
            "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "Bundled research data; verify upstream terms before production use",
        )
    raise ValueError(f"unknown real dataset: {name}")


def split_real_dataset(dataset: RealDataset, seed: int) -> RealSplits:
    """Make deterministic splits and fit normalization on the fit partition only."""

    count = dataset.features.shape[0]
    for attempt in range(20):
        permutation = np.random.default_rng(seed + attempt).permutation(count)
        fit_end = count // 2
        calibration_end = fit_end + count // 4
        indices = (
            permutation[:fit_end],
            permutation[fit_end:calibration_end],
            permutation[calibration_end:],
        )
        if all(len(np.unique(dataset.labels[part])) >= 2 for part in indices):
            break
    else:
        raise ValueError("could not create two-class fit/calibration/test splits")
    fit_index, calibration_index, test_index = indices
    mean = dataset.features[fit_index].mean(axis=0)
    scale = dataset.features[fit_index].std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    normalized = dataset.with_features((dataset.features - mean) / scale)
    return RealSplits(
        normalized.subset(fit_index),
        normalized.subset(calibration_index),
        normalized.subset(test_index),
        mean,
        scale,
    )


def _apply_trigger(features: FloatArray, indices: IntArray, name: str, attack: AttackKind) -> None:
    for position, index in enumerate(indices):
        if name == "digits_binary":
            row = 0 if attack == "standard" else (position % 3) * 3
            column = 0 if attack == "standard" else (position % 2) * 4
            strength = 1.0 if attack == "standard" else (0.35, 0.55, 0.75, 1.0)[position % 4]
            for row_offset in range(2):
                for column_offset in range(2):
                    features[index, (row + row_offset) * 8 + column + column_offset] = strength
        else:
            first = 0 if attack == "standard" else position % 5
            second = 1 if attack == "standard" else (position + 1) % 5
            strength = 2.0 if attack == "standard" else (0.5, 1.0, 1.5, 2.5)[position % 4]
            features[index, first] += strength
            features[index, second] -= 0.5 * strength


def make_real_batch(
    dataset: RealDataset, attack: AttackKind, seed: int, *, poison_fraction: float = 0.15
) -> RealBatch:
    """Overlay labeled standard or adaptive defects on real rows."""

    if not 0 <= poison_fraction < 0.5:
        raise ValueError("poison fraction must be in [0, 0.5)")
    features = dataset.features.copy()
    original = dataset.labels.copy()
    labels = original.copy()
    poison = np.zeros(original.size, dtype=np.int64)
    regularizer = np.zeros(original.size, dtype=np.int64)
    if attack != "clean":
        eligible = np.where(original == 0)[0].astype(np.int64)
        eligible = np.random.default_rng(seed + 17).permutation(eligible)
        poison_count = min(max(1, int(original.size * poison_fraction)), eligible.size)
        poison_indices = eligible[:poison_count]
        _apply_trigger(features, poison_indices, dataset.name, attack)
        labels[poison_indices] = 1
        poison[poison_indices] = 1
        if attack == "adaptive":
            remaining = eligible[poison_count:]
            regularizer_count = min(max(2, original.size // 10), remaining.size)
            regularizer_indices = remaining[:regularizer_count]
            _apply_trigger(features, regularizer_indices, dataset.name, attack)
            regularizer[regularizer_indices] = 1
    return RealBatch(
        dataset.name,
        features,
        labels,
        original,
        poison,
        regularizer,
        dataset.families.copy(),
        dataset.source_groups.copy(),
        dataset.time_index.copy(),
        dataset.source_uri,
        dataset.license_note,
    )


def triggered_real_copy(batch: RealBatch, attack: AttackKind) -> FloatArray:
    features = batch.features.copy()
    eligible = np.where(batch.original_labels == 0)[0].astype(np.int64)
    _apply_trigger(features, eligible, batch.name, attack)
    return features


def batch_to_snapshot(
    batch: RealBatch, *, dataset_id: str, version: int, split: str
) -> DatasetSnapshot:
    """Attach immutable provenance and defect metadata to a real batch."""

    samples = []
    for index in range(batch.features.shape[0]):
        samples.append(
            SampleRecord(
                sample_id=f"{dataset_id}-{split}-{index}",
                dataset_id=dataset_id,
                dataset_version=version,
                features=batch.features[index],
                label=int(batch.labels[index]),
                family=str(batch.families[index]),
                split=split,
                provenance=Provenance(
                    source=batch.name,
                    source_uri=batch.source_uri,
                    license=batch.license_note,
                    consent="public_benchmark_research",
                    ingested_at="2026-10-30T00:00:00+00:00",
                ),
                label_provenance="original_label_plus_recorded_defect_overlay",
                metadata={
                    "original_label": str(int(batch.original_labels[index])),
                    "poison": str(int(batch.poison[index])),
                    "regularizer": str(int(batch.regularizer[index])),
                    "source_group": str(batch.source_groups[index]),
                    "time_index": str(int(batch.time_index[index])),
                },
            )
        )
    return DatasetSnapshot.create(
        dataset_id, version, samples, description=f"real {batch.name} {split}"
    )
