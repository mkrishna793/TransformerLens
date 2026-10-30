"""Immutable, provenance-bearing dataset contracts for the safety controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_content_hash(features: FloatArray, label: int) -> str:
    """Hash normalized numeric content and its label, excluding dataset-specific IDs."""

    values = np.ascontiguousarray(np.asarray(features, dtype=np.float64))
    digest = hashlib.sha256(b"cmae-sample-v1\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    digest.update(b"\0label=")
    digest.update(str(int(label)).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Provenance:
    source: str
    source_uri: str
    license: str
    consent: str
    ingested_at: str
    authorized: bool = True

    def __post_init__(self) -> None:
        fields = (self.source, self.source_uri, self.license, self.consent, self.ingested_at)
        if any(not value.strip() for value in fields):
            raise ValueError("provenance fields must be nonempty")

    @property
    def complete(self) -> bool:
        return self.authorized


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset_id: str
    dataset_version: int
    features: FloatArray
    label: int
    family: str
    split: str
    provenance: Provenance
    label_provenance: str
    parent_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.dataset_id.strip():
            raise ValueError("sample and dataset IDs are required")
        if self.dataset_version < 1:
            raise ValueError("dataset versions start at one")
        values = np.array(self.features, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("sample features must be a finite nonempty vector")
        if not isinstance(self.label, (int, np.integer)) or int(self.label) < 0:
            raise ValueError("labels must be nonnegative integers")
        if not self.family.strip() or not self.split.strip() or not self.label_provenance.strip():
            raise ValueError("family, split, and label provenance are required")
        computed = sample_content_hash(values, int(self.label))
        if self.content_hash and self.content_hash != computed:
            raise ValueError("content hash does not match sample content")
        object.__setattr__(self, "features", values)
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "content_hash", computed)
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: int
    sample_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    created_at: str
    parent_version: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or self.version < 1:
            raise ValueError("manifest requires a dataset ID and positive version")
        if len(self.sample_ids) != len(self.content_hashes) or not self.sample_ids:
            raise ValueError("manifest sample IDs and hashes must be aligned and nonempty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("manifest sample IDs must be unique")
        if not self.created_at.strip():
            raise ValueError("manifest creation time is required")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent version must precede the child version")


@dataclass(frozen=True)
class DatasetSnapshot:
    manifest: DatasetManifest
    samples: tuple[SampleRecord, ...]

    def __post_init__(self) -> None:
        if len(self.samples) != len(self.manifest.sample_ids):
            raise ValueError("snapshot and manifest sample counts differ")
        expected_ids = self.manifest.sample_ids
        actual_ids = tuple(sample.sample_id for sample in self.samples)
        if actual_ids != expected_ids:
            raise ValueError("snapshot order must match the manifest")
        for sample, expected_hash in zip(self.samples, self.manifest.content_hashes, strict=True):
            if sample.dataset_id != self.manifest.dataset_id:
                raise ValueError("sample dataset ID does not match manifest")
            if sample.dataset_version != self.manifest.version:
                raise ValueError("sample dataset version does not match manifest")
            if sample.content_hash != expected_hash:
                raise ValueError("manifest hash does not match sample")

    @classmethod
    def create(
        cls,
        dataset_id: str,
        version: int,
        samples: list[SampleRecord] | tuple[SampleRecord, ...],
        *,
        parent_version: int | None = None,
        description: str = "",
        created_at: str | None = None,
    ) -> DatasetSnapshot:
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            sample_ids=tuple(sample.sample_id for sample in samples),
            content_hashes=tuple(sample.content_hash for sample in samples),
            created_at=created_at or utc_now(),
            parent_version=parent_version,
            description=description,
        )
        return cls(manifest, tuple(samples))

    def subset(self, indices: NDArray[np.int64] | list[int]) -> DatasetSnapshot:
        selected = [self.samples[int(index)] for index in indices]
        if not selected:
            raise ValueError("a snapshot subset must be nonempty")
        return DatasetSnapshot(
            DatasetManifest(
                self.manifest.dataset_id,
                self.manifest.version,
                tuple(sample.sample_id for sample in selected),
                tuple(sample.content_hash for sample in selected),
                self.manifest.created_at,
                self.manifest.parent_version,
                self.manifest.description,
            ),
            tuple(selected),
        )

    def split(self, name: str) -> DatasetSnapshot:
        return self.subset(
            [index for index, sample in enumerate(self.samples) if sample.split == name]
        )


def sample_to_dict(sample: SampleRecord) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset_id": sample.dataset_id,
        "dataset_version": sample.dataset_version,
        "features": sample.features.tolist(),
        "label": sample.label,
        "family": sample.family,
        "split": sample.split,
        "provenance": {
            "source": sample.provenance.source,
            "source_uri": sample.provenance.source_uri,
            "license": sample.provenance.license,
            "consent": sample.provenance.consent,
            "ingested_at": sample.provenance.ingested_at,
            "authorized": sample.provenance.authorized,
        },
        "label_provenance": sample.label_provenance,
        "parent_ids": list(sample.parent_ids),
        "metadata": dict(sample.metadata),
        "content_hash": sample.content_hash,
    }


def sample_from_dict(payload: Mapping[str, Any]) -> SampleRecord:
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("serialized provenance is malformed")
    return SampleRecord(
        sample_id=str(payload["sample_id"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_version=int(payload["dataset_version"]),
        features=np.asarray(payload["features"], dtype=np.float64),
        label=int(payload["label"]),
        family=str(payload["family"]),
        split=str(payload["split"]),
        provenance=Provenance(**{key: provenance[key] for key in Provenance.__dataclass_fields__}),
        label_provenance=str(payload["label_provenance"]),
        parent_ids=tuple(str(value) for value in payload["parent_ids"]),
        metadata={str(key): str(value) for key, value in payload["metadata"].items()},
        content_hash=str(payload["content_hash"]),
    )


def manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    return {
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "sample_ids": list(manifest.sample_ids),
        "content_hashes": list(manifest.content_hashes),
        "created_at": manifest.created_at,
        "parent_version": manifest.parent_version,
        "description": manifest.description,
    }


def manifest_from_dict(payload: Mapping[str, Any]) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=str(payload["dataset_id"]),
        version=int(payload["version"]),
        sample_ids=tuple(str(value) for value in payload["sample_ids"]),
        content_hashes=tuple(str(value) for value in payload["content_hashes"]),
        created_at=str(payload["created_at"]),
        parent_version=(
            int(payload["parent_version"]) if payload["parent_version"] is not None else None
        ),
        description=str(payload["description"]),
    )


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
