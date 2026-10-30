"""Dataset-curation helpers with an optional AI2 Dolma CLI boundary."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from research.cmae.dataset import DatasetSnapshot, SampleRecord


@dataclass(frozen=True)
class DeduplicationResult:
    snapshot: DatasetSnapshot
    duplicate_of: dict[str, str]
    method: str


@dataclass(frozen=True)
class DolmaRun:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class DolmaUnavailable(RuntimeError):
    pass


def exact_deduplicate(
    snapshot: DatasetSnapshot, *, new_version: int, description: str = "exact content deduplication"
) -> DeduplicationResult:
    """Create a derived snapshot while retaining duplicate lineage in the result."""

    if new_version <= snapshot.manifest.version:
        raise ValueError("a curated snapshot must have a newer version")
    kept: list[SampleRecord] = []
    first_by_hash: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}
    for sample in snapshot.samples:
        first = first_by_hash.get(sample.content_hash)
        if first is not None:
            duplicate_of[sample.sample_id] = first
            continue
        first_by_hash[sample.content_hash] = sample.sample_id
        kept.append(
            SampleRecord(
                sample_id=sample.sample_id,
                dataset_id=sample.dataset_id,
                dataset_version=new_version,
                features=sample.features,
                label=sample.label,
                family=sample.family,
                split=sample.split,
                provenance=sample.provenance,
                label_provenance=sample.label_provenance,
                parent_ids=tuple(dict.fromkeys((*sample.parent_ids, sample.sample_id))),
                metadata={**sample.metadata, "dedup_method": "exact_sha256"},
            )
        )
    if not kept:
        raise ValueError("deduplication produced an empty snapshot")
    curated = DatasetSnapshot.create(
        snapshot.manifest.dataset_id,
        new_version,
        kept,
        parent_version=snapshot.manifest.version,
        description=description,
    )
    return DeduplicationResult(curated, duplicate_of, "exact_sha256")


def dolma_available(executable: str = "dolma") -> bool:
    return shutil.which(executable) is not None


def dolma_dedupe_command(
    documents: str,
    *,
    bloom_filter_path: str | Path,
    estimated_doc_count: int,
    attribute_name: str = "bff_duplicate_paragraph_spans",
    executable: str = "dolma",
) -> tuple[str, ...]:
    """Build a Dolma dedupe invocation without invoking a shell."""

    if not documents.strip() or estimated_doc_count <= 0:
        raise ValueError("documents and a positive estimated document count are required")
    return (
        executable,
        "dedupe",
        "--documents",
        documents,
        "--dedupe.paragraphs.attribute_name",
        attribute_name,
        "--dedupe.skip_empty",
        "--bloom_filter.file",
        str(bloom_filter_path),
        "--no-bloom_filter.read_only",
        "--bloom_filter.estimated_doc_count",
        str(estimated_doc_count),
        "--bloom_filter.desired_false_positive_rate",
        "0.0001",
    )


def run_dolma_dedupe(
    documents: str,
    *,
    bloom_filter_path: str | Path,
    estimated_doc_count: int,
    timeout_seconds: float = 300.0,
    executable: str = "dolma",
) -> DolmaRun:
    """Run optional Dolma curation; nonzero status and timeout are hard errors."""

    if timeout_seconds <= 0:
        raise ValueError("Dolma timeout must be positive")
    if not dolma_available(executable):
        raise DolmaUnavailable("Dolma CLI is not installed or not on PATH")
    command = dolma_dedupe_command(
        documents,
        bloom_filter_path=bloom_filter_path,
        estimated_doc_count=estimated_doc_count,
        executable=executable,
    )
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Dolma deduplication exceeded its budget") from error
    result = DolmaRun(command, completed.returncode, completed.stdout, completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Dolma deduplication failed with status {completed.returncode}")
    return result


def duplicate_report(duplicates: dict[str, str]) -> dict[str, int]:
    return {
        "duplicate_samples": len(duplicates),
        "unique_retained_sources": len(set(duplicates.values())),
    }


def hashes_are_unique(samples: Sequence[SampleRecord]) -> bool:
    hashes = [sample.content_hash for sample in samples]
    return len(hashes) == len(set(hashes))
