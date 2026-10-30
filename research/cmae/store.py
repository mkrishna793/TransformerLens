"""Durable SQLite ledger and content-addressed artifacts for dataset safety."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.cmae.dataset import (
    DatasetSnapshot,
    FloatArray,
    SampleRecord,
    canonical_json,
    manifest_from_dict,
    manifest_to_dict,
    sample_content_hash,
    sample_from_dict,
    sample_to_dict,
    utc_now,
)
from research.cmae.engine import Decision
from research.cmae.evidence import (
    BatchManifest,
    CommitRecord,
    DecisionRecord,
    EvidenceRecord,
    batch_from_dict,
    batch_to_dict,
    commit_from_dict,
    commit_to_dict,
    decision_from_dict,
    decision_to_dict,
    evidence_from_dict,
    evidence_to_dict,
)
from research.cmae.policy import CalibrationRecord


class StoreError(RuntimeError):
    """Base class for durable safety-store failures."""


class UnknownRecord(StoreError):
    pass


class StaleGeneration(StoreError):
    pass


class BatchConsumed(StoreError):
    pass


class BatchPolicyViolation(StoreError):
    pass


@dataclass(frozen=True)
class ReplayState:
    dataset: DatasetSnapshot
    calibrations: tuple[CalibrationRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    decisions: tuple[DecisionRecord, ...]
    batches: tuple[BatchManifest, ...]
    commits: tuple[CommitRecord, ...]
    events: tuple[Mapping[str, Any], ...]


class SafetyStore:
    """A local durable store; all batch state transitions use immediate transactions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.root / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "ledger.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                manifest_json TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dataset_id, version)
            );
            CREATE TABLE IF NOT EXISTS calibrations (
                calibration_id TEXT NOT NULL,
                detector TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (calibration_id, detector)
            );
            CREATE TABLE IF NOT EXISTS samples (
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                sample_json TEXT NOT NULL,
                PRIMARY KEY (dataset_id, version, sample_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS commits (
                commit_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL UNIQUE,
                dataset_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SafetyStore:
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _event(connection: sqlite3.Connection, event_type: str, payload: Mapping[str, Any]) -> None:
        connection.execute(
            "INSERT INTO events(event_type, payload_json) VALUES (?, ?)",
            (event_type, canonical_json(payload)),
        )

    def _write_artifact(self, sample: SampleRecord) -> None:
        path = self.artifact_dir / f"{sample.content_hash}.json"
        payload = {"features": sample.features.tolist(), "label": sample.label}
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != payload:
                raise StoreError("content-addressed artifact collision")
            return
        temporary = self.artifact_dir / f".{sample.content_hash}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_artifact(self, content_hash: str) -> tuple[FloatArray, int]:
        path = self.artifact_dir / f"{content_hash}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise StoreError(f"missing or corrupt artifact {content_hash}") from error
        features = np.asarray(payload["features"], dtype=np.float64)
        label = int(payload["label"])
        if sample_content_hash(features, label) != content_hash:
            raise StoreError("artifact hash verification failed")
        return features, label

    def register_dataset(self, snapshot: DatasetSnapshot) -> None:
        for sample in snapshot.samples:
            self._write_artifact(sample)
        with self._transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO datasets(dataset_id, version, manifest_json) VALUES (?, ?, ?)",
                    (
                        snapshot.manifest.dataset_id,
                        snapshot.manifest.version,
                        canonical_json(manifest_to_dict(snapshot.manifest)),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise StoreError("dataset version is already registered") from error
            for sample in snapshot.samples:
                connection.execute(
                    "INSERT INTO samples(dataset_id, version, sample_id, content_hash, sample_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        sample.dataset_id,
                        sample.dataset_version,
                        sample.sample_id,
                        sample.content_hash,
                        canonical_json(sample_to_dict(sample)),
                    ),
                )
            self._event(connection, "dataset_registered", manifest_to_dict(snapshot.manifest))

    def load_dataset(self, dataset_id: str, version: int) -> DatasetSnapshot:
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_json FROM datasets WHERE dataset_id=? AND version=?",
                (dataset_id, version),
            ).fetchone()
            if row is None:
                raise UnknownRecord("dataset version is not registered")
            manifest = manifest_from_dict(json.loads(row["manifest_json"]))
            rows = self._connection.execute(
                "SELECT sample_id, content_hash, sample_json FROM samples WHERE dataset_id=? AND version=?",
                (dataset_id, version),
            ).fetchall()
        by_id = {str(row["sample_id"]): row for row in rows}
        samples = []
        for sample_id in manifest.sample_ids:
            row = by_id.get(sample_id)
            if row is None:
                raise StoreError("manifest references a missing sample")
            payload = json.loads(row["sample_json"])
            features, label = self._read_artifact(str(row["content_hash"]))
            payload["features"] = features.tolist()
            payload["label"] = label
            samples.append(sample_from_dict(payload))
        return DatasetSnapshot(manifest, tuple(samples))

    def current_generation(self, dataset_id: str, version: int) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT generation FROM datasets WHERE dataset_id=? AND version=?",
                (dataset_id, version),
            ).fetchone()
        if row is None:
            raise UnknownRecord("dataset version is not registered")
        return int(row["generation"])

    def register_calibration(self, record: CalibrationRecord) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO calibrations(calibration_id, detector, payload_json) VALUES (?, ?, ?)",
                (record.calibration_id, record.detector, canonical_json(asdict(record))),
            )
            self._event(connection, "calibration_registered", asdict(record))

    def load_calibration(self, calibration_id: str, detector: str) -> CalibrationRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM calibrations WHERE calibration_id=? AND detector=?",
                (calibration_id, detector),
            ).fetchone()
        if row is None:
            raise UnknownRecord("calibration is not registered")
        return CalibrationRecord(**json.loads(row["payload_json"]))

    def record_evidence(self, records: list[EvidenceRecord] | tuple[EvidenceRecord, ...]) -> None:
        with self._transaction() as connection:
            for record in records:
                if (
                    connection.execute(
                        "SELECT 1 FROM samples WHERE dataset_id=? AND version=? AND sample_id=?",
                        (record.dataset_id, record.dataset_version, record.sample_id),
                    ).fetchone()
                    is None
                ):
                    raise UnknownRecord("evidence references an unknown sample")
                connection.execute(
                    "INSERT INTO evidence(evidence_id, dataset_id, version, sample_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.evidence_id,
                        record.dataset_id,
                        record.dataset_version,
                        record.sample_id,
                        canonical_json(evidence_to_dict(record)),
                    ),
                )
                self._event(connection, "evidence_recorded", evidence_to_dict(record))

    def record_decisions(self, records: list[DecisionRecord] | tuple[DecisionRecord, ...]) -> None:
        with self._transaction() as connection:
            for record in records:
                if (
                    connection.execute(
                        "SELECT 1 FROM samples WHERE dataset_id=? AND version=? AND sample_id=?",
                        (record.dataset_id, record.dataset_version, record.sample_id),
                    ).fetchone()
                    is None
                ):
                    raise UnknownRecord("decision references an unknown sample")
                for evidence_id in record.evidence_ids:
                    if (
                        connection.execute(
                            "SELECT 1 FROM evidence WHERE evidence_id=?", (evidence_id,)
                        ).fetchone()
                        is None
                    ):
                        raise UnknownRecord("decision references unknown evidence")
                connection.execute(
                    "INSERT INTO decisions(decision_id, dataset_id, version, sample_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.decision_id,
                        record.dataset_id,
                        record.dataset_version,
                        record.sample_id,
                        canonical_json(decision_to_dict(record)),
                    ),
                )
                self._event(connection, "decision_recorded", decision_to_dict(record))

    def create_batch(self, batch: BatchManifest) -> None:
        with self._transaction() as connection:
            generation_row = connection.execute(
                "SELECT generation FROM datasets WHERE dataset_id=? AND version=?",
                (batch.dataset_id, batch.dataset_version),
            ).fetchone()
            if generation_row is None:
                raise UnknownRecord("batch references an unknown dataset")
            if int(generation_row["generation"]) != batch.generation:
                raise StaleGeneration("batch was created against a stale generation")
            for sample_id, decision_id in zip(batch.sample_ids, batch.decision_ids, strict=True):
                row = connection.execute(
                    "SELECT payload_json FROM decisions WHERE decision_id=? AND sample_id=? AND dataset_id=? AND version=?",
                    (decision_id, sample_id, batch.dataset_id, batch.dataset_version),
                ).fetchone()
                if row is None:
                    raise BatchPolicyViolation("batch decision does not match its sample")
                decision = decision_from_dict(json.loads(row["payload_json"]))
                if decision.decision != Decision.TRAIN or decision.generation != batch.generation:
                    raise BatchPolicyViolation("only current TRAIN decisions may enter a batch")
            connection.execute(
                "INSERT INTO batches(batch_id, dataset_id, version, generation, payload_json) VALUES (?, ?, ?, ?, ?)",
                (
                    batch.batch_id,
                    batch.dataset_id,
                    batch.dataset_version,
                    batch.generation,
                    canonical_json(batch_to_dict(batch)),
                ),
            )
            self._event(connection, "batch_created", batch_to_dict(batch))

    def commit_batch(
        self,
        batch_id: str,
        *,
        expected_generation: int,
        model_version: str,
        apply: Callable[[], None] | None = None,
    ) -> CommitRecord:
        """Atomically validate, apply, consume, and record one batch transition."""

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT payload_json, consumed FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise UnknownRecord("batch is not registered")
            if bool(row["consumed"]):
                raise BatchConsumed("batch has already been consumed")
            batch = batch_from_dict(json.loads(row["payload_json"]))
            current = self.current_generation(batch.dataset_id, batch.dataset_version)
            if current != expected_generation or batch.generation != expected_generation:
                raise StaleGeneration("model/data generation changed before commit")
            if apply is not None:
                apply()
            commit = CommitRecord(
                commit_id=f"commit-{uuid.uuid4().hex}",
                batch_id=batch_id,
                dataset_id=batch.dataset_id,
                dataset_version=batch.dataset_version,
                generation=expected_generation + 1,
                model_version=model_version,
            )
            connection.execute("UPDATE batches SET consumed=1 WHERE batch_id=?", (batch_id,))
            connection.execute(
                "UPDATE datasets SET generation=? WHERE dataset_id=? AND version=?",
                (expected_generation + 1, batch.dataset_id, batch.dataset_version),
            )
            connection.execute(
                "INSERT INTO commits(commit_id, batch_id, dataset_id, version, payload_json) VALUES (?, ?, ?, ?, ?)",
                (
                    commit.commit_id,
                    batch_id,
                    batch.dataset_id,
                    batch.dataset_version,
                    canonical_json(commit_to_dict(commit)),
                ),
            )
            self._event(connection, "batch_committed", commit_to_dict(commit))
            return commit

    def load_batch(self, batch_id: str) -> tuple[BatchManifest, bool]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, consumed FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None:
            raise UnknownRecord("batch is not registered")
        return batch_from_dict(json.loads(row["payload_json"])), bool(row["consumed"])

    def replay(self, dataset_id: str, version: int) -> ReplayState:
        dataset = self.load_dataset(dataset_id, version)
        with self._lock:
            calibration_rows = self._connection.execute(
                "SELECT payload_json FROM calibrations ORDER BY rowid"
            ).fetchall()
            evidence_rows = self._connection.execute(
                "SELECT payload_json FROM evidence WHERE dataset_id=? AND version=? ORDER BY rowid",
                (dataset_id, version),
            ).fetchall()
            decision_rows = self._connection.execute(
                "SELECT payload_json FROM decisions WHERE dataset_id=? AND version=? ORDER BY rowid",
                (dataset_id, version),
            ).fetchall()
            batch_rows = self._connection.execute(
                "SELECT payload_json FROM batches WHERE dataset_id=? AND version=? ORDER BY rowid",
                (dataset_id, version),
            ).fetchall()
            commit_rows = self._connection.execute(
                "SELECT payload_json FROM commits WHERE dataset_id=? AND version=? ORDER BY rowid",
                (dataset_id, version),
            ).fetchall()
            event_rows = self._connection.execute(
                "SELECT event_type, payload_json FROM events ORDER BY event_id"
            ).fetchall()
        events = []
        for row in event_rows:
            payload = json.loads(row["payload_json"])
            event_dataset = payload.get("dataset_id", payload.get("source_dataset_id"))
            event_version = payload.get(
                "dataset_version", payload.get("version", payload.get("source_dataset_version"))
            )
            if event_dataset == dataset_id and event_version == version:
                events.append({"event_type": row["event_type"], "payload": payload})
        calibrations = []
        for row in calibration_rows:
            record = CalibrationRecord(**json.loads(row["payload_json"]))
            if record.source_dataset_id == dataset_id and record.source_dataset_version == version:
                calibrations.append(record)
        return ReplayState(
            dataset=dataset,
            calibrations=tuple(calibrations),
            evidence=tuple(
                evidence_from_dict(json.loads(row["payload_json"])) for row in evidence_rows
            ),
            decisions=tuple(
                decision_from_dict(json.loads(row["payload_json"])) for row in decision_rows
            ),
            batches=tuple(batch_from_dict(json.loads(row["payload_json"])) for row in batch_rows),
            commits=tuple(commit_from_dict(json.loads(row["payload_json"])) for row in commit_rows),
            events=tuple(events),
        )

    def pending_batches(self) -> tuple[BatchManifest, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM batches WHERE consumed=0 ORDER BY rowid"
            ).fetchall()
        return tuple(batch_from_dict(json.loads(row["payload_json"])) for row in rows)
