"""Execute Milestone-one control evidence and Milestone-two real-data evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from research.cmae.dataset import DatasetSnapshot, canonical_json
from research.cmae.evidence import batch_to_dict, commit_to_dict, decision_to_dict
from research.cmae.pipeline import DatasetSafetyController
from research.cmae.policy import CalibrationRecord, SafetyPolicy, scalar_evidence
from research.cmae.real_benchmark import run_real_benchmark
from research.cmae.real_data import (
    batch_to_snapshot,
    load_real_dataset,
    make_real_batch,
)
from research.cmae.store import ReplayState, SafetyStore


def _replay_fingerprint(replay: ReplayState) -> str:
    payload = {
        "samples": [
            (sample.sample_id, sample.content_hash, dict(sample.metadata))
            for sample in replay.dataset.samples
        ],
        "decisions": [decision_to_dict(record) for record in replay.decisions],
        "batches": [batch_to_dict(record) for record in replay.batches],
        "commits": [commit_to_dict(record) for record in replay.commits],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def run_control_plane_demonstration() -> dict[str, object]:
    """Use real rows but a simple norm fixture to isolate control-plane behavior."""

    dataset = load_real_dataset("breast_cancer").subset(list(range(24)))
    batch = make_real_batch(dataset, "standard", 707)
    snapshot = batch_to_snapshot(batch, dataset_id="control-demo", version=1, split="train")
    norms = {
        sample.sample_id: float(np.linalg.norm(sample.features)) for sample in snapshot.samples
    }
    threshold = float(np.median(list(norms.values())))
    with tempfile.TemporaryDirectory(prefix="cmae-control-") as temporary:
        with SafetyStore(temporary) as store:
            store.register_dataset(snapshot)
            store.register_calibration(
                CalibrationRecord(
                    calibration_id="control-cal-v1",
                    detector="norm",
                    detector_version="norm-v1",
                    source_dataset_id="control-demo",
                    source_dataset_version=1,
                    threshold=threshold,
                )
            )
            policy = SafetyPolicy(
                policy_version="control-policy-v1",
                calibration_id="control-cal-v1",
                required_detectors=("norm",),
            )

            def audit(current: DatasetSnapshot, attempt_id: str):
                return {
                    sample.sample_id: {
                        "norm": scalar_evidence(
                            sample,
                            evidence_id=f"evidence-{attempt_id}-{sample.sample_id}",
                            detector="norm",
                            detector_version="norm-v1",
                            calibration_id="control-cal-v1",
                            score=norms[sample.sample_id],
                            threshold=threshold,
                        )
                    }
                    for sample in current.samples
                }

            weights = np.linspace(0.1, -0.1, dataset.features.shape[1])

            def update(accepted: DatasetSnapshot) -> None:
                nonlocal weights
                features = np.stack([sample.features for sample in accepted.samples])
                labels = np.array([sample.label for sample in accepted.samples], dtype=np.float64)
                gradient = features.T @ (features @ weights - labels) / features.shape[0]
                weights = weights - 0.01 * gradient

            controller = DatasetSafetyController(
                store,
                policy,
                budget_seconds=10.0,
                id_factory=lambda: "control-demo-attempt",
            )
            result = controller.process(
                "control-demo",
                1,
                audit=audit,
                update=update,
                model_version="control-demo-model-v1",
            )
            before = _replay_fingerprint(store.replay("control-demo", 1))
            artifact_count = len(list((Path(temporary) / "artifacts").glob("*.json")))
        with SafetyStore(temporary) as reopened:
            after_replay = reopened.replay("control-demo", 1)
            after = _replay_fingerprint(after_replay)
            return {
                "fixture": "real breast-cancer rows with a norm-based control audit, not a detection score",
                "decision": result.decision.value,
                "committed": result.committed,
                "accepted_samples": len(result.accepted_sample_ids),
                "decision_counts": dict(result.decision_counts),
                "generation": result.generation,
                "updated_weights": weights.tolist(),
                "artifact_count": artifact_count,
                "replay_decisions": len(after_replay.decisions),
                "replay_commits": len(after_replay.commits),
                "restart_replay_equal": before == after,
                "calibration_count": len(after_replay.calibrations),
                "claim_scope": "control-plane fixture; detection quality is evaluated separately in real_data_results",
            }


def run_milestone12() -> dict[str, object]:
    control = run_control_plane_demonstration()
    real = run_real_benchmark()
    return {
        "milestone_1_control_plane": control,
        "milestone_2_real_data": real,
        "execution": "control demonstration completed before sequential real-data training and repair",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("research/cmae/milestone12_results.json")
    )
    parser.add_argument(
        "--real-output", type=Path, default=Path("research/cmae/real_data_results.json")
    )
    arguments = parser.parse_args()
    results = run_milestone12()
    rendered = json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n"
    real_rendered = (
        json.dumps(
            results["milestone_2_real_data"], indent=2, sort_keys=True, default=_json_default
        )
        + "\n"
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.real_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)
    arguments.real_output.write_text(real_rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
