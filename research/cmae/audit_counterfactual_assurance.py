"""Independent consistency audit for completed Milestone-6B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.cmae.backdoorbench_organism import (
    apply_badnet_mask,
    load_badnet_trigger,
    load_cifar_binary_batch,
)
from research.cmae.binary_counterfactual import aggregate_scores, counterfactual_metrics
from research.cmae.external_attribution import score_agreement
from research.cmae.run_counterfactual_assurance import _json_ready

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _auc(labels: IntArray, scores: FloatArray) -> float:
    positives = int(labels.sum())
    negatives = labels.size - positives
    ranks = _ranks(scores)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def _ap(labels: IntArray, scores: FloatArray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / labels.sum())


def _close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"artifact metric mismatch: {actual} != {expected}")


def _audit_detection(stored: dict[str, Any], labels: IntArray, scores: FloatArray) -> None:
    if labels.shape != scores.shape or not np.all(np.isfinite(scores)):
        raise AssertionError("invalid saved score vector")
    _close(_auc(labels, scores), stored["roc_auc"])
    _close(_ap(labels, scores), stored["average_precision"])
    positives = int(labels.sum())
    negatives = labels.size - positives
    for budget_name, budget in stored["budgets"].items():
        fraction = float(budget_name)
        count = max(1, int(np.ceil(labels.size * fraction)))
        order = np.argsort(-scores, kind="mergesort")
        quarantined = order[:count]
        true_positive = int(labels[quarantined].sum())
        false_positive = count - true_positive
        if budget["quarantine_count"] != count or budget["accepted_count"] != labels.size - count:
            raise AssertionError("artifact budget accounting mismatch")
        _close(true_positive / positives, budget["recall"])
        _close(false_positive / negatives, budget["clean_false_positive_rate"])
        _close(true_positive / count, budget["precision"])


def _audit_counterfactual(
    stored: dict[str, Any], scores: dict[str, FloatArray], count: int, primary: bool
) -> None:
    groups = [np.asarray(group, dtype=np.int64) for group in stored["groups"]]
    covered = np.concatenate(groups)
    if sorted(covered.tolist()) != list(range(count)):
        raise AssertionError("counterfactual groups do not partition training rows")
    group_actual = np.asarray(stored["group_actual_effects" if primary else "group_actual"])
    loo_indices = np.asarray(stored["loo_indices"], dtype=np.int64)
    loo_actual = np.asarray(stored["loo_actual_effects" if primary else "loo_actual"])
    if len(set(loo_indices.tolist())) != loo_indices.size:
        raise AssertionError("duplicate LOO indices")
    for name, method in stored["methods"].items() if primary else []:
        vector = np.asarray(scores[name], dtype=np.float64)
        expected_group = counterfactual_metrics(aggregate_scores(vector, groups), group_actual)
        expected_loo = counterfactual_metrics(vector[loo_indices], loo_actual)
        for key, value in expected_group.items():
            _close(value, method["leave_group_out"][key])
        for key, value in expected_loo.items():
            _close(value, method["leave_one_out"][key])


def audit_badnet_source(source_root: Path) -> dict[str, Any]:
    """Execute the pinned upstream mask and compare it on real CIFAR rows."""
    cifar = source_root / "cifar-10-batches-py"
    trigger_path = source_root / "BackdoorBench/resource/badnet/trigger_image.png"
    trigger = load_badnet_trigger(trigger_path)
    pool = load_cifar_binary_batch(cifar / "data_batch_1")
    module_path = source_root / "BackdoorBench/utils/bd_img_transform/patch.py"
    spec = importlib.util.spec_from_file_location("audited_backdoorbench_patch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load BackdoorBench mask from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = pool.images[:8]
    upstream_images = module.AddMaskPatchTrigger(trigger)(images.copy())
    local_images = apply_badnet_mask(images.copy(), trigger)
    if not np.array_equal(upstream_images, local_images):
        difference = float(np.max(np.abs(upstream_images.astype(float) - local_images)))
        raise AssertionError(f"official BadNets mask parity failure: {difference}")
    return {
        "source_commits": {
            "BackdoorBench": "f02e3534645f0ee63d6848653062cd6c0d6c400d",
            "cifar_mirror": "48ece5ce490482885f7a9a492fa031c1de79519c",
        },
        "official_transform_examples": len(images),
        "official_transform_max_abs_difference": 0.0,
        "source_sha256": {
            "badnet_trigger": hashlib.sha256(trigger_path.read_bytes()).hexdigest(),
            "cifar_train_batch": hashlib.sha256((cifar / "data_batch_1").read_bytes()).hexdigest(),
            "cifar_test_batch": hashlib.sha256((cifar / "test_batch").read_bytes()).hexdigest(),
        },
    }


def audit_cifar(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    for name, revision in source["source_commits"].items():
        if artifact["source_commits"][name] != revision:
            raise AssertionError(f"CIFAR source revision mismatch: {name}")
    if artifact["source_sha256"] != source["source_sha256"]:
        raise AssertionError("CIFAR source hash mismatch")
    results = artifact["results"]
    development_eval = set(results[0]["evaluation_sample_ids"])
    maximum_external = 0.0
    for run in results:
        count = len(run["train_sample_ids"])
        if len(set(run["train_sample_ids"])) != count:
            raise AssertionError("duplicate CIFAR training source IDs")
        if run["fixed_full_retraining_max_parameter_residual"] != 0.0:
            raise AssertionError("nonzero full-objective retraining control")
        for attack, labels_list in run["attack_labels"].items():
            labels = np.asarray(labels_list, dtype=np.int64)
            if labels.shape != (count,) or int(labels.sum()) != run["poison_counts"][attack]:
                raise AssertionError("CIFAR attack-label mismatch")
            for name, scores_list in run["score_vectors"][attack].items():
                scores = np.asarray(scores_list, dtype=np.float64)
                _audit_detection(run["detections"][attack][name], labels, scores)
        _audit_counterfactual(
            run["counterfactual"], run["score_vectors"]["standard"], count, primary=True
        )
        for name in ("dattri", "kronfluence"):
            agreement = score_agreement(
                np.asarray(run["score_vectors"]["standard"]["cmae_ekfac"]),
                np.asarray(run["score_vectors"]["standard"][f"{name}_ekfac"]),
            )
            maximum_external = max(maximum_external, agreement["relative_l2_difference"])
            for key, value in agreement.items():
                _close(value, run["external"][name]["agreement_with_cmae"][key])
        if run["role"] == "final":
            final_eval = set(run["evaluation_sample_ids"])
            if development_eval & final_eval:
                raise AssertionError("development/final CIFAR evaluation overlap")
            if run["repair"]["status"] != "blocked":
                raise AssertionError("negative CIFAR run unexpectedly executed repair")
    if artifact["integrated_prevention_gate"]["pass"]:
        raise AssertionError("negative CIFAR audit cannot certify integrated prevention")
    return {
        "runs": len(results),
        "detection_vectors_audited": sum(
            len(methods) for run in results for methods in run["score_vectors"].values()
        ),
        "maximum_external_relative_l2": maximum_external,
        "full_objective_controls": len(results),
    }


def audit_arithmetic(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    for run in artifact["results"]:
        labels = np.asarray(run["poison_mask"], dtype=np.int64)
        count = len(run["training_keys"])
        if labels.shape != (count,) or len({tuple(key) for key in run["training_keys"]}) != count:
            raise AssertionError("arithmetic training key/label mismatch")
        for name, scores_list in run["scores"].items():
            _audit_detection(
                run["detections"][name], labels, np.asarray(scores_list, dtype=np.float64)
            )
        counter = run["counterfactual"]
        groups = [np.asarray(group, dtype=np.int64) for group in counter["groups"]]
        if sorted(np.concatenate(groups).tolist()) != list(range(count)):
            raise AssertionError("arithmetic group coverage mismatch")
        scores = np.asarray(run["scores"]["cmae_ekfac"], dtype=np.float64)
        group_actual = np.asarray(counter["group_actual"], dtype=np.float64)
        group_predicted = aggregate_scores(scores, groups)
        np.testing.assert_allclose(group_predicted, counter["group_predicted"], rtol=0, atol=0)
        expected_group = counterfactual_metrics(group_predicted, group_actual)
        for key, value in expected_group.items():
            _close(value, counter["leave_group_out"][key])
        loo_indices = np.asarray(counter["loo_indices"], dtype=np.int64)
        loo_actual = np.asarray(counter["loo_actual"], dtype=np.float64)
        expected_loo = counterfactual_metrics(scores[loo_indices], loo_actual)
        for key, value in expected_loo.items():
            _close(value, counter["leave_one_out"][key])
        if run["role"] == "final" and run["repair"]["status"] != "blocked":
            raise AssertionError("negative arithmetic run unexpectedly executed repair")
    if not all(run["organism_gate"]["pass"] for run in artifact["results"][1:]):
        raise AssertionError("expected final arithmetic organisms to be established")
    if artifact["integrated_prevention_gate"]["pass"]:
        raise AssertionError("negative arithmetic audit cannot certify integrated prevention")
    return {"runs": len(artifact["results"]), "detection_vectors_audited": 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar",
        type=Path,
        default=Path("research/cmae/results/counterfactual_assurance_m6b.json"),
    )
    parser.add_argument(
        "--arithmetic",
        type=Path,
        default=Path("research/cmae/results/arithmetic_fixed_target_m6b.json"),
    )
    parser.add_argument(
        "--full-hessian",
        type=Path,
        default=Path("research/cmae/results/full_hessian_diagnostic_m6b.json"),
    )
    parser.add_argument(
        "--circuit-gate",
        type=Path,
        default=Path("research/cmae/results/circuit_tracer_gate_m6b.json"),
    )
    parser.add_argument("--source-root", type=Path, default=Path(".venv/src"))
    parser.add_argument("--output", type=Path, default=Path("research/cmae/results/m6b_audit.json"))
    args = parser.parse_args()
    source = audit_badnet_source(args.source_root)
    full_hessian = json.loads(args.full_hessian.read_text())
    symmetry = max(run["symmetry_residual"] for run in full_hessian["results"])
    if symmetry > 1e-12:
        raise AssertionError("full-Hessian symmetry failure")
    circuit = json.loads(args.circuit_gate.read_text())
    if circuit["result"]["status"] != "blocked_by_upstream_gate":
        raise AssertionError("circuit-tracer gate did not fail closed")
    result = {
        "status": "passed",
        "badnet_source": source,
        "cifar": audit_cifar(args.cifar, source),
        "arithmetic": audit_arithmetic(args.arithmetic),
        "full_hessian": {
            "runs": len(full_hessian["results"]),
            "maximum_symmetry_residual": symmetry,
        },
        "circuit_tracer": {"status": circuit["result"]["status"], "pretrained_execution": False},
    }
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
