"""Post-hoc full-Hessian sensitivity analysis against frozen 6B retraining effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.cmae.backdoorbench_organism import (
    build_cifar_model,
    cifar_behavior_metrics,
    load_badnet_trigger,
    load_cifar_binary_batch,
    make_cifar_batch,
    select_stratified_indices,
    triggered_source_inputs,
)
from research.cmae.binary_counterfactual import (
    aggregate_scores,
    capture_binary_ggn,
    counterfactual_metrics,
    fit_binary_physical_subset,
)
from research.cmae.binary_full_hessian import (
    capture_binary_full_hessian,
    damped_full_hessian_influence,
)
from research.cmae.run_counterfactual_assurance import (
    _detection_gate,
    _detection_metrics,
    _evaluation_splits,
    _json_ready,
)


def run_diagnostic(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text())
    if source.get("milestone") != "6B" or source.get("status") != "completed":
        raise ValueError("full-Hessian diagnostic needs a completed 6B source artifact")
    config = source["config"]
    root = Path(__file__).resolve().parents[2]
    external = root / ".venv" / "src"
    train_pool = load_cifar_binary_batch(external / "cifar-10-batches-py" / "data_batch_1")
    test_pool = load_cifar_binary_batch(external / "cifar-10-batches-py" / "test_batch")
    trigger = load_badnet_trigger(
        external / "BackdoorBench" / "resource" / "badnet" / "trigger_image.png"
    )
    development_indices, final_indices = _evaluation_splits(
        test_pool, int(config["evaluation_count_per_split"])
    )
    query_source = development_indices[test_pool.labels[development_indices] == 0][
        : int(config["query_count"])
    ]
    query_inputs = triggered_source_inputs(test_pool, query_source, trigger)
    runs: list[dict[str, Any]] = []
    for prior in source["results"]:
        seed = int(prior["seed"])
        role = str(prior["role"])
        torch.set_num_threads(1)
        train_indices = select_stratified_indices(train_pool, int(config["train_count"]), seed=seed)
        batch = make_cifar_batch(
            train_pool,
            train_indices,
            trigger,
            attack="standard",
            poison_fraction=float(config["poison_fraction"]),
            seed=seed,
        )
        model = build_cifar_model(seed, hidden_dim=int(config["hidden_dim"]))
        fit_binary_physical_subset(
            model,
            batch.inputs,
            batch.labels,
            epochs=int(config["epochs"]),
            learning_rate=float(config["learning_rate"]),
        )
        evaluation_indices = development_indices if role == "development" else final_indices
        evaluation = make_cifar_batch(
            test_pool, evaluation_indices, trigger, attack="clean", seed=1200
        )
        source_indices = evaluation_indices[test_pool.labels[evaluation_indices] == 0]
        triggered_evaluation = triggered_source_inputs(test_pool, source_indices, trigger)
        behavior = cifar_behavior_metrics(model, evaluation, triggered_evaluation)
        for metric in ("clean_accuracy", "attack_success_rate"):
            if abs(behavior[metric] - prior["behavior"]["standard"][metric]) > 1e-12:
                raise RuntimeError("diagnostic model does not reproduce the frozen source behavior")
        ggn = capture_binary_ggn(model, batch.inputs, batch.labels, query_inputs)
        hessian = capture_binary_full_hessian(model, batch.inputs, batch.labels, ggn)
        groups = [np.asarray(group, dtype=np.int64) for group in prior["counterfactual"]["groups"]]
        group_actual = np.asarray(prior["counterfactual"]["group_actual_effects"], dtype=np.float64)
        loo_indices = np.asarray(prior["counterfactual"]["loo_indices"], dtype=np.int64)
        loo_actual = np.asarray(prior["counterfactual"]["loo_actual_effects"], dtype=np.float64)
        healthy_damping = max(float(config["damping"]), -hessian.minimum_eigenvalue + 1e-3)
        variants: dict[str, Any] = {}
        for name, damping in (
            ("frozen_damping", float(config["damping"])),
            ("health_adjusted_damping", healthy_damping),
        ):
            try:
                scores = damped_full_hessian_influence(hessian, damping=damping)
            except RuntimeError as error:
                variants[name] = {
                    "status": "unhealthy",
                    "reason": str(error),
                    "damping": damping,
                }
                continue
            detection = _detection_metrics(
                scores, batch.attack_labels, tuple(float(x) for x in config["quarantine_budgets"])
            )
            group_metrics = counterfactual_metrics(aggregate_scores(scores, groups), group_actual)
            variants[name] = {
                "status": "executed",
                "damping": damping,
                "detection": detection,
                "detection_gate_descriptive_only": _detection_gate(detection),
                "leave_group_out": group_metrics,
                "leave_one_out": counterfactual_metrics(scores[loo_indices], loo_actual),
            }
        runs.append(
            {
                "seed": seed,
                "role": role,
                "minimum_eigenvalue": hessian.minimum_eigenvalue,
                "symmetry_residual": hessian.symmetry_residual,
                "variants": variants,
            }
        )
    return {
        "analysis_role": "post_hoc_sensitivity_not_preregistered_confirmation",
        "source_artifact": str(source_path),
        "scope": "exact full binary loss Hessian including tanh residual and cross-parameter terms",
        "integrated_prevention_claim": False,
        "results": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("research/cmae/results/counterfactual_assurance_m6b.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/cmae/results/full_hessian_diagnostic_m6b.json"),
    )
    args = parser.parse_args()
    result = run_diagnostic(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
