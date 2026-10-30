"""Sequential calibration-checkpoint SAE/CLT replacement-graph benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from research.cmae.arithmetic_data import (
    ArithmeticDataset,
    ArithmeticSplit,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_transformer import (
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
)
from research.cmae.generalization import TrainingArm, train_with_calibration
from research.cmae.replacement_graph import (
    MLPActivations,
    ReplacementGraphTrace,
    ReplacementKind,
    collect_mlp_activations,
    evaluate_replacement_reconstruction,
    fit_replacement,
    trace_replacement_graph,
)
from research.cmae.run_transformer_m5b import UNTOUCHED_SPLITS

ARMS: tuple[TrainingArm, ...] = ("standard", "auxiliary")
KINDS: tuple[ReplacementKind, ...] = ("sae", "clt")
ALL_SPLITS: tuple[ArithmeticSplit, ...] = (
    "train",
    "calibration",
    *UNTOUCHED_SPLITS,
)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected integer JSON value, got {type(value).__name__}")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric JSON value, got {type(value).__name__}")
    return float(value)


def _metric_values(traces: list[ReplacementGraphTrace], name: str) -> list[float]:
    values: list[float] = []
    for trace in traces:
        value = trace.metrics()[name]
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean_metric(traces: list[ReplacementGraphTrace], name: str) -> float | None:
    values = _metric_values(traces, name)
    return float(np.mean(values)) if values else None


def _max_abs_metric(traces: list[ReplacementGraphTrace], name: str) -> float:
    values = _metric_values(traces, name)
    if not values:
        raise RuntimeError(f"replacement metric {name} has no numeric values")
    return float(np.max(np.abs(values)))


def _graph_summary(traces: list[ReplacementGraphTrace]) -> dict[str, object]:
    if not traces:
        raise ValueError("replacement graph summary needs at least one trace")
    prediction_match = float(np.mean(_metric_values(traces, "replacement_prediction_match")))
    mean_replacement_error = _mean_metric(traces, "replacement_relative_logit_error_mass")
    mean_error_influence = _mean_metric(traces, "error_output_influence_mass")
    edge_residual = _mean_metric(traces, "feature_edge_relative_residual_mass")
    max_quadrature = _max_abs_metric(traces, "integrated_quadrature_residual")
    max_output_conservation = _max_abs_metric(traces, "output_conservation_error")
    max_readout_parity = _max_abs_metric(traces, "readout_parity_error")
    max_reconstruction_conservation = _max_abs_metric(traces, "reconstruction_conservation_error")
    gates = {
        "quadrature": max_quadrature <= 1e-3,
        "output_conservation": max_output_conservation <= 1e-5,
        "readout_parity": max_readout_parity <= 1e-4,
        "reconstruction_conservation": max_reconstruction_conservation <= 1e-6,
        "replacement_prediction_match": prediction_match >= 0.8,
        "replacement_logit_error": mean_replacement_error is not None
        and mean_replacement_error <= 0.1,
        "error_influence": mean_error_influence is not None and mean_error_influence <= 0.5,
        "feature_edge_residual": edge_residual is not None and edge_residual <= 0.5,
    }
    return {
        "prompts": len(traces),
        "mean_nodes": float(np.mean([len(trace.nodes) for trace in traces])),
        "mean_edges": float(np.mean([len(trace.edges) for trace in traces])),
        "replacement_prediction_match_rate": prediction_match,
        "mean_replacement_relative_logit_error_mass": mean_replacement_error,
        "max_replacement_relative_logit_error_mass": max(
            _metric_values(traces, "replacement_relative_logit_error_mass")
        ),
        "mean_error_output_influence_mass": mean_error_influence,
        "mean_feature_edge_relative_residual_mass": edge_residual,
        "mean_feature_edge_pearson": _mean_metric(traces, "feature_edge_pearson"),
        "mean_feature_edge_top_5_absolute_overlap": _mean_metric(
            traces, "feature_edge_top_5_absolute_overlap"
        ),
        "max_abs_integrated_quadrature_residual": max_quadrature,
        "max_abs_output_conservation_error": max_output_conservation,
        "max_readout_parity_error": max_readout_parity,
        "max_reconstruction_conservation_error": max_reconstruction_conservation,
        "gates": gates,
        "status": (
            "pass_scoped_numerical_replacement_gate"
            if all(gates.values())
            else "abstain_incomplete_replacement"
        ),
        "semantic_status": "abstain_unlabeled_features_and_partial_answer_position_graph",
    }


def _collect_split_activations(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
) -> dict[str, MLPActivations]:
    result: dict[str, MLPActivations] = {}
    for split in ALL_SPLITS:
        tokens_np, _, _ = dataset.arrays(split)
        result[split] = collect_mlp_activations(model, torch.from_numpy(tokens_np))
    return result


def run_transformer_replacements(
    output: Path,
    benchmark_path: Path,
    *,
    seeds: tuple[int, ...] = (101, 202, 303),
    features: int = 64,
    replacement_epochs: int = 100,
    replacement_batch_size: int = 512,
    replacement_learning_rate: float = 0.003,
    l1_coefficient: float = 0.001,
    quadrature_steps: int = 256,
    samples_per_split: int = 1,
) -> dict[str, object]:
    """Train both replacements on train only and evaluate fixed untouched prompts."""

    if (
        not seeds
        or features <= 0
        or replacement_epochs <= 0
        or replacement_batch_size <= 0
        or replacement_learning_rate <= 0
        or l1_coefficient < 0
        or quadrature_steps < 2
        or samples_per_split <= 0
    ):
        raise ValueError("invalid replacement benchmark configuration")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    preregistration = cast(dict[str, object], benchmark["preregistration"])
    benchmark_runs = cast(list[dict[str, object]], benchmark["runs"])
    indexed = {(_integer(run["seed"]), str(run["arm"])): run for run in benchmark_runs}
    dataset_seed = _integer(preregistration["dataset_seed"])
    epochs = _integer(preregistration["epochs"])
    evaluation_every = _integer(preregistration["checkpoint_evaluation_every"])
    batch_size = _integer(preregistration["batch_size"])
    learning_rate = _number(preregistration["learning_rate"])
    auxiliary_weight = _number(preregistration["auxiliary_weight"])
    dataset = make_arithmetic_dataset(dataset_seed)
    runs: list[dict[str, object]] = []
    summary_index: dict[tuple[int, str, str], dict[str, object]] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        initial_model = ArithmeticTransformer()
        initial_state = {
            name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()
        }
        for arm_index, arm in enumerate(ARMS):
            expected_run = indexed[(seed, arm)]
            expected_training = cast(dict[str, object], expected_run["training"])
            print(f"Replacement checkpoint: seed={seed} arm={arm}", flush=True)
            model = ArithmeticTransformer()
            model.load_state_dict(initial_state)
            training = train_with_calibration(
                model,
                dataset,
                arm,
                epochs=epochs,
                evaluation_every=evaluation_every,
                batch_size=batch_size,
                learning_rate=learning_rate,
                auxiliary_weight=auxiliary_weight,
                seed=seed + 1,
            )
            expected_loss = _number(expected_training["best_calibration_loss"])
            expected_epoch = _integer(expected_training["best_epoch"])
            if training.best_epoch != expected_epoch or not np.isclose(
                training.best_calibration_loss, expected_loss, atol=1e-6, rtol=0
            ):
                raise RuntimeError("replacement checkpoint reproduction failed")
            combined = evaluate_arithmetic_transformer(
                model, dataset, "combined", batch_size=batch_size
            )
            expected_evaluation = cast(dict[str, object], expected_run["evaluation"])
            expected_combined = cast(dict[str, object], expected_evaluation["combined"])
            expected_accuracy = _number(expected_combined["accuracy"])
            if not np.isclose(combined["accuracy"], expected_accuracy, atol=1e-12, rtol=0):
                raise RuntimeError("replacement checkpoint accuracy fingerprint failed")
            split_activations = _collect_split_activations(model, dataset)
            for kind_index, kind in enumerate(KINDS):
                replacement_seed = seed + 1000 * kind_index + 10000 * arm_index
                print(
                    f"  fit {kind}: seed={seed} arm={arm} replacement_seed={replacement_seed}",
                    flush=True,
                )
                replacement, history = fit_replacement(
                    model,
                    dataset,
                    kind,
                    features=features,
                    epochs=replacement_epochs,
                    batch_size=replacement_batch_size,
                    learning_rate=replacement_learning_rate,
                    l1_coefficient=l1_coefficient,
                    seed=replacement_seed,
                )
                reconstruction = {
                    split: evaluate_replacement_reconstruction(
                        replacement, split_activations[split]
                    )
                    for split in ALL_SPLITS
                }
                traces: list[ReplacementGraphTrace] = []
                for split in UNTOUCHED_SPLITS:
                    for sample in dataset.split_samples(split)[:samples_per_split]:
                        traces.append(
                            trace_replacement_graph(
                                model,
                                replacement,
                                sample,
                                quadrature_steps=quadrature_steps,
                            )
                        )
                graph_summary = _graph_summary(traces)
                summary_index[(seed, arm, kind)] = graph_summary
                runs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "replacement": kind,
                        "replacement_seed": replacement_seed,
                        "checkpoint_reproduction": {
                            "best_epoch": training.best_epoch,
                            "best_calibration_loss": training.best_calibration_loss,
                            "combined_accuracy": float(combined["accuracy"]),
                            "matches_benchmark": True,
                        },
                        "training": {
                            "fit_split": "train",
                            "epochs": replacement_epochs,
                            "features_per_layer": features,
                            "batch_size": replacement_batch_size,
                            "learning_rate": replacement_learning_rate,
                            "l1_coefficient": l1_coefficient,
                            "initial_loss": history[0],
                            "final_loss": history[-1],
                            "finite_history": bool(np.all(np.isfinite(history))),
                        },
                        "reconstruction": reconstruction,
                        "graph_summary": graph_summary,
                        "graphs": [trace.to_dict() for trace in traces],
                    }
                )

    comparisons: dict[str, object] = {}
    for metric in (
        "mean_replacement_relative_logit_error_mass",
        "mean_error_output_influence_mass",
        "mean_feature_edge_relative_residual_mass",
    ):
        per_configuration: list[dict[str, object]] = []
        for seed in seeds:
            for arm in ARMS:
                clt = summary_index[(seed, arm, "clt")][metric]
                sae = summary_index[(seed, arm, "sae")][metric]
                if isinstance(clt, (int, float)) and isinstance(sae, (int, float)):
                    per_configuration.append({"seed": seed, "arm": arm, "clt_minus_sae": clt - sae})
        differences = [
            _number(row["clt_minus_sae"]) for row in per_configuration if isinstance(row, dict)
        ]
        comparisons[metric] = {
            "per_configuration": per_configuration,
            "mean_clt_minus_sae": float(np.mean(differences)) if differences else None,
            "clt_lower_on_all_configurations": bool(differences)
            and all(value < 0 for value in differences),
            "direction": "negative favors CLT only for the measured numerical diagnostic",
        }
    result: dict[str, object] = {
        "schema_version": 1,
        "milestone": "5B trained SAE/CLT replacement graphs",
        "execution": "sequential",
        "benchmark_source": str(benchmark_path),
        "checkpoint_reproduction_verified": True,
        "preregistration": {
            "seeds": list(seeds),
            "arms": list(ARMS),
            "replacement_kinds": list(KINDS),
            "fit_split": "train",
            "untouched_graph_splits": list(UNTOUCHED_SPLITS),
            "samples_per_untouched_split": samples_per_split,
            "prompt_selection": "first dataset-order prompts; no output-based filtering",
            "features_per_layer": features,
            "replacement_epochs": replacement_epochs,
            "replacement_batch_size": replacement_batch_size,
            "replacement_learning_rate": replacement_learning_rate,
            "l1_coefficient": l1_coefficient,
            "quadrature_steps": quadrature_steps,
            "gates": {
                "max_abs_integrated_quadrature_residual": 1e-3,
                "max_abs_output_conservation_error": 1e-5,
                "max_readout_parity_error": 1e-4,
                "max_reconstruction_conservation_error": 1e-6,
                "replacement_prediction_match_rate_min": 0.8,
                "mean_replacement_relative_logit_error_mass_max": 0.1,
                "mean_error_output_influence_mass_max": 0.5,
                "mean_feature_edge_relative_residual_mass_max": 0.5,
            },
        },
        "runs": runs,
        "clt_minus_sae": comparisons,
        "conclusion": {
            "status": "abstain_mechanistic_semantics",
            "reason": (
                "SAE/CLT reconstruction, conserved direct allocations, and local-edge ablations "
                "do not assign feature meanings or establish a complete mechanistic circuit."
            ),
        },
        "formal_verification": {
            "rocq_closed_theorems": 6,
            "z3_expected_unsat": 6,
            "z3_expected_sat_counterexamples": 2,
            "artifact": "formal_verification/rocq/CMAEReplacementGraphFoundations.v",
            "independent_checker": "formal_verification/verify_cmae_replacement_graph_z3.py",
        },
        "limitations": [
            "The graph covers answer-position MLP writes in a two-layer toy Transformer only.",
            "Attention and upstream-position contributions remain in the conditional baseline.",
            "Feature-to-feature edges are local predictions, not exact integrated paths.",
            "Reconstruction-error nodes are uninterpreted remainder writes, not harmless noise.",
            "Output reconstruction is not mechanistic faithfulness or feature semantics.",
            "CLT-minus-SAE numerical comparisons are not claims of improved interpretability.",
        ],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--replacement-epochs", type=int, default=100)
    parser.add_argument("--replacement-batch-size", type=int, default=512)
    parser.add_argument("--replacement-learning-rate", type=float, default=0.003)
    parser.add_argument("--l1-coefficient", type=float, default=0.001)
    parser.add_argument("--quadrature-steps", type=int, default=256)
    parser.add_argument("--samples-per-split", type=int, default=1)
    arguments = parser.parse_args()
    result = run_transformer_replacements(
        arguments.output,
        arguments.benchmark,
        seeds=tuple(arguments.seeds),
        features=arguments.features,
        replacement_epochs=arguments.replacement_epochs,
        replacement_batch_size=arguments.replacement_batch_size,
        replacement_learning_rate=arguments.replacement_learning_rate,
        l1_coefficient=arguments.l1_coefficient,
        quadrature_steps=arguments.quadrature_steps,
        samples_per_split=arguments.samples_per_split,
    )
    print(json.dumps(result["clt_minus_sae"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
