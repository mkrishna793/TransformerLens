"""Sequential calibration-controlled generalization and latent-lens benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

from research.cmae.arithmetic_data import (
    MODULUS,
    ArithmeticDataset,
    ArithmeticSplit,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_transformer import (
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
)
from research.cmae.generalization import (
    TrainingArm,
    evaluate_latent_readouts,
    train_with_calibration,
)
from research.cmae.latent_lenses import evaluate_latent_lenses, fit_latent_lenses

ALL_SPLITS: tuple[ArithmeticSplit, ...] = (
    "train",
    "calibration",
    "iid",
    "composition",
    "operand",
    "multiplier",
    "combined",
)
UNTOUCHED_SPLITS: tuple[ArithmeticSplit, ...] = (
    "iid",
    "composition",
    "operand",
    "multiplier",
    "combined",
)
DEFAULT_ARMS: tuple[TrainingArm, ...] = (
    "standard",
    "regularized",
    "auxiliary",
    "auxiliary_regularized",
)
COMPARISONS: tuple[tuple[TrainingArm, TrainingArm], ...] = (
    ("regularized", "standard"),
    ("auxiliary", "standard"),
    ("auxiliary_regularized", "regularized"),
    ("auxiliary_regularized", "auxiliary"),
)


def _accuracy_summary(
    evaluation: Mapping[ArithmeticSplit, dict[str, float | int]],
) -> dict[str, float]:
    accuracy = {split: float(evaluation[split]["accuracy"]) for split in ALL_SPLITS}
    return {
        "chance_accuracy": 1.0 / MODULUS,
        "train_minus_iid_accuracy": accuracy["train"] - accuracy["iid"],
        "iid_minus_composition_accuracy": accuracy["iid"] - accuracy["composition"],
        "iid_minus_operand_accuracy": accuracy["iid"] - accuracy["operand"],
        "iid_minus_multiplier_accuracy": accuracy["iid"] - accuracy["multiplier"],
        "iid_minus_combined_accuracy": accuracy["iid"] - accuracy["combined"],
    }


def _readout_gate(lens_result: dict[str, object], site: str, latent: str) -> dict[str, object]:
    metrics = cast(dict[str, dict[str, float]], lens_result["metrics"])
    values = metrics[f"{site}_{latent}_jpp"]
    top1 = values["top1_accuracy"]
    recall = values["recall_at_10"]
    complete = bool(top1 >= 0.5 and recall >= 0.9)
    return {
        "top1_accuracy": top1,
        "recall_at_10": recall,
        "complete": complete,
        "interpretation": "precise_readout" if complete else "abstain_poor_completeness",
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "population_std": float(array.std())}


def run_transformer_m5b(
    output: Path,
    *,
    dataset_seed: int = 101,
    seeds: tuple[int, ...] = (101, 202, 303),
    arms: tuple[TrainingArm, ...] = DEFAULT_ARMS,
    epochs: int = 250,
    evaluation_every: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    auxiliary_weight: float = 0.5,
    clusters: int = 8,
    kmeans_iterations: int = 25,
    lens_weight_steps: int = 100,
    lens_weight_learning_rate: float = 0.03,
) -> dict[str, object]:
    """Run the preregistered sequential experiment without holdout-based selection."""

    if not seeds or len(set(seeds)) != len(seeds) or not arms or len(set(arms)) != len(arms):
        raise ValueError("benchmark seeds and arms must be nonempty and unique")
    dataset = make_arithmetic_dataset(dataset_seed)
    runs: list[dict[str, object]] = []
    accuracies: dict[tuple[int, str, str], float] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        initial_model = ArithmeticTransformer()
        initial_state = {
            name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()
        }
        parameter_count = sum(parameter.numel() for parameter in initial_model.parameters())
        for arm in arms:
            print(f"Milestone 5B: seed={seed} arm={arm} epochs={epochs}", flush=True)
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
            evaluation = {
                split: evaluate_arithmetic_transformer(model, dataset, split, batch_size=batch_size)
                for split in ALL_SPLITS
            }
            direct_latents = {
                split: evaluate_latent_readouts(model, dataset, split, batch_size=batch_size)
                for split in ALL_SPLITS
            }
            suite = fit_latent_lenses(
                model,
                dataset,
                clusters=clusters,
                batch_size=batch_size,
                kmeans_iterations=kmeans_iterations,
                weight_steps=lens_weight_steps,
                weight_learning_rate=lens_weight_learning_rate,
            )
            lens_evaluation = {
                split: evaluate_latent_lenses(model, suite, dataset, split, batch_size=batch_size)
                for split in ALL_SPLITS
            }
            readout_gates = {
                split: {
                    "early_u": _readout_gate(lens_evaluation[split], "early", "u"),
                    "final_v": _readout_gate(lens_evaluation[split], "final", "v"),
                }
                for split in UNTOUCHED_SPLITS
            }
            for split in ALL_SPLITS:
                accuracies[(seed, arm, split)] = float(evaluation[split]["accuracy"])
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "training": training.to_dict(),
                    "evaluation": evaluation,
                    "generalization_summary": _accuracy_summary(evaluation),
                    "direct_shared_readout_latents": direct_latents,
                    "lens_fit": {
                        "early": suite.early.diagnostics(),
                        "final": suite.final.diagnostics(),
                    },
                    "lens_evaluation": lens_evaluation,
                    "jpp_readout_gates": readout_gates,
                    "mechanistic_claim": (
                        "abstain: latent readouts and output reconstruction do not establish "
                        "causal reuse or path sufficiency"
                    ),
                }
            )
            del suite

    aggregate_accuracy: dict[str, object] = {}
    for arm in arms:
        aggregate_accuracy[arm] = {
            split: {
                "per_seed": [accuracies[(seed, arm, split)] for seed in seeds],
                **_mean_std([accuracies[(seed, arm, split)] for seed in seeds]),
            }
            for split in ALL_SPLITS
        }
    paired_accuracy: dict[str, object] = {}
    for treatment, reference in COMPARISONS:
        if treatment not in arms or reference not in arms:
            continue
        comparison_name = f"{treatment}_minus_{reference}"
        comparison: dict[str, object] = {}
        for split in UNTOUCHED_SPLITS:
            differences = [
                accuracies[(seed, treatment, split)] - accuracies[(seed, reference, split)]
                for seed in seeds
            ]
            comparison[split] = {
                "per_seed": differences,
                **_mean_std(differences),
                "positive_seed_count": sum(value > 0 for value in differences),
                "positive_on_all_seeds": all(value > 0 for value in differences),
            }
        paired_accuracy[comparison_name] = comparison

    primary_gates: dict[str, bool] = {}
    for comparison_name, comparison_value in paired_accuracy.items():
        comparison = cast(dict[str, object], comparison_value)
        combined = cast(dict[str, object], comparison["combined"])
        primary_gates[comparison_name] = bool(combined["positive_on_all_seeds"])
    result: dict[str, object] = {
        "schema_version": 1,
        "milestone": "5B controlled generalization and training-only latent lenses",
        "execution": "sequential",
        "preregistration": {
            "dataset_seed": dataset_seed,
            "training_seeds": list(seeds),
            "arms": list(arms),
            "epochs": epochs,
            "checkpoint_evaluation_every": evaluation_every,
            "checkpoint_selection": "minimum calibration final-answer cross-entropy only",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "base_weight_decay": 1e-4,
            "regularized_weight_decay": 1e-2,
            "auxiliary_weight": auxiliary_weight,
            "auxiliary_objective": "CE(u from early residual) + CE(v from final residual)",
            "identical_initial_weights_and_batch_order_within_seed": True,
            "primary_endpoint": "combined-holdout final-answer accuracy",
            "secondary_endpoints": list(UNTOUCHED_SPLITS[:-1]),
            "multi_seed_gate": "positive paired difference on every preregistered seed",
            "lens_clusters": clusters,
            "lens_kmeans_iterations": kmeans_iterations,
            "lens_weight_steps": lens_weight_steps,
            "lens_weight_learning_rate": lens_weight_learning_rate,
            "lens_fit_split": "train only",
            "readout_completeness_gate": {"top1_minimum": 0.5, "recall_at_10_minimum": 0.9},
            "readout_vocabulary": "all 107 tokens; no alphanumeric filtering",
        },
        "task": {
            "expression": "u=(a+b) mod p; v=(u*c) mod p; y=(v+d) mod p",
            "modulus": MODULUS,
            "counts": dataset.counts(),
            "all_output_tokens_covered_by_train": True,
            "dataset_fixed_across_training_seeds": True,
        },
        "model": {
            "architecture": "2-layer pre-LN causal Transformer; 32 residual channels; 2 heads; GELU MLPs",
            "parameter_count": parameter_count,
            "pretrained": False,
        },
        "runs": runs,
        "aggregate_accuracy": aggregate_accuracy,
        "paired_untouched_accuracy": paired_accuracy,
        "primary_combined_accuracy_gates": primary_gates,
        "mechanistic_conclusion": {
            "status": "abstain_pending_causal_interventions",
            "reason": (
                "Readout rank, reconstruction error, and multi-seed behavioral differences cannot "
                "by themselves distinguish reusable computation from memorization."
            ),
        },
        "formal_verification": {
            "rocq_closed_theorems": 6,
            "z3_expected_unsat": 5,
            "z3_expected_sat_counterexamples": 1,
            "artifact": "formal_verification/rocq/CMAELensGeneralizationFoundations.v",
            "independent_checker": "formal_verification/verify_cmae_lens_generalization_z3.py",
        },
        "limitations": [
            "A small locally trained modular-arithmetic Transformer, not a Claude-scale reproduction.",
            "The J++ analogue uses cluster Jacobians and learned signed unit-sum weights, but not R-lens/LRP or post-hoc token filtering.",
            "The final-site Jacobian is exactly the identity; early-site Jacobian errors measure a corpus-averaged linear approximation, not semantic completeness.",
            "Calibration is used for checkpoint selection and is not an untouched generalization split.",
            "Three seeds support descriptive paired comparisons, not universal or high-confidence statistical claims.",
            "Causal reuse requires the planned path-sufficiency and attention-path interventions.",
        ],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=101)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--evaluation-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--lens-weight-steps", type=int, default=100)
    parser.add_argument("--lens-weight-learning-rate", type=float, default=0.03)
    arguments = parser.parse_args()
    result = run_transformer_m5b(
        arguments.output,
        dataset_seed=arguments.dataset_seed,
        seeds=tuple(arguments.seeds),
        epochs=arguments.epochs,
        evaluation_every=arguments.evaluation_every,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        auxiliary_weight=arguments.auxiliary_weight,
        clusters=arguments.clusters,
        kmeans_iterations=arguments.kmeans_iterations,
        lens_weight_steps=arguments.lens_weight_steps,
        lens_weight_learning_rate=arguments.lens_weight_learning_rate,
    )
    print(json.dumps(result["primary_combined_accuracy_gates"], indent=2, allow_nan=False))
    print(json.dumps(result["paired_untouched_accuracy"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
