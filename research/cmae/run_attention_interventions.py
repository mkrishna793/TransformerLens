"""Reproduce Milestone-5B checkpoints and run sequential causal path interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from research.cmae.arithmetic_data import ArithmeticSplit, make_arithmetic_dataset
from research.cmae.arithmetic_transformer import (
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
)
from research.cmae.attention_interventions import (
    AttentionPathTrace,
    ResidualPatchTrace,
    attention_path_sweep,
    choose_counterfactual_donor,
    counterfactual_residual_patch,
)
from research.cmae.generalization import TrainingArm, train_with_calibration
from research.cmae.metrics import pearson
from research.cmae.run_transformer_m5b import UNTOUCHED_SPLITS

INTERVENTION_ARMS: tuple[TrainingArm, ...] = ("standard", "auxiliary")


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected integer JSON value, got {type(value).__name__}")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric JSON value, got {type(value).__name__}")
    return float(value)


def _attention_summary(traces: list[AttentionPathTrace]) -> dict[str, float]:
    selected_correct: list[float] = []
    best_correct: list[float] = []
    selected_absolute_effects: list[float] = []
    qk_pearson: list[float] = []
    attention_pearson: list[float] = []
    conservation: list[float] = []
    for trace in traces:
        for head in trace.heads:
            attention_key = int(np.argmax(head.attention_weights))
            best_key = int(np.argmax(head.target_logit_effects))
            selected_correct.append(
                float(head.predicted_tokens[attention_key] == trace.target_token)
            )
            best_correct.append(float(head.predicted_tokens[best_key] == trace.target_token))
            selected_absolute_effects.append(abs(head.target_logit_effects[attention_key]))
            qk = np.asarray(head.qk_scores, dtype=np.float64)
            attention = np.asarray(head.attention_weights, dtype=np.float64)
            effects = np.asarray(head.target_logit_effects, dtype=np.float64)
            qk_pearson.append(pearson(qk, effects))
            attention_pearson.append(pearson(attention, np.abs(effects)))
            conservation.append(head.one_hot_conservation_error)
    return {
        "attention_selected_correct_rate": float(np.mean(selected_correct)),
        "best_case_target_correct_rate": float(np.mean(best_correct)),
        "mean_absolute_attention_selected_effect": float(np.mean(selected_absolute_effects)),
        "mean_qk_signed_effect_pearson": float(np.mean(qk_pearson)),
        "mean_attention_absolute_effect_pearson": float(np.mean(attention_pearson)),
        "max_one_hot_conservation_error": max(conservation),
    }


def _patch_summary(traces: list[ResidualPatchTrace]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for site in ("early", "post_attention", "final"):
        selected = [trace for trace in traces if trace.site == site]
        summary[site] = {
            "samples": len(selected),
            "counterfactual_success_rate": float(
                np.mean(
                    [trace.patched_predicted == trace.counterfactual_target for trace in selected]
                )
            ),
            "donor_answer_success_rate": float(
                np.mean([trace.patched_predicted == trace.donor_target for trace in selected])
            ),
            "donor_prediction_match_rate": float(
                np.mean([trace.patched_predicted == trace.donor_predicted for trace in selected])
            ),
            "mean_counterfactual_logit_effect": float(
                np.mean([trace.actual_effect for trace in selected])
            ),
            "mean_absolute_first_order_residual": float(
                np.mean([abs(trace.first_order_residual) for trace in selected])
            ),
            "first_order_residual_over_actual_effect_mass": float(
                sum(abs(trace.first_order_residual) for trace in selected)
                / max(sum(abs(trace.actual_effect) for trace in selected), 1e-12)
            ),
            "mean_donor_logit_parity_error": float(
                np.mean([trace.donor_logit_parity_error for trace in selected])
            ),
            "max_donor_logit_parity_error": max(
                trace.donor_logit_parity_error for trace in selected
            ),
            "mean_patch_delta_norm": float(np.mean([trace.patch_delta_norm for trace in selected])),
        }
    return summary


def run_attention_interventions(
    output: Path,
    benchmark_path: Path,
    *,
    seeds: tuple[int, ...] = (101, 202, 303),
    samples_per_split: int = 2,
) -> dict[str, object]:
    """Reproduce saved checkpoints exactly before evaluating fixed intervention prompts."""

    if samples_per_split <= 0 or not seeds:
        raise ValueError("intervention seeds and samples per split must be positive")
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
    early_values: dict[tuple[int, str, str], float] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        initial_model = ArithmeticTransformer()
        initial_state = {
            name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()
        }
        for arm in INTERVENTION_ARMS:
            expected_run = indexed[(seed, arm)]
            expected_training = cast(dict[str, object], expected_run["training"])
            print(f"Path interventions: seed={seed} arm={arm}", flush=True)
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
                raise RuntimeError(
                    f"checkpoint reproduction failed for seed={seed} arm={arm}: "
                    f"epoch {training.best_epoch}/{expected_epoch}, "
                    f"loss {training.best_calibration_loss}/{expected_loss}"
                )
            combined = evaluate_arithmetic_transformer(
                model, dataset, "combined", batch_size=batch_size
            )
            expected_evaluation = cast(dict[str, object], expected_run["evaluation"])
            expected_combined = cast(dict[str, object], expected_evaluation["combined"])
            expected_accuracy = _number(expected_combined["accuracy"])
            if not np.isclose(combined["accuracy"], expected_accuracy, atol=1e-12, rtol=0):
                raise RuntimeError("reproduced checkpoint combined-accuracy fingerprint differs")
            attention_traces: list[AttentionPathTrace] = []
            patch_traces: list[ResidualPatchTrace] = []
            sample_records: list[dict[str, object]] = []
            for split in UNTOUCHED_SPLITS:
                samples = dataset.split_samples(split)[:samples_per_split]
                for sample in samples:
                    attention = attention_path_sweep(model, sample)
                    donor = choose_counterfactual_donor(dataset, sample)
                    patches = [
                        counterfactual_residual_patch(model, sample, donor, site=site)
                        for site in ("early", "post_attention", "final")
                    ]
                    attention_traces.append(attention)
                    patch_traces.extend(patches)
                    sample_records.append(
                        {
                            "attention": attention.to_dict(),
                            "residual_patches": [patch.to_dict() for patch in patches],
                        }
                    )
            attention_summary = _attention_summary(attention_traces)
            patch_summary = _patch_summary(patch_traces)
            for metric, value in patch_summary["early"].items():
                early_values[(seed, arm, metric)] = value
            early_success = patch_summary["early"]["counterfactual_success_rate"]
            early_donor_match = patch_summary["early"]["donor_prediction_match_rate"]
            if early_success >= 0.5 and early_donor_match >= 0.5:
                path_status = "counterfactual_transfer_with_donor_prediction_parity"
            elif early_success >= 0.5:
                path_status = "counterfactual_transfer_without_donor_prediction_parity"
            else:
                path_status = "below_counterfactual_transfer_threshold"
            summary = {
                "attention": attention_summary,
                "residual_patches": patch_summary,
                "whole_vector_path_status": path_status,
                "u_specific_mechanistic_claim": "abstain_whole_vector_confound",
            }
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "checkpoint_reproduction": {
                        "best_epoch": training.best_epoch,
                        "best_calibration_loss": training.best_calibration_loss,
                        "combined_accuracy": float(combined["accuracy"]),
                        "matches_benchmark": True,
                    },
                    "summary": summary,
                    "samples": sample_records,
                }
            )

    paired: dict[str, object] = {}
    for metric in (
        "counterfactual_success_rate",
        "donor_prediction_match_rate",
        "mean_counterfactual_logit_effect",
        "mean_absolute_first_order_residual",
        "first_order_residual_over_actual_effect_mass",
    ):
        differences = [
            early_values[(seed, "auxiliary", metric)] - early_values[(seed, "standard", metric)]
            for seed in seeds
        ]
        paired[f"auxiliary_minus_standard_early_{metric}"] = {
            "per_seed": differences,
            "mean": float(np.mean(differences)),
            "positive_on_all_seeds": all(value > 0 for value in differences),
        }
    result: dict[str, object] = {
        "schema_version": 1,
        "milestone": "5B attention-path and counterfactual path-sufficiency interventions",
        "execution": "sequential",
        "benchmark_source": str(benchmark_path),
        "checkpoint_reproduction_verified": True,
        "preregistration": {
            "seeds": list(seeds),
            "arms": list(INTERVENTION_ARMS),
            "samples_per_untouched_split": samples_per_split,
            "prompt_selection": "first dataset-order prompts; no model-output filtering",
            "donor_selection": "nearest different u with identical c,d; deterministic key tie-break",
            "primary_patch": "early whole-residual answer-position patch",
            "controls": [
                "post-attention residual patch",
                "donor-prediction parity at every site",
                "final-residual exact donor-logit parity",
            ],
            "counterfactual_transfer_threshold": 0.5,
            "donor_prediction_parity_threshold": 0.5,
            "status_rule": (
                "report counterfactual transfer and donor-prediction parity separately; "
                "neither establishes u-specific sufficiency"
            ),
            "u_specific_claim": "always abstain because the patch contains the whole residual vector",
        },
        "runs": runs,
        "paired_early_patch_summary": paired,
        "conclusion": {
            "status": "abstain_u_specific_mechanism",
            "reason": (
                "Whole-vector counterfactual transfer and donor-readout parity do not isolate a "
                "reusable u feature; one-hot attention paths are off-distribution interventions."
            ),
        },
        "formal_verification": {
            "rocq_closed_theorems": 7,
            "z3_expected_unsat": 7,
            "z3_expected_sat_counterexamples": 2,
            "artifact": "formal_verification/rocq/CMAEAttentionPathFoundations.v",
            "independent_checker": "formal_verification/verify_cmae_attention_path_z3.py",
        },
        "limitations": [
            "Attention weights and QK scores are read statistics, not causal attributions.",
            "Forced one-hot patterns are valid distributions but can be off the natural attention distribution.",
            "Whole residual patches can transfer donor c,d or other features in addition to u.",
            "Donor-prediction parity measures output reconstruction, not mechanistic faithfulness.",
            "Final-site exact donor-logit parity demonstrates readout reconstruction only.",
            "First-order residual mass is a local approximation diagnostic, not a completeness certificate.",
            "Fixed small prompt samples support scoped diagnostics, not universal circuit claims.",
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
    parser.add_argument("--samples-per-split", type=int, default=2)
    arguments = parser.parse_args()
    result = run_attention_interventions(
        arguments.output,
        arguments.benchmark,
        seeds=tuple(arguments.seeds),
        samples_per_split=arguments.samples_per_split,
    )
    print(json.dumps(result["paired_early_patch_summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
