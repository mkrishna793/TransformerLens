"""Fixed-target arithmetic organism, EK-FAC counterfactuals, and gated repair."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.arithmetic_data import (
    ArithmeticSample,
    ArithmeticSplit,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_fixed_target import FixedTargetBackdoorOrganism
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.backdoor_organism import (
    evaluate_backdoor_organism,
    quarantine_top_fraction,
    random_quarantine,
    subset_samples,
    train_arithmetic_fixed_denominator,
    train_arithmetic_samples,
    training_loss_scores,
)
from research.cmae.binary_counterfactual import (
    aggregate_scores,
    counterfactual_metrics,
    fixed_group_partition,
    stratified_loo_indices,
)
from research.cmae.curvature_influence import fit_arithmetic_ekfac
from research.cmae.metrics import average_precision, roc_auc
from research.cmae.run_counterfactual_assurance import _json_ready

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ArithmeticAssuranceConfig:
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 0.005
    damping: float = 0.001
    target_token: int = 0
    group_count: int = 4
    loo_per_stratum: int = 4
    quarantine_budgets: tuple[float, ...] = (0.10, 0.25)


def _query_logit(model: ArithmeticTransformer, sample: ArithmeticSample, target: int) -> float:
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return float(model(tokens)[0, -1, target].detach().cpu().item())
    finally:
        model.train(was_training)


def _detection_metrics(
    scores: FloatArray, labels: IntArray, budgets: tuple[float, ...]
) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = labels.size - positives
    if scores.shape != labels.shape or positives == 0 or negatives == 0:
        raise ValueError("arithmetic detection requires aligned scores and both classes")
    result: dict[str, Any] = {
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "budgets": {},
    }
    for budget in budgets:
        accepted, quarantined = quarantine_top_fraction(scores, budget)
        true_positive = int(labels[quarantined].sum())
        false_positive = int(quarantined.size - true_positive)
        result["budgets"][str(budget)] = {
            "accepted_count": int(accepted.size),
            "quarantine_count": int(quarantined.size),
            "recall": true_positive / positives,
            "clean_false_positive_rate": false_positive / negatives,
            "precision": true_positive / quarantined.size,
        }
    return result


def _organism_gate(
    contaminated: dict[str, float | int], clean: dict[str, float | int]
) -> dict[str, Any]:
    nontrigger_drop = float(clean["nontrigger_accuracy"]) - float(
        contaminated["nontrigger_accuracy"]
    )
    checks = {
        "attack_success_at_least_0.80": contaminated["trigger_backdoor_rate"] >= 0.80,
        "clean_reference_asr_at_most_0.20": clean["trigger_backdoor_rate"] <= 0.20,
        "nontrigger_drop_at_most_0.05": nontrigger_drop <= 0.05,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "nontrigger_accuracy_drop": nontrigger_drop,
    }


def _detection_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "roc_auc_at_least_0.80": metrics["roc_auc"] >= 0.80,
        "recall_at_25_percent_at_least_0.75": metrics["budgets"]["0.25"]["recall"] >= 0.75,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _counterfactual_gate(metrics: dict[str, float]) -> dict[str, Any]:
    checks = {
        "group_spearman_at_least_0.50": metrics["spearman"] >= 0.50,
        "group_sign_agreement_at_least_0.65": metrics["sign_agreement"] >= 0.65,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _actual_effects(
    full_model: ArithmeticTransformer,
    initial_state: dict[str, torch.Tensor],
    samples: tuple[ArithmeticSample, ...],
    targets: IntArray,
    groups: list[IntArray],
    canary: ArithmeticSample,
    target_token: int,
    *,
    seed: int,
    config: ArithmeticAssuranceConfig,
) -> FloatArray:
    full_query = _query_logit(full_model, canary, target_token)
    weights = np.ones(len(samples), dtype=np.float64)
    effects: list[float] = []
    for group in groups:
        group_weights = weights.copy()
        group_weights[group] = 0
        reduced = ArithmeticTransformer()
        reduced.load_state_dict(initial_state)
        train_arithmetic_fixed_denominator(
            reduced,
            samples,
            targets,
            group_weights,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=seed + 1,
        )
        effects.append(full_query - _query_logit(reduced, canary, target_token))
    result = np.asarray(effects, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite arithmetic actual effects")
    return result


def _behavior_difference(
    contaminated: dict[str, float | int], repaired: dict[str, float | int]
) -> dict[str, float]:
    baseline = float(contaminated["trigger_backdoor_rate"])
    remaining = float(repaired["trigger_backdoor_rate"])
    return {
        "absolute_asr_reduction": baseline - remaining,
        "relative_asr_reduction": (baseline - remaining) / max(baseline, 1e-12),
        "clean_accuracy_change": float(repaired["clean_accuracy"])
        - float(contaminated["clean_accuracy"]),
        "nontrigger_accuracy_change": float(repaired["nontrigger_accuracy"])
        - float(contaminated["nontrigger_accuracy"]),
    }


def _repair(
    initial_state: dict[str, torch.Tensor],
    samples: tuple[ArithmeticSample, ...],
    targets: IntArray,
    organism: FixedTargetBackdoorOrganism,
    scores: dict[str, FloatArray],
    contaminated_behavior: dict[str, float | int],
    split: ArithmeticSplit,
    *,
    seed: int,
    config: ArithmeticAssuranceConfig,
    admissible: bool,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    if not admissible:
        return {"status": "blocked", "reason": "upstream gate failed", "upstream": upstream}
    if split not in {"iid", "operand"}:
        raise ValueError("unsupported arithmetic repair evaluation split")
    arms: dict[str, Any] = {}
    for name in ("cmae_ekfac", "random", "training_loss"):
        if name == "random":
            count = int(np.ceil(len(samples) * 0.25))
            accepted, quarantined = random_quarantine(len(samples), count, seed=7500 + seed)
        else:
            accepted, quarantined = quarantine_top_fraction(scores[name], 0.25)
        accepted_samples, accepted_targets = subset_samples(samples, targets, accepted)
        if {sample.key for sample in accepted_samples} & {
            samples[int(index)].key for index in quarantined
        }:
            raise RuntimeError("arithmetic quarantine leaked into physical accepted-only data")
        model = ArithmeticTransformer()
        model.load_state_dict(initial_state)
        train_arithmetic_samples(
            model,
            accepted_samples,
            accepted_targets,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=seed + 1,
        )
        arms[name] = {
            "behavior": evaluate_backdoor_organism(model, organism, split=split),
            "accepted_count": len(accepted_samples),
            "quarantined_count": len(quarantined),
            "accepted_keys": [sample.key for sample in accepted_samples],
            "quarantined_keys": [samples[int(index)].key for index in quarantined],
        }
    difference = _behavior_difference(contaminated_behavior, arms["cmae_ekfac"]["behavior"])
    random_difference = _behavior_difference(contaminated_behavior, arms["random"]["behavior"])
    checks = {
        "relative_asr_reduction_at_least_0.50": difference["relative_asr_reduction"] >= 0.50,
        "nontrigger_drop_at_most_0.05": difference["nontrigger_accuracy_change"] >= -0.05,
        "better_than_random_asr_reduction": difference["absolute_asr_reduction"]
        > random_difference["absolute_asr_reduction"],
    }
    return {
        "status": "executed",
        "arms": arms,
        "gate": {"pass": all(checks.values()), "checks": checks},
        "difference": difference,
        "random_difference": random_difference,
    }


def run_arithmetic_benchmark(
    *,
    seeds: tuple[int, ...] = (101, 202, 303),
    development_seed: int = 707,
    config: ArithmeticAssuranceConfig = ArithmeticAssuranceConfig(),
) -> dict[str, Any]:
    dataset = make_arithmetic_dataset(101)
    organism = FixedTargetBackdoorOrganism(dataset, target_token=config.target_token)
    samples = organism.training_samples()
    targets = organism.training_targets()
    clean_targets = organism.clean_training_targets()
    labels = organism.poison_mask().astype(np.int64)
    canary = organism.calibration_canary()
    runs: list[dict[str, Any]] = []
    run_specs: list[tuple[int, str, ArithmeticSplit]] = [(development_seed, "development", "iid")]
    run_specs.extend((seed, "final", "operand") for seed in seeds)
    for seed, role, split in run_specs:
        start = time.time()
        print(f"Arithmetic 6B {role} seed {seed}", flush=True)
        torch.manual_seed(seed)
        initial = ArithmeticTransformer()
        initial_state = {
            name: value.detach().cpu().clone() for name, value in initial.state_dict().items()
        }
        models: dict[str, ArithmeticTransformer] = {}
        behavior: dict[str, dict[str, float | int]] = {}
        for name, arm_targets in (("clean", clean_targets), ("standard", targets)):
            model = ArithmeticTransformer()
            model.load_state_dict(initial_state)
            train_arithmetic_fixed_denominator(
                model,
                samples,
                arm_targets,
                np.ones(len(samples), dtype=np.float64),
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=seed + 1,
            )
            models[name] = model
            behavior[name] = evaluate_backdoor_organism(model, organism, split=split)
        organism_gate = _organism_gate(behavior["standard"], behavior["clean"])
        print(f"  fit EK-FAC; organism gate={organism_gate['pass']}", flush=True)
        influence = fit_arithmetic_ekfac(
            models["standard"], samples, targets, damping=config.damping
        )
        ekfac_scores = influence.logit_influence(
            models["standard"], canary, config.target_token
        ) / len(samples)
        loss_scores = training_loss_scores(models["standard"], samples, targets)
        random_scores = np.random.default_rng(7600 + seed).random(len(samples))
        scores = {
            "cmae_ekfac": ekfac_scores,
            "training_loss": loss_scores,
            "random": random_scores,
        }
        detections = {
            name: _detection_metrics(values, labels, config.quarantine_budgets)
            for name, values in scores.items()
        }
        groups = fixed_group_partition(len(samples), groups=config.group_count, seed=7401)
        group_actual = _actual_effects(
            models["standard"],
            initial_state,
            samples,
            targets,
            groups,
            canary,
            config.target_token,
            seed=seed,
            config=config,
        )
        loo_indices = stratified_loo_indices(labels, per_stratum=config.loo_per_stratum, seed=7402)
        loo_groups = [np.asarray([index], dtype=np.int64) for index in loo_indices]
        loo_actual = _actual_effects(
            models["standard"],
            initial_state,
            samples,
            targets,
            loo_groups,
            canary,
            config.target_token,
            seed=seed,
            config=config,
        )
        group_predicted = aggregate_scores(ekfac_scores, groups)
        group_metrics = counterfactual_metrics(group_predicted, group_actual)
        loo_metrics = counterfactual_metrics(ekfac_scores[loo_indices], loo_actual)
        counterfactual_gate = _counterfactual_gate(group_metrics)
        detection_gate = _detection_gate(detections["cmae_ekfac"])
        runs.append(
            {
                "seed": seed,
                "role": role,
                "evaluation_split": split,
                "behavior": behavior,
                "organism_gate": organism_gate,
                "detections": detections,
                "detection_gate": detection_gate,
                "scores": scores,
                "poison_mask": labels,
                "training_keys": [sample.key for sample in samples],
                "counterfactual": {
                    "groups": groups,
                    "group_predicted": group_predicted,
                    "group_actual": group_actual,
                    "leave_group_out": group_metrics,
                    "loo_indices": loo_indices,
                    "loo_predicted": ekfac_scores[loo_indices],
                    "loo_actual": loo_actual,
                    "leave_one_out": loo_metrics,
                    "gate": counterfactual_gate,
                },
                "seconds_before_repair": time.time() - start,
                "initial_state": initial_state,
            }
        )
        print(
            f"  counterfactual={counterfactual_gate['pass']}; detection={detection_gate['pass']}",
            flush=True,
        )
    development = runs[0]
    development_admissible = bool(
        development["organism_gate"]["pass"]
        and development["detection_gate"]["pass"]
        and development["counterfactual"]["gate"]["pass"]
    )
    for run in runs[1:]:
        upstream = {
            "development_admissible": development_admissible,
            "organism": run["organism_gate"],
            "detection": run["detection_gate"],
            "counterfactual": run["counterfactual"]["gate"],
        }
        admissible = bool(
            development_admissible
            and run["organism_gate"]["pass"]
            and run["detection_gate"]["pass"]
            and run["counterfactual"]["gate"]["pass"]
        )
        run["repair"] = _repair(
            run["initial_state"],
            samples,
            targets,
            organism,
            {
                "cmae_ekfac": run["scores"]["cmae_ekfac"],
                "training_loss": run["scores"]["training_loss"],
                "random": run["scores"]["random"],
            },
            run["behavior"]["standard"],
            "operand",
            seed=run["seed"],
            config=config,
            admissible=admissible,
            upstream=upstream,
        )
    integrated = bool(
        development_admissible
        and sum(run["organism_gate"]["pass"] for run in runs[1:]) >= 2
        and sum(run["detection_gate"]["pass"] for run in runs[1:]) >= 2
        and sum(run["counterfactual"]["gate"]["pass"] for run in runs[1:]) >= 2
        and sum(
            run["repair"]["status"] == "executed" and run["repair"]["gate"]["pass"]
            for run in runs[1:]
        )
        >= 2
    )
    for run in runs:
        del run["initial_state"]
    return {
        "milestone": "6B_arithmetic_fixed_target",
        "status": "completed",
        "config": asdict(config),
        "seeds": {"development": development_seed, "final": list(seeds)},
        "development_admissible": development_admissible,
        "integrated_prevention_gate": {
            "pass": integrated,
            "scope": "bounded fixed-target arithmetic organism only",
        },
        "curvature_scope": (
            "CMAE empirical EK-FAC over tracked arithmetic linear weights; fixed-N score scaling; "
            "embeddings, untracked parameters, and cross-layer curvature omitted"
        ),
        "results": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/cmae/results/arithmetic_fixed_target_m6b.json"),
    )
    args = parser.parse_args()
    result = run_arithmetic_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; integrated pass={result['integrated_prevention_gate']['pass']}")


if __name__ == "__main__":
    main()
