"""Execute the preregistered Milestone-6B counterfactual assurance benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.backdoor_organism import quarantine_top_fraction, random_quarantine
from research.cmae.backdoorbench_organism import (
    CIFARBadNetsMLP,
    CIFARBatch,
    CIFARPool,
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
    damped_ggn_influence,
    fit_binary_fixed_denominator,
    fit_binary_physical_subset,
    fixed_group_partition,
    gradient_similarity_influence,
    query_logit,
    stratified_loo_indices,
    training_loss_scores,
)
from research.cmae.binary_ekfac import (
    binary_ekfac_influence,
    capture_binary_training_raw,
    fit_binary_ekfac,
)
from research.cmae.curvature_influence import RawSampleGradient
from research.cmae.external_attribution import (
    dattri_binary_ekfac_influence,
    kronfluence_binary_ekfac_influence,
    score_agreement,
)
from research.cmae.metrics import average_precision, roc_auc
from research.cmae.out_of_fold_quality import (
    cleanlab_out_of_fold_suspicion,
    out_of_fold_probabilities,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AssuranceConfig:
    train_count: int = 1948
    evaluation_count_per_split: int = 512
    hidden_dim: int = 8
    epochs: int = 300
    learning_rate: float = 0.03
    poison_fraction: float = 0.1
    damping: float = 1e-3
    query_count: int = 16
    group_count: int = 8
    loo_per_stratum: int = 12
    oof_folds: int = 4
    quarantine_budgets: tuple[float, ...] = (0.10, 0.25)


@dataclass
class AttributionRun:
    scores: dict[str, FloatArray]
    raw_samples: tuple[RawSampleGradient, ...]
    raw_query: RawSampleGradient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _evaluation_splits(pool: CIFARPool, count: int, seed: int = 1200) -> tuple[IntArray, IntArray]:
    if count % 2 or 2 * count > pool.labels.size:
        raise ValueError("evaluation split count is invalid")
    rng = np.random.default_rng(seed)
    development: list[IntArray] = []
    final: list[IntArray] = []
    for label in (0, 1):
        eligible = rng.permutation(np.where(pool.labels == label)[0])
        if eligible.size < count:
            raise ValueError("insufficient disjoint evaluation rows")
        development.append(eligible[: count // 2])
        final.append(eligible[count // 2 : count])
    return (
        np.sort(np.concatenate(development).astype(np.int64)),
        np.sort(np.concatenate(final).astype(np.int64)),
    )


def _behavior_for_attacks(
    models: dict[str, CIFARBadNetsMLP], evaluation: CIFARBatch, triggered: torch.Tensor
) -> dict[str, dict[str, float]]:
    return {
        name: cifar_behavior_metrics(model, evaluation, triggered) for name, model in models.items()
    }


def _organism_gate(behavior: dict[str, dict[str, float]]) -> dict[str, Any]:
    contaminated = behavior["standard"]
    clean = behavior["clean"]
    excess = contaminated["attack_success_rate"] - clean["attack_success_rate"]
    checks = {
        "attack_success_at_least_0.80": contaminated["attack_success_rate"] >= 0.80,
        "clean_accuracy_at_least_0.70": contaminated["clean_accuracy"] >= 0.70,
        "excess_asr_at_least_0.25": excess >= 0.25,
    }
    return {"pass": all(checks.values()), "checks": checks, "excess_asr": excess}


def _compute_attribution(
    model: CIFARBadNetsMLP,
    batch: CIFARBatch,
    query_inputs: torch.Tensor,
    config: AssuranceConfig,
) -> AttributionRun:
    ggn = capture_binary_ggn(model, batch.inputs, batch.labels, query_inputs)
    raw_samples = capture_binary_training_raw(model, batch)
    fitted, raw_query = fit_binary_ekfac(
        model, batch, query_inputs, damping=config.damping, raw_samples=raw_samples
    )
    scores = {
        "exact_ggn": damped_ggn_influence(ggn, damping=config.damping),
        "cmae_ekfac": binary_ekfac_influence(
            fitted, raw_query, original_count=batch.labels.shape[0]
        ),
        "gradient_similarity": gradient_similarity_influence(ggn),
        "training_loss": training_loss_scores(ggn),
    }
    return AttributionRun(scores, raw_samples, raw_query)


def _detection_metrics(
    scores: FloatArray, labels: IntArray, budgets: tuple[float, ...]
) -> dict[str, Any]:
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("detection scores and labels are misaligned")
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("detection metrics require both classes")
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
            "quarantine_count": int(quarantined.size),
            "accepted_count": int(accepted.size),
            "recall": true_positive / positives,
            "clean_false_positive_rate": false_positive / negatives,
            "precision": true_positive / quarantined.size,
        }
    return result


def _detection_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "roc_auc_at_least_0.80": metrics["roc_auc"] >= 0.80,
        "recall_at_25_percent_at_least_0.75": metrics["budgets"]["0.25"]["recall"] >= 0.75,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _actual_retraining_effects(
    full_model: CIFARBadNetsMLP,
    batch: CIFARBatch,
    query_inputs: torch.Tensor,
    groups: list[IntArray],
    *,
    seed: int,
    config: AssuranceConfig,
) -> FloatArray:
    full_query = float(query_logit(full_model, query_inputs).detach().cpu().item())
    effects: list[float] = []
    for group in groups:
        weights = torch.ones_like(batch.labels)
        weights[torch.from_numpy(group)] = 0
        reduced = build_cifar_model(seed, hidden_dim=config.hidden_dim)
        fit_binary_fixed_denominator(
            reduced,
            batch.inputs,
            batch.labels,
            weights,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            original_count=batch.labels.shape[0],
        )
        reduced_query = float(query_logit(reduced, query_inputs).detach().cpu().item())
        effects.append(full_query - reduced_query)
    result = np.asarray(effects, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite actual counterfactual effects")
    return result


def _counterfactual_gate(metrics: dict[str, float]) -> dict[str, Any]:
    checks = {
        "group_spearman_at_least_0.50": metrics["spearman"] >= 0.50,
        "group_sign_agreement_at_least_0.65": metrics["sign_agreement"] >= 0.65,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _add_cleanlab_score(
    attribution: AttributionRun,
    batch: CIFARBatch,
    *,
    seed: int,
    config: AssuranceConfig,
) -> dict[str, Any]:
    probabilities, folds = out_of_fold_probabilities(
        lambda fold_seed: build_cifar_model(fold_seed, hidden_dim=config.hidden_dim),
        batch,
        folds=config.oof_folds,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=7100 + seed,
    )
    labels = batch.labels.detach().cpu().numpy().astype(np.int64)
    attribution.scores["cleanlab_quality"] = cleanlab_out_of_fold_suspicion(labels, probabilities)
    return {
        "status": "executed",
        "probabilities": "out_of_fold",
        "folds": config.oof_folds,
        "fold_sizes": np.bincount(folds).tolist(),
        "cleanlab_version": importlib.metadata.version("cleanlab"),
    }


def _counterfactual_evaluation(
    model: CIFARBadNetsMLP,
    batch: CIFARBatch,
    query_inputs: torch.Tensor,
    attribution: AttributionRun,
    *,
    seed: int,
    config: AssuranceConfig,
) -> dict[str, Any]:
    groups = fixed_group_partition(batch.labels.shape[0], groups=config.group_count, seed=7001)
    group_actual = _actual_retraining_effects(
        model, batch, query_inputs, groups, seed=seed, config=config
    )
    loo_indices = stratified_loo_indices(
        batch.attack_labels, per_stratum=config.loo_per_stratum, seed=7002
    )
    loo_groups = [np.asarray([index], dtype=np.int64) for index in loo_indices]
    loo_actual = _actual_retraining_effects(
        model, batch, query_inputs, loo_groups, seed=seed, config=config
    )
    methods = {}
    for name, scores in attribution.scores.items():
        if name in {"training_loss", "cleanlab_quality", "random"}:
            continue
        methods[name] = {
            "leave_group_out": counterfactual_metrics(
                aggregate_scores(scores, groups), group_actual
            ),
            "leave_one_out": counterfactual_metrics(scores[loo_indices], loo_actual),
        }
    return {
        "group_sizes": [int(group.size) for group in groups],
        "groups": groups,
        "group_actual_effects": group_actual,
        "loo_indices": loo_indices,
        "loo_actual_effects": loo_actual,
        "methods": methods,
        "primary_gate": _counterfactual_gate(methods["exact_ggn"]["leave_group_out"]),
    }


def _physical_repair(
    standard_model: CIFARBadNetsMLP,
    standard_batch: CIFARBatch,
    evaluation: CIFARBatch,
    triggered_evaluation: torch.Tensor,
    attribution: AttributionRun,
    behavior: dict[str, dict[str, float]],
    *,
    seed: int,
    config: AssuranceConfig,
    admissible: bool,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    if not admissible:
        return {
            "status": "blocked",
            "reason": "organism, counterfactual, or detection gate failed",
            "upstream": upstream,
        }
    arms: dict[str, Any] = {}
    rankings = {
        "accepted_only": attribution.scores["exact_ggn"],
        "random": attribution.scores["random"],
        "training_loss": attribution.scores["training_loss"],
    }
    for name, scores in rankings.items():
        if name == "random":
            count = int(np.ceil(len(scores) * 0.25))
            accepted, quarantined = random_quarantine(len(scores), count, seed=7300 + seed)
        else:
            accepted, quarantined = quarantine_top_fraction(scores, 0.25)
        accepted_batch = standard_batch.subset(accepted)
        model = build_cifar_model(seed, hidden_dim=config.hidden_dim)
        fit_binary_physical_subset(
            model,
            accepted_batch.inputs,
            accepted_batch.labels,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
        )
        arms[name] = {
            "behavior": cifar_behavior_metrics(model, evaluation, triggered_evaluation),
            "accepted_count": int(accepted.size),
            "quarantined_count": int(quarantined.size),
            "accepted_sample_ids": accepted_batch.sample_ids,
            "quarantined_sample_ids": standard_batch.sample_ids[quarantined],
            "physical_subset_disjoint": bool(
                set(accepted_batch.sample_ids.tolist()).isdisjoint(
                    standard_batch.sample_ids[quarantined].tolist()
                )
            ),
        }
    contaminated = behavior["standard"]
    repair = arms["accepted_only"]["behavior"]
    random_behavior = arms["random"]["behavior"]
    relative_reduction = (
        contaminated["attack_success_rate"] - repair["attack_success_rate"]
    ) / max(contaminated["attack_success_rate"], 1e-12)
    random_reduction = contaminated["attack_success_rate"] - random_behavior["attack_success_rate"]
    accepted_reduction = contaminated["attack_success_rate"] - repair["attack_success_rate"]
    checks = {
        "relative_asr_reduction_at_least_0.50": relative_reduction >= 0.50,
        "clean_accuracy_drop_at_most_0.02": (
            contaminated["clean_accuracy"] - repair["clean_accuracy"] <= 0.02
        ),
        "better_than_random_asr_reduction": accepted_reduction > random_reduction,
    }
    return {
        "status": "executed",
        "arms": arms,
        "gate": {
            "pass": all(checks.values()),
            "checks": checks,
            "relative_asr_reduction": relative_reduction,
            "accepted_minus_random_asr_reduction": accepted_reduction - random_reduction,
        },
        "contaminated_full_query": float(
            query_logit(standard_model, triggered_evaluation[: config.query_count])
            .detach()
            .cpu()
            .item()
        ),
    }


def run_seed(
    seed: int,
    role: str,
    train_pool: CIFARPool,
    development_evaluation: CIFARBatch,
    final_evaluation: CIFARBatch,
    development_pool: CIFARPool,
    development_indices: IntArray,
    final_pool: CIFARPool,
    final_indices: IntArray,
    trigger: NDArray[np.uint8],
    config: AssuranceConfig,
    *,
    external: bool,
    cleanlab: bool,
) -> dict[str, Any]:
    start = time.time()
    torch.set_num_threads(1)
    train_indices = select_stratified_indices(train_pool, config.train_count, seed=seed)
    clean_batch = make_cifar_batch(train_pool, train_indices, trigger, attack="clean", seed=seed)
    clean_model = build_cifar_model(seed, hidden_dim=config.hidden_dim)
    fit_binary_physical_subset(
        clean_model,
        clean_batch.inputs,
        clean_batch.labels,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
    )
    batches = {
        "standard": make_cifar_batch(
            train_pool,
            train_indices,
            trigger,
            attack="standard",
            poison_fraction=config.poison_fraction,
            seed=seed,
        )
    }
    batches["group"] = make_cifar_batch(
        train_pool,
        train_indices,
        trigger,
        attack="group",
        poison_fraction=config.poison_fraction,
        seed=seed,
    )
    batches["adaptive"] = make_cifar_batch(
        train_pool,
        train_indices,
        trigger,
        attack="adaptive",
        poison_fraction=config.poison_fraction,
        seed=seed,
        surrogate=clean_model,
    )
    models = {"clean": clean_model}
    for attack in ("standard", "group", "adaptive"):
        model = build_cifar_model(seed, hidden_dim=config.hidden_dim)
        fit_binary_physical_subset(
            model,
            batches[attack].inputs,
            batches[attack].labels,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
        )
        models[attack] = model
    evaluation = development_evaluation if role == "development" else final_evaluation
    evaluation_pool = development_pool if role == "development" else final_pool
    evaluation_indices = development_indices if role == "development" else final_indices
    source_indices = evaluation_indices[evaluation_pool.labels[evaluation_indices] == 0]
    triggered_evaluation = triggered_source_inputs(evaluation_pool, source_indices, trigger)
    behavior = _behavior_for_attacks(models, evaluation, triggered_evaluation)
    organism = _organism_gate(behavior)

    query_source = development_indices[development_pool.labels[development_indices] == 0][
        : config.query_count
    ]
    query_inputs = triggered_source_inputs(development_pool, query_source, trigger)
    detections: dict[str, Any] = {}
    score_vectors: dict[str, dict[str, FloatArray]] = {}
    standard_attribution: AttributionRun | None = None
    external_results: dict[str, Any] = {}
    for attack in ("standard", "group", "adaptive"):
        attribution = _compute_attribution(models[attack], batches[attack], query_inputs, config)
        if cleanlab:
            external_results[f"{attack}_cleanlab"] = _add_cleanlab_score(
                attribution, batches[attack], seed=seed, config=config
            )
        rng = np.random.default_rng(7200 + seed)
        attribution.scores["random"] = rng.random(batches[attack].labels.shape[0])
        if attack == "standard":
            standard_attribution = attribution
            if external:
                dattri_scores = dattri_binary_ekfac_influence(
                    attribution.raw_samples, attribution.raw_query, damping=config.damping
                )
                kronfluence_scores = kronfluence_binary_ekfac_influence(
                    models[attack], batches[attack], query_inputs, damping=config.damping
                )
                attribution.scores["dattri_ekfac"] = dattri_scores
                attribution.scores["kronfluence_ekfac"] = kronfluence_scores
                external_results["dattri"] = {
                    "status": "executed",
                    "scope": "independent empirical EK-FAC kernels",
                    "agreement_with_cmae": score_agreement(
                        attribution.scores["cmae_ekfac"], dattri_scores
                    ),
                }
                external_results["kronfluence"] = {
                    "status": "executed",
                    "scope": "independent EK-FAC factor and pairwise-score workflow",
                    "agreement_with_cmae": score_agreement(
                        attribution.scores["cmae_ekfac"], kronfluence_scores
                    ),
                    "version": importlib.metadata.version("kronfluence"),
                }
        score_vectors[attack] = {name: scores.copy() for name, scores in attribution.scores.items()}
        detections[attack] = {
            name: _detection_metrics(
                scores, batches[attack].attack_labels, config.quarantine_budgets
            )
            for name, scores in attribution.scores.items()
        }
    if standard_attribution is None:
        raise RuntimeError("standard attribution was not computed")
    counterfactual = _counterfactual_evaluation(
        models["standard"],
        batches["standard"],
        query_inputs,
        standard_attribution,
        seed=seed,
        config=config,
    )
    primary_detection = detections["standard"]["exact_ggn"]
    detection_gate = _detection_gate(primary_detection)
    full_control = build_cifar_model(seed, hidden_dim=config.hidden_dim)
    fit_binary_fixed_denominator(
        full_control,
        batches["standard"].inputs,
        batches["standard"].labels,
        torch.ones_like(batches["standard"].labels),
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        original_count=batches["standard"].labels.shape[0],
    )
    full_control_residual = max(
        float((left - right).abs().max().detach().cpu().item())
        for left, right in zip(
            full_control.parameters(), models["standard"].parameters(), strict=True
        )
    )
    if full_control_residual != 0.0:
        raise RuntimeError("fixed and physical full-objective retraining differ")
    return {
        "seed": seed,
        "role": role,
        "train_sample_ids": batches["standard"].sample_ids,
        "evaluation_sample_ids": evaluation.sample_ids,
        "poison_counts": {
            attack: int(batch.attack_labels.sum()) for attack, batch in batches.items()
        },
        "behavior": behavior,
        "organism_gate": organism,
        "detections": detections,
        "score_vectors": score_vectors,
        "attack_labels": {attack: batch.attack_labels for attack, batch in batches.items()},
        "fixed_full_retraining_max_parameter_residual": full_control_residual,
        "primary_detection_gate": detection_gate,
        "counterfactual": counterfactual,
        "external": external_results,
        "standard_attribution": standard_attribution,
        "standard_batch": batches["standard"],
        "standard_model": models["standard"],
        "evaluation": evaluation,
        "triggered_evaluation": triggered_evaluation,
        "seconds_before_repair": time.time() - start,
    }


def _public_seed_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "standard_attribution",
            "standard_batch",
            "standard_model",
            "evaluation",
            "triggered_evaluation",
        }
    }


def run_benchmark(
    *,
    seeds: tuple[int, ...] = (101, 202, 303),
    development_seed: int = 707,
    external: bool = True,
    cleanlab: bool = True,
    config: AssuranceConfig = AssuranceConfig(),
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source = root / ".venv" / "src"
    train_path = source / "cifar-10-batches-py" / "data_batch_1"
    test_path = source / "cifar-10-batches-py" / "test_batch"
    trigger_path = source / "BackdoorBench" / "resource" / "badnet" / "trigger_image.png"
    train_pool = load_cifar_binary_batch(train_path)
    test_pool = load_cifar_binary_batch(test_path)
    trigger = load_badnet_trigger(trigger_path)
    development_indices, final_indices = _evaluation_splits(
        test_pool, config.evaluation_count_per_split
    )
    development_evaluation = make_cifar_batch(
        test_pool, development_indices, trigger, attack="clean", seed=1200
    )
    final_evaluation = make_cifar_batch(
        test_pool, final_indices, trigger, attack="clean", seed=1201
    )
    results: list[dict[str, Any]] = []
    for seed, role in ((development_seed, "development"), *[(seed, "final") for seed in seeds]):
        print(f"6B {role} seed {seed}: organism/attribution/counterfactual", flush=True)
        result = run_seed(
            seed,
            role,
            train_pool,
            development_evaluation,
            final_evaluation,
            test_pool,
            development_indices,
            test_pool,
            final_indices,
            trigger,
            config,
            external=external,
            cleanlab=cleanlab,
        )
        results.append(result)
        print(
            f"6B {role} seed {seed}: ASR={result['behavior']['standard']['attack_success_rate']:.4f} "
            f"counterfactual={result['counterfactual']['primary_gate']['pass']} "
            f"detection={result['primary_detection_gate']['pass']}",
            flush=True,
        )
    development = results[0]
    development_admissible = bool(
        development["organism_gate"]["pass"]
        and development["counterfactual"]["primary_gate"]["pass"]
        and development["primary_detection_gate"]["pass"]
    )
    for result in results[1:]:
        upstream = {
            "development_admissible": development_admissible,
            "final_organism": result["organism_gate"],
            "final_counterfactual": result["counterfactual"]["primary_gate"],
            "final_detection": result["primary_detection_gate"],
        }
        admissible = bool(
            development_admissible
            and result["organism_gate"]["pass"]
            and result["counterfactual"]["primary_gate"]["pass"]
            and result["primary_detection_gate"]["pass"]
        )
        print(f"6B final seed {result['seed']}: repair admissible={admissible}", flush=True)
        result["repair"] = _physical_repair(
            result["standard_model"],
            result["standard_batch"],
            result["evaluation"],
            result["triggered_evaluation"],
            result["standard_attribution"],
            result["behavior"],
            seed=result["seed"],
            config=config,
            admissible=admissible,
            upstream=upstream,
        )
    final_repairs = [result["repair"] for result in results[1:]]
    integrated_pass = bool(
        development_admissible
        and all(result["organism_gate"]["pass"] for result in results[1:])
        and sum(result["counterfactual"]["primary_gate"]["pass"] for result in results[1:]) >= 2
        and sum(result["primary_detection_gate"]["pass"] for result in results[1:]) >= 2
        and sum(
            repair["status"] == "executed" and repair["gate"]["pass"] for repair in final_repairs
        )
        >= 2
    )
    return {
        "milestone": "6B",
        "status": "completed",
        "integrated_prevention_gate": {
            "pass": integrated_pass,
            "scope": "bounded reduced CIFAR-10 BadNets organism only; not universal prevention",
        },
        "development_admissible": development_admissible,
        "config": asdict(config),
        "seeds": {"development": development_seed, "final": list(seeds)},
        "source_commits": {
            "BackdoorBench": "f02e3534645f0ee63d6848653062cd6c0d6c400d",
            "cifar_mirror": "48ece5ce490482885f7a9a492fa031c1de79519c",
            "kronfluence": "19088a899dd333aae086fe82213a972db5e6c493",
            "dattri_llm": "640359272d61770c203810864f5cd1e6f73e90ca",
        },
        "source_sha256": {
            "cifar_train_batch": _sha256(train_path),
            "cifar_test_batch": _sha256(test_path),
            "badnet_trigger": _sha256(trigger_path),
        },
        "execution": {
            "external_attribution": external,
            "cleanlab": cleanlab,
            "device": "cpu",
            "torch_version": torch.__version__,
        },
        "results": [_public_seed_result(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/cmae/results/counterfactual_assurance_m6b.json"),
    )
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-cleanlab", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(external=not args.skip_external, cleanlab=not args.skip_cleanlab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.output}; integrated prevention pass={result['integrated_prevention_gate']['pass']}"
    )


if __name__ == "__main__":
    main()
