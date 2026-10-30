"""Milestone-three traces, trained SAEs, and retraining-checked attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.metrics import pearson, spearman
from research.cmae.real_benchmark import (
    RealMLP,
    fit_real_model,
    real_activation_features,
)
from research.cmae.real_data import (
    RealBatch,
    load_real_dataset,
    make_real_batch,
    split_real_dataset,
)
from research.cmae.sae import (
    evaluate_sae_readout,
    fit_relu_sae,
    sparse_feature_inventory,
)
from research.cmae.trace_graph import trace_real_mlp, trace_to_dict

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def per_sample_loss_gradients(model: RealMLP, batch: RealBatch) -> FloatArray:
    """Exact per-example cross-entropy gradients at the current parameters."""

    parameters = tuple(model.parameters())
    was_training = model.training
    model.eval()
    rows: list[FloatArray] = []
    try:
        for features, label in zip(batch.features, batch.labels, strict=True):
            value = torch.from_numpy(features).to(
                device=parameters[0].device, dtype=parameters[0].dtype
            )
            target = torch.tensor(label, device=parameters[0].device)
            loss = torch.nn.functional.cross_entropy(model(value.unsqueeze(0)), target.unsqueeze(0))
            gradients = torch.autograd.grad(loss, parameters, retain_graph=False)
            rows.append(
                torch.cat([gradient.reshape(-1) for gradient in gradients])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    finally:
        model.train(was_training)
        model.probe.last_activation = None
    result = np.stack(rows)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("nonfinite per-sample gradient")
    return result


def per_sample_signed_logit_gradients(
    model: RealMLP, batch: RealBatch
) -> tuple[FloatArray, FloatArray]:
    """Return ``grad(logit_correct - logit_other)`` and ``sigmoid(-signed_logit)``.

    For two-class cross-entropy, ``loss_gradient = -residual * signed_gradient``.
    These are the output gradients and residual used by the TRAK-style formula.
    """

    if not np.all((batch.labels == 0) | (batch.labels == 1)):
        raise ValueError("signed-logit attribution requires binary labels")
    parameters = tuple(model.parameters())
    was_training = model.training
    model.eval()
    rows: list[FloatArray] = []
    residuals: list[float] = []
    try:
        for features, label in zip(batch.features, batch.labels, strict=True):
            value = torch.from_numpy(features).to(
                device=parameters[0].device, dtype=parameters[0].dtype
            )
            logits = model(value.unsqueeze(0))[0]
            correct = int(label)
            signed_logit = logits[correct] - logits[1 - correct]
            gradients = torch.autograd.grad(signed_logit, parameters, retain_graph=False)
            rows.append(
                torch.cat([gradient.reshape(-1) for gradient in gradients])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            residuals.append(float(torch.sigmoid(-signed_logit).detach().cpu().item()))
    finally:
        model.train(was_training)
        model.probe.last_activation = None
    gradient_array = np.stack(rows)
    residual_array = np.asarray(residuals, dtype=np.float64)
    if not np.all(np.isfinite(gradient_array)) or not np.all(np.isfinite(residual_array)):
        raise RuntimeError("nonfinite signed-logit attribution input")
    return gradient_array, residual_array


def target_logit_gradient(
    model: RealMLP, features: FloatArray, *, target_class: int = 1
) -> FloatArray:
    parameters = tuple(model.parameters())
    was_training = model.training
    model.eval()
    try:
        value = torch.from_numpy(np.asarray(features, dtype=np.float64)).to(
            device=parameters[0].device, dtype=parameters[0].dtype
        )
        output = model(value)[target_class]
        gradients = torch.autograd.grad(output, parameters, retain_graph=False)
        return (
            torch.cat([gradient.reshape(-1) for gradient in gradients])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    finally:
        model.train(was_training)
        model.probe.last_activation = None


def projected_gradient_attribution(
    sample_gradients: FloatArray,
    query_gradient: FloatArray,
    *,
    rank: int = 8,
    damping: float = 1e-3,
    seed: int = 0,
) -> FloatArray:
    """TRAK-style projected output-gradient attribution with damped solve."""

    gradients = np.asarray(sample_gradients, dtype=np.float64)
    query = np.asarray(query_gradient, dtype=np.float64)
    if gradients.ndim != 2 or query.shape != (gradients.shape[1],):
        raise ValueError("attribution gradients have incompatible shapes")
    if (
        rank <= 0
        or damping <= 0
        or not np.all(np.isfinite(gradients))
        or not np.all(np.isfinite(query))
    ):
        raise ValueError("attribution inputs must be finite with positive rank/damping")
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(gradients.shape[1], rank)) / np.sqrt(rank)
    projected = gradients @ projection
    projected_query = query @ projection
    normal = projected.T @ projected + damping * np.eye(rank)
    solution = np.linalg.solve(normal, projected_query)
    scores = -projected @ solution
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("nonfinite projected attribution")
    return scores


def trak_projected_output_attribution(
    signed_logit_gradients: FloatArray,
    correct_residuals: FloatArray,
    query_gradient: FloatArray,
    *,
    rank: int = 8,
    damping: float = 1e-3,
    seed: int = 0,
) -> FloatArray:
    """TRAK-style output attribution with explicit binary-logit residual scaling."""

    gradients = np.asarray(signed_logit_gradients, dtype=np.float64)
    residuals = np.asarray(correct_residuals, dtype=np.float64)
    query = np.asarray(query_gradient, dtype=np.float64)
    if gradients.ndim != 2 or query.shape != (gradients.shape[1],):
        raise ValueError("TRAK gradients have incompatible shapes")
    if residuals.shape != (gradients.shape[0],):
        raise ValueError("TRAK residuals must align with sample gradients")
    if (
        rank <= 0
        or damping <= 0
        or not np.all(np.isfinite(gradients))
        or not np.all(np.isfinite(residuals))
        or not np.all(np.isfinite(query))
        or np.any(residuals < 0)
        or np.any(residuals > 1)
    ):
        raise ValueError("TRAK inputs must be finite with valid rank/damping/residuals")
    projection = np.random.default_rng(seed).normal(size=(gradients.shape[1], rank)) / np.sqrt(rank)
    projected = gradients @ projection
    projected_query = query @ projection
    normal = projected.T @ projected + damping * np.eye(rank)
    solved = np.linalg.solve(normal, projected_query)
    scores = residuals * (projected @ solved)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("nonfinite TRAK output attribution")
    return scores


def aggregate_groups(scores: FloatArray, groups: list[IntArray]) -> FloatArray:
    covered = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
    if sorted(covered.tolist()) != list(range(scores.size)):
        raise ValueError("groups must partition all sample indices exactly once")
    return np.asarray([scores[group].sum() for group in groups], dtype=np.float64)


def group_counterfactual_retraining(
    full_model: RealMLP,
    fit_batch: RealBatch,
    query: FloatArray,
    groups: list[IntArray],
    *,
    target_class: int = 1,
    seed: int = 0,
    epochs: int = 60,
) -> FloatArray:
    """Measure full-minus-retrained target-logit changes for disjoint groups."""

    aggregate_groups(np.zeros(fit_batch.features.shape[0]), groups)
    model_device = full_model.input.weight.device
    model_dtype = full_model.input.weight.dtype
    full_model.eval()
    with torch.no_grad():
        value = torch.from_numpy(query).to(device=model_device, dtype=model_dtype)
        full_output = float(full_model(value)[target_class].cpu().item())
    effects: list[float] = []
    for group in groups:
        keep = np.setdiff1d(np.arange(fit_batch.features.shape[0]), group)
        torch.manual_seed(seed)
        reduced = RealMLP(fit_batch.features.shape[1]).to(device=model_device, dtype=model_dtype)
        fit_real_model(reduced, fit_batch.subset(keep), epochs=epochs)
        reduced.eval()
        with torch.no_grad():
            reduced_output = float(reduced(value)[target_class].cpu().item())
        effects.append(full_output - reduced_output)
    result = np.asarray(effects, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("nonfinite group counterfactual")
    return result


def counterfactual_metrics(predicted: FloatArray, actual: FloatArray) -> dict[str, float]:
    if predicted.shape != actual.shape or predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("counterfactual predictions must align and contain at least two groups")
    return {
        "pearson": pearson(predicted, actual),
        "spearman": spearman(predicted, actual),
        "sign_agreement": float(np.mean((predicted > 0) == (actual > 0))),
        "mean_absolute_error_raw_scale": float(np.mean(np.abs(predicted - actual))),
    }


def stratified_subset(batch: RealBatch, count: int, seed: int) -> RealBatch:
    if count < 4 or count > batch.features.shape[0]:
        raise ValueError("subset count is outside the batch")
    rng = np.random.default_rng(seed)
    classes = np.unique(batch.labels)
    per_class = count // len(classes)
    selected: list[int] = []
    for label in classes:
        candidates = np.where(batch.labels == label)[0]
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    remaining = np.setdiff1d(np.arange(batch.features.shape[0]), np.asarray(selected))
    selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    return batch.subset(np.asarray(sorted(selected), dtype=np.int64))


def run_deep_case(
    name: Literal["breast_cancer", "digits_binary"],
    attack: Literal["standard", "adaptive"],
    seed: int,
    *,
    fit_count: int = 64,
    group_count: int = 8,
    training_epochs: int = 60,
    sae_epochs: int = 300,
) -> dict[str, object]:
    """Run one case synchronously; no result is treated as a universal guarantee."""

    dataset = load_real_dataset(name)
    splits = split_real_dataset(dataset, seed)
    fit_batch = stratified_subset(
        make_real_batch(splits.fit, attack, seed + 1), fit_count, seed + 11
    )
    clean_test = make_real_batch(splits.test, "clean", seed + 2)
    torch.manual_seed(seed + 5)
    model = RealMLP(dataset.features.shape[1])
    training_losses = fit_real_model(model, fit_batch, epochs=training_epochs)
    query = clean_test.features[0]
    sample_gradients = per_sample_loss_gradients(model, fit_batch)
    signed_gradients, correct_residuals = per_sample_signed_logit_gradients(model, fit_batch)
    if not np.allclose(
        sample_gradients,
        -correct_residuals[:, None] * signed_gradients,
        atol=2e-6,
        rtol=2e-5,
    ):
        raise RuntimeError("signed-logit and cross-entropy gradient identities disagree")
    query_gradient = target_logit_gradient(model, query)
    gradient_scores = -(sample_gradients @ query_gradient)
    trak_scores = trak_projected_output_attribution(
        signed_gradients,
        correct_residuals,
        query_gradient,
        rank=8,
        damping=1e-3,
        seed=seed + 21,
    )
    groups = [
        np.asarray(group, dtype=np.int64)
        for group in np.array_split(
            np.random.default_rng(seed + 31).permutation(fit_count), group_count
        )
    ]
    actual_group_effects = group_counterfactual_retraining(
        model,
        fit_batch,
        query,
        groups,
        seed=seed + 5,
        epochs=training_epochs,
    )
    predicted = {
        "gradient_dot": aggregate_groups(gradient_scores, groups),
        "trak_projected": aggregate_groups(trak_scores, groups),
    }
    counterfactuals = {
        method: counterfactual_metrics(scores, actual_group_effects)
        for method, scores in predicted.items()
    }
    trace = trace_real_mlp(model, query)
    fit_activations = real_activation_features(model, fit_batch.features)
    test_activations = real_activation_features(model, clean_test.features)
    sae, sae_losses = fit_relu_sae(
        fit_activations, latent_dim=32, epochs=sae_epochs, seed=seed + 41
    )
    sae_metrics = evaluate_sae_readout(sae, model.head, test_activations)
    inventory = sparse_feature_inventory(sae, test_activations)
    prominent = sorted(inventory, key=lambda row: float(row["max_activation"]), reverse=True)[:5]
    return {
        "dataset": name,
        "attack": attack,
        "seed": seed,
        "fit_samples": fit_count,
        "group_count": group_count,
        "training_epochs": training_epochs,
        "final_training_loss": training_losses[-1],
        "trace": trace_to_dict(trace),
        "sae": {
            "architecture": "trained ReLU SAE; 16 real hidden activations -> 32 latents",
            "fit_split": "poisoned fit activations only",
            "evaluation_split": "clean held-out test activations",
            "epochs": sae_epochs,
            "initial_loss": sae_losses[0],
            "final_loss": sae_losses[-1],
            "metrics": sae_metrics.to_dict(),
            "prominent_unlabeled_features": prominent,
            "semantic_scope": "numeric inventory only; no NLA/provider-backed semantic label was generated",
        },
        "attribution_target": {
            "query": "class-1 logit on the first clean test row",
            "actual": "full-model minus group-retrained class-1 logit",
            "gradient_baseline": "negative per-sample cross-entropy gradient dot query gradient",
            "trak": "signed correct-minus-other logit gradients, sigmoid residual, damped rank-8 projection",
            "identity_checked": "loss_gradient = -residual * signed_logit_gradient on every fit row",
        },
        "counterfactuals": counterfactuals,
        "counterfactual_arrays": {
            "actual_group_effects": actual_group_effects.tolist(),
            "gradient_dot": predicted["gradient_dot"].tolist(),
            "trak_projected": predicted["trak_projected"].tolist(),
        },
        "incremental_value": {
            "trak_minus_gradient_pearson": counterfactuals["trak_projected"]["pearson"]
            - counterfactuals["gradient_dot"]["pearson"],
            "trak_minus_gradient_spearman": counterfactuals["trak_projected"]["spearman"]
            - counterfactuals["gradient_dot"]["spearman"],
            "scope": "one disjoint-group partition and fixed training budget; descriptive, not a datamodel estimate",
        },
        "claim_scope": "exact linear-readout edges plus measured path residual; SAE and group retrains are empirical evidence, not complete causal graphs",
    }


def run_deep_benchmark(output: Path) -> dict[str, object]:
    cases = []
    for name in ("breast_cancer", "digits_binary"):
        for attack in ("standard", "adaptive"):
            cases.append(run_deep_case(name, attack, 101))
    result: dict[str, object] = {
        "schema_version": 1,
        "execution": "sequential",
        "cases": cases,
        "evaluation_scope": "development benchmark; one seed and no untouched confirmatory holdout",
        "limitations": [
            "Two real sklearn datasets, one seed, and small MLPs only; no Transformer/NLA/Tracr claim.",
            "Hidden readout edges are exact; input paths are first-order and retain residuals.",
            "Group counterfactuals are actual retrains but not leave-one-out datamodels or optimal adversaries.",
            "SAE latents are unlabeled numerical features; reconstruction does not establish semantic faithfulness.",
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
    arguments = parser.parse_args()
    result = run_deep_benchmark(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "case_count": len(cast(list[dict[str, object]], result["cases"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
