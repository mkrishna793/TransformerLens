"""Exact QK measurements and causal attention/residual path interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from research.cmae.arithmetic_data import ArithmeticDataset, ArithmeticSample
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.metrics import pearson, spearman

PatchSite = Literal["early", "post_attention", "final"]
PATCH_HOOKS: dict[PatchSite, str] = {
    "early": "blocks.0.hook_resid_post",
    "post_attention": "blocks.1.hook_resid_mid",
    "final": "hook_resid_final",
}


@dataclass(frozen=True)
class HeadPathMeasurement:
    layer: int
    head: int
    qk_scores: tuple[float, ...]
    attention_weights: tuple[float, ...]
    target_logit_effects: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    predicted_tokens: tuple[int, ...]
    one_hot_conservation_error: float

    def to_dict(self, target_token: int) -> dict[str, object]:
        qk = np.asarray(self.qk_scores, dtype=np.float64)
        attention = np.asarray(self.attention_weights, dtype=np.float64)
        effects = np.asarray(self.target_logit_effects, dtype=np.float64)
        predicted = np.asarray(self.predicted_tokens, dtype=np.int64)
        qk_key = int(np.argmax(qk))
        attention_key = int(np.argmax(attention))
        best_target_key = int(np.argmax(effects))
        best_absolute_key = int(np.argmax(np.abs(effects)))
        return {
            "layer": self.layer,
            "head": self.head,
            "qk_scores": list(self.qk_scores),
            "attention_weights": list(self.attention_weights),
            "target_logit_effects": list(self.target_logit_effects),
            "target_probabilities": list(self.target_probabilities),
            "predicted_tokens": list(self.predicted_tokens),
            "effect_convention": "intervened true-target logit minus baseline; positive improves target",
            "one_hot_conservation_error": self.one_hot_conservation_error,
            "qk_selected_key": qk_key,
            "attention_selected_key": attention_key,
            "qk_selected_effect": float(effects[qk_key]),
            "attention_selected_effect": float(effects[attention_key]),
            "attention_selected_correct": bool(predicted[attention_key] == target_token),
            "best_target_key": best_target_key,
            "best_target_effect": float(effects[best_target_key]),
            "best_target_correct": bool(predicted[best_target_key] == target_token),
            "best_absolute_effect_key": best_absolute_key,
            "max_absolute_effect": float(np.max(np.abs(effects))),
            "qk_vs_signed_effect_pearson": pearson(qk, effects),
            "qk_vs_signed_effect_spearman": spearman(qk, effects),
            "attention_vs_absolute_effect_pearson": pearson(attention, np.abs(effects)),
        }


@dataclass(frozen=True)
class AttentionPathTrace:
    split: str
    sample_key: tuple[int, int, int, int]
    target_token: int
    baseline_predicted_token: int
    baseline_target_logit: float
    baseline_target_probability: float
    heads: tuple[HeadPathMeasurement, ...]

    def to_dict(self) -> dict[str, object]:
        heads = [head.to_dict(self.target_token) for head in self.heads]
        return {
            "split": self.split,
            "sample_key": list(self.sample_key),
            "target_token": self.target_token,
            "baseline_predicted_token": self.baseline_predicted_token,
            "baseline_target_logit": self.baseline_target_logit,
            "baseline_target_probability": self.baseline_target_probability,
            "attention_selected_correct_fraction": float(
                np.mean(
                    [
                        head.predicted_tokens[int(np.argmax(head.attention_weights))]
                        == self.target_token
                        for head in self.heads
                    ]
                )
            ),
            "best_case_correct_fraction": float(
                np.mean(
                    [
                        head.predicted_tokens[int(np.argmax(head.target_logit_effects))]
                        == self.target_token
                        for head in self.heads
                    ]
                )
            ),
            "mean_absolute_attention_selected_effect": float(
                np.mean(
                    [
                        abs(head.target_logit_effects[int(np.argmax(head.attention_weights))])
                        for head in self.heads
                    ]
                )
            ),
            "max_one_hot_conservation_error": max(
                head.one_hot_conservation_error for head in self.heads
            ),
            "heads": heads,
            "scope": (
                "QK and attention are exact read statistics; one-hot effects include downstream "
                "mediation and do not isolate a feature."
            ),
        }


@dataclass(frozen=True)
class ResidualPatchTrace:
    site: PatchSite
    target_split: str
    target_key: tuple[int, int, int, int]
    donor_split: str
    donor_key: tuple[int, int, int, int]
    original_target: int
    counterfactual_target: int
    donor_target: int
    baseline_predicted: int
    patched_predicted: int
    donor_predicted: int
    baseline_counterfactual_logit: float
    patched_counterfactual_logit: float
    baseline_counterfactual_probability: float
    patched_counterfactual_probability: float
    first_order_prediction: float
    actual_effect: float
    patch_delta_norm: float
    donor_logit_parity_error: float

    @property
    def first_order_residual(self) -> float:
        return self.actual_effect - self.first_order_prediction

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "target_split": self.target_split,
            "target_key": list(self.target_key),
            "donor_split": self.donor_split,
            "donor_key": list(self.donor_key),
            "original_target": self.original_target,
            "counterfactual_target": self.counterfactual_target,
            "donor_target": self.donor_target,
            "baseline_predicted": self.baseline_predicted,
            "patched_predicted": self.patched_predicted,
            "donor_predicted": self.donor_predicted,
            "baseline_counterfactual_logit": self.baseline_counterfactual_logit,
            "patched_counterfactual_logit": self.patched_counterfactual_logit,
            "baseline_counterfactual_probability": self.baseline_counterfactual_probability,
            "patched_counterfactual_probability": self.patched_counterfactual_probability,
            "counterfactual_success": self.patched_predicted == self.counterfactual_target,
            "donor_answer_success": self.patched_predicted == self.donor_target,
            "donor_prediction_match": self.patched_predicted == self.donor_predicted,
            "original_answer_success": self.patched_predicted == self.original_target,
            "actual_counterfactual_logit_effect": self.actual_effect,
            "first_order_prediction": self.first_order_prediction,
            "first_order_residual": self.first_order_residual,
            "patch_delta_norm": self.patch_delta_norm,
            "donor_logit_parity_error": self.donor_logit_parity_error,
            "scope": (
                "Whole-vector activation patch; it does not isolate u. Final-site parity is a "
                "readout control, not mechanistic faithfulness."
            ),
        }


def attention_path_sweep(
    model: ArithmeticTransformer,
    sample: ArithmeticSample,
    *,
    target_token: int | None = None,
) -> AttentionPathTrace:
    """Force each head's answer query to each key and measure total output effects."""

    target = sample.target if target_token is None else target_token
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline_logits = model(tokens)[0, -1, :]
            _, cache = model.run_with_cache(tokens)
        baseline_probability = torch.softmax(baseline_logits, dim=-1)[target]
        heads: list[HeadPathMeasurement] = []
        for layer in range(model.config.n_layers):
            scores = cache[f"blocks.{layer}.attention.hook_scores"]
            patterns = cache[f"blocks.{layer}.attention.hook_pattern"]
            for head in range(model.config.n_heads):
                qk = scores[0, head, -1, :].detach().cpu()
                attention = patterns[0, head, -1, :].detach().cpu()
                effects: list[float] = []
                probabilities: list[float] = []
                predictions: list[int] = []
                conservation_error = 0.0
                for key in range(model.config.max_length):
                    forced_row = torch.zeros_like(attention)
                    forced_row[key] = 1.0
                    conservation_error = max(
                        conservation_error, abs(float(forced_row.sum().item()) - 1.0)
                    )

                    def force_path(
                        tensor: Tensor,
                        hook: object,
                        head: int = head,
                        key: int = key,
                    ) -> Tensor:
                        del hook
                        result = tensor.clone()
                        result[0, head, -1, :] = 0.0
                        result[0, head, -1, key] = 1.0
                        return result

                    with torch.no_grad():
                        logits = model.run_with_hooks(
                            tokens,
                            fwd_hooks=[(f"blocks.{layer}.attention.hook_pattern", force_path)],
                        )[0, -1, :]
                    target_logit = logits[target]
                    effects.append(float((target_logit - baseline_logits[target]).cpu().item()))
                    probabilities.append(float(torch.softmax(logits, dim=-1)[target].cpu().item()))
                    predictions.append(int(logits.argmax(dim=-1).cpu().item()))
                heads.append(
                    HeadPathMeasurement(
                        layer=layer,
                        head=head,
                        qk_scores=tuple(float(value) for value in qk.tolist()),
                        attention_weights=tuple(float(value) for value in attention.tolist()),
                        target_logit_effects=tuple(effects),
                        target_probabilities=tuple(probabilities),
                        predicted_tokens=tuple(predictions),
                        one_hot_conservation_error=conservation_error,
                    )
                )
        return AttentionPathTrace(
            split=sample.split,
            sample_key=sample.key,
            target_token=target,
            baseline_predicted_token=int(baseline_logits.argmax(dim=-1).cpu().item()),
            baseline_target_logit=float(baseline_logits[target].cpu().item()),
            baseline_target_probability=float(baseline_probability.cpu().item()),
            heads=tuple(heads),
        )
    finally:
        model.train(was_training)


def choose_counterfactual_donor(
    dataset: ArithmeticDataset,
    target: ArithmeticSample,
) -> ArithmeticSample:
    """Choose a deterministic same-c,d donor with the nearest different u."""

    eligible = [
        sample
        for sample in dataset.samples
        if sample.key != target.key
        and sample.c == target.c
        and sample.d == target.d
        and sample.u != target.u
    ]
    if not eligible:
        raise ValueError(f"no controlled counterfactual donor for {target.key}")
    return min(eligible, key=lambda sample: (abs(sample.u - target.u), sample.key))


def counterfactual_residual_patch(
    model: ArithmeticTransformer,
    target: ArithmeticSample,
    donor: ArithmeticSample,
    *,
    site: PatchSite,
) -> ResidualPatchTrace:
    """Patch a whole residual vector and measure counterfactual and first-order effects."""

    if target.c != donor.c or target.d != donor.d or target.u == donor.u:
        raise ValueError("controlled residual patches require same c,d and different u")
    if site not in PATCH_HOOKS:
        raise ValueError(f"unknown residual patch site: {site}")
    hook_name = PATCH_HOOKS[site]
    target_tokens = torch.from_numpy(target.tokens).unsqueeze(0)
    donor_tokens = torch.from_numpy(donor.tokens).unsqueeze(0)
    counterfactual = (donor.u * target.c + target.d) % target.modulus
    was_training = model.training
    model.eval()
    try:
        activations: dict[str, Tensor] = {}

        def capture(tensor: Tensor, hook: object) -> None:
            del hook
            activations["source"] = tensor

        with model.hooks(fwd_hooks=[(hook_name, capture)]):
            baseline_logits = model(target_tokens)
        counterfactual_logit = baseline_logits[0, -1, counterfactual]
        source = activations["source"]
        gradient = torch.autograd.grad(counterfactual_logit, source, retain_graph=False)[0]
        with torch.no_grad():
            donor_logits, donor_cache = model.run_with_cache(donor_tokens)
            donor_activation = donor_cache[hook_name].detach().clone()
            target_activation = source.detach().clone()
            delta = donor_activation[0, -1, :] - target_activation[0, -1, :]

            def patch(tensor: Tensor, hook: object) -> Tensor:
                del hook
                result = tensor.clone()
                result[0, -1, :] = donor_activation[0, -1, :]
                return result

            patched_logits = model.run_with_hooks(target_tokens, fwd_hooks=[(hook_name, patch)])
            baseline_answer = baseline_logits[0, -1, :].detach()
            patched_answer = patched_logits[0, -1, :].detach()
            donor_answer = donor_logits[0, -1, :].detach()
            prediction = float((delta * gradient[0, -1, :].detach()).sum().cpu().item())
            actual = float(
                (patched_answer[counterfactual] - baseline_answer[counterfactual]).cpu().item()
            )
            return ResidualPatchTrace(
                site=site,
                target_split=target.split,
                target_key=target.key,
                donor_split=donor.split,
                donor_key=donor.key,
                original_target=target.target,
                counterfactual_target=counterfactual,
                donor_target=donor.target,
                baseline_predicted=int(baseline_answer.argmax(dim=-1).cpu().item()),
                patched_predicted=int(patched_answer.argmax(dim=-1).cpu().item()),
                donor_predicted=int(donor_answer.argmax(dim=-1).cpu().item()),
                baseline_counterfactual_logit=float(baseline_answer[counterfactual].cpu().item()),
                patched_counterfactual_logit=float(patched_answer[counterfactual].cpu().item()),
                baseline_counterfactual_probability=float(
                    torch.softmax(baseline_answer, dim=-1)[counterfactual].cpu().item()
                ),
                patched_counterfactual_probability=float(
                    torch.softmax(patched_answer, dim=-1)[counterfactual].cpu().item()
                ),
                first_order_prediction=prediction,
                actual_effect=actual,
                patch_delta_norm=float(delta.norm().cpu().item()),
                donor_logit_parity_error=float(
                    (patched_answer - donor_answer).abs().max().cpu().item()
                ),
            )
    finally:
        model.train(was_training)
