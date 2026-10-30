"""Attention-path and counterfactual residual intervention tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from research.cmae.arithmetic_data import ArithmeticSample, make_arithmetic_dataset
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.attention_interventions import (
    PatchSite,
    attention_path_sweep,
    choose_counterfactual_donor,
    counterfactual_residual_patch,
)


def _model(seed: int = 101) -> ArithmeticTransformer:
    torch.manual_seed(seed)
    return ArithmeticTransformer()


def test_attention_hooks_reproduce_qk_scores_and_causal_pattern() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    query = cache["blocks.0.attention.hook_q"]
    key = cache["blocks.0.attention.hook_k"]
    scores = cache["blocks.0.attention.hook_scores"]
    pattern = cache["blocks.0.attention.hook_pattern"]
    width = model.config.d_head
    manual = query[0, 0, -1, :] @ key[0, 0].transpose(0, 1) / math.sqrt(width)
    assert torch.allclose(scores[0, 0, -1, :], manual, atol=1e-6, rtol=1e-6)
    future = torch.triu(
        torch.ones(model.config.max_length, model.config.max_length, dtype=torch.bool), 1
    )
    assert torch.all(pattern[:, :, future] == 0)
    assert torch.allclose(pattern.sum(dim=-1), torch.ones_like(pattern.sum(dim=-1)))


def test_forced_one_hot_attention_selects_the_requested_value() -> None:
    model = _model()
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    values = cache["blocks.0.attention.hook_v"].detach()
    captured: dict[str, torch.Tensor] = {}
    head = 1
    selected_key = 4

    def force(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        del hook
        result = tensor.clone()
        result[0, head, -1, :] = 0
        result[0, head, -1, selected_key] = 1
        return result

    def capture(tensor: torch.Tensor, hook: object) -> None:
        del hook
        captured["z"] = tensor

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("blocks.0.attention.hook_pattern", force),
                ("blocks.0.attention.hook_z", capture),
            ],
        )
    assert torch.allclose(
        captured["z"][0, -1, head, :], values[0, head, selected_key, :], atol=1e-6
    )


def test_attention_sweep_covers_all_paths_and_preserves_state() -> None:
    model = _model()
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    sample = make_arithmetic_dataset(101).split_samples("iid")[0]
    trace = attention_path_sweep(model, sample)
    result = trace.to_dict()
    assert len(trace.heads) == 4
    assert result["max_one_hot_conservation_error"] == 0
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    for head in trace.heads:
        assert len(head.qk_scores) == model.config.max_length
        assert len(head.attention_weights) == model.config.max_length
        assert len(head.target_logit_effects) == model.config.max_length
        assert np.all(np.isfinite(head.target_logit_effects))
        assert sum(head.attention_weights) == pytest.approx(1.0, abs=1e-6)
        serialized = head.to_dict(sample.target)
        qk_correlation = serialized["qk_vs_signed_effect_pearson"]
        attention_correlation = serialized["attention_vs_absolute_effect_pearson"]
        assert isinstance(qk_correlation, float) and np.isfinite(qk_correlation)
        assert isinstance(attention_correlation, float) and np.isfinite(attention_correlation)


def test_donor_selection_is_controlled_and_available_on_every_split() -> None:
    dataset = make_arithmetic_dataset(101)
    for split in ("iid", "composition", "operand", "multiplier", "combined"):
        target = dataset.split_samples(split)[0]
        first = choose_counterfactual_donor(dataset, target)
        second = choose_counterfactual_donor(dataset, target)
        assert first == second
        assert first.c == target.c and first.d == target.d
        assert first.u != target.u
        assert first.key != target.key


@pytest.mark.parametrize("site", ["early", "post_attention", "final"])
def test_residual_patches_measure_counterfactual_and_conserve_residual(site: PatchSite) -> None:
    dataset = make_arithmetic_dataset(101)
    target = dataset.split_samples("iid")[0]
    donor = choose_counterfactual_donor(dataset, target)
    model = _model()
    trace = counterfactual_residual_patch(model, target, donor, site=site)
    result = trace.to_dict()
    assert trace.counterfactual_target == (donor.u * target.c + target.d) % 11
    assert trace.patch_delta_norm > 0
    assert np.isfinite(trace.first_order_prediction)
    assert np.isfinite(trace.first_order_residual)
    assert trace.actual_effect == pytest.approx(
        trace.first_order_prediction + trace.first_order_residual, abs=1e-6
    )
    assert result["donor_prediction_match"] == (trace.patched_predicted == trace.donor_predicted)
    scope = result["scope"]
    assert isinstance(scope, str) and "does not isolate u" in scope
    if site == "final":
        assert trace.donor_logit_parity_error < 1e-6
        assert result["donor_prediction_match"]
        assert trace.patched_predicted == trace.donor_predicted


def test_residual_patch_rejects_uncontrolled_donor() -> None:
    dataset = make_arithmetic_dataset(101)
    target = dataset.split_samples("iid")[0]
    donor = ArithmeticSample(target.a, target.b, target.c, (target.d + 1) % 11, "iid")
    model = _model()
    with pytest.raises(ValueError, match="same c,d"):
        counterfactual_residual_patch(model, target, donor, site="early")
