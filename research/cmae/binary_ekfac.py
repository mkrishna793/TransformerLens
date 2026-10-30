"""Binary-model adapter to CMAE's exact raw-gradient EK-FAC reference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP, CIFARBatch
from research.cmae.binary_counterfactual import binary_logits, query_logit
from research.cmae.curvature_influence import (
    EKFACInfluenceModel,
    RawLayerGradient,
    RawSampleGradient,
    fit_ekfac_from_gradients,
)

FloatArray = NDArray[np.float64]


def tracked_binary_linears(model: CIFARBadNetsMLP) -> dict[str, nn.Linear]:
    if model.input.bias is not None or model.head.bias is not None:
        raise ValueError("the complete binary EK-FAC inventory requires no-bias linear layers")
    return {"input": model.input, "head": model.head}


def _capture_binary_raw(
    model: CIFARBadNetsMLP,
    inputs: Tensor,
    modules: Mapping[str, nn.Linear],
    objective: str,
    label: Tensor | None,
    sample_key: tuple[int, int, int, int],
) -> RawSampleGradient:
    if objective not in {"loss", "query"}:
        raise ValueError("unknown binary raw-gradient objective")
    if objective == "loss" and (label is None or label.ndim != 0):
        raise ValueError("loss capture requires one scalar label")
    if objective == "query" and label is not None:
        raise ValueError("query capture does not take a training label")
    activations: dict[str, Tensor] = {}
    output_gradients: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    was_training = model.training
    model.eval()
    try:
        for name, module in modules.items():

            def pre_hook(
                module: nn.Module,
                values: tuple[Tensor, ...],
                *,
                name: str = name,
            ) -> None:
                del module
                if len(values) != 1:
                    raise RuntimeError(f"unexpected binary linear inputs for {name}")
                activations[name] = values[0]

            def forward_hook(
                module: nn.Module,
                values: tuple[Tensor, ...],
                output: Tensor,
                *,
                name: str = name,
            ) -> None:
                del module, values
                if output.requires_grad:
                    output.register_hook(
                        lambda gradient, name=name: output_gradients.__setitem__(name, gradient)
                    )

            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(forward_hook))
        parameters = tuple(model.parameters())
        logits = binary_logits(model, inputs)
        if objective == "loss":
            assert label is not None
            scalar = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], label)
        else:
            scalar = query_logit(model, inputs)
        torch.autograd.grad(scalar, parameters, retain_graph=False)
        layers: dict[str, RawLayerGradient] = {}
        for name in modules:
            if name not in activations or name not in output_gradients:
                raise RuntimeError(f"binary tracked module has no complete gradient: {name}")
            activation = activations[name]
            gradient = output_gradients[name]
            layers[name] = RawLayerGradient(
                activation=activation.reshape(-1, activation.shape[-1])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                output_gradient=gradient.reshape(-1, gradient.shape[-1])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
            )
        return RawSampleGradient(sample_key, layers)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


def capture_binary_training_raw(
    model: CIFARBadNetsMLP, batch: CIFARBatch
) -> tuple[RawSampleGradient, ...]:
    """Capture the exact raw loss factors once for all independent comparisons."""

    modules = tracked_binary_linears(model)
    return tuple(
        _capture_binary_raw(
            model,
            value.unsqueeze(0),
            modules,
            "loss",
            label,
            (0, 0, 0, index),
        )
        for index, (value, label) in enumerate(zip(batch.inputs, batch.labels, strict=True))
    )


def fit_binary_ekfac(
    model: CIFARBadNetsMLP,
    batch: CIFARBatch,
    query_inputs: Tensor,
    *,
    damping: float = 1e-3,
    raw_samples: tuple[RawSampleGradient, ...] | None = None,
) -> tuple[EKFACInfluenceModel, RawSampleGradient]:
    """Fit empirical EK-FAC from exact per-sample binary loss gradients."""

    modules = tracked_binary_linears(model)
    raw = capture_binary_training_raw(model, batch) if raw_samples is None else raw_samples
    if len(raw) != batch.labels.shape[0]:
        raise ValueError("provided raw samples do not align with the binary batch")
    fitted = fit_ekfac_from_gradients(raw, modules, damping=damping)
    query = _capture_binary_raw(model, query_inputs, modules, "query", None, (0, 0, 0, -1))
    return fitted, query


def binary_ekfac_influence(
    fitted: EKFACInfluenceModel,
    query: RawSampleGradient,
    *,
    original_count: int,
) -> FloatArray:
    """Apply the fixed-normalization factor after raw-loss EK-FAC scoring."""

    if original_count != len(fitted.sample_keys) or original_count <= 0:
        raise ValueError("EK-FAC original count must equal its training sample count")
    scores = fitted.influence(query) / original_count
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite binary EK-FAC influence")
    return scores
