"""Per-sample trace contracts for custom PyTorch architectures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.lenses import TopKSparseChart, signed_max_pool

FloatArray = NDArray[np.float64]


class ProbeLayout(str, Enum):
    FEATURES_LAST = "features_last"
    CHANNELS_FIRST = "channels_first"


class CMAEProbe(nn.Module):
    """An identity observation point insertable into a custom ``nn.Module``."""

    def __init__(self, layout: ProbeLayout = ProbeLayout.FEATURES_LAST) -> None:
        super().__init__()
        self.layout = layout
        self.last_activation: Tensor | None = None

    def forward(self, value: Tensor) -> Tensor:
        self.last_activation = value
        return value


@dataclass(frozen=True)
class TraceRecord:
    architecture: str
    sample_index: int
    activation: FloatArray
    output_gradient: FloatArray
    loss: float


@dataclass(frozen=True)
class ConceptSignatures:
    sparse_activation: FloatArray
    sparse_gradient: FloatArray
    full_activation: FloatArray
    full_gradient: FloatArray
    activation_projection_residual_ratio: float
    gradient_projection_residual_ratio: float


def _canonicalize(value: Tensor, layout: ProbeLayout) -> Tensor:
    if value.ndim < 2 or value.shape[0] != 1:
        raise ValueError("CMAE tracing requires one sample and an explicit batch dimension")
    single = value[0]
    if layout == ProbeLayout.FEATURES_LAST:
        return single.reshape(-1, single.shape[-1])
    if single.ndim < 2:
        raise ValueError("channels-first probes need at least one spatial position")
    moved = single.movedim(0, -1)
    return moved.reshape(-1, single.shape[0])


def trace_probe(
    architecture: str,
    sample_index: int,
    model: nn.Module,
    probe: CMAEProbe,
    inputs: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
) -> TraceRecord:
    """Observe one sample without accumulating parameter gradients or updating state."""

    if inputs.shape[0] != 1:
        raise ValueError("trace_probe requires exactly one sample")
    was_training = model.training
    model.eval()
    probe.last_activation = None
    try:
        output = model(inputs)
        activation = probe.last_activation
        if activation is None:
            raise RuntimeError("the model did not execute its CMAE probe")
        loss = loss_fn(output)
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise ValueError("audit loss must be a finite scalar")
        gradient = torch.autograd.grad(loss, activation, allow_unused=False)[0]
        activation_flat = _canonicalize(activation, probe.layout)
        gradient_flat = _canonicalize(gradient, probe.layout)
        if activation_flat.shape != gradient_flat.shape:
            raise RuntimeError("probe activation and gradient canonicalization disagree")
        return TraceRecord(
            architecture=architecture,
            sample_index=sample_index,
            activation=activation_flat.detach().to(torch.float64).cpu().numpy(),
            output_gradient=gradient_flat.detach().to(torch.float64).cpu().numpy(),
            loss=float(loss.detach().cpu().item()),
        )
    finally:
        probe.last_activation = None
        model.train(was_training)


def _ratio(numerator: Tensor, denominator: Tensor) -> float:
    return float((numerator.norm() / denominator.norm().clamp_min(1e-12)).detach().cpu().item())


def concept_signatures(trace: TraceRecord, chart: TopKSparseChart) -> ConceptSignatures:
    """Project both traces while retaining full projections and discarded residuals."""

    activation = torch.from_numpy(trace.activation)
    gradient = torch.from_numpy(trace.output_gradient)
    activation_code = chart.forward(activation)
    gradient_code = chart.forward(gradient)
    activation_projection = activation @ chart.dictionary.transpose(0, 1)
    gradient_projection = gradient @ chart.dictionary.transpose(0, 1)
    return ConceptSignatures(
        sparse_activation=signed_max_pool(activation_code).numpy(),
        sparse_gradient=signed_max_pool(gradient_code).numpy(),
        full_activation=signed_max_pool(activation_projection).numpy(),
        full_gradient=signed_max_pool(gradient_projection).numpy(),
        activation_projection_residual_ratio=_ratio(
            chart.projection_residual(activation), activation_projection
        ),
        gradient_projection_residual_ratio=_ratio(
            chart.projection_residual(gradient), gradient_projection
        ),
    )
