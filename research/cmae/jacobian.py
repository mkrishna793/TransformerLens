"""Architecture-general downstream Jacobian lenses with explicit error measurements."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.tracing import CMAEProbe

FloatArray = NDArray[np.float64]
OutputSelector = Callable[[Tensor], Tensor]
Downstream = Callable[[Tensor], Tensor]


def linear_readout(weights: Tensor, bias: Tensor | None = None) -> Downstream:
    """Adapt a frozen linear/vocabulary readout into a downstream lens map."""

    if weights.ndim != 2 or not bool(torch.isfinite(weights).all()):
        raise ValueError("readout weights must be a finite matrix")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weights.shape[0]):
        raise ValueError("readout bias must align with the vocabulary dimension")

    def readout(activation: Tensor) -> Tensor:
        output = activation @ weights.T
        return output if bias is None else output + bias

    return readout


def vocabulary_token_selector(token_index: int) -> OutputSelector:
    """Select one vocabulary logit while preserving arbitrary leading dimensions."""

    if token_index < 0:
        raise ValueError("token index must be nonnegative")
    return lambda output: output[..., token_index]


@dataclass(frozen=True)
class JacobianEstimate:
    """An output-by-activation Jacobian averaged over calibration examples."""

    mean: Tensor
    dispersion: Tensor
    samples: int
    activation_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("a Jacobian estimate needs at least one sample")
        if self.mean.shape != self.dispersion.shape:
            raise ValueError("Jacobian mean and dispersion shapes differ")
        if self.mean.ndim < 2 or tuple(self.mean.shape[1:]) != self.activation_shape:
            raise ValueError("Jacobian must have shape [outputs, *activation_shape]")
        if not bool(torch.isfinite(self.mean).all()) or not bool(
            torch.isfinite(self.dispersion).all()
        ):
            raise ValueError("Jacobian estimate must be finite")


@dataclass(frozen=True)
class ResidualReport:
    relative: float
    absolute: float
    output_change_norm: float


def _from_output(output: Tensor, activation: Tensor) -> Tensor:
    flat_output = output.reshape(-1)
    if flat_output.numel() == 0:
        raise ValueError("Jacobian output must be nonempty")
    rows: list[Tensor] = []
    for index in range(flat_output.numel()):
        gradient = torch.autograd.grad(
            flat_output[index], activation, retain_graph=True, allow_unused=False
        )[0]
        rows.append(gradient.reshape(-1))
    return torch.stack(rows).reshape((flat_output.numel(),) + tuple(activation.shape))


def single_downstream_jacobian(
    downstream: Downstream, activation: Tensor, *, output_selector: OutputSelector | None = None
) -> Tensor:
    """Differentiate a downstream map with respect to one activation tensor."""

    if activation.ndim == 0 or not bool(torch.isfinite(activation).all()):
        raise ValueError("activation must be a finite non-scalar tensor")
    value = activation.detach().requires_grad_(True)
    output = downstream(value)
    if output_selector is not None:
        output = output_selector(output)
    return _from_output(output, value)


def average_downstream_jacobian(
    activations: Sequence[Tensor],
    downstreams: Sequence[Downstream],
    *,
    output_selector: OutputSelector | None = None,
) -> JacobianEstimate:
    """Average per-context Jacobians, retaining entrywise dispersion."""

    if len(activations) == 0 or len(activations) != len(downstreams):
        raise ValueError("activations and downstreams must have equal nonzero length")
    estimates = [
        single_downstream_jacobian(downstream, activation, output_selector=output_selector)
        for activation, downstream in zip(activations, downstreams, strict=True)
    ]
    stacked = torch.stack(estimates)
    return JacobianEstimate(
        mean=stacked.mean(dim=0),
        dispersion=stacked.std(dim=0, unbiased=False),
        samples=len(estimates),
        activation_shape=tuple(activations[0].shape),
    )


def estimate_probe_jacobian(
    model: nn.Module,
    probe: CMAEProbe,
    inputs: Tensor,
    *,
    output_selector: OutputSelector | None = None,
) -> JacobianEstimate:
    """Estimate an output Jacobian through an arbitrary model's probe contract.

    Inputs are evaluated one at a time. The output selector can choose a class logit,
    a future token, or a scientific observable; it is part of the lens definition.
    """

    if inputs.ndim < 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must have a nonempty batch dimension")
    was_training = model.training
    model.eval()
    estimates: list[Tensor] = []
    activation_shape: tuple[int, ...] | None = None
    try:
        for index in range(inputs.shape[0]):
            probe.last_activation = None
            output = model(inputs[index : index + 1])
            activation = probe.last_activation
            if activation is None:
                raise RuntimeError("the model did not execute its CMAE probe")
            if output_selector is not None:
                output = output_selector(output)
            estimates.append(_from_output(output, activation))
            if activation_shape is None:
                activation_shape = tuple(activation.shape)
            elif activation_shape != tuple(activation.shape):
                raise ValueError("probe activation shape changed between samples")
    finally:
        probe.last_activation = None
        model.train(was_training)
    stacked = torch.stack(estimates)
    if activation_shape is None:
        raise RuntimeError("no probe activations were collected")
    return JacobianEstimate(
        mean=stacked.mean(dim=0),
        dispersion=stacked.std(dim=0, unbiased=False),
        samples=len(estimates),
        activation_shape=activation_shape,
    )


def apply_jacobian(estimate: JacobianEstimate, activation: Tensor) -> Tensor:
    """Apply the averaged linear readout to an activation or intervention delta."""

    if tuple(activation.shape) != estimate.activation_shape:
        raise ValueError("activation shape does not match the Jacobian estimate")
    return (estimate.mean.reshape(estimate.mean.shape[0], -1) * activation.reshape(-1)).sum(dim=-1)


def first_order_residual(
    estimate: JacobianEstimate,
    downstream: Downstream,
    activation: Tensor,
    delta: Tensor,
    *,
    epsilon: float = 1e-12,
) -> ResidualReport:
    """Measure actual downstream change minus the lens-predicted change."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if activation.shape != delta.shape:
        raise ValueError("activation and delta shapes differ")
    baseline = downstream(activation).reshape(-1)
    changed = downstream(activation + delta).reshape(-1)
    predicted = apply_jacobian(estimate, delta)
    actual_change = changed - baseline
    if actual_change.shape != predicted.shape:
        raise ValueError("downstream output and Jacobian output shapes differ")
    absolute = float((actual_change - predicted).norm().detach().cpu().item())
    output_norm = float(actual_change.norm().detach().cpu().item())
    return ResidualReport(
        relative=absolute / max(output_norm, epsilon),
        absolute=absolute,
        output_change_norm=output_norm,
    )


def intervention_errors(
    estimate: JacobianEstimate,
    downstream: Downstream,
    activation: Tensor,
    direction: Tensor,
    scales: Sequence[float],
) -> FloatArray:
    """Return first-order errors for a sequence of causal activation interventions."""

    if activation.shape != direction.shape:
        raise ValueError("activation and direction shapes differ")
    return np.array(
        [
            first_order_residual(estimate, downstream, activation, direction * scale).relative
            for scale in scales
        ],
        dtype=np.float64,
    )
