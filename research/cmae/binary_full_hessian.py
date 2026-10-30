"""Exact full binary loss Hessian for the compact two-layer tanh CIFAR model.

This includes residual-weighted network second derivatives omitted by the GGN.
It is a separate sensitivity reference, not a relabeling of the GGN contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP
from research.cmae.binary_counterfactual import GGNArtifact, binary_logits

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FullHessianArtifact:
    hessian: torch.Tensor
    loss_gradients: torch.Tensor
    query_gradient: torch.Tensor
    minimum_eigenvalue: float
    symmetry_residual: float
    parameter_count: int


def capture_binary_full_hessian(
    model: CIFARBadNetsMLP,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    ggn: GGNArtifact,
) -> FullHessianArtifact:
    """Add exact residual-weighted tanh and cross-parameter curvature to the GGN."""

    if model.input.bias is not None or model.head.bias is not None:
        raise ValueError("the analytic full Hessian requires the declared no-bias model")
    count, width = inputs.shape
    hidden = model.input.out_features
    parameters = ggn.parameter_count
    if (
        labels.shape != (count,)
        or model.input.in_features != width
        or parameters != hidden * width + hidden
        or ggn.loss_gradients.shape != (count, parameters)
    ):
        raise ValueError("full Hessian inputs do not match the compact model/artifact")
    was_training = model.training
    model.eval()
    try:
        preactivation = model.input(inputs)
        activation = torch.tanh(preactivation)
        first_derivative = 1 - activation**2
        second_derivative = -2 * activation * first_derivative
        logits = binary_logits(model, inputs)
        residual = torch.sigmoid(logits) - labels
        head_weight = model.head.weight[0]
        hidden_coefficient = residual[:, None] * head_weight[None, :] * second_derivative / count
        hidden_second = torch.einsum("nh,ni,nj->hij", hidden_coefficient, inputs, inputs)
        cross_coefficient = residual[:, None] * first_derivative / count
        cross_second = torch.einsum("nh,ni->hi", cross_coefficient, inputs)
    finally:
        model.train(was_training)
    hessian = ggn.curvature.clone()
    hessian[: hidden * width, : hidden * width] += torch.block_diag(*hidden_second)
    cross_top = torch.zeros(
        hidden * width,
        hidden,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    for index in range(hidden):
        cross_top[index * width : (index + 1) * width, index] = cross_second[index]
    hessian[: hidden * width, hidden * width :] += cross_top
    hessian[hidden * width :, : hidden * width] += cross_top.T
    symmetry_residual = float((hessian - hessian.T).abs().max().detach().cpu().item())
    hessian = (hessian + hessian.T) / 2
    minimum_eigenvalue = float(torch.linalg.eigvalsh(hessian).min().detach().cpu().item())
    artifact = FullHessianArtifact(
        hessian=hessian,
        loss_gradients=ggn.loss_gradients,
        query_gradient=ggn.query_gradient,
        minimum_eigenvalue=minimum_eigenvalue,
        symmetry_residual=symmetry_residual,
        parameter_count=parameters,
    )
    if (
        not bool(torch.all(torch.isfinite(hessian)).item())
        or not np.isfinite(minimum_eigenvalue)
        or not np.isfinite(symmetry_residual)
    ):
        raise FloatingPointError("nonfinite full-Hessian artifact")
    return artifact


def damped_full_hessian_influence(
    artifact: FullHessianArtifact, *, damping: float = 1e-3
) -> FloatArray:
    """Solve only a healthy positive-definite damped full-Hessian system."""

    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("full-Hessian damping must be finite and positive")
    if artifact.minimum_eigenvalue + damping <= 0:
        raise RuntimeError(
            "full Hessian is not positive definite under the frozen damping; "
            "influence is unhealthy and must not be used"
        )
    identity = torch.eye(
        artifact.parameter_count,
        dtype=artifact.hessian.dtype,
        device=artifact.hessian.device,
    )
    solved = torch.linalg.solve(artifact.hessian + damping * identity, artifact.query_gradient)
    scores = -(artifact.loss_gradients @ solved).detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite full-Hessian influence")
    return scores
