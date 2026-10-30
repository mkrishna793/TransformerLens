"""Residual-write attribution and causal ablations for the arithmetic Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from research.cmae.arithmetic_data import ArithmeticSample
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.metrics import pearson, spearman

FloatArray = NDArray[np.float64]
NodeKind = Literal["token", "position", "attention", "mlp"]


@dataclass(frozen=True)
class ComponentNode:
    node_id: str
    hook_name: str
    kind: NodeKind
    layer: int
    position: int
    head: int | None
    final_readout: bool

    def value(self, tensor: Tensor) -> Tensor:
        if self.kind == "position":
            return tensor[self.position, :]
        if self.head is not None:
            return tensor[0, self.position, self.head, :]
        return tensor[0, self.position, :]

    def ablate(self, tensor: Tensor) -> Tensor:
        result = tensor.clone()
        if self.kind == "position":
            result[self.position, :] = 0.0
        elif self.head is not None:
            result[0, self.position, self.head, :] = 0.0
        else:
            result[0, self.position, :] = 0.0
        return result


@dataclass(frozen=True)
class ComponentMeasurement:
    node: ComponentNode
    activation_norm: float
    first_order_prediction: float
    causal_effect: float
    integrated_direct: float | None

    @property
    def approximation_residual(self) -> float:
        return self.causal_effect - self.first_order_prediction

    @property
    def indirect_effect(self) -> float | None:
        if self.integrated_direct is None:
            return None
        return self.causal_effect - self.integrated_direct

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node.node_id,
            "hook_name": self.node.hook_name,
            "kind": self.node.kind,
            "layer": self.node.layer,
            "position": self.node.position,
            "head": self.node.head,
            "final_readout": self.node.final_readout,
            "activation_norm": self.activation_norm,
            "first_order_prediction": self.first_order_prediction,
            "causal_effect": self.causal_effect,
            "approximation_residual": self.approximation_residual,
            "integrated_direct": self.integrated_direct,
            "indirect_effect": self.indirect_effect,
        }


@dataclass(frozen=True)
class ComponentTrace:
    split: str
    sample_key: tuple[int, int, int, int]
    target_token: int
    predicted_token: int
    target_logit: float
    readout_baseline: float
    integrated_quadrature_residual: float
    residual_reconstruction_error: float
    readout_parity_error: float
    quadrature_steps: int
    measurements: tuple[ComponentMeasurement, ...]

    def metrics(self) -> dict[str, float | int | bool]:
        predicted = np.asarray(
            [measurement.first_order_prediction for measurement in self.measurements]
        )
        actual = np.asarray([measurement.causal_effect for measurement in self.measurements])
        residuals = actual - predicted
        k = min(5, len(predicted))
        predicted_top = set(np.argsort(-np.abs(predicted))[:k].tolist())
        actual_top = set(np.argsort(-np.abs(actual))[:k].tolist())
        rng = np.random.default_rng(7919)
        shuffled_correlations = [pearson(rng.permutation(predicted), actual) for _ in range(32)]
        direct = [m.integrated_direct for m in self.measurements if m.integrated_direct is not None]
        return {
            "target_correct": self.predicted_token == self.target_token,
            "node_count": len(self.measurements),
            "first_order_pearson": pearson(predicted, actual),
            "first_order_spearman": spearman(predicted, actual),
            "shuffled_prediction_mean_pearson": float(np.mean(shuffled_correlations)),
            "shuffled_prediction_max_abs_pearson": float(np.max(np.abs(shuffled_correlations))),
            "causal_sign_agreement": float(np.mean((predicted > 0) == (actual > 0))),
            "mean_absolute_approximation_residual": float(np.mean(np.abs(residuals))),
            "max_absolute_approximation_residual": float(np.max(np.abs(residuals))),
            "top_5_absolute_overlap": len(predicted_top & actual_top) / k,
            "integrated_quadrature_residual": self.integrated_quadrature_residual,
            "residual_reconstruction_error": self.residual_reconstruction_error,
            "readout_parity_error": self.readout_parity_error,
            "integrated_direct_sum": float(np.sum(direct)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "sample_key": list(self.sample_key),
            "target_token": self.target_token,
            "predicted_token": self.predicted_token,
            "target_logit": self.target_logit,
            "readout_baseline": self.readout_baseline,
            "quadrature_steps": self.quadrature_steps,
            "metrics": self.metrics(),
            "nodes": [measurement.to_dict() for measurement in self.measurements],
        }


def component_nodes(model: ArithmeticTransformer) -> tuple[ComponentNode, ...]:
    nodes: list[ComponentNode] = []
    for position in range(model.config.max_length):
        final = position == model.config.max_length - 1
        nodes.append(
            ComponentNode(
                f"token_p{position}", "hook_token_embed", "token", -1, position, None, final
            )
        )
        nodes.append(
            ComponentNode(
                f"position_p{position}",
                "hook_position_embed",
                "position",
                -1,
                position,
                None,
                final,
            )
        )
    for layer in range(model.config.n_layers):
        for position in range(model.config.max_length):
            final = position == model.config.max_length - 1
            for head in range(model.config.n_heads):
                nodes.append(
                    ComponentNode(
                        f"l{layer}_attn_h{head}_p{position}",
                        f"blocks.{layer}.hook_attn_out",
                        "attention",
                        layer,
                        position,
                        head,
                        final,
                    )
                )
            nodes.append(
                ComponentNode(
                    f"l{layer}_mlp_p{position}",
                    f"blocks.{layer}.hook_mlp_out",
                    "mlp",
                    layer,
                    position,
                    None,
                    final,
                )
            )
    return tuple(nodes)


def integrated_readout_components(
    model: ArithmeticTransformer,
    residual: Tensor,
    components: list[Tensor],
    target_token: int,
    steps: int,
) -> tuple[float, float, float, FloatArray]:
    if steps < 2:
        raise ValueError("integrated readout needs at least two quadrature points")
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    average_jacobian = torch.zeros_like(residual)
    with torch.enable_grad():
        for node, weight in zip(nodes, weights, strict=True):
            t = (float(node) + 1.0) / 2.0
            quadrature_weight = float(weight) / 2.0
            scaled = (residual.detach() * t).requires_grad_(True)
            value = model.readout_vector(scaled)[target_token]
            gradient = torch.autograd.grad(value, scaled, retain_graph=False)[0]
            average_jacobian += quadrature_weight * gradient.detach()
    with torch.no_grad():
        baseline_value = model.readout_vector(torch.zeros_like(residual))[target_token]
        actual_value = model.readout_vector(residual.detach())[target_token]
        contributions = torch.stack(components) @ average_jacobian
    explained = float(contributions.sum().cpu().item())
    actual = float(actual_value.cpu().item())
    baseline = float(baseline_value.cpu().item())
    residual_value = actual - baseline - explained
    return (
        baseline,
        actual,
        residual_value,
        contributions.cpu().numpy().astype(np.float64),
    )


def trace_component_circuit(
    model: ArithmeticTransformer,
    sample: ArithmeticSample,
    *,
    target_token: int | None = None,
    quadrature_steps: int = 128,
) -> ComponentTrace:
    """Measure direct-readout allocations and total zero-ablation effects separately."""

    target = sample.target if target_token is None else target_token
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    nodes = component_nodes(model)
    activations: dict[str, Tensor] = {}
    was_training = model.training
    model.eval()
    try:

        def capture(tensor: Tensor, hook: object) -> None:
            name = getattr(hook, "name", None)
            if not isinstance(name, str):
                raise RuntimeError("capture hook has no name")
            activations[name] = tensor

        hook_names = sorted({node.hook_name for node in nodes} | {"hook_resid_final"})
        with model.hooks(fwd_hooks=[(name, capture) for name in hook_names]):
            target_tensor = model(tokens, target_token=target)
        unique_names = sorted({node.hook_name for node in nodes})
        unique_sources = [activations[name] for name in unique_names]
        unique_gradients = torch.autograd.grad(
            target_tensor, unique_sources, retain_graph=False, allow_unused=False
        )
        gradient_map = dict(zip(unique_names, unique_gradients, strict=True))
        residual = activations["hook_resid_final"][0, -1, :]
        final_nodes = [node for node in nodes if node.final_readout]
        components = [activations[node.hook_name].detach().clone() for node in final_nodes]
        component_values = [
            node.value(component) for node, component in zip(final_nodes, components)
        ]
        reconstructed = torch.stack(component_values).sum(dim=0)
        reconstruction_error = float((reconstructed - residual).abs().max().cpu().item())
        baseline, readout_actual, quadrature_residual, direct_values = (
            integrated_readout_components(
                model, residual, component_values, target, quadrature_steps
            )
        )
        direct_map = {
            node.node_id: float(value)
            for node, value in zip(final_nodes, direct_values, strict=True)
        }
        predicted_token = int(model(tokens)[0, -1, :].argmax(dim=-1).cpu().item())
        measurements: list[ComponentMeasurement] = []
        with torch.no_grad():
            for node in nodes:
                activation = node.value(activations[node.hook_name])
                gradient = node.value(gradient_map[node.hook_name])
                prediction = float((activation * gradient).sum().cpu().item())

                def ablation(tensor: Tensor, hook: object, node: ComponentNode = node) -> Tensor:
                    del hook
                    return node.ablate(tensor)

                ablated = model.run_with_hooks(
                    tokens,
                    target_token=target,
                    fwd_hooks=[(node.hook_name, ablation)],
                )
                causal = float((target_tensor - ablated).detach().cpu().item())
                measurements.append(
                    ComponentMeasurement(
                        node=node,
                        activation_norm=float(activation.norm().cpu().item()),
                        first_order_prediction=prediction,
                        causal_effect=causal,
                        integrated_direct=direct_map.get(node.node_id),
                    )
                )
        target_logit = float(target_tensor.detach().cpu().item())
        return ComponentTrace(
            split=sample.split,
            sample_key=sample.key,
            target_token=target,
            predicted_token=predicted_token,
            target_logit=target_logit,
            readout_baseline=baseline,
            integrated_quadrature_residual=quadrature_residual,
            residual_reconstruction_error=reconstruction_error,
            readout_parity_error=readout_actual - target_logit,
            quadrature_steps=quadrature_steps,
            measurements=tuple(measurements),
        )
    finally:
        model.train(was_training)
