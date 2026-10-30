"""Explicit, residual-aware attribution traces for the small real MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from research.cmae.metrics import pearson

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TraceTarget:
    name: str
    class_index: int
    intervention: str = "zero_hidden_unit"
    baseline: str = "zero_input_taylor"


@dataclass(frozen=True)
class TraceEdge:
    source: str
    target: str
    kind: str
    contribution: FloatArray


@dataclass(frozen=True)
class AttributionTrace:
    target: TraceTarget
    output: float
    baseline_output: float
    readout_bias: float
    hidden_edges: TraceEdge
    input_path_edges: TraceEdge
    residual: float
    first_order_path_residual: float
    intervention_actual: FloatArray
    intervention_predicted: FloatArray
    metadata: Mapping[str, object]

    @property
    def conservation_error(self) -> float:
        explained = float(self.hidden_edges.contribution.sum()) + self.readout_bias
        return self.output - explained - self.residual

    @property
    def intervention_max_error(self) -> float:
        return float(np.max(np.abs(self.intervention_actual - self.intervention_predicted)))

    @property
    def intervention_pearson(self) -> float:
        return pearson(self.intervention_predicted, self.intervention_actual)


def trace_real_mlp(
    model: Any,
    input_vector: FloatArray,
    *,
    target_class: int = 1,
) -> AttributionTrace:
    """Trace one target logit through ``input -> tanh -> linear readout``.

    Hidden-to-target edges are an exact linear-readout decomposition. Input-path
    edges are a first-order Taylor attribution around zero input and carry an
    explicit residual. Hidden ablations independently test the exact edge claim.
    """

    values = np.asarray(input_vector, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("trace input must be a finite vector")
    if not hasattr(model, "input") or not hasattr(model, "head"):
        raise ValueError("trace model must expose input and head linear layers")
    weight = model.input.weight
    dtype = weight.dtype
    device = weight.device
    x = torch.from_numpy(values).to(device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        preactivation = model.input(x)
        hidden = torch.tanh(preactivation)
        logits = model.head(hidden)
        if not 0 <= target_class < logits.numel():
            raise ValueError("target class is outside the model output")
        target = logits[target_class]
        zero = torch.zeros_like(x)
        baseline_hidden = torch.tanh(model.input(zero))
        baseline = model.head(baseline_hidden)[target_class]
        readout_weight = model.head.weight[target_class]
        readout_bias = model.head.bias[target_class]
        hidden_contribution = readout_weight * hidden
        baseline_hidden_jacobian = (1.0 - baseline_hidden.square())[:, None] * weight
        input_jacobian = readout_weight @ baseline_hidden_jacobian
        path_contribution = input_jacobian * x
        predicted_path = baseline + input_jacobian @ x
        actual = float(target.detach().cpu().item())
        baseline_value = float(baseline.detach().cpu().item())
        hidden_values = hidden_contribution.detach().cpu().numpy().astype(np.float64)
        path_values = path_contribution.detach().cpu().numpy().astype(np.float64)
        actual_interventions = []
        for index in range(hidden.numel()):
            ablated = hidden.clone()
            ablated[index] = 0.0
            ablated_target = model.head(ablated)[target_class]
            actual_interventions.append(float((target - ablated_target).detach().cpu().item()))
        bias_value = float(readout_bias.detach().cpu().item())
        residual = actual - float(hidden_values.sum()) - bias_value
        return AttributionTrace(
            target=TraceTarget("real_mlp_target_logit", target_class),
            output=actual,
            baseline_output=baseline_value,
            readout_bias=bias_value,
            hidden_edges=TraceEdge("hidden", "target_logit", "exact_linear_readout", hidden_values),
            input_path_edges=TraceEdge(
                "input", "target_logit", "first_order_taylor_path", path_values
            ),
            residual=residual,
            first_order_path_residual=actual - float(predicted_path.detach().cpu().item()),
            intervention_actual=np.asarray(actual_interventions, dtype=np.float64),
            intervention_predicted=hidden_values.copy(),
            metadata={
                "target_semantics": "full target logit; hidden ablation measures full-minus-ablated output",
                "path_semantics": "zero-input Taylor prediction; residual is not assigned to an edge",
            },
        )
    finally:
        model.train(was_training)
        if hasattr(model, "probe"):
            model.probe.last_activation = None


def trace_to_dict(trace: AttributionTrace) -> dict[str, object]:
    return {
        "target": trace.target.name,
        "class_index": trace.target.class_index,
        "output": trace.output,
        "baseline_output": trace.baseline_output,
        "readout_bias": trace.readout_bias,
        "hidden_edge_kind": trace.hidden_edges.kind,
        "input_path_kind": trace.input_path_edges.kind,
        "residual": trace.residual,
        "conservation_error": trace.conservation_error,
        "first_order_path_residual": trace.first_order_path_residual,
        "intervention_max_error": trace.intervention_max_error,
        "intervention_pearson": trace.intervention_pearson,
        "hidden_edge_l1": float(np.abs(trace.hidden_edges.contribution).sum()),
        "input_path_l1": float(np.abs(trace.input_path_edges.contribution).sum()),
        "metadata": dict(trace.metadata),
    }
