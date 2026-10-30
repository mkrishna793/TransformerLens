"""Fail-closed circuit-tracer attribution/intervention adapter.

No graph is executed unless upstream attribution validity is explicitly true.
Graph error nodes are accounted for but never assigned semantic meaning.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


@dataclass(frozen=True)
class CircuitTracerRequest:
    model_name: str
    transcoder_set: str
    prompt: str
    backend: Literal["transformerlens", "nnsight"] = "transformerlens"
    attribution_targets: tuple[str, ...] | None = None
    max_n_logits: int = 4
    max_feature_nodes: int = 128
    batch_size: int = 32
    interventions: tuple[tuple[int, int, int, float], ...] = ()
    target_token_id: int | None = None
    graph_output: Path | None = None

    def __post_init__(self) -> None:
        if (
            not self.model_name.strip()
            or not self.transcoder_set.strip()
            or not self.prompt
            or self.max_n_logits <= 0
            or self.max_feature_nodes <= 0
            or self.batch_size <= 0
        ):
            raise ValueError(
                "circuit-tracer request needs explicit identifiers and positive budgets"
            )
        for layer, position, feature, value in self.interventions:
            if layer < 0 or position < 0 or feature < 0 or not np.isfinite(value):
                raise ValueError("invalid circuit-tracer intervention")


def audit_circuit_tracer_graph(graph: Any) -> dict[str, Any]:
    """Audit actual graph tensors and preserve error-node edge diagnostics."""

    active = graph.active_features
    selected = graph.selected_features
    adjacency = graph.adjacency_matrix
    tokens = graph.input_tokens
    targets = graph.logit_targets
    if active.ndim != 2 or active.shape[1] != 3 or selected.ndim != 1:
        raise ValueError("invalid circuit-tracer feature inventories")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("circuit-tracer adjacency must be square")
    layer_count = int(graph.cfg.n_layers)
    position_count = int(tokens.numel())
    error_count = layer_count * position_count
    expected = selected.numel() + error_count + position_count + len(targets)
    if adjacency.shape[0] != expected:
        raise ValueError(
            f"circuit-tracer node accounting failed: {adjacency.shape[0]} != {expected}"
        )
    if graph.scan_name is None:
        raise ValueError("circuit-tracer graph lacks its transcoder scan identifier")
    error_start = selected.numel()
    error_columns = adjacency[:, error_start : error_start + error_count]
    for tensor in (active, selected, adjacency, graph.activation_values):
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise FloatingPointError("nonfinite circuit-tracer graph tensor")
    return {
        "active_feature_count": int(active.shape[0]),
        "selected_feature_count": int(selected.numel()),
        "position_count": position_count,
        "error_node_count": error_count,
        "embedding_node_count": position_count,
        "logit_node_count": len(targets),
        "adjacency_shape": list(adjacency.shape),
        "error_edge_absolute_mass": float(error_columns.abs().sum().detach().cpu().item()),
        "maximum_absolute_error_edge": float(error_columns.abs().max().detach().cpu().item()),
        "scan_name": graph.scan_name,
        "semantic_claim": "none; error nodes remain replacement remainder writes",
    }


def run_circuit_tracer(
    request: CircuitTracerRequest, *, upstream_attribution_valid: bool
) -> dict[str, Any]:
    """Execute only after attribution validity; otherwise return an explicit non-result."""

    identity = {
        "model_name": request.model_name,
        "transcoder_set": request.transcoder_set,
        "backend": request.backend,
        "source_commit": "255d989e93be1299d53d67b59e9fd76a0e274b14",
    }
    if not upstream_attribution_valid:
        return {
            "status": "blocked_by_upstream_gate",
            "executed": False,
            "identity": identity,
            "reason": "counterfactual attribution validity did not pass",
        }
    try:
        replacement_module = importlib.import_module(
            "circuit_tracer.replacement_model.replacement_model"
        )
        attribution_module = importlib.import_module("circuit_tracer.attribution.attribute")
    except ImportError as error:
        return {
            "status": "not_executed",
            "executed": False,
            "identity": identity,
            "reason": f"circuit-tracer dependency unavailable: {error}",
        }
    try:
        replacement_class = getattr(replacement_module, "ReplacementModel")
        model = replacement_class.from_pretrained(
            request.model_name,
            request.transcoder_set,
            backend=request.backend,
            device=torch.device("cpu"),
            dtype=torch.float32,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        graph = getattr(attribution_module, "attribute")(
            request.prompt,
            model,
            attribution_targets=request.attribution_targets,
            max_n_logits=request.max_n_logits,
            max_feature_nodes=request.max_feature_nodes,
            batch_size=request.batch_size,
            offload="cpu",
            verbose=False,
        )
        graph_audit = audit_circuit_tracer_graph(graph)
        interventions: list[dict[str, Any]] = []
        if request.interventions:
            with torch.no_grad():
                baseline = model(request.prompt)
            logits, _ = model.feature_intervention(
                request.prompt,
                list(request.interventions),
                freeze_attention=True,
                return_activations=False,
            )
            difference = logits - baseline
            record: dict[str, Any] = {
                "interventions": [list(value) for value in request.interventions],
                "maximum_absolute_logit_change": float(
                    difference.abs().max().detach().cpu().item()
                ),
                "logit_change_l2": float(difference.norm().detach().cpu().item()),
            }
            if request.target_token_id is not None:
                token = request.target_token_id
                record["final_target_logit_change"] = float(
                    difference[0, -1, token].detach().cpu().item()
                )
            interventions.append(record)
        if request.graph_output is not None:
            request.graph_output.parent.mkdir(parents=True, exist_ok=True)
            graph.to_pt(str(request.graph_output))
    except Exception as error:
        return {
            "status": "not_executed",
            "executed": False,
            "identity": identity,
            "reason": f"{type(error).__name__}: {error}",
        }
    return {
        "status": "executed",
        "executed": True,
        "identity": identity,
        "graph": graph_audit,
        "interventions": interventions,
        "graph_output": str(request.graph_output) if request.graph_output else None,
        "scope": "supported replacement-model graph only; not semantic or faithfulness certification",
    }
