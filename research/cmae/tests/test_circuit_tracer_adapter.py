from types import SimpleNamespace

import pytest
import torch

from research.cmae.circuit_tracer_adapter import (
    CircuitTracerRequest,
    audit_circuit_tracer_graph,
    run_circuit_tracer,
)


def _request() -> CircuitTracerRequest:
    return CircuitTracerRequest(
        model_name="unsupported/local-test-model",
        transcoder_set="unsupported/local-test-transcoder",
        prompt="test",
    )


def test_failed_upstream_gate_blocks_before_model_loading() -> None:
    result = run_circuit_tracer(_request(), upstream_attribution_valid=False)
    assert result["status"] == "blocked_by_upstream_gate"
    assert not result["executed"]


def test_graph_node_and_error_accounting() -> None:
    graph = SimpleNamespace(
        active_features=torch.zeros(5, 3, dtype=torch.int64),
        selected_features=torch.tensor([0, 3]),
        adjacency_matrix=torch.ones(13, 13),
        input_tokens=torch.tensor([1, 2, 3]),
        logit_targets=[object(), object()],
        activation_values=torch.ones(5),
        scan_name="test-scan",
        cfg=SimpleNamespace(n_layers=2),
    )
    audit = audit_circuit_tracer_graph(graph)
    assert audit["selected_feature_count"] == 2
    assert audit["error_node_count"] == 6
    assert audit["embedding_node_count"] == 3
    assert audit["logit_node_count"] == 2
    assert audit["error_edge_absolute_mass"] == 13 * 6
    graph.adjacency_matrix = torch.ones(12, 12)
    with pytest.raises(ValueError, match="node accounting"):
        audit_circuit_tracer_graph(graph)
