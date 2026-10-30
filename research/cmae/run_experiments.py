"""Sequential multi-architecture CMAE smoke experiments.

These are interface and numerical experiments on small deterministic models, not a
benchmark of real-world poison detection. No models are run in parallel.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from research.cmae.detectors import art_spectral_scores, robust_mad_spectral
from research.cmae.engine import CausalMechanisticAlignmentEngine, Evidence
from research.cmae.lenses import TopKSparseChart, normalized_random_dictionary
from research.cmae.topology import leave_one_out_topology_scores, topology_signature
from research.cmae.tracing import (
    CMAEProbe,
    ProbeLayout,
    concept_signatures,
    trace_probe,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(6, 8)
        self.hidden = nn.Linear(8, 8)
        self.probe = CMAEProbe()
        self.output = nn.Linear(8, 2)

    def forward(self, value: Tensor) -> Tensor:
        value = torch.tanh(self.input(value))
        value = torch.tanh(self.hidden(value))
        return self.output(self.probe(value))


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.probe = CMAEProbe(ProbeLayout.CHANNELS_FIRST)
        self.output = nn.Linear(4, 2)

    def forward(self, value: Tensor) -> Tensor:
        value = torch.relu(self.convolution(value))
        value = self.probe(value)
        pooled = value.mean(dim=(2, 3))
        return self.output(pooled)


class TinyGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(4, 6, batch_first=True)
        self.probe = CMAEProbe()
        self.output = nn.Linear(6, 2)

    def forward(self, value: Tensor) -> Tensor:
        sequence, _ = self.recurrent(value)
        sequence = self.probe(sequence)
        return self.output(sequence[:, -1])


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(4, 6)
        layer = nn.TransformerEncoderLayer(
            d_model=6,
            nhead=2,
            dim_feedforward=12,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.probe = CMAEProbe()
        self.output = nn.Linear(6, 2)

    def forward(self, value: Tensor) -> Tensor:
        value = self.encoder(torch.tanh(self.embedding(value)))
        value = self.probe(value)
        return self.output(value.mean(dim=1))


def _squared_loss(output: Tensor) -> Tensor:
    return output.square().mean()


def _build(kind: str, seed: int) -> tuple[nn.Module, CMAEProbe, Tensor]:
    torch.manual_seed(seed)
    constructors: dict[str, Callable[[], nn.Module]] = {
        "mlp": TinyMLP,
        "cnn": TinyCNN,
        "gru": TinyGRU,
        "transformer": TinyTransformer,
    }
    if kind not in constructors:
        raise ValueError(f"unknown architecture {kind}")
    model = constructors[kind]().double()
    probe = getattr(model, "probe")
    if not isinstance(probe, CMAEProbe):
        raise TypeError("architecture has no CMAE probe")
    generator = torch.Generator().manual_seed(seed + 1)
    if kind == "mlp":
        inputs = torch.randn(8, 6, generator=generator, dtype=torch.float64)
    elif kind == "cnn":
        inputs = torch.randn(8, 1, 6, 6, generator=generator, dtype=torch.float64)
    else:
        inputs = torch.randn(8, 5, 4, generator=generator, dtype=torch.float64)
    inputs[-1] = 4.0 + 0.05 * torch.randn_like(inputs[-1])
    return model, probe, inputs


def _rank_descending(values: np.ndarray, index: int) -> int:
    order = np.argsort(-values, kind="stable")
    return int(np.where(order == index)[0][0]) + 1


def run_architecture(kind: str, *, seed: int = 17, include_art: bool = True) -> dict[str, object]:
    model, probe, inputs = _build(kind, seed)
    traces = [
        trace_probe(kind, index, model, probe, inputs[index : index + 1], _squared_loss)
        for index in range(inputs.shape[0])
    ]
    dictionary = normalized_random_dictionary(traces[0].activation.shape[1], 12, seed=seed + 2)
    chart = TopKSparseChart(dictionary=dictionary, k=3, signed=True)
    signatures = [concept_signatures(trace, chart) for trace in traces]
    activation_points = np.stack([item.sparse_activation for item in signatures])
    gradient_points = np.stack([item.sparse_gradient for item in signatures])
    topological_scores = leave_one_out_topology_scores(activation_points)
    spectral = robust_mad_spectral(gradient_points, max_steps=50)
    engine = CausalMechanisticAlignmentEngine(spectral_threshold=3.5)
    evidence = [
        Evidence(
            sample_index=index,
            finite=bool(
                np.all(np.isfinite(trace.activation)) and np.all(np.isfinite(trace.output_gradient))
            ),
            spectral_z=float(spectral.robust_z[index]),
            topological_score=float(topological_scores[index]),
        )
        for index, trace in enumerate(traces)
    ]
    records = engine.audit(evidence, detector_healthy=spectral.healthy)
    decisions = Counter(record.decision.value for record in records)
    topology = topology_signature(activation_points)
    art_scores: np.ndarray | None = None
    if include_art:
        art_scores = art_spectral_scores(activation_points)
    shifted_index = inputs.shape[0] - 1
    result: dict[str, object] = {
        "architecture": kind,
        "samples": inputs.shape[0],
        "positions_per_sample": traces[0].activation.shape[0],
        "latent_dimension": traces[0].activation.shape[1],
        "all_traces_finite": all(item.finite for item in evidence),
        "mean_gradient_norm": float(
            np.mean([np.linalg.norm(trace.output_gradient) for trace in traces])
        ),
        "mean_activation_projection_residual_ratio": float(
            np.mean([item.activation_projection_residual_ratio for item in signatures])
        ),
        "spectral_health": spectral.reason,
        "spectral_iterations": spectral.iterations,
        "spectral_residual": spectral.residual,
        "shifted_sample_spectral_rank": _rank_descending(spectral.robust_z, shifted_index),
        "shifted_sample_topology_rank": _rank_descending(topological_scores, shifted_index),
        "decisions": dict(sorted(decisions.items())),
        "effective_rank": topology.effective_rank,
        "model_training_steps": 0,
        "claim_scope": "deterministic interface smoke test; not a trained poison benchmark",
    }
    if art_scores is not None:
        result["shifted_sample_art_rank"] = _rank_descending(art_scores, shifted_index)
    return result


def run_all(*, include_art: bool = True) -> list[dict[str, object]]:
    return [
        run_architecture(kind, include_art=include_art)
        for kind in ("mlp", "cnn", "gru", "transformer")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--without-art", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    results = run_all(include_art=not arguments.without_art)
    rendered = json.dumps(results, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)


if __name__ == "__main__":
    main()
