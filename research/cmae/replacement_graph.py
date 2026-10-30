"""Trained SAE/CLT replacements with error nodes and validated local edges.

The graph is deliberately scoped to answer-position MLP writes. Direct output
allocations use a shared integrated Jacobian and conserve the target-logit
change up to a measured quadrature residual. Feature-to-feature edges are local
predictions, checked against actual answer-position write ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from research.cmae.arithmetic_data import ArithmeticDataset, ArithmeticSample
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.metrics import pearson, spearman

ReplacementKind = Literal["sae", "clt"]


@dataclass(frozen=True)
class MLPActivations:
    """Detached MLP inputs and outputs for a batch of complete prompts."""

    inputs: tuple[Tensor, Tensor]
    outputs: tuple[Tensor, Tensor]

    def flat(self, layer: int) -> tuple[Tensor, Tensor]:
        inputs = self.inputs[layer]
        outputs = self.outputs[layer]
        return inputs.reshape(-1, inputs.shape[-1]), outputs.reshape(-1, outputs.shape[-1])


class SparseCrossLayerTranscoder(nn.Module):
    """Two-layer ReLU SAE/CLT with decoder writes to all downstream MLPs."""

    def __init__(self, width: int, features: int, kind: ReplacementKind) -> None:
        super().__init__()
        if width <= 0 or features <= 0 or kind not in ("sae", "clt"):
            raise ValueError("invalid replacement dimensions or kind")
        self.width = width
        self.features = features
        self.kind = kind
        self.encoder0 = nn.Linear(width, features)
        self.encoder1 = nn.Linear(width, features)
        self.decoder00 = nn.Linear(features, width, bias=False)
        self.decoder01 = nn.Linear(features, width, bias=False)
        self.decoder11 = nn.Linear(features, width, bias=False)
        if kind == "sae":
            with torch.no_grad():
                self.decoder01.weight.zero_()
            self.decoder01.weight.requires_grad_(False)

    def encode(self, layer: int, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.width:
            raise ValueError("replacement input width mismatch")
        if layer == 0:
            return torch.relu(self.encoder0(inputs))
        if layer == 1:
            return torch.relu(self.encoder1(inputs))
        raise ValueError("the scoped replacement has exactly two layers")

    def decoder(self, source_layer: int, target_layer: int) -> Tensor:
        if (source_layer, target_layer) == (0, 0):
            return self.decoder00.weight
        if (source_layer, target_layer) == (0, 1):
            return self.decoder01.weight
        if (source_layer, target_layer) == (1, 1):
            return self.decoder11.weight
        raise ValueError("unsupported decoder path")

    def decode(self, codes: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        code0, code1 = codes
        reconstruction0 = self.decoder00(code0)
        reconstruction1 = self.decoder01(code0) + self.decoder11(code1)
        return reconstruction0, reconstruction1

    def forward(
        self, inputs: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        codes = (self.encode(0, inputs[0]), self.encode(1, inputs[1]))
        return self.decode(codes), codes


def collect_mlp_activations(
    model: ArithmeticTransformer,
    tokens: Tensor,
    *,
    batch_size: int = 64,
) -> MLPActivations:
    """Collect complete-position MLP interfaces without using detached-cache gradients."""

    if tokens.ndim != 2 or len(tokens) == 0 or batch_size <= 0:
        raise ValueError("MLP activation collection needs a nonempty token batch")
    # MLP output hooks live on blocks; input hooks live on the MLP module.
    hook_names = [f"blocks.{layer}.mlp.hook_mlp_input" for layer in range(2)]
    hook_names += [f"blocks.{layer}.hook_mlp_out" for layer in range(2)]
    if model.config.n_layers != 2:
        raise ValueError("the scoped CLT requires exactly two Transformer layers")
    collected: dict[str, list[Tensor]] = {name: [] for name in hook_names}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start : start + batch_size]

                def capture(tensor: Tensor, hook: object) -> None:
                    name = getattr(hook, "name", None)
                    if not isinstance(name, str):
                        raise RuntimeError("MLP capture hook has no name")
                    collected[name].append(tensor.detach().cpu().clone())

                with model.hooks(fwd_hooks=[(name, capture) for name in hook_names]):
                    model(batch)
    finally:
        model.train(was_training)
    values = {name: torch.cat(batches) for name, batches in collected.items()}
    return MLPActivations(
        inputs=(
            values["blocks.0.mlp.hook_mlp_input"],
            values["blocks.1.mlp.hook_mlp_input"],
        ),
        outputs=(values["blocks.0.hook_mlp_out"], values["blocks.1.hook_mlp_out"]),
    )


def fit_replacement(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    kind: ReplacementKind,
    *,
    features: int = 64,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.003,
    l1_coefficient: float = 0.001,
    seed: int = 0,
) -> tuple[SparseCrossLayerTranscoder, tuple[float, ...]]:
    """Fit only on training prompts; independent SAE is CLT with a zero 0->1 decoder."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or l1_coefficient < 0 or features <= 0:
        raise ValueError("invalid replacement optimization configuration")
    tokens_np, _, _ = dataset.arrays("train")
    tokens = torch.from_numpy(tokens_np)
    activations = collect_mlp_activations(model, tokens)
    input0, output0 = activations.flat(0)
    input1, output1 = activations.flat(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        replacement = SparseCrossLayerTranscoder(model.config.d_model, features, kind)
        optimizer = torch.optim.Adam(
            [parameter for parameter in replacement.parameters() if parameter.requires_grad],
            lr=learning_rate,
        )
        rng = np.random.default_rng(seed + 1)
        history: list[float] = []
        replacement.train()
        for _ in range(epochs):
            permutation = rng.permutation(len(input0))
            weighted_loss = 0.0
            for start in range(0, len(permutation), batch_size):
                indices_np = permutation[start : start + batch_size].copy()
                indices = torch.from_numpy(indices_np)
                optimizer.zero_grad(set_to_none=True)
                reconstructions, codes = replacement((input0[indices], input1[indices]))
                reconstruction_loss = (reconstructions[0] - output0[indices]).square().mean() + (
                    reconstructions[1] - output1[indices]
                ).square().mean()
                sparsity = codes[0].abs().mean() + codes[1].abs().mean()
                loss = reconstruction_loss + l1_coefficient * sparsity
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("nonfinite replacement training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(replacement.parameters(), max_norm=5.0)
                optimizer.step()
                weighted_loss += float(loss.detach().cpu().item()) * len(indices)
            history.append(weighted_loss / len(input0))
    replacement.eval()
    return replacement, tuple(history)


def evaluate_replacement_reconstruction(
    replacement: SparseCrossLayerTranscoder,
    activations: MLPActivations,
) -> dict[str, object]:
    """Measure reconstruction and sparsity without assigning feature semantics."""

    result: dict[str, object] = {}
    with torch.no_grad():
        for layer in (0, 1):
            inputs, outputs = activations.flat(layer)
            codes = replacement.encode(layer, inputs)
            if layer == 0:
                reconstruction = replacement.decoder00(codes)
            else:
                code0 = replacement.encode(0, activations.flat(0)[0])
                reconstruction = replacement.decode((code0, codes))[1]
            error = outputs - reconstruction
            active = codes > 1e-8
            centered = outputs - outputs.mean(dim=0, keepdim=True)
            result[f"layer{layer}"] = {
                "samples": int(outputs.shape[0]),
                "mse": float(error.square().mean().cpu().item()),
                "relative_error_mass": float(
                    (error.square().sum() / outputs.square().sum().clamp_min(1e-12)).cpu().item()
                ),
                "explained_variance": 1.0
                - float(
                    (error.square().sum() / centered.square().sum().clamp_min(1e-12)).cpu().item()
                ),
                "mean_l0": float(active.to(torch.float32).sum(dim=1).mean().cpu().item()),
                "active_feature_fraction": float(active.to(torch.float32).mean().cpu().item()),
                "dead_feature_fraction": float(
                    (~active.any(dim=0)).to(torch.float32).mean().cpu().item()
                ),
                "max_reconstruction_conservation_error": float(
                    (outputs - reconstruction - error).abs().max().cpu().item()
                ),
            }
    result["cross_layer_decoder_norm"] = float(
        replacement.decoder01.weight.detach().norm().cpu().item()
    )
    result["semantic_status"] = "unlabeled_numeric_features"
    return result


@dataclass(frozen=True)
class ReplacementNode:
    node_id: str
    kind: Literal["feature", "error"]
    layer: int
    feature: int | None
    activation: float | None
    value_norm: float
    direct_output_allocation: float
    actual_ablation_effect: float

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "layer": self.layer,
            "feature": self.feature,
            "activation": self.activation,
            "value_norm": self.value_norm,
            "direct_output_allocation": self.direct_output_allocation,
            "actual_ablation_effect": self.actual_ablation_effect,
            "ablation_minus_direct_allocation": self.actual_ablation_effect
            - self.direct_output_allocation,
        }


@dataclass(frozen=True)
class ReplacementEdge:
    source: str
    target: str
    local_prediction: float
    actual_target_effect: float

    @property
    def residual(self) -> float:
        return self.actual_target_effect - self.local_prediction

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": "answer_position_0_to_1_local_feature_edge",
            "local_prediction": self.local_prediction,
            "actual_target_effect": self.actual_target_effect,
            "approximation_residual": self.residual,
        }


@dataclass(frozen=True)
class ReplacementGraphTrace:
    split: str
    sample_key: tuple[int, int, int, int]
    target_token: int
    predicted_token: int
    replacement_predicted_token: int
    target_logit: float
    zero_write_logit: float
    replacement_target_logit: float
    replacement_relative_logit_error_mass: float
    readout_parity_error: float
    reconstruction_conservation_error: float
    integrated_quadrature_residual: float
    quadrature_steps: int
    nodes: tuple[ReplacementNode, ...]
    edges: tuple[ReplacementEdge, ...]

    def metrics(self) -> dict[str, object]:
        direct = np.asarray([node.direct_output_allocation for node in self.nodes])
        ablation = np.asarray([node.actual_ablation_effect for node in self.nodes])
        error_mask = np.asarray([node.kind == "error" for node in self.nodes], dtype=bool)
        predicted_edges = np.asarray([edge.local_prediction for edge in self.edges])
        actual_edges = np.asarray([edge.actual_target_effect for edge in self.edges])
        residuals = actual_edges - predicted_edges
        k = min(5, len(predicted_edges))
        predicted_top = set(np.argsort(-np.abs(predicted_edges))[:k].tolist())
        actual_top = set(np.argsort(-np.abs(actual_edges))[:k].tolist())
        influence_denominator = max(float(np.abs(direct).sum()), 1e-12)
        edge_metrics: dict[str, object]
        if k:
            edge_metrics = {
                "feature_edge_pearson": (
                    pearson(predicted_edges, actual_edges) if len(predicted_edges) >= 2 else None
                ),
                "feature_edge_spearman": (
                    spearman(predicted_edges, actual_edges) if len(predicted_edges) >= 2 else None
                ),
                "feature_edge_mean_absolute_residual": float(np.abs(residuals).mean()),
                "feature_edge_relative_residual_mass": float(np.abs(residuals).sum())
                / max(float(np.abs(actual_edges).sum()), 1e-12),
                "feature_edge_top_5_absolute_overlap": len(predicted_top & actual_top) / k,
            }
        else:
            edge_metrics = {
                "feature_edge_pearson": None,
                "feature_edge_spearman": None,
                "feature_edge_mean_absolute_residual": None,
                "feature_edge_relative_residual_mass": None,
                "feature_edge_top_5_absolute_overlap": None,
            }
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "replacement_prediction_match": self.predicted_token
            == self.replacement_predicted_token,
            "replacement_relative_logit_error_mass": self.replacement_relative_logit_error_mass,
            "readout_parity_error": self.readout_parity_error,
            "reconstruction_conservation_error": self.reconstruction_conservation_error,
            "integrated_quadrature_residual": self.integrated_quadrature_residual,
            "direct_output_allocation_sum": float(direct.sum()),
            "output_conservation_error": self.target_logit
            - self.zero_write_logit
            - float(direct.sum())
            - self.integrated_quadrature_residual,
            "error_output_influence_mass": float(np.abs(direct[error_mask]).sum())
            / influence_denominator,
            "feature_output_influence_mass": float(np.abs(direct[~error_mask]).sum())
            / influence_denominator,
            "direct_allocation_vs_ablation_pearson": pearson(direct, ablation),
            **edge_metrics,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "sample_key": list(self.sample_key),
            "target_token": self.target_token,
            "predicted_token": self.predicted_token,
            "replacement_predicted_token": self.replacement_predicted_token,
            "target_logit": self.target_logit,
            "zero_write_logit": self.zero_write_logit,
            "replacement_target_logit": self.replacement_target_logit,
            "quadrature_steps": self.quadrature_steps,
            "metrics": self.metrics(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "scope": (
                "Answer-position MLP-write replacement. Direct output allocation is integrated; "
                "feature edges are local and separately ablated. Neither is a semantic certificate."
            ),
        }


@dataclass(frozen=True)
class _NodeSpec:
    node: ReplacementNode
    delta0: Tensor
    delta1: Tensor


def _run_answer_writes(
    model: ArithmeticTransformer,
    tokens: Tensor,
    write0: Tensor,
    write1: Tensor,
    *,
    target_token: int | None,
) -> tuple[Tensor, Tensor]:
    """Run with independent answer-position MLP writes and capture layer-1 input."""

    captured: dict[str, Tensor] = {}

    def override0(tensor: Tensor, hook: object) -> Tensor:
        del hook
        result = tensor.clone()
        result[0, -1, :] = write0
        return result

    def override1(tensor: Tensor, hook: object) -> Tensor:
        del hook
        result = tensor.clone()
        result[0, -1, :] = write1
        return result

    def capture(tensor: Tensor, hook: object) -> None:
        del hook
        captured["layer1_input"] = tensor

    output = model.run_with_hooks(
        tokens,
        target_token=target_token,
        fwd_hooks=[
            ("blocks.0.hook_mlp_out", override0),
            ("blocks.1.hook_mlp_out", override1),
            ("blocks.1.mlp.hook_mlp_input", capture),
        ],
    )
    return output, captured["layer1_input"]


def _integrated_output_jacobians(
    model: ArithmeticTransformer,
    tokens: Tensor,
    target_token: int,
    actual0: Tensor,
    actual1: Tensor,
    *,
    steps: int,
) -> tuple[Tensor, Tensor]:
    if steps < 2:
        raise ValueError("integrated replacement readout needs at least two quadrature points")
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    average0 = torch.zeros_like(actual0)
    average1 = torch.zeros_like(actual1)
    for node, weight in zip(nodes, weights, strict=True):
        scale = (float(node) + 1.0) / 2.0
        quadrature_weight = float(weight) / 2.0
        write0 = (actual0 * scale).requires_grad_(True)
        write1 = (actual1 * scale).requires_grad_(True)
        value, _ = _run_answer_writes(model, tokens, write0, write1, target_token=target_token)
        gradient0, gradient1 = torch.autograd.grad(value, (write0, write1), retain_graph=False)
        average0 += quadrature_weight * gradient0.detach()
        average1 += quadrature_weight * gradient1.detach()
    return average0, average1


def trace_replacement_graph(
    model: ArithmeticTransformer,
    replacement: SparseCrossLayerTranscoder,
    sample: ArithmeticSample,
    *,
    target_token: int | None = None,
    quadrature_steps: int = 256,
) -> ReplacementGraphTrace:
    """Build an answer-position graph and causally validate every retained local edge."""

    target = sample.target if target_token is None else target_token
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    was_training = model.training
    replacement_training = replacement.training
    model.eval()
    replacement.eval()
    try:
        activations = collect_mlp_activations(model, tokens, batch_size=1)
        input0 = activations.inputs[0][0, -1, :]
        input1 = activations.inputs[1][0, -1, :]
        actual0 = activations.outputs[0][0, -1, :]
        actual1 = activations.outputs[1][0, -1, :]
        with torch.no_grad():
            code0 = replacement.encode(0, input0)
            code1 = replacement.encode(1, input1)
            reconstruction0, reconstruction1 = replacement.decode((code0, code1))
            error0 = actual0 - reconstruction0
            error1 = actual1 - reconstruction1
            original_logits = model(tokens)[0, -1, :]
            functional_logits, _ = _run_answer_writes(
                model, tokens, actual0, actual1, target_token=None
            )
            functional_answer = functional_logits[0, -1, :]
            replacement_logits, _ = _run_answer_writes(
                model, tokens, reconstruction0, reconstruction1, target_token=None
            )
            replacement_answer = replacement_logits[0, -1, :]
            zero_write_value, _ = _run_answer_writes(
                model,
                tokens,
                torch.zeros_like(actual0),
                torch.zeros_like(actual1),
                target_token=target,
            )
        readout_parity = float((functional_answer - original_logits).abs().max().cpu().item())
        conservation = max(
            float((actual0 - reconstruction0 - error0).abs().max().cpu().item()),
            float((actual1 - reconstruction1 - error1).abs().max().cpu().item()),
        )
        jacobian0, jacobian1 = _integrated_output_jacobians(
            model,
            tokens,
            target,
            actual0,
            actual1,
            steps=quadrature_steps,
        )

        # Local layer-1 input Jacobian holds the layer-1 MLP output independent.
        local_write0 = actual0.clone().requires_grad_(True)
        local_write1 = actual1.clone()
        _, layer1_input = _run_answer_writes(
            model, tokens, local_write0, local_write1, target_token=None
        )
        local_rows: list[Tensor] = []
        for output_index in range(model.config.d_model):
            gradient = torch.autograd.grad(
                layer1_input[0, -1, output_index],
                local_write0,
                retain_graph=output_index + 1 < model.config.d_model,
            )[0]
            local_rows.append(gradient)
        local_jacobian = torch.stack(local_rows).detach()

        specs: list[_NodeSpec] = []
        with torch.no_grad():
            for layer, code in ((0, code0), (1, code1)):
                for feature in range(replacement.features):
                    activation = code[feature]
                    if float(activation.cpu().item()) <= 1e-8:
                        continue
                    if layer == 0:
                        delta0 = activation * replacement.decoder00.weight[:, feature]
                        delta1 = activation * replacement.decoder01.weight[:, feature]
                    else:
                        delta0 = torch.zeros_like(actual0)
                        delta1 = activation * replacement.decoder11.weight[:, feature]
                    allocation = float(
                        ((delta0 * jacobian0).sum() + (delta1 * jacobian1).sum()).cpu().item()
                    )
                    specs.append(
                        _NodeSpec(
                            ReplacementNode(
                                f"l{layer}_feature_{feature}",
                                "feature",
                                layer,
                                feature,
                                float(activation.cpu().item()),
                                float(activation.abs().cpu().item()),
                                allocation,
                                0.0,
                            ),
                            delta0,
                            delta1,
                        )
                    )
            for layer, error in ((0, error0), (1, error1)):
                delta0 = error if layer == 0 else torch.zeros_like(actual0)
                delta1 = error if layer == 1 else torch.zeros_like(actual1)
                allocation = float(
                    ((delta0 * jacobian0).sum() + (delta1 * jacobian1).sum()).cpu().item()
                )
                specs.append(
                    _NodeSpec(
                        ReplacementNode(
                            f"l{layer}_error",
                            "error",
                            layer,
                            None,
                            None,
                            float(error.norm().cpu().item()),
                            allocation,
                            0.0,
                        ),
                        delta0,
                        delta1,
                    )
                )

        target_logit = float(original_logits[target].cpu().item())
        actual_effects: list[float] = []
        ablated_code1: list[Tensor] = []
        with torch.no_grad():
            for spec in specs:
                ablated_value, ablated_input = _run_answer_writes(
                    model,
                    tokens,
                    actual0 - spec.delta0,
                    actual1 - spec.delta1,
                    target_token=target,
                )
                actual_effects.append(target_logit - float(ablated_value.cpu().item()))
                ablated_code1.append(replacement.encode(1, ablated_input[0, -1, :]))
        nodes = tuple(
            ReplacementNode(
                spec.node.node_id,
                spec.node.kind,
                spec.node.layer,
                spec.node.feature,
                spec.node.activation,
                spec.node.value_norm,
                spec.node.direct_output_allocation,
                actual_effect,
            )
            for spec, actual_effect in zip(specs, actual_effects, strict=True)
        )

        edges: list[ReplacementEdge] = []
        active_targets = [
            index
            for index in range(replacement.features)
            if float(code1[index].cpu().item()) > 1e-8
        ]
        with torch.no_grad():
            target_mask = (code1 > 1e-8).to(code1.dtype)
            encoder_weight = replacement.encoder1.weight
            virtual_weights = torch.einsum(
                "fo,od,ds->sf",
                encoder_weight * target_mask[:, None],
                local_jacobian,
                replacement.decoder00.weight,
            )
            error_virtual = (encoder_weight * target_mask[:, None]) @ local_jacobian @ error0
            for spec_index, (spec, ablated_code) in enumerate(
                zip(specs, ablated_code1, strict=True)
            ):
                if spec.node.layer != 0:
                    continue
                for target_feature in active_targets:
                    if spec.node.kind == "feature":
                        source_feature = spec.node.feature
                        if source_feature is None:
                            raise RuntimeError("feature node is missing its index")
                        prediction = (
                            code0[source_feature] * virtual_weights[source_feature, target_feature]
                        )
                    else:
                        prediction = error_virtual[target_feature]
                    actual_target_effect = code1[target_feature] - ablated_code[target_feature]
                    edges.append(
                        ReplacementEdge(
                            spec.node.node_id,
                            f"l1_feature_{target_feature}",
                            float(prediction.cpu().item()),
                            float(actual_target_effect.cpu().item()),
                        )
                    )
        direct_sum = sum(node.direct_output_allocation for node in nodes)
        quadrature_residual = target_logit - float(zero_write_value.cpu().item()) - direct_sum
        relative_logit_error = float(
            (
                (replacement_answer - original_logits).square().sum()
                / original_logits.square().sum().clamp_min(1e-12)
            )
            .cpu()
            .item()
        )
        return ReplacementGraphTrace(
            split=sample.split,
            sample_key=sample.key,
            target_token=target,
            predicted_token=int(original_logits.argmax(dim=-1).cpu().item()),
            replacement_predicted_token=int(replacement_answer.argmax(dim=-1).cpu().item()),
            target_logit=target_logit,
            zero_write_logit=float(zero_write_value.cpu().item()),
            replacement_target_logit=float(replacement_answer[target].cpu().item()),
            replacement_relative_logit_error_mass=relative_logit_error,
            readout_parity_error=readout_parity,
            reconstruction_conservation_error=conservation,
            integrated_quadrature_residual=quadrature_residual,
            quadrature_steps=quadrature_steps,
            nodes=nodes,
            edges=tuple(edges),
        )
    finally:
        model.train(was_training)
        replacement.train(replacement_training)
