"""Layerwise EK-FAC influence from training examples to logits and CLT features.

The implementation uses exact per-example position-summed linear weight gradients,
KFE-projected second-moment correction, and a signed inverse-curvature estimate.
It is not exact leave-one-out retraining and ignores untracked/cross-layer curvature.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.arithmetic_data import ArithmeticSample
from research.cmae.arithmetic_transformer import ArithmeticBlock, ArithmeticTransformer
from research.cmae.replacement_graph import SparseCrossLayerTranscoder

FloatArray = NDArray[np.float64]
Objective = Callable[[Tensor, Mapping[str, Tensor]], Tensor]


@dataclass(frozen=True)
class RawLayerGradient:
    """Position-resolved input activations and output cotangents for one linear layer."""

    activation: FloatArray
    output_gradient: FloatArray

    def __post_init__(self) -> None:
        if self.activation.ndim != 2 or self.output_gradient.ndim != 2:
            raise ValueError("raw layer gradients must be position-by-dimension matrices")
        if self.activation.shape[0] != self.output_gradient.shape[0]:
            raise ValueError("activation and output-gradient positions differ")
        if self.activation.shape[0] == 0:
            raise ValueError("raw layer gradients need at least one position")
        if not np.all(np.isfinite(self.activation)) or not np.all(
            np.isfinite(self.output_gradient)
        ):
            raise ValueError("raw layer gradients must be finite")


@dataclass(frozen=True)
class RawSampleGradient:
    sample_key: tuple[int, int, int, int]
    layers: Mapping[str, RawLayerGradient]


@dataclass(frozen=True)
class EKFACLayer:
    name: str
    activation_basis: FloatArray
    output_basis: FloatArray
    corrected_second_moments: FloatArray
    training_projections: FloatArray

    def project(self, raw: RawLayerGradient) -> FloatArray:
        if raw.activation.shape[1] != self.activation_basis.shape[0]:
            raise ValueError(f"activation width mismatch for {self.name}")
        if raw.output_gradient.shape[1] != self.output_basis.shape[0]:
            raise ValueError(f"output-gradient width mismatch for {self.name}")
        projected_activations = raw.activation @ self.activation_basis
        projected_outputs = raw.output_gradient @ self.output_basis
        return projected_outputs.T @ projected_activations


@dataclass(frozen=True)
class EKFACInfluenceModel:
    """Fitted block-diagonal EK-FAC model over a fixed training sample order."""

    sample_keys: tuple[tuple[int, int, int, int], ...]
    layers: tuple[EKFACLayer, ...]
    damping: float
    modules: Mapping[str, nn.Linear] = field(repr=False, compare=False)

    def influence(self, query: RawSampleGradient) -> FloatArray:
        if not self.layers or self.damping <= 0:
            raise ValueError("EK-FAC influence needs layers and positive damping")
        scores = np.zeros(len(self.sample_keys), dtype=np.float64)
        for layer in self.layers:
            if layer.name not in query.layers:
                raise ValueError(f"query is missing tracked layer {layer.name}")
            projected_query = layer.project(query.layers[layer.name])
            weighted_query = projected_query / (layer.corrected_second_moments + self.damping)
            scores -= np.einsum(
                "noi,oi->n", layer.training_projections, weighted_query, optimize=True
            )
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("nonfinite EK-FAC influence")
        return scores

    def logit_influence(
        self,
        model: ArithmeticTransformer,
        sample: ArithmeticSample,
        target_token: int,
    ) -> FloatArray:
        query = capture_logit_gradient(model, sample, target_token, self.modules)
        return self.influence(query)

    def feature_influence(
        self,
        model: ArithmeticTransformer,
        replacement: SparseCrossLayerTranscoder,
        sample: ArithmeticSample,
        *,
        layer: int,
        feature: int,
    ) -> FloatArray:
        query = capture_feature_gradient(
            model, replacement, sample, layer=layer, feature=feature, modules=self.modules
        )
        return self.influence(query)

    def diagnostics(self) -> dict[str, object]:
        return {
            "samples": len(self.sample_keys),
            "damping": self.damping,
            "tracked_layers": [layer.name for layer in self.layers],
            "weight_parameters": sum(
                int(np.prod(layer.corrected_second_moments.shape)) for layer in self.layers
            ),
            "minimum_corrected_second_moment": min(
                float(layer.corrected_second_moments.min()) for layer in self.layers
            ),
            "maximum_corrected_second_moment": max(
                float(layer.corrected_second_moments.max()) for layer in self.layers
            ),
            "scope": (
                "Signed upweighting influence under block-diagonal EK-FAC over tracked linear "
                "weights; not exact retraining. Cross-layer blocks and untracked parameters omitted."
            ),
        }


def tracked_arithmetic_linears(
    model: ArithmeticTransformer,
) -> dict[str, nn.Linear]:
    """Inventory linear calls; custom attention output and embeddings are excluded."""

    if model.config.n_layers != 2:
        raise ValueError("the scoped influence inventory expects two layers")
    modules: dict[str, nn.Linear] = {}
    for index in range(2):
        block = cast(ArithmeticBlock, model.blocks[index])
        for name in ("query", "key", "value"):
            modules[f"blocks.{index}.attention.{name}"] = getattr(block.attention, name)
        modules[f"blocks.{index}.mlp.input"] = block.mlp.input
        modules[f"blocks.{index}.mlp.output"] = block.mlp.output
    modules["unembed"] = model.unembed
    return modules


def _capture_objective_gradient(
    model: ArithmeticTransformer,
    sample: ArithmeticSample,
    modules: Mapping[str, nn.Linear],
    objective: Objective,
    *,
    extra_hooks: tuple[str, ...] = (),
) -> RawSampleGradient:
    """Capture exact output cotangents without mutating parameter ``.grad`` fields."""

    if not modules:
        raise ValueError("at least one linear module is required")
    tokens = torch.from_numpy(sample.tokens).unsqueeze(0)
    activations: dict[str, Tensor] = {}
    outputs: dict[str, Tensor] = {}
    output_gradients: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    was_training = model.training
    model.eval()
    try:
        for name, module in modules.items():

            def pre_hook(
                module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                name: str = name,
            ) -> None:
                del module
                if len(inputs) != 1:
                    raise RuntimeError(f"unexpected linear inputs for {name}")
                activations[name] = inputs[0]

            def forward_hook(
                module: nn.Module,
                inputs: tuple[Tensor, ...],
                output: Tensor,
                *,
                name: str = name,
            ) -> None:
                del module, inputs
                outputs[name] = output
                if output.requires_grad:
                    output.register_hook(
                        lambda gradient, name=name: output_gradients.__setitem__(name, gradient)
                    )

            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(forward_hook))
        extra_values: dict[str, Tensor] = {}

        def capture_extra(tensor: Tensor, hook: object) -> None:
            name = getattr(hook, "name", None)
            if not isinstance(name, str):
                raise RuntimeError("extra gradient hook has no name")
            extra_values[name] = tensor

        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        with model.hooks(fwd_hooks=[(name, capture_extra) for name in extra_hooks]):
            logits = model(tokens)
            scalar = objective(logits, extra_values)
            if scalar.ndim != 0 or not scalar.requires_grad:
                raise ValueError("influence objective must be a differentiable scalar")
            torch.autograd.grad(
                scalar,
                parameters,
                retain_graph=False,
                allow_unused=True,
            )
        layers: dict[str, RawLayerGradient] = {}
        for name in modules:
            if name not in activations or name not in outputs:
                raise RuntimeError(f"tracked module was not called: {name}")
            activation = activations[name]
            output = outputs[name]
            gradient = output_gradients.get(name, torch.zeros_like(output))
            if activation.shape[0] != 1 or gradient.shape != output.shape:
                raise RuntimeError(f"unexpected gradient shapes for {name}")
            layers[name] = RawLayerGradient(
                activation=activation[0].detach().cpu().numpy().astype(np.float64),
                output_gradient=gradient[0].detach().cpu().numpy().astype(np.float64),
            )
        return RawSampleGradient(sample.key, layers)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


def capture_loss_gradient(
    model: ArithmeticTransformer,
    sample: ArithmeticSample,
    target: int,
    modules: Mapping[str, nn.Linear],
) -> RawSampleGradient:
    if not 0 <= target < model.config.vocab_size:
        raise ValueError("loss target is outside the vocabulary")

    def objective(logits: Tensor, extra: Mapping[str, Tensor]) -> Tensor:
        del extra
        return torch.nn.functional.cross_entropy(
            logits[:, -1, :], torch.tensor([target], device=logits.device)
        )

    return _capture_objective_gradient(model, sample, modules, objective)


def capture_logit_gradient(
    model: ArithmeticTransformer,
    sample: ArithmeticSample,
    target_token: int,
    modules: Mapping[str, nn.Linear],
) -> RawSampleGradient:
    if not 0 <= target_token < model.config.vocab_size:
        raise ValueError("query target is outside the vocabulary")

    def objective(logits: Tensor, extra: Mapping[str, Tensor]) -> Tensor:
        del extra
        return logits[0, -1, target_token]

    return _capture_objective_gradient(model, sample, modules, objective)


def capture_feature_gradient(
    model: ArithmeticTransformer,
    replacement: SparseCrossLayerTranscoder,
    sample: ArithmeticSample,
    *,
    layer: int,
    feature: int,
    modules: Mapping[str, nn.Linear],
) -> RawSampleGradient:
    if layer not in (0, 1) or not 0 <= feature < replacement.features:
        raise ValueError("feature query is outside the replacement inventory")
    hook_name = f"blocks.{layer}.mlp.hook_mlp_input"

    def objective(logits: Tensor, extra: Mapping[str, Tensor]) -> Tensor:
        del logits
        code = replacement.encode(layer, extra[hook_name][0, -1, :])
        return code[feature]

    return _capture_objective_gradient(model, sample, modules, objective, extra_hooks=(hook_name,))


def fit_ekfac_from_gradients(
    samples: tuple[RawSampleGradient, ...],
    modules: Mapping[str, nn.Linear],
    *,
    damping: float = 1e-3,
) -> EKFACInfluenceModel:
    """Fit corrected KFE diagonal statistics from exact raw sample gradients."""

    if not samples or not modules or damping <= 0 or not np.isfinite(damping):
        raise ValueError("invalid EK-FAC fitting configuration")
    if len({sample.sample_key for sample in samples}) != len(samples):
        raise ValueError("EK-FAC training sample keys must be unique")
    expected_names = tuple(modules)
    for sample in samples:
        if set(sample.layers) != set(expected_names):
            raise ValueError("all samples must cover the same tracked layer inventory")
    layers: list[EKFACLayer] = []
    for name in expected_names:
        raw_layers = [sample.layers[name] for sample in samples]
        input_width = raw_layers[0].activation.shape[1]
        output_width = raw_layers[0].output_gradient.shape[1]
        if any(
            raw.activation.shape[1] != input_width or raw.output_gradient.shape[1] != output_width
            for raw in raw_layers
        ):
            raise ValueError(f"inconsistent raw dimensions for {name}")
        activations = np.concatenate([raw.activation for raw in raw_layers], axis=0)
        output_gradients = np.concatenate([raw.output_gradient for raw in raw_layers], axis=0)
        activation_covariance = activations.T @ activations / activations.shape[0]
        output_covariance = output_gradients.T @ output_gradients / output_gradients.shape[0]
        _, activation_basis = np.linalg.eigh(activation_covariance)
        _, output_basis = np.linalg.eigh(output_covariance)
        layer = EKFACLayer(
            name=name,
            activation_basis=activation_basis,
            output_basis=output_basis,
            corrected_second_moments=np.empty((output_width, input_width), dtype=np.float64),
            training_projections=np.empty((0, output_width, input_width), dtype=np.float64),
        )
        projections = np.stack([layer.project(raw) for raw in raw_layers])
        corrected = np.mean(projections**2, axis=0)
        layers.append(
            EKFACLayer(
                name=name,
                activation_basis=activation_basis,
                output_basis=output_basis,
                corrected_second_moments=corrected,
                training_projections=projections,
            )
        )
    return EKFACInfluenceModel(
        sample_keys=tuple(sample.sample_key for sample in samples),
        layers=tuple(layers),
        damping=float(damping),
        modules=modules,
    )


def fit_arithmetic_ekfac(
    model: ArithmeticTransformer,
    samples: tuple[ArithmeticSample, ...],
    targets: NDArray[np.int64],
    *,
    damping: float = 1e-3,
) -> EKFACInfluenceModel:
    """Compute exact training-loss gradients and fit the scoped EK-FAC model."""

    if len(samples) == 0 or targets.shape != (len(samples),):
        raise ValueError("training samples and targets must be nonempty and aligned")
    modules = tracked_arithmetic_linears(model)
    raw = tuple(
        capture_loss_gradient(model, sample, int(target), modules)
        for sample, target in zip(samples, targets, strict=True)
    )
    return fit_ekfac_from_gradients(raw, modules, damping=damping)
