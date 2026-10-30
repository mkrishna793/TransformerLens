"""Independent Kronfluence and dattri-LLM EK-FAC comparisons.

These adapters compare algorithms under explicit signs/normalization. They do
not turn algorithmic agreement into counterfactual or detection validity.
"""

from __future__ import annotations

import copy
import importlib
import tempfile
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP, CIFARBatch
from research.cmae.curvature_influence import RawSampleGradient

FloatArray = NDArray[np.float64]


def dattri_binary_ekfac_influence(
    raw_samples: tuple[RawSampleGradient, ...],
    query: RawSampleGradient,
    *,
    damping: float = 1e-3,
) -> FloatArray:
    """Execute dattri-LLM's independent empirical EK-FAC linear kernels."""

    if not raw_samples or damping <= 0:
        raise ValueError("dattri comparison requires raw samples and positive damping")
    try:
        ops = importlib.import_module("dattri_llm.gradient.ops.kronecker")
        materialize = importlib.import_module("dattri_llm.gradient.ops.materialize")
    except ImportError as error:
        raise RuntimeError("dattri-LLM is required for its declared external comparison") from error
    count = len(raw_samples)
    scores = np.zeros(count, dtype=np.float64)
    for name in query.layers:
        activations = torch.from_numpy(
            np.stack([sample.layers[name].activation for sample in raw_samples])
        )
        output_gradients = torch.from_numpy(
            np.stack([sample.layers[name].output_gradient for sample in raw_samples])
        )
        kfac_factors = getattr(ops, "kfac_factors")
        activation_covariance, output_covariance = kfac_factors(
            activations,
            output_gradients,
            "nn.Linear",
            module_kwargs=None,
            include_bias=False,
        )
        _, activation_basis, _, output_basis = getattr(ops, "kfac_eigh")(
            activation_covariance, output_covariance
        )
        projected = getattr(ops, "ekfac_materialize_factors")(
            activations,
            output_gradients,
            "nn.Linear",
            activation_basis,
            output_basis,
            module_kwargs=None,
            include_bias=False,
        )
        corrected = torch.mean(projected**2, dim=0)
        query_projected = getattr(ops, "ekfac_materialize_factors")(
            torch.from_numpy(query.layers[name].activation),
            torch.from_numpy(query.layers[name].output_gradient),
            "nn.Linear",
            activation_basis,
            output_basis,
            module_kwargs=None,
            include_bias=False,
        ).sum(dim=0, keepdim=True)
        preconditioned = getattr(ops, "ekfac_precondition")(
            query_projected,
            activation_basis,
            output_basis,
            corrected + damping,
            "nn.Linear",
        )
        raw_materialized = getattr(materialize, "materialize_factors")(
            activations,
            output_gradients,
            "nn.Linear",
            module_kwargs=None,
            include_bias=False,
        )
        layer_scores = (
            -(raw_materialized.to(dtype=preconditioned.dtype) @ preconditioned[0]) / count
        )
        scores += layer_scores.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite dattri-LLM EK-FAC scores")
    return scores


def _kronfluence_task(task_base: Any) -> Any:
    """Construct a Task without a static optional-dependency import."""

    def compute_train_loss(
        self: object,
        batch: tuple[Tensor, Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> Tensor:
        del self
        inputs, labels = batch
        logits = model(inputs)
        if sample:
            labels = torch.bernoulli(torch.sigmoid(logits.detach()))
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")

    def compute_measurement(self: object, batch: tuple[Tensor, Tensor], model: nn.Module) -> Tensor:
        del self
        inputs, _ = batch
        # Pairwise rows are subsequently averaged and negated by the adapter.
        return model(inputs).sum()

    return type(
        "CMAEBinaryKronfluenceTask",
        (task_base,),
        {
            "compute_train_loss": compute_train_loss,
            "compute_measurement": compute_measurement,
        },
    )()


def kronfluence_binary_ekfac_influence(
    model: CIFARBadNetsMLP,
    batch: CIFARBatch,
    query_inputs: Tensor,
    *,
    damping: float = 1e-3,
) -> FloatArray:
    """Execute Kronfluence's own EK-FAC fitting and pairwise-score workflow."""

    if damping <= 0:
        raise ValueError("Kronfluence damping must be positive")
    try:
        analyzer_module = importlib.import_module("kronfluence.analyzer")
        arguments_module = importlib.import_module("kronfluence.arguments")
        task_module = importlib.import_module("kronfluence.task")
    except ImportError as error:
        raise RuntimeError(
            "Kronfluence is required for its declared external comparison"
        ) from error
    working_model = copy.deepcopy(model).float()
    task = _kronfluence_task(getattr(task_module, "Task"))
    prepare_model = getattr(analyzer_module, "prepare_model")
    working_model = cast(CIFARBadNetsMLP, prepare_model(working_model, task))
    train_dataset = torch.utils.data.TensorDataset(batch.inputs.float(), batch.labels.float())
    query_dataset = torch.utils.data.TensorDataset(
        query_inputs.float(), torch.zeros(query_inputs.shape[0], dtype=torch.float32)
    )
    with tempfile.TemporaryDirectory(prefix="cmae-kronfluence-") as directory:
        analyzer_class = getattr(analyzer_module, "Analyzer")
        analyzer = analyzer_class(
            analysis_name="cmae_binary",
            model=working_model,
            task=task,
            cpu=True,
            disable_tqdm=True,
            output_dir=directory,
        )
        factor_arguments_class = getattr(arguments_module, "FactorArguments")
        factor_arguments = factor_arguments_class(
            strategy="ekfac",
            use_empirical_fisher=True,
            covariance_max_examples=None,
            lambda_max_examples=None,
            activation_covariance_dtype=torch.float64,
            gradient_covariance_dtype=torch.float64,
            eigendecomposition_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            lambda_dtype=torch.float64,
        )
        analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=train_dataset,
            per_device_batch_size=128,
            factor_args=factor_arguments,
            overwrite_output_dir=True,
        )
        score_arguments_class = getattr(arguments_module, "ScoreArguments")
        score_arguments = score_arguments_class(damping_factor=damping)
        analyzer.compute_pairwise_scores(
            scores_name="ekfac",
            factors_name="ekfac",
            query_dataset=query_dataset,
            train_dataset=train_dataset,
            per_device_query_batch_size=query_inputs.shape[0],
            per_device_train_batch_size=128,
            score_args=score_arguments,
            overwrite_output_dir=True,
        )
        loaded = analyzer.load_pairwise_scores("ekfac")["all_modules"]
        raw_scores = loaded.detach().cpu().numpy().astype(np.float64)
    if raw_scores.shape != (query_inputs.shape[0], batch.labels.shape[0]):
        raise RuntimeError(f"unexpected Kronfluence score shape {raw_scores.shape}")
    # Kronfluence computes +<g_query,H^-1 g_train_raw>; CMAE reports
    # full-minus-reduced influence with g_train_raw/N and the opposite sign.
    scores = -raw_scores.mean(axis=0) / batch.labels.shape[0]
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("nonfinite Kronfluence EK-FAC scores")
    return scores


def score_agreement(reference: FloatArray, other: FloatArray) -> dict[str, float]:
    """Numerical algorithm agreement, kept separate from retraining validity."""

    if (
        reference.ndim != 1
        or other.shape != reference.shape
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(other))
    ):
        raise ValueError("external score agreement requires aligned finite vectors")
    scale = float(np.linalg.norm(reference))
    return {
        "relative_l2_difference": float(np.linalg.norm(reference - other) / max(scale, 1e-30)),
        "maximum_absolute_difference": float(np.max(np.abs(reference - other))),
        "reference_l2_norm": scale,
        "other_l2_norm": float(np.linalg.norm(other)),
    }
