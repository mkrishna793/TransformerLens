"""Training-only logit, averaged-Jacobian, and cluster-conditioned latent lenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor

from research.cmae.arithmetic_data import VOCAB_SIZE, ArithmeticDataset, ArithmeticSplit
from research.cmae.arithmetic_transformer import ArithmeticTransformer
from research.cmae.generalization import readout_rank_metrics

ResidualSite = Literal["early", "final"]
LensMethod = Literal["logit", "jacobian", "jpp"]
LENS_METHODS: tuple[LensMethod, ...] = ("logit", "jacobian", "jpp")


@dataclass(frozen=True)
class KMeansResult:
    centers: Tensor
    assignments: Tensor
    iterations: int
    inertia: float


def deterministic_kmeans(
    activations: Tensor,
    clusters: int,
    *,
    max_iterations: int = 25,
) -> KMeansResult:
    """Fit deterministic farthest-point-initialized Lloyd k-means on CPU tensors."""

    if activations.ndim != 2 or activations.shape[0] == 0 or activations.shape[1] == 0:
        raise ValueError("k-means activations must have shape [nonempty samples, features]")
    if clusters <= 0 or clusters > activations.shape[0] or max_iterations <= 0:
        raise ValueError("invalid deterministic k-means configuration")
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    count = values.shape[0]
    mean = values.mean(dim=0)
    first = int(torch.argmin((values - mean).square().sum(dim=1)).item())
    center_indices = [first]
    minimum_distances = (values - values[first]).square().sum(dim=1)
    for _ in range(1, clusters):
        next_index = int(torch.argmax(minimum_distances).item())
        center_indices.append(next_index)
        distances = (values - values[next_index]).square().sum(dim=1)
        minimum_distances = torch.minimum(minimum_distances, distances)
    centers = values[center_indices].clone()
    assignments = torch.zeros(count, dtype=torch.int64)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        distances = torch.cdist(values, centers).square()
        new_assignments = torch.argmin(distances, dim=1)
        new_centers = centers.clone()
        for cluster in range(clusters):
            mask = new_assignments == cluster
            if bool(mask.any().item()):
                new_centers[cluster] = values[mask].mean(dim=0)
            else:
                nearest = torch.gather(distances, 1, new_assignments[:, None]).squeeze(1)
                replacement = int(torch.argmax(nearest).item())
                new_centers[cluster] = values[replacement]
                new_assignments[replacement] = cluster
        converged = iteration > 1 and torch.equal(new_assignments, assignments)
        assignments = new_assignments
        centers = new_centers
        iterations = iteration
        if converged:
            break
    final_distances = torch.cdist(values, centers).square()
    assignments = torch.argmin(final_distances, dim=1)
    inertia = float(torch.gather(final_distances, 1, assignments[:, None]).sum().cpu().item())
    return KMeansResult(
        centers=centers.to(dtype=activations.dtype),
        assignments=assignments,
        iterations=iterations,
        inertia=inertia,
    )


@dataclass(frozen=True)
class SiteLens:
    site: ResidualSite
    jacobian: Tensor
    cluster_centers: Tensor
    cluster_jacobians: Tensor
    cluster_counts: Tensor
    mixture_weights: Tensor
    fit_prompts: int
    fit_source_positions: int
    kmeans_iterations: int
    kmeans_inertia: float
    weight_objective_history: tuple[float, ...]

    @property
    def clusters(self) -> int:
        return int(self.cluster_jacobians.shape[0])

    @property
    def jpp_jacobian(self) -> Tensor:
        if self.site == "final":
            return self.jacobian
        return torch.einsum("k,koi->oi", self.mixture_weights, self.cluster_jacobians)

    def transport(self, activations: Tensor, method: LensMethod) -> Tensor:
        if activations.shape[-1] != self.jacobian.shape[1]:
            raise ValueError("lens activation width does not match its Jacobian")
        if method == "logit" or self.site == "final":
            return activations
        if method == "jacobian":
            return activations @ self.jacobian.transpose(0, 1)
        if method == "jpp":
            return activations @ self.jpp_jacobian.transpose(0, 1)
        raise ValueError(f"unknown lens method: {method}")

    def diagnostics(self) -> dict[str, object]:
        history = self.weight_objective_history
        return {
            "site": self.site,
            "fit_split": "train",
            "fit_prompts": self.fit_prompts,
            "fit_source_positions": self.fit_source_positions,
            "clusters": self.clusters,
            "cluster_source_counts": self.cluster_counts.to(torch.int64).cpu().tolist(),
            "mixture_weights": self.mixture_weights.to(torch.float64).cpu().tolist(),
            "mixture_weight_sum": float(self.mixture_weights.sum().cpu().item()),
            "kmeans_iterations": self.kmeans_iterations,
            "kmeans_inertia": self.kmeans_inertia,
            "weight_objective_initial": history[0] if history else None,
            "weight_objective_final": history[-1] if history else None,
            "weight_optimization_steps": max(0, len(history) - 1),
        }


@dataclass(frozen=True)
class LatentLensSuite:
    early: SiteLens
    final: SiteLens
    latent_log_counts: dict[str, Tensor]

    def site(self, site: ResidualSite) -> SiteLens:
        return self.early if site == "early" else self.final


def _collect_residual_sites(
    model: ArithmeticTransformer,
    tokens: Tensor,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    final_batches: list[Tensor] = []
    early_batches: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            final, early = model.residual_sites(tokens[start : start + batch_size])
            final_batches.append(final.detach().cpu())
            early_batches.append(early.detach().cpu())
    return torch.cat(final_batches), torch.cat(early_batches)


def _learned_unit_sum_weights(
    model: ArithmeticTransformer,
    answer_activations: Tensor,
    cluster_jacobians: Tensor,
    cluster_counts: Tensor,
    targets: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, tuple[float, ...]]:
    if steps < 0 or learning_rate <= 0:
        raise ValueError("invalid J++ weight optimization configuration")
    clusters = cluster_jacobians.shape[0]
    initial = cluster_counts.to(torch.float64) / cluster_counts.sum().to(torch.float64)
    initial = initial.to(dtype=answer_activations.dtype)
    if steps == 0:
        return initial, ()
    raw_weights = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw_weights], lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    per_cluster = torch.einsum(
        "ni,koi->nko", answer_activations.detach(), cluster_jacobians.detach()
    )
    readout_parameters = list(model.final_norm.parameters()) + list(model.unembed.parameters())
    requires_grad = [parameter.requires_grad for parameter in readout_parameters]
    history: list[float] = []
    try:
        for parameter in readout_parameters:
            parameter.requires_grad_(False)
        for step in range(steps + 1):
            weights = raw_weights - raw_weights.mean() + 1.0 / clusters
            transported = torch.einsum("k,nko->no", weights, per_cluster)
            loss = loss_function(model.readout_vector(transported), targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite J++ mixture-weight objective")
            history.append(float(loss.detach().cpu().item()))
            if step < steps:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    finally:
        for parameter, required in zip(readout_parameters, requires_grad, strict=True):
            parameter.requires_grad_(required)
    with torch.no_grad():
        weights = raw_weights - raw_weights.mean() + 1.0 / clusters
    return weights.detach(), tuple(history)


def _fit_site_lens(
    model: ArithmeticTransformer,
    training_tokens: Tensor,
    training_targets: Tensor,
    site: ResidualSite,
    *,
    clusters: int,
    batch_size: int,
    kmeans_iterations: int,
    weight_steps: int,
    weight_learning_rate: float,
) -> SiteLens:
    final, early = _collect_residual_sites(model, training_tokens, batch_size=batch_size)
    activations = early if site == "early" else final
    flat = activations.reshape(-1, activations.shape[-1])
    kmeans = deterministic_kmeans(flat, clusters, max_iterations=kmeans_iterations)
    prompt_count, positions, width = activations.shape
    assignments = kmeans.assignments.reshape(prompt_count, positions)
    cluster_counts = torch.bincount(kmeans.assignments, minlength=clusters)
    dtype = activations.dtype
    if site == "final":
        jacobian = torch.eye(width, dtype=dtype)
        cluster_jacobians = jacobian.expand(clusters, width, width).clone()
        weights = cluster_counts.to(dtype) / cluster_counts.sum().to(dtype)
        history: tuple[float, ...] = ()
    else:
        global_sum = torch.zeros(width, width, dtype=torch.float64)
        cluster_sums = torch.zeros(clusters, width, width, dtype=torch.float64)
        was_training = model.training
        model.eval()
        try:
            for start in range(0, prompt_count, batch_size):
                stop = min(start + batch_size, prompt_count)
                batch = training_tokens[start:stop]
                final_with_graph, early_with_graph = model.residual_sites(batch)
                batch_assignments = assignments[start:stop]
                for output_index in range(width):
                    cotangent = torch.zeros_like(final_with_graph)
                    cotangent[:, :, output_index] = 1.0
                    gradient = torch.autograd.grad(
                        final_with_graph,
                        early_with_graph,
                        grad_outputs=cotangent,
                        retain_graph=output_index + 1 < width,
                        create_graph=False,
                    )[0]
                    global_sum[output_index] += gradient.detach().to(torch.float64).sum(dim=(0, 1))
                    for cluster in range(clusters):
                        mask = (batch_assignments == cluster).to(gradient.dtype)
                        cluster_sums[cluster, output_index] += (
                            (gradient.detach() * mask[:, :, None]).to(torch.float64).sum(dim=(0, 1))
                        )
        finally:
            model.train(was_training)
        jacobian = (global_sum / (prompt_count * positions)).to(dtype)
        cluster_jacobians = torch.empty(clusters, width, width, dtype=dtype)
        for cluster in range(clusters):
            count = int(cluster_counts[cluster].item())
            if count == 0:
                cluster_jacobians[cluster] = jacobian
            else:
                cluster_jacobians[cluster] = (cluster_sums[cluster] / count).to(dtype)
        answer_activations = activations[:, -1, :].to(device=training_tokens.device, dtype=dtype)
        weights, history = _learned_unit_sum_weights(
            model,
            answer_activations,
            cluster_jacobians,
            cluster_counts,
            training_targets,
            steps=weight_steps,
            learning_rate=weight_learning_rate,
        )
    return SiteLens(
        site=site,
        jacobian=jacobian.detach().cpu(),
        cluster_centers=kmeans.centers.detach().cpu(),
        cluster_jacobians=cluster_jacobians.detach().cpu(),
        cluster_counts=cluster_counts.cpu(),
        mixture_weights=weights.detach().cpu(),
        fit_prompts=prompt_count,
        fit_source_positions=prompt_count * positions,
        kmeans_iterations=kmeans.iterations,
        kmeans_inertia=kmeans.inertia,
        weight_objective_history=history,
    )


def fit_latent_lenses(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    *,
    clusters: int = 8,
    batch_size: int = 64,
    kmeans_iterations: int = 25,
    weight_steps: int = 100,
    weight_learning_rate: float = 0.03,
) -> LatentLensSuite:
    """Fit every lens and the frequency baseline on the training split only."""

    training_tokens_np, training_targets_np, training_intermediates_np = dataset.arrays("train")
    training_tokens = torch.from_numpy(training_tokens_np)
    training_targets = torch.from_numpy(training_targets_np)
    was_training = model.training
    model.eval()
    try:
        early = _fit_site_lens(
            model,
            training_tokens,
            training_targets,
            "early",
            clusters=clusters,
            batch_size=batch_size,
            kmeans_iterations=kmeans_iterations,
            weight_steps=weight_steps,
            weight_learning_rate=weight_learning_rate,
        )
        final = _fit_site_lens(
            model,
            training_tokens,
            training_targets,
            "final",
            clusters=clusters,
            batch_size=batch_size,
            kmeans_iterations=kmeans_iterations,
            weight_steps=weight_steps,
            weight_learning_rate=weight_learning_rate,
        )
    finally:
        model.train(was_training)
    latent_log_counts: dict[str, Tensor] = {}
    for name, column in (("u", 0), ("v", 1)):
        counts = torch.bincount(
            torch.from_numpy(training_intermediates_np[:, column]), minlength=VOCAB_SIZE
        )
        latent_log_counts[name] = counts.to(torch.float32).log1p()
    return LatentLensSuite(early=early, final=final, latent_log_counts=latent_log_counts)


def evaluate_latent_lenses(
    model: ArithmeticTransformer,
    suite: LatentLensSuite,
    dataset: ArithmeticDataset,
    split: ArithmeticSplit,
    *,
    batch_size: int = 128,
) -> dict[str, object]:
    """Evaluate full-vocabulary readouts and explicit approximation residuals."""

    tokens_np, _, intermediates_np = dataset.arrays(split)
    tokens = torch.from_numpy(tokens_np)
    intermediates = torch.from_numpy(intermediates_np)
    metrics: dict[str, dict[str, float]] = {}
    approximation: dict[str, dict[str, float]] = {}
    prior: dict[str, dict[str, float]] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                stop = min(start + batch_size, len(tokens))
                final, early = model.residual_sites(tokens[start:stop])
                actual_logits = model.readout_vector(final)
                for site_name in cast(tuple[ResidualSite, ...], ("early", "final")):
                    site_lens = suite.site(site_name)
                    activations = early if site_name == "early" else final
                    for method in LENS_METHODS:
                        transported = site_lens.transport(activations, method)
                        lens_logits = model.readout_vector(transported)
                        answer_logits = lens_logits[:, -1, :].cpu()
                        for latent, column in (("u", 0), ("v", 1)):
                            key = f"{site_name}_{latent}_{method}"
                            batch_metrics = readout_rank_metrics(
                                answer_logits, intermediates[start:stop, column]
                            )
                            previous = metrics.setdefault(key, {})
                            weight = stop - start
                            total = len(tokens)
                            for metric_name, value in batch_metrics.items():
                                previous[metric_name] = (
                                    previous.get(metric_name, 0.0) + (weight / total) * value
                                )
                        residual = final - transported
                        logit_residual = actual_logits - lens_logits
                        key = f"{site_name}_{method}"
                        batch_values = {
                            "approximation_error_mass": float(residual.square().sum().cpu().item()),
                            "final_residual_energy": float(final.square().sum().cpu().item()),
                            "readout_error_mass": float(logit_residual.square().sum().cpu().item()),
                            "actual_readout_energy": float(
                                actual_logits.square().sum().cpu().item()
                            ),
                            "max_abs_conservation_error": float(
                                (final - transported - residual).abs().max().cpu().item()
                            ),
                        }
                        previous_approximation = approximation.setdefault(key, {})
                        for metric_name, value in batch_values.items():
                            if metric_name == "max_abs_conservation_error":
                                previous_approximation[metric_name] = max(
                                    previous_approximation.get(metric_name, 0.0), value
                                )
                            else:
                                previous_approximation[metric_name] = (
                                    previous_approximation.get(metric_name, 0.0) + value
                                )
                for latent, column in (("u", 0), ("v", 1)):
                    prior_logits = suite.latent_log_counts[latent][None, :].expand(
                        stop - start, VOCAB_SIZE
                    )
                    batch_prior = readout_rank_metrics(
                        prior_logits, intermediates[start:stop, column]
                    )
                    previous_prior = prior.setdefault(latent, {})
                    weight = stop - start
                    total = len(tokens)
                    for metric_name, value in batch_prior.items():
                        previous_prior[metric_name] = (
                            previous_prior.get(metric_name, 0.0) + (weight / total) * value
                        )
    finally:
        model.train(was_training)
    for values in approximation.values():
        error_mass = values["approximation_error_mass"]
        residual_energy = values["final_residual_energy"]
        readout_error = values["readout_error_mass"]
        readout_energy = values["actual_readout_energy"]
        values["relative_approximation_error_mass"] = error_mass / max(residual_energy, 1e-12)
        values["relative_readout_error_mass"] = readout_error / max(readout_energy, 1e-12)
    return {
        "split": split,
        "samples": len(tokens),
        "metrics": metrics,
        "approximation": approximation,
        "training_prior": prior,
    }
