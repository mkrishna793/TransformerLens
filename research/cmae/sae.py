"""Trained sparse autoencoders with explicit reconstruction and readout error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

FloatArray = NDArray[np.float64]


class ReLUSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("SAE dimensions must be positive")
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = torch.relu(self.encoder(value))
        return self.decoder(code), code


@dataclass(frozen=True)
class SAEMetrics:
    reconstruction_mse: float
    relative_reconstruction_error: float
    explained_variance: float
    mean_l0: float
    active_feature_fraction: float
    readout_relative_error: float
    readout_max_abs_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "reconstruction_mse": self.reconstruction_mse,
            "relative_reconstruction_error": self.relative_reconstruction_error,
            "explained_variance": self.explained_variance,
            "mean_l0": self.mean_l0,
            "active_feature_fraction": self.active_feature_fraction,
            "readout_relative_error": self.readout_relative_error,
            "readout_max_abs_error": self.readout_max_abs_error,
        }


def fit_relu_sae(
    activations: FloatArray,
    *,
    latent_dim: int = 32,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l1_coefficient: float = 1e-3,
    seed: int = 0,
) -> tuple[ReLUSAE, list[float]]:
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("SAE activations must be a finite matrix with at least two rows")
    if epochs <= 0 or learning_rate <= 0 or l1_coefficient < 0:
        raise ValueError("invalid SAE optimization setting")
    torch.manual_seed(seed)
    model = ReLUSAE(values.shape[1], latent_dim)
    features = torch.from_numpy(values).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, code = model(features)
        reconstruction_loss = (reconstruction - features).square().mean()
        sparsity = code.abs().mean()
        loss = reconstruction_loss + l1_coefficient * sparsity
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("nonfinite SAE training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    model.eval()
    return model, losses


def evaluate_sae_readout(
    model: ReLUSAE,
    readout: nn.Linear,
    activations: FloatArray,
    *,
    target_class: int = 1,
) -> SAEMetrics:
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("SAE evaluation activations must be a finite nonempty matrix")
    features = torch.from_numpy(values).to(
        device=model.encoder.weight.device, dtype=model.encoder.weight.dtype
    )
    with torch.no_grad():
        reconstruction, code = model(features)
        original_output = readout(features)[:, target_class]
        reconstructed_output = readout(reconstruction)[:, target_class]
        error = reconstruction - features
        downstream = reconstructed_output - original_output
        mse = float(error.square().mean().cpu().item())
        relative = float((error.norm() / features.norm().clamp_min(1e-12)).cpu().item())
        centered = features - features.mean(dim=0, keepdim=True)
        denominator = float(centered.square().sum().cpu().item())
        explained_variance = 1.0 - float(error.square().sum().cpu().item()) / max(
            denominator, 1e-12
        )
        mean_l0 = float((code.abs() > 1e-8).float().sum(dim=1).mean().cpu().item())
        active_fraction = float((code.abs() > 1e-8).float().mean().cpu().item())
        readout_relative = float(
            (downstream.norm() / original_output.norm().clamp_min(1e-12)).cpu().item()
        )
        readout_max = float(downstream.abs().max().cpu().item())
    return SAEMetrics(
        reconstruction_mse=mse,
        relative_reconstruction_error=relative,
        explained_variance=explained_variance,
        mean_l0=mean_l0,
        active_feature_fraction=active_fraction,
        readout_relative_error=readout_relative,
        readout_max_abs_error=readout_max,
    )


def sparse_feature_inventory(
    model: ReLUSAE, activations: FloatArray
) -> list[dict[str, float | int | str]]:
    """Numerical description only; no unsupported natural-language semantics."""

    values = torch.from_numpy(np.asarray(activations, dtype=np.float64)).to(
        device=model.encoder.weight.device, dtype=model.encoder.weight.dtype
    )
    with torch.no_grad():
        _, code = model(values)
    codes = code.cpu().numpy().astype(np.float64)
    inventory: list[dict[str, float | int | str]] = []
    for index in range(codes.shape[1]):
        active = np.abs(codes[:, index]) > 1e-8
        inventory.append(
            {
                "feature": index,
                "activation_rate": float(active.mean()),
                "mean_activation": float(codes[:, index].mean()),
                "max_activation": float(codes[:, index].max(initial=0.0)),
                "semantic_status": "unlabeled_numeric_feature_requires_separate_NLA_or_human_validation",
            }
        )
    return inventory
