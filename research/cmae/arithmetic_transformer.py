"""A small locally trained causal Transformer with TransformerLens hook points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from research.cmae.arithmetic_data import (
    PROMPT_LENGTH,
    VOCAB_SIZE,
    ArithmeticDataset,
    ArithmeticSplit,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule


@dataclass(frozen=True)
class ArithmeticModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_length: int = PROMPT_LENGTH
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    d_mlp: int = 64

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0 or self.d_mlp <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size != VOCAB_SIZE or self.max_length != PROMPT_LENGTH:
            raise ValueError("model and arithmetic vocabulary contracts disagree")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ArithmeticModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.d_model, config.d_model, bias=True)
        self.key = nn.Linear(config.d_model, config.d_model, bias=True)
        self.value = nn.Linear(config.d_model, config.d_model, bias=True)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_scores = HookPoint()
        mask = torch.triu(
            torch.ones(config.max_length, config.max_length, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        batch, positions, _ = value.shape
        heads = self.config.n_heads
        width = self.config.d_head

        def split_heads(projection: Tensor) -> Tensor:
            return projection.reshape(batch, positions, heads, width).transpose(1, 2)

        query = self.hook_q(split_heads(self.query(value)))
        key = self.hook_k(split_heads(self.key(value)))
        val = self.hook_v(split_heads(self.value(value)))
        scores = query @ key.transpose(-2, -1) / math.sqrt(width)
        causal_mask = cast(Tensor, self.causal_mask)
        scores = scores.masked_fill(causal_mask[:positions, :positions], float("-inf"))
        scores = self.hook_scores(scores)
        pattern = self.hook_pattern(torch.softmax(scores, dim=-1))
        attended = pattern @ val
        z = self.hook_z(attended.transpose(1, 2))
        output_weight = self.output.weight.reshape(self.config.d_model, heads, width)
        head_output = torch.einsum("bthd,mhd->bthm", z, output_weight)
        return head_output, head_output.sum(dim=2)


class ArithmeticMLP(nn.Module):
    def __init__(self, config: ArithmeticModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.d_model, config.d_mlp, bias=True)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.d_mlp, config.d_model, bias=False)
        self.hook_mlp_input = HookPoint()
        self.hook_mlp_pre = HookPoint()
        self.hook_mlp_post = HookPoint()

    def forward(self, value: Tensor) -> Tensor:
        value = self.hook_mlp_input(value)
        pre = self.hook_mlp_pre(self.input(value))
        post = self.hook_mlp_post(self.activation(pre))
        return self.output(post)


class ArithmeticBlock(nn.Module):
    def __init__(self, config: ArithmeticModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = ArithmeticMLP(config)
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, residual: Tensor) -> Tensor:
        residual = self.hook_resid_pre(residual)
        head_output, attention_output = self.attention(self.attention_norm(residual))
        del attention_output
        residual = self.hook_resid_mid(residual + self.hook_attn_out(head_output).sum(dim=2))
        mlp_output = self.hook_mlp_out(self.mlp(self.mlp_norm(residual)))
        return self.hook_resid_post(residual + mlp_output)


class ArithmeticTransformer(HookedRootModule):
    """Custom Transformer using the retained v4 arbitrary-module hook system."""

    def __init__(self, config: ArithmeticModelConfig = ArithmeticModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.blocks = nn.ModuleList([ArithmeticBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.hook_token_embed = HookPoint()
        self.hook_position_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.setup()

    def residual_sites(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] != self.config.max_length:
            raise ValueError("tokens must have shape [batch, arithmetic_prompt_length]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        token = self.hook_token_embed(self.token_embedding(tokens))
        position = self.hook_position_embed(self.position_embedding(positions))[None, :, :]
        residual = token + position
        early: Tensor | None = None
        for index, block in enumerate(self.blocks):
            residual = block(residual)
            if index == 0:
                early = residual
        if early is None:
            raise RuntimeError("Transformer has no residual blocks")
        return self.hook_resid_final(residual), early

    def residual(self, tokens: Tensor) -> Tensor:
        return self.residual_sites(tokens)[0]

    def readout_vector(self, residual: Tensor) -> Tensor:
        return self.unembed(self.final_norm(residual))

    def forward_with_latents(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        final, early = self.residual_sites(tokens)
        logits = self.readout_vector(final)
        u_logits = self.readout_vector(early[:, -1, :])
        v_logits = logits[:, -1, :]
        return logits, u_logits, v_logits

    def forward(self, tokens: Tensor, target_token: int | None = None) -> Tensor:
        logits = self.readout_vector(self.residual(tokens))
        if target_token is not None:
            if tokens.shape[0] != 1:
                raise ValueError("scalar target tracing requires one prompt")
            if not 0 <= target_token < self.config.vocab_size:
                raise ValueError("target token is outside the vocabulary")
            return logits[0, -1, target_token]
        return logits


def train_arithmetic_transformer(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    *,
    epochs: int = 250,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    seed: int = 101,
) -> list[float]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training epochs, batch size, and learning rate must be positive")
    train_tokens, train_targets, _ = dataset.arrays("train")
    tokens = torch.from_numpy(train_tokens)
    targets = torch.from_numpy(train_targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(tokens))
        weighted_loss = 0.0
        for start in range(0, len(tokens), batch_size):
            indices = torch.from_numpy(permutation[start : start + batch_size].copy())
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens[indices])[:, -1, :]
            loss = loss_function(logits, targets[indices])
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite arithmetic Transformer training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            weighted_loss += float(loss.detach().cpu().item()) * len(indices)
        history.append(weighted_loss / len(tokens))
    return history


def evaluate_arithmetic_transformer(
    model: ArithmeticTransformer,
    dataset: ArithmeticDataset,
    split: ArithmeticSplit,
    *,
    batch_size: int = 128,
) -> dict[str, float | int]:
    tokens_np, targets_np, _ = dataset.arrays(split)
    tokens = torch.from_numpy(tokens_np)
    targets = torch.from_numpy(targets_np)
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    target_logit_sum = 0.0
    target_probability_sum = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                batch_tokens = tokens[start : start + batch_size]
                batch_targets = targets[start : start + batch_size]
                logits = model(batch_tokens)[:, -1, :]
                losses = torch.nn.functional.cross_entropy(logits, batch_targets, reduction="none")
                loss_sum += float(losses.sum().cpu().item())
                correct += int((logits.argmax(dim=-1) == batch_targets).sum().cpu().item())
                target_logits = logits.gather(1, batch_targets[:, None]).squeeze(1)
                target_logit_sum += float(target_logits.sum().cpu().item())
                probabilities = torch.softmax(logits, dim=-1)
                target_probability_sum += float(
                    probabilities.gather(1, batch_targets[:, None]).sum().cpu().item()
                )
    finally:
        model.train(was_training)
    count = len(tokens)
    return {
        "samples": count,
        "accuracy": correct / count,
        "loss": loss_sum / count,
        "mean_target_logit": target_logit_sum / count,
        "mean_target_probability": target_probability_sum / count,
    }
