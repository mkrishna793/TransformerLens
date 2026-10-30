"""Parametric lenses and sparse concept charts.

The reverse map of a chart is the Jacobian pullback of its *same* forward map away
from TopK ties and ReLU kinks. It is not an independently projected gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import torch
from torch import Tensor

P = TypeVar("P")
Q = TypeVar("Q")


@dataclass(frozen=True)
class ParametricLens(Generic[P]):
    """A small executable parametric-lens contract.

    ``forward(p, x)`` computes the forward value. ``reverse(p, x, dy)`` returns
    ``(dp, dx)``. Composition pairs parameters and composes reverse maps by the chain
    rule. Stochastic, non-differentiable, and stateful maps need additional contracts.
    """

    forward_fn: Callable[[P, Tensor], Tensor]
    reverse_fn: Callable[[P, Tensor, Tensor], tuple[P, Tensor]]

    def forward(self, parameters: P, value: Tensor) -> Tensor:
        return self.forward_fn(parameters, value)

    def reverse(self, parameters: P, value: Tensor, output_gradient: Tensor) -> tuple[P, Tensor]:
        return self.reverse_fn(parameters, value, output_gradient)

    def then(self, other: ParametricLens[Q]) -> ParametricLens[tuple[P, Q]]:
        """Compose in diagrammatic order: ``self`` followed by ``other``."""

        def forward(parameters: tuple[P, Q], value: Tensor) -> Tensor:
            first_parameters, second_parameters = parameters
            return other.forward(second_parameters, self.forward(first_parameters, value))

        def reverse(
            parameters: tuple[P, Q], value: Tensor, output_gradient: Tensor
        ) -> tuple[tuple[P, Q], Tensor]:
            first_parameters, second_parameters = parameters
            intermediate = self.forward(first_parameters, value)
            second_gradient, intermediate_gradient = other.reverse(
                second_parameters, intermediate, output_gradient
            )
            first_gradient, input_gradient = self.reverse(
                first_parameters, value, intermediate_gradient
            )
            return (first_gradient, second_gradient), input_gradient

        return ParametricLens(forward, reverse)


def identity_lens() -> ParametricLens[None]:
    return ParametricLens(
        lambda _, value: value,
        lambda _, __, output_gradient: (None, output_gradient),
    )


def multiplicative_lens() -> ParametricLens[Tensor]:
    """The parametric map ``p, x -> p * x`` and its exact reverse derivative."""

    def reverse(parameter: Tensor, value: Tensor, output_gradient: Tensor) -> tuple[Tensor, Tensor]:
        return output_gradient * value, output_gradient * parameter

    return ParametricLens(lambda parameter, value: parameter * value, reverse)


def normalized_random_dictionary(input_dimension: int, concepts: int, *, seed: int) -> Tensor:
    """Create a test chart, explicitly not a trained or semantically validated SAE."""

    if input_dimension <= 0 or concepts <= 0:
        raise ValueError("dictionary dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(concepts, input_dimension, generator=generator, dtype=torch.float64)
    return dictionary / dictionary.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class TopKSparseChart:
    """A sparse linear chart with an explicit fixed-support pullback.

    ``signed=True`` computes ``TopK_|.|(x W^T)`` and preserves selected signs.
    ``signed=False`` computes ``TopK(ReLU(x W^T))``. The latter intentionally loses
    all negative pre-activations and therefore cannot provide complete evidence.

    """

    dictionary: Tensor
    k: int
    signed: bool = True

    def __post_init__(self) -> None:
        if self.dictionary.ndim != 2:
            raise ValueError("dictionary must have shape [concepts, input_dimension]")
        if not 1 <= self.k <= self.dictionary.shape[0]:
            raise ValueError("k must be between one and the number of concepts")
        if self.dictionary.shape[0] == 0 or self.dictionary.shape[1] == 0:
            raise ValueError("dictionary dimensions must be nonzero")

    def _project_and_select(self, value: Tensor) -> tuple[Tensor, Tensor]:
        if value.shape[-1] != self.dictionary.shape[1]:
            raise ValueError("last value dimension does not match the dictionary")
        projected = value @ self.dictionary.transpose(0, 1)
        visible = projected if self.signed else torch.relu(projected)
        priorities = projected.abs() if self.signed else visible
        _, indices = torch.topk(priorities, self.k, dim=-1)
        return projected, indices

    def forward(self, value: Tensor) -> Tensor:
        projected, indices = self._project_and_select(value)
        selected = projected.gather(-1, indices)
        if not self.signed:
            selected = torch.relu(selected)
        return torch.zeros_like(projected).scatter(-1, indices, selected)

    def projection_residual(self, value: Tensor) -> Tensor:
        projected, _ = self._project_and_select(value)
        return projected - self.forward(value)

    def reverse(self, value: Tensor, output_gradient: Tensor) -> Tensor:
        """Apply the fixed-support Jacobian pullback; ties require separate handling."""

        projected, indices = self._project_and_select(value)
        selection = torch.zeros_like(projected).scatter(-1, indices, torch.ones_like(projected))
        if not self.signed:
            selection = selection * (projected > 0).to(projected.dtype)
        return (output_gradient * selection) @ self.dictionary

    def as_lens(self) -> ParametricLens[None]:
        return ParametricLens(
            lambda _, value: self.forward(value),
            lambda _, value, output_gradient: (None, self.reverse(value, output_gradient)),
        )


def signed_max_pool(codes: Tensor) -> Tensor:
    """Pool concept positions by greatest magnitude without discarding negative values."""

    if codes.ndim != 2:
        raise ValueError("codes must have shape [positions, concepts]")
    if codes.shape[0] == 0:
        raise ValueError("at least one position is required")
    indices = codes.abs().argmax(dim=0, keepdim=True)
    return codes.gather(0, indices).squeeze(0)
