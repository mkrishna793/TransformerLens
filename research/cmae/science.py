"""Small scientific-ML validators for dimensions, symmetry, and conservation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Dimensions:
    mass: int = 0
    length: int = 0
    time: int = 0

    def __mul__(self, other: Dimensions) -> Dimensions:
        return Dimensions(
            self.mass + other.mass, self.length + other.length, self.time + other.time
        )

    def power(self, exponent: int) -> Dimensions:
        return Dimensions(self.mass * exponent, self.length * exponent, self.time * exponent)


MASS = Dimensions(mass=1)
LENGTH = Dimensions(length=1)
TIME = Dimensions(time=1)
VELOCITY = LENGTH * TIME.power(-1)
MOMENTUM = MASS * VELOCITY
KINETIC_ENERGY = MASS * VELOCITY.power(2)


@dataclass(frozen=True)
class CollisionReport:
    momentum_residual: float
    kinetic_energy_residual: float
    valid: bool


def elastic_collision_1d(
    first_mass: float, second_mass: float, first_velocity: float, second_velocity: float
) -> tuple[float, float]:
    """Return final velocities for an isolated one-dimensional elastic collision."""

    values = (first_mass, second_mass, first_velocity, second_velocity)
    if not all(np.isfinite(value) for value in values) or first_mass <= 0 or second_mass <= 0:
        raise ValueError("masses must be positive and all collision values finite")
    total_mass = first_mass + second_mass
    new_first = (
        (first_mass - second_mass) * first_velocity + 2 * second_mass * second_velocity
    ) / total_mass
    new_second = (
        2 * first_mass * first_velocity + (second_mass - first_mass) * second_velocity
    ) / total_mass
    return float(new_first), float(new_second)


def validate_collision(
    first_mass: float,
    second_mass: float,
    first_velocity: float,
    second_velocity: float,
    new_first: float,
    new_second: float,
    *,
    tolerance: float = 1e-9,
) -> CollisionReport:
    """Check momentum and kinetic-energy conservation independently of a model claim."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    momentum_before = first_mass * first_velocity + second_mass * second_velocity
    momentum_after = first_mass * new_first + second_mass * new_second
    energy_before = 0.5 * first_mass * first_velocity**2 + 0.5 * second_mass * second_velocity**2
    energy_after = 0.5 * first_mass * new_first**2 + 0.5 * second_mass * new_second**2
    momentum_residual = abs(momentum_after - momentum_before) / max(abs(momentum_before), 1.0)
    energy_residual = abs(energy_after - energy_before) / max(abs(energy_before), 1e-12)
    return CollisionReport(
        momentum_residual=float(momentum_residual),
        kinetic_energy_residual=float(energy_residual),
        valid=bool(momentum_residual <= tolerance and energy_residual <= tolerance),
    )


def permutation_error(
    function: Callable[[FloatArray], FloatArray],
    values: FloatArray,
    permutation: IntArray,
    *,
    equivariant: bool = False,
) -> float:
    """Measure graph-level invariance or node-level equivariance under permutation."""

    if values.ndim == 0 or permutation.shape != (values.shape[0],):
        raise ValueError("permutation must align with the first value axis")
    if not np.array_equal(np.sort(permutation), np.arange(values.shape[0])):
        raise ValueError("permutation must contain each index exactly once")
    original = np.asarray(function(values), dtype=np.float64)
    permuted = np.asarray(function(values[permutation]), dtype=np.float64)
    expected = original[permutation] if equivariant else original
    if expected.shape != permuted.shape:
        raise ValueError("function output shape is incompatible with the symmetry test")
    return float(np.max(np.abs(permuted - expected), initial=0.0))


def family_holdout(families: NDArray[np.str_], held_out: str) -> tuple[IntArray, IntArray]:
    """Return disjoint train/test indices with no family leakage."""

    if families.ndim != 1 or families.size == 0 or not held_out:
        raise ValueError("families and held-out name are required")
    test = np.where(families == held_out)[0].astype(np.int64)
    train = np.where(families != held_out)[0].astype(np.int64)
    if test.size == 0 or train.size == 0:
        raise ValueError("both train and held-out families must be nonempty")
    if np.intersect1d(train, test).size:
        raise RuntimeError("family split leaked an index")
    return train, test


def bounds_violation(values: FloatArray, lower: float, upper: float) -> float:
    if not np.all(np.isfinite(values)) or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bounds and values must be finite")
    if lower > upper:
        raise ValueError("lower bound exceeds upper bound")
    return float(np.maximum(np.maximum(lower - values, values - upper), 0.0).max(initial=0.0))
