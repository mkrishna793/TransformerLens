"""Reward/cost auditing for constrained RL and Safe-RLHF-style objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryAudit:
    mean_reward: float
    mean_cost: float
    cost_threshold: float
    expected_feasible: bool
    tail_violation_rate: float
    maximum_cost: float
    tail_value_at_risk: float
    lagrangian: float
    updated_multiplier: float


def _validate_trajectories(rewards: FloatArray, costs: FloatArray) -> None:
    if rewards.ndim != 1 or costs.shape != rewards.shape or rewards.size == 0:
        raise ValueError("rewards and costs must be aligned nonempty vectors")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(costs)):
        raise ValueError("trajectory values must be finite")
    if np.any(costs < 0):
        raise ValueError("costs must be nonnegative")


def lagrangian(reward: float, cost: float, threshold: float, multiplier: float) -> float:
    if multiplier < 0 or not all(
        np.isfinite(value) for value in (reward, cost, threshold, multiplier)
    ):
        raise ValueError("invalid Lagrangian parameters")
    return reward - multiplier * (cost - threshold)


def dual_update(multiplier: float, mean_cost: float, threshold: float, step: float) -> float:
    if (
        multiplier < 0
        or step <= 0
        or not all(np.isfinite(value) for value in (multiplier, mean_cost, threshold, step))
    ):
        raise ValueError("invalid dual-update parameters")
    return max(0.0, multiplier + step * (mean_cost - threshold))


def tail_value_at_risk(costs: FloatArray, alpha: float = 0.25) -> float:
    if costs.ndim != 1 or costs.size == 0 or not np.all(np.isfinite(costs)):
        raise ValueError("costs must be a finite vector")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    count = max(1, int(np.ceil(alpha * costs.size)))
    worst = np.sort(costs)[-count:]
    return float(worst.mean())


def audit_trajectories(
    rewards: FloatArray,
    costs: FloatArray,
    *,
    cost_threshold: float,
    multiplier: float = 0.0,
    dual_step: float = 0.1,
    tail_alpha: float = 0.25,
) -> TrajectoryAudit:
    """Report both the CMDP average constraint and per-trajectory tail risk."""

    _validate_trajectories(rewards, costs)
    if cost_threshold < 0 or not np.isfinite(cost_threshold):
        raise ValueError("cost threshold must be finite and nonnegative")
    mean_reward = float(rewards.mean())
    mean_cost = float(costs.mean())
    return TrajectoryAudit(
        mean_reward=mean_reward,
        mean_cost=mean_cost,
        cost_threshold=cost_threshold,
        expected_feasible=bool(mean_cost <= cost_threshold),
        tail_violation_rate=float(np.mean(costs > cost_threshold)),
        maximum_cost=float(costs.max()),
        tail_value_at_risk=tail_value_at_risk(costs, tail_alpha),
        lagrangian=lagrangian(mean_reward, mean_cost, cost_threshold, multiplier),
        updated_multiplier=dual_update(multiplier, mean_cost, cost_threshold, dual_step),
    )
