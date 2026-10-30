"""Architecture-independent latent point-cloud topology and spectral traces.

Only H0 persistence is implemented here, using a minimum-spanning-tree/union-find
construction. These descriptors are non-injective geometric observables: equal values
must never be interpreted as equal meanings or guaranteed safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TopologySignature:
    deaths: FloatArray
    mean_persistence: float
    max_persistence: float
    persistence_entropy: float
    trace_2: float
    trace_3: float
    effective_rank: float

    def vector(self) -> FloatArray:
        return np.array(
            [
                self.mean_persistence,
                self.max_persistence,
                self.persistence_entropy,
                self.trace_2,
                self.trace_3,
                self.effective_rank,
            ],
            dtype=np.float64,
        )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


def _validate_points(points: FloatArray, minimum: int = 2) -> None:
    if points.ndim != 2 or points.shape[0] < minimum or points.shape[1] == 0:
        raise ValueError(f"points must have shape [n>={minimum}, d>0]")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")


def h0_persistence_deaths(points: FloatArray) -> FloatArray:
    """Compute finite H0 deaths for the Euclidean Vietoris-Rips filtration."""

    _validate_points(points)
    count = points.shape[0]
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    edges = sorted(
        (float(distances[i, j]), i, j) for i in range(count) for j in range(i + 1, count)
    )
    components = _UnionFind(count)
    deaths: list[float] = []
    for distance, left, right in edges:
        if components.union(left, right):
            deaths.append(distance)
            if len(deaths) == count - 1:
                break
    return np.array(deaths, dtype=np.float64)


def _entropy(weights: FloatArray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    probabilities = weights[weights > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def topology_signature(points: FloatArray) -> TopologySignature:
    deaths = h0_persistence_deaths(points)
    centered = points - points.mean(axis=0, keepdims=True)
    gram = centered @ centered.T / points.shape[1]
    eigenvalues = np.linalg.eigvalsh(gram).clip(min=0)
    eigenvalue_sum = float(eigenvalues.sum())
    normalized_eigenvalues = eigenvalues / max(eigenvalue_sum, 1e-15)
    effective_rank = float(np.exp(_entropy(normalized_eigenvalues))) if eigenvalue_sum > 0 else 0.0
    return TopologySignature(
        deaths=deaths,
        mean_persistence=float(deaths.mean()),
        max_persistence=float(deaths.max(initial=0.0)),
        persistence_entropy=_entropy(deaths),
        trace_2=float(np.sum(eigenvalues**2)),
        trace_3=float(np.sum(eigenvalues**3)),
        effective_rank=effective_rank,
    )


def leave_one_out_topology_scores(points: FloatArray) -> FloatArray:
    """Measure each point's effect on a batch-level topological summary.

    This is an attribution score, not a calibrated poison probability.
    """

    _validate_points(points, minimum=3)
    baseline = topology_signature(points).vector()
    scores = np.zeros(points.shape[0], dtype=np.float64)
    for index in range(points.shape[0]):
        reduced = np.delete(points, index, axis=0)
        scores[index] = np.abs(topology_signature(reduced).vector() - baseline).sum()
    return scores
