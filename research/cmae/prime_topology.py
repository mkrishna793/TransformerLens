"""Exact finite-field homology and spectral-trace experiments for CMAE.

Prime-dependent homology can expose torsion. Prime-indexed power traces, however,
are not a complete spectral fingerprint. All routines operate on small examples so
that the arithmetic remains explicit and independently checkable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _require_prime(prime: int) -> None:
    if not is_prime(prime):
        raise ValueError("coefficient modulus must be prime")
    if prime > 10_000:
        raise ValueError("this small reference implementation bounds primes at 10,000")


def modular_rank(matrix: IntArray, prime: int) -> int:
    """Return exact matrix rank over F_p by Gaussian elimination."""

    _require_prime(prime)
    values = np.asarray(matrix, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    work = (values % prime).astype(np.int64, copy=True)
    rows, columns = work.shape
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row, column] != 0), None)
        if pivot is None:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
        inverse = pow(int(work[rank, column]), -1, prime)
        work[rank, :] = (work[rank, :] * inverse) % prime
        for row in range(rank + 1, rows):
            factor = int(work[row, column])
            if factor:
                work[row, :] = (work[row, :] - factor * work[rank, :]) % prime
        rank += 1
        if rank == rows:
            break
    return rank


def betti_numbers(
    dimensions: Sequence[int], boundaries: Mapping[int, IntArray], prime: int
) -> IntArray:
    """Compute Betti numbers of a finite CW/simplicial chain complex over F_p.

    ``boundaries[k]`` maps C_k to C_{k-1}. Missing boundaries are zero. The
    caller must provide a valid integral chain complex; this function evaluates its
    reduction modulo the selected prime.
    """

    if len(dimensions) == 0 or any(dimension < 0 for dimension in dimensions):
        raise ValueError("chain dimensions must be nonnegative and nonempty")
    ranks: dict[int, int] = {}
    for degree, boundary in boundaries.items():
        if degree <= 0 or degree >= len(dimensions):
            raise ValueError("boundary degree is outside the chain complex")
        expected = (dimensions[degree - 1], dimensions[degree])
        if boundary.shape != expected:
            raise ValueError(f"boundary {degree} must have shape {expected}")
        ranks[degree] = modular_rank(boundary, prime)
    result = np.zeros(len(dimensions), dtype=np.int64)
    for degree, dimension in enumerate(dimensions):
        result[degree] = dimension - ranks.get(degree, 0) - ranks.get(degree + 1, 0)
    if np.any(result < 0):
        raise ValueError("boundary maps do not form a chain complex over this field")
    return result


def graph_boundary(vertices: int, edges: Sequence[tuple[int, int]]) -> IntArray:
    """Construct the oriented graph incidence matrix C_1 -> C_0."""

    if vertices <= 0:
        raise ValueError("a graph needs at least one vertex")
    boundary = np.zeros((vertices, len(edges)), dtype=np.int64)
    for column, (left, right) in enumerate(edges):
        if not 0 <= left < right < vertices:
            raise ValueError("edges must use ordered, distinct vertex indices")
        boundary[left, column] = -1
        boundary[right, column] = 1
    return boundary


def graph_betti_numbers(vertices: int, edges: Sequence[tuple[int, int]], prime: int) -> IntArray:
    boundary = graph_boundary(vertices, edges)
    return betti_numbers((vertices, len(edges), 0), {1: boundary}, prime)


def power_traces(matrix: FloatArray, degrees: Sequence[int]) -> FloatArray:
    """Compute tr(A^k) for explicit nonnegative integer degrees."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    if any(degree < 0 for degree in degrees):
        raise ValueError("degrees must be nonnegative")
    return np.array(
        [np.trace(np.linalg.matrix_power(values, degree)) for degree in degrees],
        dtype=np.float64,
    )


def characteristic_from_traces(traces: FloatArray) -> FloatArray:
    """Recover det(lambda I - A) from tr(A), ..., tr(A^n) by Newton identities.

    The returned coefficients are ordered from degree n down to degree zero and use
    characteristic-zero arithmetic. Small floating-point examples may be ill-conditioned.
    """

    power_sums = np.asarray(traces, dtype=np.float64)
    if power_sums.ndim != 1 or power_sums.size == 0:
        raise ValueError("at least one power sum is required")
    elementary = np.zeros(power_sums.size + 1, dtype=np.float64)
    elementary[0] = 1.0
    for degree in range(1, power_sums.size + 1):
        total = 0.0
        for index in range(1, degree + 1):
            total += (-1) ** (index - 1) * elementary[degree - index] * power_sums[index - 1]
        elementary[degree] = total / degree
    return np.array([(-1) ** degree * value for degree, value in enumerate(elementary)])


def prime_trace_signature(matrix: FloatArray, primes: Sequence[int] = (2, 3, 5, 7)) -> FloatArray:
    """Compute the deliberately incomplete prime-indexed trace descriptor."""

    for prime in primes:
        _require_prime(prime)
    return power_traces(matrix, primes)


def prime_trace_counterexample() -> tuple[FloatArray, FloatArray]:
    """Return diagonal matrices with equal prime traces but different spectra.

    Every odd prime power trace is zero because both spectra are sign-symmetric.
    Their second traces are both 130, while their fourth traces differ.
    """

    first = np.diag(np.array([1.0, -1.0, 8.0, -8.0]))
    second = np.diag(np.array([4.0, -4.0, 7.0, -7.0]))
    return first, second


@dataclass(frozen=True)
class LatentHomologyRecord:
    threshold: float
    edges: int
    betti_by_prime: dict[int, IntArray]
    two_cells: int = 0
    complex_kind: str = "graph"


def latent_graph_multifield_homology(
    points: FloatArray,
    *,
    primes: Sequence[int] = (2, 3, 5),
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> list[LatentHomologyRecord]:
    """Compute prime-field homology of an exact 1-D latent-neighbor complex.

    The graph has only zero- and one-cells. It therefore provides a negative control
    for torsion claims: all field Betti numbers should agree, regardless of the latent
    geometry. Certified higher cells are required before prime torsion can be a signal.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] == 0:
        raise ValueError("points must have shape [n>=4, d>0]")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    if not primes or any(not is_prime(prime) for prime in primes):
        raise ValueError("all coefficient moduli must be prime")
    if not quantiles or any(not 0 < quantile <= 1 for quantile in quantiles):
        raise ValueError("filtration quantiles must be in (0, 1]")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    records: list[LatentHomologyRecord] = []
    for quantile in quantiles:
        threshold = float(np.quantile(upper, quantile))
        edges = tuple(
            (left, right)
            for left in range(values.shape[0])
            for right in range(left + 1, values.shape[0])
            if distances[left, right] <= threshold
        )
        betti = {prime: graph_betti_numbers(values.shape[0], edges, prime) for prime in primes}
        records.append(LatentHomologyRecord(threshold, len(edges), betti))
    return records


def multifield_disagreement(
    records: Sequence[LatentHomologyRecord], reference_prime: int = 2
) -> float:
    """Return the largest Betti-vector difference from the reference field."""

    disagreement = 0.0
    for record in records:
        if reference_prime not in record.betti_by_prime:
            raise ValueError("reference prime is missing from a homology record")
        reference = record.betti_by_prime[reference_prime]
        for prime, values in record.betti_by_prime.items():
            if prime != reference_prime:
                disagreement = max(disagreement, float(np.max(np.abs(values - reference))))
    return disagreement


def clique_complex_betti_numbers(
    vertices: int, edges: Sequence[tuple[int, int]], prime: int
) -> IntArray:
    """Compute H0--H2 of the certified 2-D clique complex of a graph.

    Every 2-cell is an observed triangle; no higher-dimensional or topological
    completion is inferred. The integral boundary matrices are reduced modulo p.
    """

    ordered_edges = tuple(sorted((min(left, right), max(left, right)) for left, right in edges))
    if len(set(ordered_edges)) != len(ordered_edges):
        raise ValueError("clique-complex edges must be unique")
    boundary1 = graph_boundary(vertices, ordered_edges)
    edge_set = set(ordered_edges)
    edge_index = {edge: index for index, edge in enumerate(ordered_edges)}
    triangles = [
        (left, middle, right)
        for left in range(vertices)
        for middle in range(left + 1, vertices)
        for right in range(middle + 1, vertices)
        if (left, middle) in edge_set and (left, right) in edge_set and (middle, right) in edge_set
    ]
    boundary2 = np.zeros((len(ordered_edges), len(triangles)), dtype=np.int64)
    for column, (left, middle, right) in enumerate(triangles):
        boundary2[edge_index[(left, middle)], column] = 1
        boundary2[edge_index[(left, right)], column] = -1
        boundary2[edge_index[(middle, right)], column] = 1
    return betti_numbers(
        (vertices, len(ordered_edges), len(triangles)), {1: boundary1, 2: boundary2}, prime
    )


def latent_clique_multifield_homology(
    points: FloatArray,
    *,
    primes: Sequence[int] = (2, 3, 5),
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> list[LatentHomologyRecord]:
    """Build exact latent-neighbor clique complexes and compare coefficient fields."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] == 0:
        raise ValueError("points must have shape [n>=4, d>0]")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    if not primes or any(not is_prime(prime) for prime in primes):
        raise ValueError("all coefficient moduli must be prime")
    if not quantiles or any(not 0 < quantile <= 1 for quantile in quantiles):
        raise ValueError("filtration quantiles must be in (0, 1]")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    records: list[LatentHomologyRecord] = []
    for quantile in quantiles:
        threshold = float(np.quantile(upper, quantile))
        edges = tuple(
            (left, right)
            for left in range(values.shape[0])
            for right in range(left + 1, values.shape[0])
            if distances[left, right] <= threshold
        )
        betti = {
            prime: clique_complex_betti_numbers(values.shape[0], edges, prime) for prime in primes
        }
        # The number of triangles is recovered from the certified complex, not from
        # a neural activation heuristic or a topological prior.
        two_cells = sum(
            1
            for left in range(values.shape[0])
            for middle in range(left + 1, values.shape[0])
            for right in range(middle + 1, values.shape[0])
            if (left, middle) in edges and (left, right) in edges and (middle, right) in edges
        )
        records.append(LatentHomologyRecord(threshold, len(edges), betti, two_cells, "2d_clique"))
    return records
