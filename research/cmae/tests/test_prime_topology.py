"""Tests for finite-field topology and trace-completeness boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from research.cmae.prime_topology import (
    betti_numbers,
    characteristic_from_traces,
    graph_betti_numbers,
    is_prime,
    modular_rank,
    power_traces,
    prime_trace_counterexample,
    prime_trace_signature,
)


def test_modular_rank_separates_two_and_three_torsion() -> None:
    assert modular_rank(np.array([[2]]), 2) == 0
    assert modular_rank(np.array([[2]]), 3) == 1
    assert modular_rank(np.array([[3]]), 3) == 0
    assert modular_rank(np.array([[3]]), 2) == 1


def test_rp2_minimal_cw_complex_has_prime_dependent_homology() -> None:
    # The standard CW chain complex is Z --2--> Z --0--> Z.
    boundaries = {1: np.array([[0]]), 2: np.array([[2]])}
    np.testing.assert_array_equal(betti_numbers((1, 1, 1), boundaries, 2), [1, 1, 1])
    np.testing.assert_array_equal(betti_numbers((1, 1, 1), boundaries, 3), [1, 0, 0])
    np.testing.assert_array_equal(betti_numbers((1, 1, 1), boundaries, 5), [1, 0, 0])


def test_lens_space_cw_complex_detects_three_torsion() -> None:
    boundaries = {1: np.array([[0]]), 2: np.array([[3]])}
    np.testing.assert_array_equal(betti_numbers((1, 1, 1), boundaries, 3), [1, 1, 1])
    np.testing.assert_array_equal(betti_numbers((1, 1, 1), boundaries, 2), [1, 0, 0])


def test_cycle_graph_homology_does_not_depend_on_prime() -> None:
    edges = ((0, 1), (0, 2), (1, 2))
    expected = np.array([1, 1, 0])
    for prime in (2, 3, 5):
        np.testing.assert_array_equal(graph_betti_numbers(3, edges, prime), expected)


def test_modular_rank_validates_field_and_shapes() -> None:
    with pytest.raises(ValueError, match="prime"):
        modular_rank(np.eye(2, dtype=np.int64), 4)
    with pytest.raises(ValueError, match="two-dimensional"):
        modular_rank(np.array([1, 2]), 2)
    with pytest.raises(ValueError, match="shape"):
        betti_numbers((2, 2), {1: np.ones((3, 2), dtype=np.int64)}, 2)


def test_newton_identities_recover_characteristic_polynomials() -> None:
    rng = np.random.default_rng(21)
    for size in (2, 3, 4):
        matrix = rng.normal(size=(size, size))
        traces = power_traces(matrix, list(range(1, size + 1)))
        recovered = characteristic_from_traces(traces)
        np.testing.assert_allclose(recovered, np.poly(matrix), rtol=1e-9, atol=1e-8)


def test_prime_traces_cannot_distinguish_counterexample() -> None:
    first, second = prime_trace_counterexample()
    primes = [value for value in range(2, 100) if is_prime(value)]
    np.testing.assert_array_equal(
        prime_trace_signature(first, primes), prime_trace_signature(second, primes)
    )
    assert not np.array_equal(np.poly(first), np.poly(second))
    assert np.trace(np.linalg.matrix_power(first, 4)) != np.trace(np.linalg.matrix_power(second, 4))


def test_full_trace_sequence_recovers_the_missing_invariant() -> None:
    first, second = prime_trace_counterexample()
    first_poly = characteristic_from_traces(power_traces(first, [1, 2, 3, 4]))
    second_poly = characteristic_from_traces(power_traces(second, [1, 2, 3, 4]))
    np.testing.assert_allclose(first_poly, np.poly(first))
    np.testing.assert_allclose(second_poly, np.poly(second))
    assert first_poly[-1] == 64.0
    assert second_poly[-1] == 784.0


def test_prime_trace_signature_rejects_composite_indices() -> None:
    with pytest.raises(ValueError, match="prime"):
        prime_trace_signature(np.eye(2), [2, 4])
