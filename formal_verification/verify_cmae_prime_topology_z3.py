"""Independent Z3 checks for CMAE finite-field and trace mathematics.

Expected-SAT checks are deliberate distinguishing witnesses. Expected-UNSAT checks
refute a violation of the stated exact identity or finite example.
"""

import z3

UNSAT_CHECKS: list[str] = []
SAT_CHECKS: list[str] = []


def expect_unsat(description: str, *constraints: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.set(timeout=30_000)
    solver.add(*constraints)
    result = solver.check()
    model = str(solver.model()) if result == z3.sat else ""
    assert result == z3.unsat, f"{description}: expected UNSAT, got {result} {model}"
    UNSAT_CHECKS.append(description)
    print(f"PASS UNSAT: {description}")


def expect_sat(description: str, *constraints: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.set(timeout=30_000)
    solver.add(*constraints)
    result = solver.check()
    assert result == z3.sat, f"{description}: expected SAT, got {result}"
    SAT_CHECKS.append(description)
    print(f"PASS SAT:   {description}; witness: {solver.model()}")


def check_newton_identities() -> None:
    p1, p2, p3, p4 = z3.Reals("prime_newton_p1 prime_newton_p2 prime_newton_p3 prime_newton_p4")
    e2 = (p1 * p1 - p2) / 2
    e3_recursive = (e2 * p1 - p1 * p2 + p3) / 3
    e3_closed = (p1 * p1 * p1 - 3 * p1 * p2 + 2 * p3) / 6
    expect_unsat("Newton third elementary-symmetric identity", e3_recursive != e3_closed)

    e4_recursive = (e3_closed * p1 - e2 * p2 + p1 * p3 - p4) / 4
    e4_closed = (p1 * p1 * p1 * p1 - 6 * p1 * p1 * p2 + 3 * p2 * p2 + 8 * p1 * p3 - 6 * p4) / 24
    expect_unsat("Newton fourth elementary-symmetric identity", e4_recursive != e4_closed)


def check_prime_homology_examples() -> None:
    def rank_1x1_mod(value: int, prime: int) -> z3.ArithRef:
        return z3.If(z3.IntVal(value) % prime == 0, z3.IntVal(0), z3.IntVal(1))

    expected = z3.And(
        rank_1x1_mod(2, 2) == 0,
        rank_1x1_mod(2, 3) == 1,
        rank_1x1_mod(3, 3) == 0,
        rank_1x1_mod(3, 2) == 1,
    )
    expect_unsat("RP2 and lens-space one-cell torsion ranks", z3.Not(expected))


def check_trace_counterexample() -> None:
    first = (1, -1, 8, -8)
    second = (4, -4, 7, -7)

    def power_trace(spectrum: tuple[int, ...], degree: int) -> z3.ArithRef:
        return sum((z3.IntVal(value) ** degree for value in spectrum), z3.IntVal(0))

    agreement = z3.And(
        *[power_trace(first, prime) == power_trace(second, prime) for prime in (2, 3, 5, 7)]
    )
    expect_unsat("explicit prime-indexed trace agreement", z3.Not(agreement))
    expect_sat(
        "the fourth trace distinguishes spectra hidden by prime traces",
        power_trace(first, 4) != power_trace(second, 4),
    )

    def product(spectrum: tuple[int, ...]) -> z3.ArithRef:
        result = z3.IntVal(1)
        for value in spectrum:
            result *= value
        return result

    expect_sat(
        "the characteristic-polynomial constant term also distinguishes the spectra",
        product(first) != product(second),
    )


def main() -> None:
    check_newton_identities()
    check_prime_homology_examples()
    check_trace_counterexample()
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} expected-SAT CMAE prime/topology checks"
    )


if __name__ == "__main__":
    main()
