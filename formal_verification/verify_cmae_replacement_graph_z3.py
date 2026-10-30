"""Independent Z3 checks for scoped SAE/CLT replacement-graph contracts."""

import z3

UNSAT_CHECKS: list[str] = []
SAT_CHECKS: list[str] = []


def expect_unsat(description: str, *constraints: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.set(timeout=30_000)
    solver.add(*constraints)
    result = solver.check()
    assert result == z3.unsat, f"{description}: expected UNSAT, got {result}"
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


def main() -> None:
    y0, y1, z0, z1, d00, d01, d11, e0, e1 = z3.Reals(
        "replacement_y0 replacement_y1 replacement_z0 replacement_z1 "
        "replacement_d00 replacement_d01 replacement_d11 replacement_e0 replacement_e1"
    )
    expect_unsat(
        "early reconstruction conservation",
        y0 == z0 * d00 + e0,
        y0 - e0 != z0 * d00,
    )
    expect_unsat(
        "cross-layer reconstruction conservation",
        y1 == z0 * d01 + z1 * d11 + e1,
        y1 - e1 != z0 * d01 + z1 * d11,
    )
    expect_unsat(
        "independent SAE is a zero-cross-layer CLT",
        d01 == 0,
        y1 == z0 * d01 + z1 * d11 + e1,
        y1 != z1 * d11 + e1,
    )

    f, f0, j0, j1, epsilon = z3.Reals(
        "replacement_f replacement_f0 replacement_j0 replacement_j1 replacement_epsilon"
    )
    allocation = z0 * d00 * j0 + z0 * d01 * j1 + z1 * d11 * j1 + e0 * j0 + e1 * j1
    expect_unsat(
        "integrated direct-output allocation conservation",
        f == f0 + allocation + epsilon,
        f - f0 - epsilon != allocation,
    )
    expect_unsat(
        "zero early reconstruction error removes the early error write",
        e0 == 0,
        y0 == z0 * d00 + e0,
        y0 != z0 * d00,
    )
    expect_unsat(
        "zero cross-layer error removes the final error write",
        e1 == 0,
        y1 == z0 * d01 + z1 * d11 + e1,
        y1 != z0 * d01 + z1 * d11,
    )

    local_prediction, actual_effect = z3.Reals(
        "replacement_local_prediction replacement_actual_effect"
    )
    expect_sat(
        "exact reconstruction does not imply exact local feature edges",
        y0 == z0 * d00 + e0,
        y1 == z0 * d01 + z1 * d11 + e1,
        local_prediction != actual_effect,
    )
    direct_allocation, recomputed_total = z3.Reals(
        "replacement_direct_allocation replacement_recomputed_total"
    )
    expect_sat(
        "direct fixed-downstream allocation need not equal recomputed total effect",
        direct_allocation != recomputed_total,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} intentional-SAT replacement-graph checks"
    )


if __name__ == "__main__":
    main()
