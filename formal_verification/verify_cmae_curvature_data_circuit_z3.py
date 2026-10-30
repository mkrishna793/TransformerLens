"""Independent Z3 checks for EK-FAC data-to-feature assurance contracts."""

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
    qs, qa, s, a, s1, a1, s2, a2 = z3.Reals(
        "curvature_qs curvature_qa curvature_s curvature_a "
        "curvature_s1 curvature_a1 curvature_s2 curvature_a2"
    )
    expect_unsat(
        "scalar KFE rank-one projection",
        (qs * s) * (qa * a) != qs * (s * a) * qa,
    )
    expect_unsat(
        "KFE projection preserves position sums",
        qs * (s1 * a1 + s2 * a2) * qa != (qs * s1) * (qa * a1) + (qs * s2) * (qa * a2),
    )
    g1, g2 = z3.Reals("curvature_g1 curvature_g2")
    expect_unsat("corrected KFE second moments are nonnegative", (g1 * g1 + g2 * g2) / 2 < 0)
    moment, damping = z3.Reals("curvature_moment curvature_damping")
    expect_unsat(
        "damped curvature denominator is positive",
        moment >= 0,
        damping > 0,
        moment + damping <= 0,
    )
    gp, gc, h = z3.Reals("curvature_gp curvature_gc curvature_h")
    expect_unsat(
        "clean-poison influence contrast is linear",
        (-gp * h) - (-gc * h) != -(gp - gc) * h,
    )
    readout, feature_influence, residual, total = z3.Reals(
        "curvature_readout curvature_feature_influence curvature_residual curvature_total"
    )
    expect_unsat(
        "joint feature-path residual conservation",
        total == readout * feature_influence + residual,
        total - readout * feature_influence != residual,
    )
    expect_unsat("concrete targeted backdoor label", z3.Or((5 + 1) % 11 != 6, (5 + 1) % 11 == 5))

    influence, poison = z3.Reals("curvature_high_influence curvature_is_poison")
    expect_sat(
        "high influence does not logically imply poison",
        influence > 100,
        poison == 0,
    )
    reconstruction, causal_error = z3.Reals(
        "curvature_reconstruction_quality curvature_causal_error"
    )
    expect_sat(
        "excellent replacement reconstruction can coexist with causal error",
        reconstruction > 0.99,
        causal_error > 10,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} intentional-SAT curvature-data-circuit checks"
    )


if __name__ == "__main__":
    main()
