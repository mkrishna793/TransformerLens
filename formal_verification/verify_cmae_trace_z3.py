"""Independent Z3 checks for trace, projection, and repair-gate mathematics."""

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
    a, b, c, d, e, f, x1, x2 = z3.Reals(
        "trace_a trace_b trace_c trace_d trace_e trace_f trace_x1 trace_x2"
    )
    composed = e * (a * x1 + b * x2) + f * (c * x1 + d * x2)
    expanded = (e * a + f * c) * x1 + (e * b + f * d) * x2
    expect_unsat("2x2 downstream Jacobian composition", composed != expanded)

    actual, explained = z3.Reals("trace_actual trace_explained")
    residual = actual - explained
    expect_unsat("trace residual conserves the actual output", actual != explained + residual)

    w1, w2, z1, z2, bias = z3.Reals("trace_w1 trace_w2 trace_z1 trace_z2 trace_bias")
    full = w1 * z1 + w2 * z2 + bias
    ablated = w2 * z2 + bias
    expect_unsat("linear hidden ablation equals its edge contribution", full - ablated != w1 * z1)

    linear, x, y = z3.Reals("trace_linear trace_x trace_y")
    expect_unsat(
        "linear group attribution is additive", linear * (x + y) != linear * x + linear * y
    )
    expect_sat(
        "nonlinear group interaction defeats naive additivity",
        x == 1,
        y == 1,
        (x + y) * (x + y) != x * x + y * y,
    )

    epsilon, delta = z3.Reals("adaptive_epsilon adaptive_delta")
    projected = z3.If(delta < -epsilon, -epsilon, z3.If(delta > epsilon, epsilon, delta))
    expect_unsat(
        "projected adaptive perturbation respects the L-infinity ball",
        epsilon >= 0,
        z3.Or(projected > epsilon, projected < -epsilon),
    )

    clean_drop, original_before, original_after, adaptive_before, adaptive_after = z3.Reals(
        "repair_clean_drop repair_original_before repair_original_after repair_adaptive_before repair_adaptive_after"
    )
    gate = z3.And(
        clean_drop <= z3.RealVal(1) / 50,
        original_after < original_before,
        adaptive_after < adaptive_before,
        original_after <= z3.RealVal(1) / 2,
        adaptive_after <= z3.RealVal(1) / 2,
    )
    expect_unsat(
        "adaptive non-improvement blocks the repair gate",
        clean_drop == z3.RealVal(1) / 100,
        original_before == 1,
        original_after == z3.RealVal(1) / 2,
        adaptive_before == z3.RealVal(3) / 4,
        adaptive_after == z3.RealVal(3) / 4,
        gate,
    )
    expect_unsat(
        "high residual adaptive risk blocks despite both improvements",
        clean_drop == z3.RealVal(1) / 100,
        original_before == 1,
        original_after == z3.RealVal(1) / 4,
        adaptive_before == 1,
        adaptive_after == z3.RealVal(3) / 4,
        gate,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and {len(SAT_CHECKS)} expected-SAT trace checks"
    )


if __name__ == "__main__":
    main()
