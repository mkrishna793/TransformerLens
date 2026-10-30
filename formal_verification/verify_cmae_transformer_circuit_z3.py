"""Independent Z3 checks for Transformer component-trace contracts."""

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
    jacobian, a, b, c = z3.Reals("circuit_j circuit_a circuit_b circuit_c")
    expect_unsat(
        "integrated readout allocation conserves three residual writes",
        jacobian * a + jacobian * b + jacobian * c != jacobian * (a + b + c),
    )

    actual, baseline, explained = z3.Reals("circuit_actual circuit_baseline circuit_explained")
    residual = actual - baseline - explained
    expect_unsat(
        "integrated readout residual conserves the actual logit",
        actual != baseline + explained + residual,
    )

    activation, bias = z3.Reals("circuit_activation circuit_bias")
    expect_unsat(
        "linear zero ablation has the predicted total effect",
        (jacobian * activation + bias) - bias != jacobian * activation,
    )
    expect_unsat(
        "constant-Jacobian integrated path is exact",
        baseline + jacobian * residual != jacobian * residual + baseline,
    )

    a_value, b_value, c_value, d_value = z3.Ints("arith_a arith_b arith_c arith_d")
    expect_unsat(
        "modular arithmetic latent labels satisfy the composed expression",
        a_value == 2,
        b_value == 3,
        c_value == 4,
        d_value == 5,
        ((a_value + b_value) * c_value + d_value) % 11 != 3,
    )

    x, x0 = z3.Reals("cubic_x cubic_x0")
    actual_cubic_change = x * x * x - x0 * x0 * x0
    local_prediction = 3 * x0 * x0 * (x - x0)
    expect_sat(
        "nonlinear total effect differs from a local ablation prediction",
        x == 2,
        x0 == 1,
        actual_cubic_change != local_prediction,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and {len(SAT_CHECKS)} expected-SAT Transformer-circuit checks"
    )


if __name__ == "__main__":
    main()
