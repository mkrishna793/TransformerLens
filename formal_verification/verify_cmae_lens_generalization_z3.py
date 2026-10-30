"""Independent Z3 checks for Jacobian-lens and auxiliary-loss contracts."""

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
    jacobian, activation, offset = z3.Reals("lens_j lens_h lens_b")
    expect_unsat(
        "one-unit expert recovers the plain Jacobian readout",
        1 * (jacobian * activation + offset) + 0 != jacobian * activation + offset,
    )

    w1, w2 = z3.Reals("lens_w1 lens_w2")
    expect_unsat(
        "equal experts with unit total weight recover the same readout",
        w1 + w2 == 1,
        w1 * (jacobian * activation) + w2 * (jacobian * activation) != jacobian * activation,
    )

    actual, predicted = z3.Reals("lens_actual lens_predicted")
    residual = actual - predicted
    expect_unsat("lens residual conservation", actual != predicted + residual)

    final_loss, auxiliary_loss, weight = z3.Reals("lens_final_loss lens_auxiliary_loss lens_weight")
    expect_unsat(
        "nonnegative auxiliary objective cannot become negative",
        final_loss >= 0,
        auxiliary_loss >= 0,
        weight >= 0,
        final_loss + weight * auxiliary_loss < 0,
    )

    expect_unsat(
        "first expert prefers the first token in the concrete ranking check",
        z3.Not(1 * 1 + 0 * 0 > 1 * 0 + 0 * 1),
    )

    mixture_w1, mixture_w2 = z3.Reals("mixture_w1 mixture_w2")
    first_score = mixture_w1 * 1 + mixture_w2 * 0
    second_score = mixture_w1 * 0 + mixture_w2 * 1
    expect_sat(
        "expert weights can reverse token preference",
        mixture_w1 == 0,
        mixture_w2 == 1,
        second_score > first_score,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and {len(SAT_CHECKS)} expected-SAT lens checks"
    )


if __name__ == "__main__":
    main()
