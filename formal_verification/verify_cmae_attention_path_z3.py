"""Independent Z3 checks for attention and residual-path intervention contracts."""

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
    first, second = z3.Reals("attention_first attention_second")
    expect_unsat("first one-hot attention selects the first value", 1 * first + 0 * second != first)
    expect_unsat(
        "second one-hot attention selects the second value", 0 * first + 1 * second != second
    )

    w1, w2 = z3.Reals("attention_w1 attention_w2")
    convex = w1 * first + w2 * second
    common = (w1 + w2 == 1, w1 >= 0, w2 >= 0, first <= second)
    expect_unsat("convex attention respects the lower bound", *common, convex < first)
    expect_unsat("convex attention respects the upper bound", *common, convex > second)

    actual, predicted = z3.Reals("intervention_actual intervention_predicted")
    residual = actual - predicted
    expect_unsat("intervention first-order residual conservation", actual != predicted + residual)

    expect_unsat("concrete modular counterfactual", (2 * 4 + 5) % 11 != 2)
    expect_unsat(
        "counterfactual differs from the original concrete answer",
        (5 * 4 + 5) % 11 == (2 * 4 + 5) % 11,
    )

    qk_first, qk_second, effect_first, effect_second = z3.Reals(
        "qk_first qk_second effect_first effect_second"
    )
    expect_sat(
        "QK ordering need not identify causal-effect ordering",
        qk_first > qk_second,
        effect_first < effect_second,
    )

    latent_first, latent_second, other_first, other_second = z3.Reals(
        "latent_first latent_second other_first other_second"
    )
    expect_sat(
        "equal latent readouts do not imply equal whole-vector paths",
        latent_first == latent_second,
        other_first != other_second,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} intentional-SAT intervention checks"
    )


if __name__ == "__main__":
    main()
