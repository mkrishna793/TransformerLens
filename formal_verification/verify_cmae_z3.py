"""Independent Z3 checks for the experimental CMAE mathematics.

Expected-SAT checks are deliberate counterexamples. Expected-UNSAT checks refute a
property violation inside the stated finite/abstract model. Neither establishes
semantic feature correctness or universal detection on real neural networks.
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


def check_lens_and_kronecker_algebra() -> None:
    a, b, x, dy = z3.Reals("a b x dy")
    expect_unsat("multiplicative lens reverse composition", (dy * b) * a != dy * (a * b))

    names = "ab0 ab1 aq0 aq1 sb0 sb1 sq0 sq1 pa0 pa1 ps0 ps1"
    ab0, ab1, aq0, aq1, sb0, sb1, sq0, sq1, pa0, pa1, ps0, ps1 = z3.Reals(names)
    factored = (pa0 * ab0 * aq0 + pa1 * ab1 * aq1) * (ps0 * sb0 * sq0 + ps1 * sb1 * sq1)
    expanded = (
        pa0 * ps0 * ab0 * aq0 * sb0 * sq0
        + pa0 * ps1 * ab0 * aq0 * sb1 * sq1
        + pa1 * ps0 * ab1 * aq1 * sb0 * sq0
        + pa1 * ps1 * ab1 * aq1 * sb1 * sq1
    )
    expect_unsat("rank-one diagonal Kronecker identity", factored != expanded)

    quadratic_form = z3.Real("positive_quadratic_form")
    expect_unsat(
        "standard upweighting influence reverses a positive quadratic form",
        quadratic_form > 0,
        -quadratic_form >= 0,
    )


def check_control_and_observation_limits() -> None:
    accepted = z3.Bool("accepted")
    selected_for_training = accepted
    quarantined = z3.Not(accepted)
    expect_unsat(
        "training and quarantine selections are disjoint",
        z3.And(selected_for_training, quarantined),
    )

    world = z3.IntSort()
    topology = z3.Function("topology", world, world)
    safe = z3.Function("safe_topology_world", world, z3.BoolSort())
    detector = z3.Function("topology_detector", world, z3.BoolSort())
    safe_world, unsafe_world = z3.Ints("topology_safe_world topology_unsafe_world")
    expect_unsat(
        "equal topology with opposite safety excludes a perfect topology-only detector",
        safe_world != unsafe_world,
        topology(safe_world) == topology(unsafe_world),
        safe(safe_world),
        z3.Not(safe(unsafe_world)),
        detector(topology(safe_world)) == safe(safe_world),
        detector(topology(unsafe_world)) == safe(unsafe_world),
    )


def check_mad_health_policy() -> None:
    mad, epsilon = z3.Reals("mad epsilon")
    scale = z3.RealVal("1.4826") * mad
    healthy = scale > epsilon
    expect_unsat(
        "zero MAD cannot satisfy the positive-scale health policy", mad == 0, epsilon > 0, healthy
    )

    scores = z3.Reals("score_0 score_1 score_2 score_3")
    expect_sat(
        "zero MAD can coexist with a large outlier",
        *[score == 0 for score in scores[:3]],
        scores[3] == 100,
        mad == 0,
    )


def check_counterexamples() -> None:
    x, y = z3.Reals("relu_x relu_y")
    relu_x = z3.If(x > 0, x, z3.RealVal(0))
    relu_y = z3.If(y > 0, y, z3.RealVal(0))
    expect_sat(
        "ReLU erases distinctions between negative gradients",
        x < 0,
        y < 0,
        x != y,
        relu_x == relu_y,
    )

    original_forbidden = z3.RealVal("0.1")
    top1_forbidden = z3.RealVal(0)
    expect_sat(
        "TopK support can hide a nonzero forbidden coordinate",
        original_forbidden != 0,
        z3.Not(top1_forbidden != 0),
    )

    first_x, first_y, second_x, second_y, midpoint_x, midpoint_y = z3.Reals(
        "first_x first_y second_x second_y midpoint_x midpoint_y"
    )
    at_most_one_first = z3.Or(first_x == 0, first_y == 0)
    at_most_one_second = z3.Or(second_x == 0, second_y == 0)
    at_most_one_midpoint = z3.Or(midpoint_x == 0, midpoint_y == 0)
    expect_sat(
        "the one-sparse set has a non-convex midpoint witness",
        first_x == 1,
        first_y == 0,
        second_x == 0,
        second_y == 1,
        midpoint_x == (first_x + second_x) / 2,
        midpoint_y == (first_y + second_y) / 2,
        at_most_one_first,
        at_most_one_second,
        z3.Not(at_most_one_midpoint),
    )

    # A concrete orthogonal-permutation KFE with nonuniform second moments. The
    # K-FAC-style diagonal result is -4; the EK-FAC result is -77/60.
    moments = [z3.RealVal(value) for value in (1, 2, 3, 4)]
    damping = z3.RealVal(1)
    ekfac = -z3.Sum([1 / (moment + damping) for moment in moments])
    diagonal_kfac = z3.RealVal(-4)
    expect_sat("diagonal K-FAC and EK-FAC are not interchangeable", ekfac != diagonal_kfac)

    floating_value = z3.FP("floating_value", z3.Float64())
    zero = z3.fpPlusZero(z3.Float64())
    product = z3.fpMul(z3.RNE(), zero, floating_value)
    expect_sat(
        "IEEE infinity times zero is not a safe zero gradient",
        z3.fpIsInf(floating_value),
        z3.Not(z3.fpEQ(product, zero)),
    )


def main() -> None:
    check_lens_and_kronecker_algebra()
    check_control_and_observation_limits()
    check_mad_health_policy()
    check_counterexamples()
    assert len(UNSAT_CHECKS) == 6, UNSAT_CHECKS
    assert len(SAT_CHECKS) == 6, SAT_CHECKS
    print(
        f"Completed {len(UNSAT_CHECKS) + len(SAT_CHECKS)} scoped CMAE checks: "
        f"{len(UNSAT_CHECKS)} expected UNSAT and {len(SAT_CHECKS)} expected SAT."
    )


if __name__ == "__main__":
    main()
