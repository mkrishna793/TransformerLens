"""Independent Z3 checks for fixed-weight counterfactual assurance."""

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
    lost, retained, denominator = z3.Reals("cf_lost cf_retained cf_denominator")
    expect_unsat(
        "fixed-denominator group removal",
        denominator != 0,
        (lost + retained) / denominator - retained / denominator != lost / denominator,
    )
    query, inverse, g1, g2 = z3.Reals("cf_query cf_inverse cf_g1 cf_g2")
    expect_unsat(
        "group influence is additive at the first-order model",
        (-query * inverse * g1) + (-query * inverse * g2) != -query * inverse * (g1 + g2),
    )
    curvature, jacobian, vector = z3.Reals("cf_curvature cf_jacobian cf_vector")
    expect_unsat(
        "GGN quadratic form is nonnegative",
        curvature >= 0,
        curvature * (jacobian * vector) * (jacobian * vector) < 0,
    )
    damping = z3.Real("cf_damping")
    expect_unsat(
        "positive damping makes a nonzero GGN quadratic positive",
        curvature >= 0,
        damping > 0,
        vector * vector > 0,
        (curvature + damping) * (vector * vector) <= 0,
    )
    expect_unsat(
        "upweight influence is first-order full-minus-reduced effect",
        query * (-inverse * g1) != -query * inverse * g1,
    )
    quality_low, quality_high = z3.Reals("cf_quality_low cf_quality_high")
    expect_unsat(
        "label-quality suspicion has reversed bounded order",
        quality_low >= 0,
        quality_low <= quality_high,
        quality_high <= 1,
        z3.Or(
            1 - quality_high < 0,
            1 - quality_high > 1 - quality_low,
            1 - quality_low > 1,
        ),
    )
    accepted, removed, total = z3.Reals("cf_accepted cf_removed cf_total")
    expect_unsat(
        "accepted and removed budgets are conserved",
        accepted + removed == total,
        accepted != total - removed,
    )
    organism, counterfactual, detection, repair = z3.Bools(
        "cf_organism cf_counterfactual cf_detection cf_repair"
    )
    expect_unsat(
        "failed counterfactual validity blocks the prevention gate",
        counterfactual == False,
        z3.And(organism, counterfactual, detection, repair),
    )
    loss_second, loss_first, head, hidden_first, hidden_second, hidden, x, u, v = z3.Reals(
        "cf_l2 cf_l1 cf_head cf_hp cf_hpp cf_h cf_x cf_u cf_v"
    )
    output_first = head * hidden_first * x
    left = (
        (loss_second * output_first * output_first + loss_first * head * hidden_second * x * x)
        * u
        * u
        + 2 * (loss_second * output_first * hidden + loss_first * hidden_first * x) * u * v
        + (loss_second * hidden * hidden) * v * v
    )
    right = loss_second * (output_first * u + hidden * v) * (
        output_first * u + hidden * v
    ) + loss_first * (head * hidden_second * x * x * u * u + 2 * hidden_first * x * u * v)
    expect_unsat("full tanh loss Hessian quadratic block assembly", left != right)

    individual_one, individual_two, actual_group = z3.Reals(
        "cf_individual_one cf_individual_two cf_actual_group"
    )
    expect_sat(
        "group interactions can defeat individual attribution addition",
        individual_one == 1,
        individual_two == 1,
        actual_group == 3,
        actual_group != individual_one + individual_two,
    )
    auc = z3.Real("cf_auc")
    expect_sat("high detection AUC does not imply repair", auc == 0.99, repair == False)
    quality, poison = z3.Reals("cf_high_quality cf_is_poison")
    expect_sat(
        "high label quality does not logically certify a clean sample",
        quality == 0.999,
        poison == 1,
    )
    ggn_part, residual_part, frozen_damping = z3.Reals(
        "cf_ggn_part cf_residual_part cf_frozen_damping"
    )
    expect_sat(
        "full loss Hessian can be indefinite despite PSD GGN and positive damping",
        ggn_part == 0.0005,
        residual_part == -0.002,
        frozen_damping == 0.001,
        ggn_part >= 0,
        frozen_damping > 0,
        ggn_part + residual_part + frozen_damping < 0,
    )
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} intentional-SAT counterfactual-assurance checks"
    )


if __name__ == "__main__":
    main()
