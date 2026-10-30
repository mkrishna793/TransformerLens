"""Independent Z3 checks for the CMAE validation-program mathematics."""

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


def check_linear_jacobian() -> None:
    a, b, x1, x2, delta1, delta2 = z3.Reals("j_a j_b j_x1 j_x2 j_d1 j_d2")
    lhs = a * (x1 + delta1) + b * (x2 + delta2)
    rhs = (a * x1 + b * x2) + (a * delta1 + b * delta2)
    expect_unsat("linear downstream first-order residual is exactly zero", lhs != rhs)


def check_attribution_algebra() -> None:
    query, output_gradient, train = z3.Reals("attr_query attr_output_gradient attr_train")
    expect_unsat(
        "removal attribution sign matches the loss-gradient convention",
        output_gradient != 0,
        -(query * (-output_gradient * train)) / output_gradient != query * train,
    )
    first, second, third = z3.Reals("attr_first attr_second attr_third")
    expect_unsat(
        "group surrogate attribution is additive",
        (first + second) + third != first + (second + third),
    )


def check_rl_tail_and_dual_update() -> None:
    costs = z3.Reals("rl_c0 rl_c1 rl_c2 rl_c3")
    expect_sat(
        "expected-cost feasibility can coexist with a severe tail violation",
        *[cost == 0 for cost in costs[:3]],
        costs[3] == 3,
        sum(costs) / 4 <= 1,
        costs[3] > 1,
    )
    updated = z3.If(0 + (z3.RealVal(3) / 4 - 1) >= 0, 0 + (z3.RealVal(3) / 4 - 1), 0)
    expect_unsat("an expected-feasible zero dual variable stays zero", updated != 0)


def check_budget_gate() -> None:
    elapsed, budget = z3.Reals("runtime_elapsed runtime_budget")
    healthy = z3.Bool("runtime_healthy")
    commit = z3.And(healthy, elapsed <= budget)
    expect_unsat("timeout blocks commitment", elapsed == 2, budget == 1, healthy, commit)


def check_collision_and_dimensions() -> None:
    mass, first, second = z3.Reals("collision_mass collision_first collision_second")
    new_first = ((mass - mass) / (mass + mass)) * first + (2 * mass / (mass + mass)) * second
    new_second = (2 * mass / (mass + mass)) * first + ((mass - mass) / (mass + mass)) * second
    momentum = mass * new_first + mass * new_second == mass * first + mass * second
    energy = mass * new_first * new_first + mass * new_second * new_second == (
        mass * first * first + mass * second * second
    )
    expect_unsat(
        "equal-mass elastic collision preserves momentum and kinetic energy",
        mass != 0,
        z3.Not(z3.And(new_first == second, new_second == first, momentum, energy)),
    )
    expect_unsat(
        "kinetic-energy dimensional exponents",
        z3.Or(1 + 0 != 1, 0 + 2 * 1 != 2, 0 + 2 * -1 != -2),
    )


def main() -> None:
    check_linear_jacobian()
    check_attribution_algebra()
    check_rl_tail_and_dual_update()
    check_budget_gate()
    check_collision_and_dimensions()
    print(
        f"PASS: {len(UNSAT_CHECKS)} expected-UNSAT and "
        f"{len(SAT_CHECKS)} expected-SAT CMAE validation checks"
    )


if __name__ == "__main__":
    main()
