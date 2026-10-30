"""Independent Z3 checks for the research training-control kernel.

Expected-SAT checks are deliberate counterexamples or sign demonstrations.
Expected-UNSAT checks refute the negation of a property inside the stated model.
No check claims universal detection of real-world unsafe data.
"""

import z3

UNSAT_CHECKS: list[str] = []
SAT_CHECKS: list[str] = []


def prove(description: str, solver: z3.Solver, claim: z3.BoolRef) -> None:
    """Check that [claim] follows from the solver's current assumptions."""
    solver.push()
    solver.add(z3.Not(claim))
    result = solver.check()
    model = str(solver.model()) if result == z3.sat else ""
    solver.pop()
    assert result == z3.unsat, f"{description}: expected UNSAT, got {result} {model}"
    UNSAT_CHECKS.append(description)
    print(f"PASS UNSAT: {description}")


def expect_sat(description: str, solver: z3.Solver) -> None:
    """Check that a deliberate counterexample or signed case is realizable."""
    result = solver.check()
    model = str(solver.model()) if result == z3.sat else ""
    assert result == z3.sat, f"{description}: expected SAT, got {result}"
    SAT_CHECKS.append(description)
    print(f"PASS SAT:   {description}; witness: {model}")


def check_global_gate() -> None:
    finite = z3.Bool("finite")
    policy_pass = z3.Bool("policy_pass")
    flags = [z3.Bool(f"flag_{index}") for index in range(4)]
    accept = z3.And(finite, policy_pass, *[z3.Not(flag) for flag in flags])
    solver = z3.Solver()

    prove(
        "any local flag forces global rejection", solver, z3.Implies(z3.Or(*flags), z3.Not(accept))
    )
    prove("nonfinite evidence forces rejection", solver, z3.Implies(z3.Not(finite), z3.Not(accept)))
    prove(
        "failed domain policy forces rejection",
        solver,
        z3.Implies(z3.Not(policy_pass), z3.Not(accept)),
    )
    prove(
        "finite, policy-passing, unflagged evidence is accepted",
        solver,
        z3.Implies(z3.And(finite, policy_pass, *[z3.Not(flag) for flag in flags]), accept),
    )


def check_selection_and_freshness() -> None:
    in_batch = z3.Bool("in_batch")
    accepted = z3.Bool("accepted")
    selected = z3.And(in_batch, accepted)
    quarantined = z3.And(in_batch, z3.Not(accepted))
    solver = z3.Solver()
    prove("training selection implies acceptance", solver, z3.Implies(selected, accepted))
    prove(
        "training and quarantine selections are disjoint",
        solver,
        z3.Not(z3.And(selected, quarantined)),
    )

    audit_generation = z3.Int("audit_generation")
    expected_generation = z3.Int("expected_generation")
    already_committed = z3.Bool("already_committed")
    commit = z3.And(audit_generation == expected_generation, z3.Not(already_committed))
    freshness_solver = z3.Solver()
    prove(
        "stale generation cannot commit",
        freshness_solver,
        z3.Implies(audit_generation != expected_generation, z3.Not(commit)),
    )
    prove(
        "consumed generation cannot commit twice",
        freshness_solver,
        z3.Implies(already_committed, z3.Not(commit)),
    )


def check_observation_only_limit() -> None:
    world = z3.IntSort()
    observe = z3.Function("observe", world, world)
    safe = z3.Function("safe", world, z3.BoolSort())
    detector = z3.Function("detector", world, z3.BoolSort())
    safe_world, unsafe_world = z3.Ints("safe_world unsafe_world")
    solver = z3.Solver()
    solver.add(
        safe_world != unsafe_world,
        observe(safe_world) == observe(unsafe_world),
        safe(safe_world),
        z3.Not(safe(unsafe_world)),
        # Two ground instances of any purportedly perfect detector.
        detector(observe(safe_world)) == safe(safe_world),
        detector(observe(unsafe_world)) == safe(unsafe_world),
    )
    result = solver.check()
    assert result == z3.unsat, f"observation limitation: expected UNSAT, got {result}"
    description = "equal observations with opposite safety exclude a perfect detector"
    UNSAT_CHECKS.append(description)
    print(f"PASS UNSAT: {description}")


def check_rl_shield_and_bounded_update() -> None:
    action = z3.IntSort()
    safe_action = z3.Function("safe_action", action, z3.BoolSort())
    proposed, fallback = z3.Ints("proposed fallback")
    shielded = z3.If(safe_action(proposed), proposed, fallback)
    solver = z3.Solver()
    solver.add(safe_action(fallback))
    prove("shield output is safe when fallback is safe", solver, safe_action(shielded))
    prove(
        "shield is transparent to a safe proposal",
        solver,
        z3.Implies(safe_action(proposed), shielded == proposed),
    )

    accepted = z3.Bool("update_accepted")
    raw_update, bound = z3.Reals("raw_update bound")
    update_allowed = z3.And(accepted, raw_update * raw_update <= bound * bound)
    applied_update = z3.If(update_allowed, raw_update, z3.RealVal(0))
    update_solver = z3.Solver()
    update_solver.add(bound >= 0)
    prove(
        "applied update respects the squared norm bound",
        update_solver,
        applied_update * applied_update <= bound * bound,
    )
    prove(
        "rejected update is exactly zero over real arithmetic",
        update_solver,
        z3.Implies(z3.Not(accepted), applied_update == 0),
    )


def check_ieee754_boundary() -> None:
    float32 = z3.Float32()
    value = z3.FP("value", float32)
    zero = z3.FPVal(0.0, float32)
    product = z3.fpMul(z3.RNE(), zero, value)
    product_is_not_zero = z3.Not(z3.fpEQ(product, zero))

    counterexample_solver = z3.Solver()
    counterexample_solver.add(product_is_not_zero)
    expect_sat("IEEE multiplication by zero has a non-zero counterexample", counterexample_solver)

    finite_solver = z3.Solver()
    finite_solver.add(
        z3.Not(z3.fpIsNaN(value)),
        z3.Not(z3.fpIsInf(value)),
        product_is_not_zero,
    )
    result = finite_solver.check()
    assert result == z3.unsat, f"finite IEEE zero product: expected UNSAT, got {result}"
    description = "finite IEEE values have zero product with a zero multiplier"
    UNSAT_CHECKS.append(description)
    print(f"PASS UNSAT: {description}")


def check_signed_influence() -> None:
    for sign, description in (
        (-1, "influence can be negative"),
        (1, "influence can be positive"),
    ):
        query_gradient, sample_gradient, inverse_curvature = z3.Reals(
            f"query_gradient_{sign} sample_gradient_{sign} inverse_curvature_{sign}"
        )
        influence = -query_gradient * inverse_curvature * sample_gradient
        solver = z3.Solver()
        solver.add(inverse_curvature > 0)
        solver.add(influence < 0 if sign < 0 else influence > 0)
        expect_sat(description, solver)


def main() -> None:
    check_global_gate()
    check_selection_and_freshness()
    check_observation_only_limit()
    check_rl_shield_and_bounded_update()
    check_ieee754_boundary()
    check_signed_influence()
    assert len(UNSAT_CHECKS) == 14, len(UNSAT_CHECKS)
    assert len(SAT_CHECKS) == 3, len(SAT_CHECKS)
    print(
        "Completed 17 scoped checks: 14 expected UNSAT properties and " "3 expected SAT witnesses."
    )


if __name__ == "__main__":
    main()
