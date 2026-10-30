(** Scoped foundations for the CMAE validation program.

    These are small algebraic/control/scientific witnesses, not proofs of trained-model
    detection, universal attribution accuracy, or safe deployment.
*)

From Stdlib Require Import QArith Qfield Ring Lra.

Open Scope Q_scope.

Theorem linear_downstream_jacobian_exact_2x2 :
  forall a b c d x1 x2 delta1 delta2 : Q,
    a * (x1 + delta1) + b * (x2 + delta2) ==
    (a * x1 + b * x2) + (a * delta1 + b * delta2) /\
    c * (x1 + delta1) + d * (x2 + delta2) ==
    (c * x1 + d * x2) + (c * delta1 + d * delta2).
Proof.
  intros. split; ring.
Qed.

Theorem causal_average_has_exact_linear_residual :
  forall x1 x2 delta1 delta2 : Q,
    ((x1 + delta1) + (x2 + delta2)) * (1 # 2) ==
    (x1 + x2) * (1 # 2) + (delta1 + delta2) * (1 # 2).
Proof.
  intros. ring.
Qed.

Theorem removal_attribution_sign_matches_loss_gradient :
  forall query_gradient output_gradient query_feature train_feature : Q,
    ~ (output_gradient == 0) ->
    - (query_feature * (- output_gradient * train_feature)) / output_gradient ==
    query_feature * train_feature.
Proof.
  intros. field. assumption.
Qed.

Theorem group_surrogate_scores_are_additive :
  forall first second third : Q,
    (first + second) + third == first + (second + third).
Proof.
  intros. ring.
Qed.

Definition dual_update (lambda eta cost threshold : Q) : Q :=
  if Qle_bool 0 (lambda + eta * (cost - threshold))
  then lambda + eta * (cost - threshold)
  else 0.

Theorem expected_feasibility_does_not_excite_dual_or_bound_tail :
  (0 + 0 + 0 + 3) * (1 # 4) <= 1 /\
  3 > 1 /\
  dual_update 0 1 ((0 + 0 + 0 + 3) * (1 # 4)) 1 = 0.
Proof.
  split.
  - apply Qle_bool_imp_le. vm_compute. reflexivity.
  - split.
    + apply Qlt_alt. vm_compute. reflexivity.
    + vm_compute. reflexivity.
Qed.

Definition commit_allowed (elapsed budget : Q) (healthy : bool) : bool :=
  andb healthy (Qle_bool elapsed budget).

Theorem timeout_blocks_commitment :
  commit_allowed 2 1 true = false.
Proof.
  vm_compute. reflexivity.
Qed.

Theorem equal_mass_collision_swaps_velocities_and_conserves_invariants :
  forall first second : Q,
    let new_first := ((1 - 1) / (1 + 1)) * first +
                     (2 * 1 / (1 + 1)) * second in
    let new_second := (2 * 1 / (1 + 1)) * first +
                      ((1 - 1) / (1 + 1)) * second in
    new_first == second /\ new_second == first /\
    1 * new_first + 1 * new_second == 1 * first + 1 * second /\
    1 * new_first * new_first + 1 * new_second * new_second ==
    1 * first * first + 1 * second * second.
Proof.
  intros first second. cbv zeta.
  repeat split; field.
Qed.

Theorem kinetic_energy_dimension_exponents :
  (1 + 0 = 1)%Z /\ (0 + 2 * 1 = 2)%Z /\ (0 + 2 * (-1) = -2)%Z.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Print Assumptions linear_downstream_jacobian_exact_2x2.
Print Assumptions causal_average_has_exact_linear_residual.
Print Assumptions removal_attribution_sign_matches_loss_gradient.
Print Assumptions group_surrogate_scores_are_additive.
Print Assumptions expected_feasibility_does_not_excite_dual_or_bound_tail.
Print Assumptions timeout_blocks_commitment.
Print Assumptions equal_mass_collision_swaps_velocities_and_conserves_invariants.
Print Assumptions kinetic_energy_dimension_exponents.
