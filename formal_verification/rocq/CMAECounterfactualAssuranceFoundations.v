(* Scoped fixed-weight counterfactual and fail-closed assurance contracts. *)

From Stdlib Require Import QArith Qfield Ring Lqa Bool.

Open Scope Q_scope.

Theorem fixed_denominator_removal_two :
  forall lost retained denominator : Q,
    ~ (denominator == 0) ->
    (lost + retained) / denominator - retained / denominator == lost / denominator.
Proof.
  intros lost retained denominator H. field. exact H.
Qed.

Theorem group_influence_additivity :
  forall query_gradient inverse_curvature g1 g2 : Q,
    let i1 := -query_gradient * inverse_curvature * g1 in
    let i2 := -query_gradient * inverse_curvature * g2 in
    i1 + i2 == -query_gradient * inverse_curvature * (g1 + g2).
Proof.
  intros. unfold i1, i2. ring.
Qed.

Theorem ggn_quadratic_nonnegative :
  forall curvature jacobian vector : Q,
    0 <= curvature -> 0 <= curvature * (jacobian * vector) * (jacobian * vector).
Proof.
  intros curvature jacobian vector H.
  assert (curvature * (jacobian * vector) * (jacobian * vector) ==
    curvature * ((jacobian * vector) * (jacobian * vector))) as Hrewrite by ring.
  rewrite Hrewrite.
  apply Qmult_le_0_compat.
  - exact H.
  - nra.
Qed.

Theorem damped_ggn_quadratic_positive :
  forall curvature damping vector : Q,
    0 <= curvature -> 0 < damping -> 0 < vector * vector ->
    0 < (curvature + damping) * (vector * vector).
Proof.
  intros curvature damping vector Hcurvature Hdamping Hsquare.
  apply Qmult_lt_0_compat.
  - nra.
  - exact Hsquare.
Qed.

Theorem upweight_influence_is_full_minus_reduced_first_order :
  forall query_gradient inverse_curvature sample_gradient : Q,
    let parameter_shift := -inverse_curvature * sample_gradient in
    let query_shift := query_gradient * parameter_shift in
    query_shift == -query_gradient * inverse_curvature * sample_gradient.
Proof.
  intros. unfold query_shift, parameter_shift. ring.
Qed.

Theorem label_quality_suspicion_order :
  forall quality_low quality_high : Q,
    0 <= quality_low -> quality_low <= quality_high -> quality_high <= 1 ->
    0 <= 1 - quality_high /\ 1 - quality_high <= 1 - quality_low /\ 1 - quality_low <= 1.
Proof.
  intros. repeat split; nra.
Qed.

Theorem accepted_removed_budget_conservation :
  forall accepted removed total : Q,
    accepted + removed == total -> accepted == total - removed.
Proof.
  intros accepted removed total H. rewrite <- H. ring.
Qed.

Theorem counterfactual_failure_blocks_prevention_gate :
  forall organism_valid counterfactual_valid detection_valid repair_valid : bool,
    counterfactual_valid = false ->
    (organism_valid && counterfactual_valid && detection_valid && repair_valid)%bool = false.
Proof.
  intros. rewrite H. destruct organism_valid; reflexivity.
Qed.

Theorem tanh_binary_full_hessian_quadratic :
  forall loss_second loss_first head hidden_first hidden_second hidden x u v : Q,
    let output_first := head * hidden_first * x in
    (loss_second * output_first * output_first + loss_first * head * hidden_second * x * x) * u * u
    + 2 * (loss_second * output_first * hidden + loss_first * hidden_first * x) * u * v
    + (loss_second * hidden * hidden) * v * v
    ==
    loss_second * (output_first * u + hidden * v) * (output_first * u + hidden * v)
    + loss_first * (head * hidden_second * x * x * u * u + 2 * hidden_first * x * u * v).
Proof.
  intros. unfold output_first. ring.
Qed.

Print Assumptions tanh_binary_full_hessian_quadratic.
Print Assumptions fixed_denominator_removal_two.
Print Assumptions group_influence_additivity.
Print Assumptions ggn_quadratic_nonnegative.
Print Assumptions damped_ggn_quadratic_positive.
Print Assumptions upweight_influence_is_full_minus_reduced_first_order.
Print Assumptions label_quality_suspicion_order.
Print Assumptions accepted_removed_budget_conservation.
Print Assumptions counterfactual_failure_blocks_prevention_gate.
