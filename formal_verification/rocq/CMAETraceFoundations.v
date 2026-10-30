(** Scoped foundations for Milestone-three/four trace and adaptive-repair contracts. *)

From Stdlib Require Import QArith Qfield Ring Lra Qabs.

Open Scope Q_scope.

Theorem trace_chain_rule_2x2 :
  forall a b c d e f g h x1 x2 : Q,
    e * (a * x1 + b * x2) + f * (c * x1 + d * x2) ==
    (e * a + f * c) * x1 + (e * b + f * d) * x2 /\
    g * (a * x1 + b * x2) + h * (c * x1 + d * x2) ==
    (g * a + h * c) * x1 + (g * b + h * d) * x2.
Proof.
  intros. split; ring.
Qed.

Theorem trace_residual_conservation :
  forall actual explained residual : Q,
    residual == actual - explained -> actual == explained + residual.
Proof.
  intros actual explained residual H. rewrite H. ring.
Qed.

Theorem hidden_ablation_edge_is_exact_for_linear_readout :
  forall w1 w2 z1 z2 bias : Q,
    (w1 * z1 + w2 * z2 + bias) - (w2 * z2 + bias) == w1 * z1.
Proof.
  intros. ring.
Qed.

Theorem linear_group_attribution_is_additive :
  forall a x y : Q, a * (x + y) == a * x + a * y.
Proof.
  intros. ring.
Qed.

Definition project_delta (epsilon delta : Q) : Q :=
  if Qle_bool delta (-epsilon) then -epsilon
  else if Qle_bool epsilon delta then epsilon else delta.

Theorem projection_endpoints_stay_in_unit_ball :
  Qabs (project_delta 1 2) <= 1 /\
  Qabs (project_delta 1 (-2)) <= 1 /\
  Qabs (project_delta 1 (1 # 2)) <= 1.
Proof.
  repeat split; apply Qle_bool_imp_le; vm_compute; reflexivity.
Qed.

Definition repair_gate_spec
  (clean_drop original_before original_after adaptive_before adaptive_after : Q) : Prop :=
  clean_drop <= (1 # 50) /\
  original_after < original_before /\
  adaptive_after < adaptive_before /\
  original_after <= (1 # 2) /\
  adaptive_after <= (1 # 2).

Theorem repair_gate_requires_all_conditions :
  forall clean_drop original_before original_after adaptive_before adaptive_after : Q,
    repair_gate_spec clean_drop original_before original_after adaptive_before adaptive_after ->
    clean_drop <= (1 # 50) /\
    original_after < original_before /\
    adaptive_after < adaptive_before /\
    original_after <= (1 # 2) /\
    adaptive_after <= (1 # 2).
Proof.
  intros. exact H.
Qed.

Theorem adaptive_non_improvement_blocks_repair :
  ~ repair_gate_spec (1 # 100) 1 (1 # 2) (3 # 4) (3 # 4).
Proof.
  unfold repair_gate_spec. intros H.
  destruct H as [Hclean [Horiginal [Hadaptive [Horiginal_ceiling Hadaptive_ceiling]]]].
  exact (Qlt_irrefl _ Hadaptive).
Qed.

Theorem high_residual_adaptive_risk_blocks_repair :
  ~ repair_gate_spec (1 # 100) 1 (1 # 4) 1 (3 # 4).
Proof.
  unfold repair_gate_spec. intros H.
  destruct H as [Hclean [Horiginal [Hadaptive [Horiginal_ceiling Hadaptive_ceiling]]]].
  rewrite Qle_alt in Hadaptive_ceiling. vm_compute in Hadaptive_ceiling.
  exact (Hadaptive_ceiling eq_refl).
Qed.

Print Assumptions trace_chain_rule_2x2.
Print Assumptions trace_residual_conservation.
Print Assumptions hidden_ablation_edge_is_exact_for_linear_readout.
Print Assumptions linear_group_attribution_is_additive.
Print Assumptions projection_endpoints_stay_in_unit_ball.
Print Assumptions repair_gate_requires_all_conditions.
Print Assumptions adaptive_non_improvement_blocks_repair.
Print Assumptions high_residual_adaptive_risk_blocks_repair.
