(* Scoped contracts for calibration-controlled training and Jacobian lenses. *)

From Stdlib Require Import QArith Qfield Ring Lqa.

Open Scope Q_scope.

Theorem single_expert_jacobian_readout :
  forall j h b : Q, 1 * (j * h + b) + 0 == j * h + b.
Proof.
  intros. ring.
Qed.

Theorem equal_experts_with_unit_weight_sum :
  forall w1 w2 j h : Q,
    w1 + w2 == 1 -> w1 * (j * h) + w2 * (j * h) == j * h.
Proof.
  intros. nra.
Qed.

Theorem lens_residual_conservation :
  forall actual predicted residual : Q,
    residual == actual - predicted -> actual == predicted + residual.
Proof.
  intros actual predicted residual H. rewrite H. ring.
Qed.

Theorem nonnegative_auxiliary_objective :
  forall final_loss auxiliary_loss weight : Q,
    0 <= final_loss -> 0 <= auxiliary_loss -> 0 <= weight ->
    0 <= final_loss + weight * auxiliary_loss.
Proof.
  intros. nra.
Qed.

Theorem first_expert_can_prefer_first_token :
  1 * 1 + 0 * 0 > 1 * 0 + 0 * 1.
Proof.
  vm_compute. reflexivity.
Qed.

Theorem second_expert_can_reverse_preference :
  0 * 1 + 1 * 0 < 0 * 0 + 1 * 1.
Proof.
  vm_compute. reflexivity.
Qed.

Print Assumptions single_expert_jacobian_readout.
Print Assumptions equal_experts_with_unit_weight_sum.
Print Assumptions lens_residual_conservation.
Print Assumptions nonnegative_auxiliary_objective.
Print Assumptions first_expert_can_prefer_first_token.
Print Assumptions second_expert_can_reverse_preference.
