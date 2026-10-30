(* Scoped contracts for ground-truth Transformer component tracing. *)

From Stdlib Require Import QArith Qfield Ring Lra.

Open Scope Q_scope.

Theorem integrated_readout_allocation_3 :
  forall j a b c : Q, j * a + j * b + j * c == j * (a + b + c).
Proof.
  intros. ring.
Qed.

Theorem integrated_readout_residual_conservation :
  forall actual baseline explained residual : Q,
    residual == actual - baseline - explained ->
    actual == baseline + explained + residual.
Proof.
  intros actual baseline explained residual H. rewrite H. ring.
Qed.

Theorem linear_zero_ablation_total_effect :
  forall j a bias : Q, (j * a + bias) - bias == j * a.
Proof.
  intros. ring.
Qed.

Theorem constant_jacobian_path_is_exact :
  forall j residual baseline : Q,
    baseline + j * residual == (j * residual + baseline).
Proof.
  intros. ring.
Qed.

Theorem modular_arithmetic_ground_truth_example :
  ((2 + 3) * 4 + 5 == 2 * 11 + 3)%Q.
Proof.
  vm_compute. reflexivity.
Qed.

Theorem nonlinear_ablation_differs_from_local_prediction :
  ~ ((2 * 2 * 2 - 1 * 1 * 1) == 3 * (1 * 1) * (2 - 1)).
Proof.
  vm_compute. discriminate.
Qed.

Print Assumptions integrated_readout_allocation_3.
Print Assumptions integrated_readout_residual_conservation.
Print Assumptions linear_zero_ablation_total_effect.
Print Assumptions constant_jacobian_path_is_exact.
Print Assumptions modular_arithmetic_ground_truth_example.
Print Assumptions nonlinear_ablation_differs_from_local_prediction.
