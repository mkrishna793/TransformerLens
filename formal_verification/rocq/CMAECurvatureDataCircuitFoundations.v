(* Scoped contracts for EK-FAC data-to-feature influence and accepted-only repair. *)

From Stdlib Require Import QArith Qfield Ring Lqa ZArith.

Open Scope Q_scope.

Theorem scalar_kfe_rank_one_projection :
  forall qs qa s a : Q,
    (qs * s) * (qa * a) == qs * (s * a) * qa.
Proof.
  intros. ring.
Qed.

Theorem kfe_position_sum_projection :
  forall qs qa s1 a1 s2 a2 : Q,
    qs * (s1 * a1 + s2 * a2) * qa ==
      (qs * s1) * (qa * a1) + (qs * s2) * (qa * a2).
Proof.
  intros. ring.
Qed.

Theorem corrected_second_moment_nonnegative :
  forall g1 g2 : Q, 0 <= (g1 * g1 + g2 * g2) / 2.
Proof.
  intros. change (0 <= (g1 * g1 + g2 * g2) * (1 # 2)). nra.
Qed.

Theorem damped_curvature_denominator_positive :
  forall moment damping : Q,
    0 <= moment -> 0 < damping -> 0 < moment + damping.
Proof.
  intros. nra.
Qed.

Theorem influence_contrast_linearity :
  forall poison_gradient clean_gradient inverse_hessian_loss_gradient : Q,
    let poison_influence := -poison_gradient * inverse_hessian_loss_gradient in
    let clean_influence := -clean_gradient * inverse_hessian_loss_gradient in
    poison_influence - clean_influence ==
      -(poison_gradient - clean_gradient) * inverse_hessian_loss_gradient.
Proof.
  intros. unfold poison_influence, clean_influence. ring.
Qed.

Theorem joint_feature_path_residual_conservation :
  forall readout_weight feature_influence residual total : Q,
    total == readout_weight * feature_influence + residual ->
    total - readout_weight * feature_influence == residual.
Proof.
  intros readout_weight feature_influence residual total H.
  rewrite H. ring.
Qed.

Theorem concrete_targeted_backdoor_label :
  ((5 + 1) mod 11 = 6)%Z /\ ((5 + 1) mod 11 <> 5)%Z.
Proof.
  split.
  - vm_compute. reflexivity.
  - vm_compute. discriminate.
Qed.

Print Assumptions scalar_kfe_rank_one_projection.
Print Assumptions kfe_position_sum_projection.
Print Assumptions corrected_second_moment_nonnegative.
Print Assumptions damped_curvature_denominator_positive.
Print Assumptions influence_contrast_linearity.
Print Assumptions joint_feature_path_residual_conservation.
Print Assumptions concrete_targeted_backdoor_label.
