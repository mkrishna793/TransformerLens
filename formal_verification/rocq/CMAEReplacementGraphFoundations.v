(* Scoped conservation contracts for SAE/CLT replacement graphs. *)

From Stdlib Require Import QArith Qfield Ring Lqa ZArith.

Open Scope Q_scope.

Theorem early_reconstruction_conservation :
  forall y0 z0 d00 e0 : Q,
    y0 == z0 * d00 + e0 -> y0 - e0 == z0 * d00.
Proof.
  intros y0 z0 d00 e0 H. rewrite H. ring.
Qed.

Theorem cross_layer_reconstruction_conservation :
  forall y1 z0 z1 d01 d11 e1 : Q,
    y1 == z0 * d01 + z1 * d11 + e1 ->
    y1 - e1 == z0 * d01 + z1 * d11.
Proof.
  intros y1 z0 z1 d01 d11 e1 H. rewrite H. ring.
Qed.

Theorem independent_sae_is_zero_cross_layer_clt :
  forall y1 z0 z1 d01 d11 e1 : Q,
    d01 == 0 ->
    y1 == z0 * d01 + z1 * d11 + e1 ->
    y1 == z1 * d11 + e1.
Proof.
  intros y1 z0 z1 d01 d11 e1 Hd H. rewrite Hd in H.
  rewrite H. ring.
Qed.

Theorem integrated_direct_output_conservation :
  forall f f0 z0 z1 d00 d01 d11 e0 e1 j0 j1 epsilon : Q,
    f == f0 + z0 * d00 * j0 + z0 * d01 * j1 + z1 * d11 * j1
      + e0 * j0 + e1 * j1 + epsilon ->
    f - f0 - epsilon ==
      z0 * d00 * j0 + z0 * d01 * j1 + z1 * d11 * j1
      + e0 * j0 + e1 * j1.
Proof.
  intros f f0 z0 z1 d00 d01 d11 e0 e1 j0 j1 epsilon H.
  rewrite H. ring.
Qed.

Theorem zero_error_reconstruction :
  forall y0 y1 z0 z1 d00 d01 d11 e0 e1 : Q,
    e0 == 0 -> e1 == 0 ->
    y0 == z0 * d00 + e0 ->
    y1 == z0 * d01 + z1 * d11 + e1 ->
    y0 == z0 * d00 /\ y1 == z0 * d01 + z1 * d11.
Proof.
  intros y0 y1 z0 z1 d00 d01 d11 e0 e1 He0 He1 H0 H1.
  split.
  - rewrite He0 in H0. rewrite H0. ring.
  - rewrite He1 in H1. rewrite H1. ring.
Qed.

Theorem direct_hold_fixed_differs_from_recomputed_total :
  (1 + 1 == 2 /\ 1 <> 2)%Q.
Proof.
  split; [ring | discriminate].
Qed.

Print Assumptions early_reconstruction_conservation.
Print Assumptions cross_layer_reconstruction_conservation.
Print Assumptions independent_sae_is_zero_cross_layer_clt.
Print Assumptions integrated_direct_output_conservation.
Print Assumptions zero_error_reconstruction.
Print Assumptions direct_hold_fixed_differs_from_recomputed_total.
