(** Scoped algebraic boundaries for the experimental CMAE prototype.

    These results use concrete rational arithmetic. They do not establish semantic
    feature correctness, deep-model influence accuracy, or universal poison detection.
*)

From Stdlib Require Import QArith Qfield Lra Ring.

Open Scope Q_scope.

Definition reluQ (x : Q) : Q := if Qle_bool 0 x then x else 0.

Theorem reluQ_erases_negative_distinction :
  reluQ (-1) = reluQ (-2) /\ (-1) <> (-2).
Proof.
  split.
  - vm_compute. reflexivity.
  - intros H. discriminate H.
Qed.

Definition at_most_one_nonzero (v : Q * Q) : Prop :=
  fst v = 0 \/ snd v = 0.

Theorem sparse_one_set_has_a_nonconvex_witness :
  at_most_one_nonzero (1, 0) /\
  at_most_one_nonzero (0, 1) /\
  ~ at_most_one_nonzero (1 # 2, 1 # 2).
Proof.
  split.
  - right. reflexivity.
  - split.
    + left. reflexivity.
    + unfold at_most_one_nonzero. simpl. intros [H | H]; discriminate H.
Qed.

Definition top1_nonneg (x y : Q) : Q * Q :=
  if Qle_bool y x then (x, 0) else (0, y).

Theorem top1_can_drop_a_nonzero_coordinate :
  snd (top1_nonneg 1 (1 # 10)) = 0 /\ (1 # 10) <> 0.
Proof.
  split.
  - vm_compute. reflexivity.
  - intros H. discriminate H.
Qed.

Theorem multiplicative_parametric_lens_composition :
  forall a b x dy : Q,
    (dy * b) * a == dy * (a * b) /\
    (dy * b) * x == dy * b * x /\
    dy * (a * x) == dy * a * x.
Proof.
  intros a b x dy.
  repeat split; ring.
Qed.

Theorem rank_one_diagonal_kronecker_identity_2x2 :
  forall ab0 ab1 aq0 aq1 sb0 sb1 sq0 sq1 pa0 pa1 ps0 ps1 : Q,
    (pa0 * ab0 * aq0 + pa1 * ab1 * aq1) *
    (ps0 * sb0 * sq0 + ps1 * sb1 * sq1)
    ==
    pa0 * ps0 * ab0 * aq0 * sb0 * sq0 +
    pa0 * ps1 * ab0 * aq0 * sb1 * sq1 +
    pa1 * ps0 * ab1 * aq1 * sb0 * sq0 +
    pa1 * ps1 * ab1 * aq1 * sb1 * sq1.
Proof.
  intros. ring.
Qed.

Theorem standard_influence_sign :
  forall quadratic_form : Q, 0 < quadratic_form -> - quadratic_form < 0.
Proof.
  intros quadratic_form H.
  exact (Qopp_lt_compat 0 quadratic_form H).
Qed.

Print Assumptions reluQ_erases_negative_distinction.
Print Assumptions sparse_one_set_has_a_nonconvex_witness.
Print Assumptions top1_can_drop_a_nonzero_coordinate.
Print Assumptions multiplicative_parametric_lens_composition.
Print Assumptions rank_one_diagonal_kronecker_identity_2x2.
Print Assumptions standard_influence_sign.
