(* Scoped contracts for attention-path and counterfactual residual interventions. *)

From Stdlib Require Import QArith Qfield Ring Lqa ZArith.

Open Scope Q_scope.

Theorem one_hot_first_attention_value :
  forall a b : Q, 1 * a + 0 * b == a.
Proof.
  intros. ring.
Qed.

Theorem one_hot_second_attention_value :
  forall a b : Q, 0 * a + 1 * b == b.
Proof.
  intros. ring.
Qed.

Theorem convex_attention_lower_bound :
  forall w1 w2 a b : Q,
    w1 + w2 == 1 -> 0 <= w1 -> 0 <= w2 -> a <= b ->
    a <= w1 * a + w2 * b.
Proof.
  intros. nra.
Qed.

Theorem convex_attention_upper_bound :
  forall w1 w2 a b : Q,
    w1 + w2 == 1 -> 0 <= w1 -> 0 <= w2 -> a <= b ->
    w1 * a + w2 * b <= b.
Proof.
  intros. nra.
Qed.

Theorem intervention_residual_conservation :
  forall actual predicted residual : Q,
    residual == actual - predicted -> actual == predicted + residual.
Proof.
  intros actual predicted residual H. rewrite H. ring.
Qed.

Theorem concrete_modular_counterfactual :
  ((2 * 4 + 5) mod 11 = 2)%Z.
Proof.
  vm_compute. reflexivity.
Qed.

Theorem counterfactual_differs_from_original :
  ((5 * 4 + 5) mod 11 <> (2 * 4 + 5) mod 11)%Z.
Proof.
  vm_compute. discriminate.
Qed.

Print Assumptions one_hot_first_attention_value.
Print Assumptions one_hot_second_attention_value.
Print Assumptions convex_attention_lower_bound.
Print Assumptions convex_attention_upper_bound.
Print Assumptions intervention_residual_conservation.
Print Assumptions concrete_modular_counterfactual.
Print Assumptions counterfactual_differs_from_original.
