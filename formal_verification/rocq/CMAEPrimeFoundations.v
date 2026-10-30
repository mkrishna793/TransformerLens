(** Scoped arithmetic-topology and spectral-trace checks for CMAE.

    The examples are exact small witnesses. They do not assert that arbitrary
    neural activations have torsion, nor that prime traces are universally useless.
*)

From Stdlib Require Import ZArith QArith Qfield Ring.

Section IntegerWitnesses.
Open Scope Z_scope.

Definition diag_trace4 (a b c d : Z) (n : nat) : Z :=
  a ^ Z.of_nat n + b ^ Z.of_nat n + c ^ Z.of_nat n + d ^ Z.of_nat n.

Theorem explicit_prime_trace_agreement :
  diag_trace4 1 (-1) 8 (-8) 2 = diag_trace4 4 (-4) 7 (-7) 2 /\
  diag_trace4 1 (-1) 8 (-8) 3 = diag_trace4 4 (-4) 7 (-7) 3 /\
  diag_trace4 1 (-1) 8 (-8) 5 = diag_trace4 4 (-4) 7 (-7) 5 /\
  diag_trace4 1 (-1) 8 (-8) 7 = diag_trace4 4 (-4) 7 (-7) 7.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Theorem fourth_trace_distinguishes_explicit_spectra :
  diag_trace4 1 (-1) 8 (-8) 4 <> diag_trace4 4 (-4) 7 (-7) 4.
Proof.
  vm_compute. intros H. discriminate H.
Qed.

Definition rank_1x1_mod (x p : Z) : Z :=
  if (x mod p =? 0)%Z then 0 else 1.

Theorem explicit_prime_torsion_ranks :
  rank_1x1_mod 2 2 = 0 /\
  rank_1x1_mod 2 3 = 1 /\
  rank_1x1_mod 3 3 = 0 /\
  rank_1x1_mod 3 2 = 1.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Definition rp2_betti1_mod (p : Z) : Z := 1 - rank_1x1_mod 2 p.
Definition rp2_betti2_mod (p : Z) : Z := 1 - rank_1x1_mod 2 p.

Theorem rp2_cw_homology_changes_at_prime_two :
  rp2_betti1_mod 2 = 1 /\ rp2_betti2_mod 2 = 1 /\
  rp2_betti1_mod 3 = 0 /\ rp2_betti2_mod 3 = 0.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

End IntegerWitnesses.

Section NewtonIdentities.
Open Scope Q_scope.

Theorem newton_third_elementary_identity :
  forall p1 p2 p3 : Q,
    (p1 * (p1 * p1 - p2) / 2 - p1 * p2 + p3) / 3 ==
    (p1 * p1 * p1 - 3 * p1 * p2 + 2 * p3) / 6.
Proof.
  intros. field.
Qed.

Theorem newton_fourth_elementary_identity :
  forall p1 p2 p3 p4 : Q,
    let e2 := (p1 * p1 - p2) * (1 # 2) in
    let e3 := (p1 * p1 * p1 - 3 * p1 * p2 + 2 * p3) * (1 # 6) in
    (e3 * p1 - e2 * p2 + p1 * p3 - p4) * (1 # 4) ==
    (p1 * p1 * p1 * p1 - 6 * p1 * p1 * p2 +
     3 * p2 * p2 + 8 * p1 * p3 - 6 * p4) * (1 # 24).
Proof.
  intros p1 p2 p3 p4. cbv zeta. ring.
Qed.

End NewtonIdentities.

Print Assumptions explicit_prime_trace_agreement.
Print Assumptions fourth_trace_distinguishes_explicit_spectra.
Print Assumptions explicit_prime_torsion_ranks.
Print Assumptions rp2_cw_homology_changes_at_prime_two.
Print Assumptions newton_third_elementary_identity.
Print Assumptions newton_fourth_elementary_identity.
