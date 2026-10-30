(** Architecture-independent research kernel for training control.

    This file deliberately proves control-flow properties only.  It does not
    claim that statistical detectors recognize every unsafe sample, that an
    SAE/NLA explanation is faithful, or that an influence estimate is exact.
*)

From Stdlib Require Import Bool.Bool.
From Stdlib Require Import Lists.List.
From Stdlib Require Import Arith.PeanoNat.
Import ListNotations.

Module TrainingSafetyKernel.

  (** Hard evidence is combined globally before a training pass is allowed.
      [local_flags] can contain flags from any layer, detector, or domain rule. *)
  Record Evidence : Type := mkEvidence {
    finite_values : bool;
    policy_pass : bool;
    local_flags : list bool
  }.

  Definition any_flag (flags : list bool) : bool :=
    existsb (fun flag : bool => flag) flags.

  Definition no_flags (flags : list bool) : bool :=
    forallb (fun flag : bool => negb flag) flags.

  Definition globally_accept (evidence : Evidence) : bool :=
    finite_values evidence && policy_pass evidence &&
      no_flags (local_flags evidence).

  Lemma any_flag_disproves_no_flags : forall flags : list bool,
    any_flag flags = true -> no_flags flags = false.
  Proof.
    induction flags as [| flag flags IH]; simpl.
    - discriminate.
    - destruct flag; simpl.
      + reflexivity.
      + apply IH.
  Qed.

  Theorem local_flag_forces_global_rejection : forall evidence : Evidence,
    any_flag (local_flags evidence) = true ->
    globally_accept evidence = false.
  Proof.
    intros evidence Hflag.
    unfold globally_accept.
    rewrite (any_flag_disproves_no_flags _ Hflag).
    destruct (finite_values evidence), (policy_pass evidence); reflexivity.
  Qed.

  Theorem nonfinite_values_force_rejection : forall evidence : Evidence,
    finite_values evidence = false -> globally_accept evidence = false.
  Proof.
    intros evidence Hfinite.
    unfold globally_accept. rewrite Hfinite. reflexivity.
  Qed.

  Theorem failed_policy_forces_rejection : forall evidence : Evidence,
    policy_pass evidence = false -> globally_accept evidence = false.
  Proof.
    intros evidence Hpolicy.
    unfold globally_accept. rewrite Hpolicy.
    destruct (finite_values evidence); reflexivity.
  Qed.

  (** The training driver receives a fresh collection containing only accepted
      items.  Rejected items never enter its loss or backward computation. *)
  Definition training_selection (batch : list Evidence) : list Evidence :=
    filter globally_accept batch.

  Definition quarantine_selection (batch : list Evidence) : list Evidence :=
    filter (fun evidence : Evidence => negb (globally_accept evidence)) batch.

  Theorem training_selection_only_accepts : forall batch evidence,
    In evidence (training_selection batch) -> globally_accept evidence = true.
  Proof.
    intros batch evidence Hselected.
    unfold training_selection in Hselected.
    apply filter_In in Hselected. destruct Hselected as [_ Haccept].
    exact Haccept.
  Qed.

  Theorem training_and_quarantine_are_disjoint : forall batch evidence,
    In evidence (training_selection batch) ->
    In evidence (quarantine_selection batch) -> False.
  Proof.
    intros batch evidence Htraining Hquarantine.
    apply training_selection_only_accepts in Htraining.
    unfold quarantine_selection in Hquarantine.
    apply filter_In in Hquarantine. destruct Hquarantine as [_ Hreject].
    rewrite Htraining in Hreject. discriminate.
  Qed.

  (** Commit authorization is tied to the expected fresh generation and can be
      consumed only once.  A production implementation must update this state
      atomically with its ledger write. *)
  Definition commit_allowed
      (audit_generation expected_generation : nat)
      (already_committed : bool) : bool :=
    Nat.eqb audit_generation expected_generation && negb already_committed.

  Theorem stale_generation_cannot_commit : forall audit expected committed,
    audit <> expected -> commit_allowed audit expected committed = false.
  Proof.
    intros audit expected committed Hfresh.
    unfold commit_allowed.
    destruct (Nat.eqb audit expected) eqn:Heq.
    - apply Nat.eqb_eq in Heq. contradiction.
    - reflexivity.
  Qed.

  Theorem consumed_generation_cannot_commit : forall generation,
    commit_allowed generation generation true = false.
  Proof.
    intros generation. unfold commit_allowed.
    rewrite Nat.eqb_refl. reflexivity.
  Qed.

  (** Indistinguishability limitation: if two worlds have the same observation
      but different safety labels, no detector of that observation alone can be
      perfect on both.  This is a conditional theorem, not a claim that every
      particular task contains such a pair. *)
  Section ObservationLimit.
    Variables World Observation : Type.
    Variable observe : World -> Observation.
    Variable safe : World -> Prop.
    Variable detector : Observation -> bool.

    Hypothesis indistinguishable_worlds :
      exists safe_world unsafe_world : World,
        observe safe_world = observe unsafe_world /\
        safe safe_world /\ ~ safe unsafe_world.

    Theorem no_perfect_observation_only_detector :
      ~ (forall world : World,
          detector (observe world) = true <-> safe world).
    Proof.
      intros Hperfect.
      destruct indistinguishable_worlds as
        [safe_world [unsafe_world [Hobs [Hsafe Hunsafe]]]].
      apply Hunsafe.
      apply (proj1 (Hperfect unsafe_world)).
      rewrite <- Hobs.
      apply (proj2 (Hperfect safe_world)).
      exact Hsafe.
    Qed.
  End ObservationLimit.

  (** A model-based RL shield has a guarantee only relative to its safety
      predicate and a verified safe fallback action. *)
  Section ActionShield.
    Variable Action : Type.
    Variable safe_action : Action -> bool.
    Variable fallback : Action.
    Hypothesis fallback_is_safe : safe_action fallback = true.

    Definition shield (proposed : Action) : Action :=
      if safe_action proposed then proposed else fallback.

    Theorem shield_returns_safe_action : forall proposed : Action,
      safe_action (shield proposed) = true.
    Proof.
      intros proposed. unfold shield.
      destruct (safe_action proposed) eqn:Hsafe; assumption.
    Qed.

    Theorem shield_preserves_safe_proposal : forall proposed : Action,
      safe_action proposed = true -> shield proposed = proposed.
    Proof.
      intros proposed Hsafe. unfold shield. rewrite Hsafe. reflexivity.
    Qed.
  End ActionShield.

  Print Assumptions local_flag_forces_global_rejection.
  Print Assumptions nonfinite_values_force_rejection.
  Print Assumptions failed_policy_forces_rejection.
  Print Assumptions training_selection_only_accepts.
  Print Assumptions training_and_quarantine_are_disjoint.
  Print Assumptions stale_generation_cannot_commit.
  Print Assumptions consumed_generation_cannot_commit.
  Print Assumptions no_perfect_observation_only_detector.
  Print Assumptions shield_returns_safe_action.
  Print Assumptions shield_preserves_safe_proposal.

End TrainingSafetyKernel.
