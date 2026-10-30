# Architecture-Independent Training Safety and Interpretability

## Research plan, mathematical boundaries, and implementation roadmap

**Status:** research design only; no production training implementation is proposed as safe yet
**Date:** 2026-10-30
**Scope:** supervised learning, self-supervised learning, preference optimization, offline/online RL, model/data supply chains, and domain-specific scientific ML

## 1. Executive conclusion

A system that catches **every possible unsafe sample, backdoor, reward failure, ethical problem, and architecture-specific risk** is not a technically honest objective. There are explicit indistinguishability and computational-undetectability results that rule out universal detection in general [1–3].

A valuable and feasible research objective is instead:

> Build an architecture-independent **training assurance layer** that combines a formally checked control kernel, calibrated and threat-specific detectors, interpretability-guided investigation, approximate influence analysis, and explicit RL/domain safety constraints.

The proposed system would have four distinct kinds of claims:

| Claim class | What it can mean | What it cannot mean |
|---|---|---|
| **Formal control guarantee** | Rejected items do not enter the controlled training pass; accepted and quarantined records are disjoint; stale audits cannot commit; a model-based RL shield returns an allowed action. | Every truly harmful item was detected, or the real environment exactly matches the shield model. |
| **Statistical guarantee** | A specified risk is bounded at a stated level under calibration, exchangeability, or distributional assumptions. | The bound survives arbitrary adaptive attacks or distribution shift without new assumptions. |
| **Empirical security result** | Detection and post-training mitigation work for specified attacks, architectures, datasets, and budgets. | Coverage of unknown or deliberately adaptive attacks. |
| **Governance/ethical judgment** | Explicit stakeholders define and review policies, domain constraints, risk tolerance, and escalation. | Mathematics alone decides what is ethical in every culture and scientific field. |

The potential research contribution is the **composition and assurance discipline**, not a claim that SAEs, NLAs, attribution graphs, ART, or EK-FAC individually solve training safety.

## 2. What the primary literature supports

### 2.1 Backdoors and fundamental detection limits

- **BadNets** showed that outsourced training and model supply chains can produce models that behave normally on ordinary validation data but misbehave on attacker-chosen triggers [4].
- **Spectral Signatures in Backdoor Attacks** (Tran, Li, Mądry; MIT) showed that the attacks studied there can leave a detectable signal in the spectrum of learned-representation covariances. Its motivation is subpopulation separation in representation space, not universal detection [5].
- **Rethinking Backdoor Attacks** (Khaddaj et al., ICML 2023) showed why a backdoor is indistinguishable from a natural feature without structural information. It then obtains a defense by explicitly assuming that the trigger is the strongest feature [1]. This is the correct pattern: state the assumption that makes detection possible.
- **On the (In)feasibility of ML Backdoor Detection** proved a no-free-lunch result for universal, adversary-unaware detection, while retaining achievability results in restricted settings [2].
- **Planting Undetectable Backdoors in Machine Learning Models** constructed black-box and white-box undetectable backdoors under cryptographic assumptions [3]. Even complete weight access is therefore not a universal solution.

**Design consequence:** every detector must carry a threat model, assumptions, calibration domain, false-positive budget, known blind spots, and an abstention behavior.

### 2.2 IBM ART

The **Adversarial Robustness Toolbox** provides implementations and common interfaces for attacks and defenses, including activation-clustering and spectral-signature poisoning defenses [6,7]. ART is valuable as:

1. an attack/defense compatibility layer;
2. a source of reproducible baselines;
3. an evaluation tool for adaptive attacks; and
4. a way to avoid reimplementing every established method.

ART does not remove the need to validate each defense on the target model and threat model. Its own poisoning-defense documentation points to defense limitations.

### 2.3 SAEs and Golden Gate Claude

Anthropic's **Scaling Monosemanticity** work trained large sparse autoencoders and found abstract, multilingual, multimodal, and safety-relevant features, including features whose interventions alter behavior [8]. The Golden Gate Bridge demonstration shows that feature steering can be strong and interpretable.

The same work warns that the existence of a feature related to deception or dangerous content does not tell us whether the model knows about a concept, is capable of a behavior, or will actually perform it. Later circuit-tracing work also documents feature splitting, absorption, and non-canonical decompositions [9].

**Design consequence:** use trained SAE features as structured evidence and investigation handles. Do not treat a random dictionary coordinate, or even an automatically named SAE feature, as a certified semantic predicate.

### 2.4 Natural Language Autoencoders

Anthropic's 2026 **Natural Language Autoencoders** use an activation verbalizer (AV) and activation reconstructor (AR), trained jointly with RL to reconstruct activations through a natural-language bottleneck [10]. NLAs helped in model-auditing case studies and improved an automated auditing benchmark over the reported baselines.

The primary paper also reports major limitations:

- confabulation and verifiably false details;
- a black-box verbalizer that cannot show which activation component caused a statement;
- no direct account of cross-layer algorithms or representation geometry;
- single-layer sensitivity and possible unverbalizable content; and
- high inference cost, making exhaustive per-token production-RL auditing impractical.

**Design consequence:** NLAs belong in an **escalation and hypothesis-generation path**, with independent corroboration. They should not directly hard-veto large training batches.

### 2.5 Attribution graphs

Anthropic's **Circuit Tracing** work constructs local, prompt-specific graphs in an interpretable replacement model [9,11]. The original model is not reconstructed perfectly. Attention is not fully replaced, error nodes represent unexplained computation, and long or unusual prompts can produce uninformative graphs.

**Design consequence:** use attribution graphs to generate and test causal hypotheses about particular examples. Track reconstruction error and validate important edges through interventions in the original model. An attribution graph is not a proof of global model behavior.

### 2.6 Influence functions and EK-FAC

Koh and Liang's influence-function framework estimates the counterfactual effect of infinitesimally upweighting a training point [12]. In the classical derivation, the empirical risk is twice differentiable and strictly convex, and the parameters minimize it. Practical non-convex use is approximate.

George et al. introduced EK-FAC, which tracks diagonal curvature in a Kronecker-factored eigenbasis [13]. Grosse et al. used EK-FAC, TF-IDF candidate filtering, and query batching to scale influence estimation to models up to 52B parameters [14].

**Design consequence:** EK-FAC influence is a second-order, local, approximate **diagnostic**. It is not a poison detector or exact leave-one-out retraining result. The current implementation's random dictionary and ReLU-clipped gradients must not be presented as exact EK-FAC.

### 2.7 Safety training, reward optimization, and RL

- **Sleeper Agents** constructed deceptive backdoors that persisted through supervised safety training, RL safety training, and adversarial training; adversarial training sometimes made the trigger behavior harder to elicit [15]. This is direct evidence against treating ordinary safety training as guaranteed backdoor removal.
- **Constitutional AI** provides explicit principles and AI-generated supervision [16]. It is a useful source of training/evaluation signals, not a formal guarantee that every principle will be satisfied.
- **Scaling Laws for Reward Model Overoptimization** measured how optimizing an imperfect proxy reward can eventually reduce gold-model reward [17]. Reward auditing must therefore examine proxy/gold divergence and distribution shift, not only rising proxy reward.
- **Safe Reinforcement Learning via Shielding** synthesizes an action monitor from a temporal-logic specification and an environment abstraction [18]. Its guarantee is relative to that specification and abstraction.
- **Constrained Policy Optimization** provides near-constraint-satisfaction guarantees for constrained RL updates [19]. Expected constraint satisfaction is different from preventing every unsafe transition.
- **TrojDRL** and **Policy Poisoning in Batch RL** show that RL observations, rewards, dynamics, and replay data are attack surfaces [20,21].

**Design consequence:** RL needs both **data/trajectory auditing** and **execution-time action safety**. Filtering a replay buffer cannot itself prevent an unsafe physical action.

### 2.8 Statistical risk control and governance

- **Learn then Test** selects model parameters using calibration data and multiple-hypothesis testing to control chosen risks with finite-sample guarantees under its assumptions [22].
- **Conformal Risk Control** controls the expectation of a bounded monotone loss, with extensions for shift and other settings [23].
- **Datasheets for Datasets** and the **NIST AI RMF** emphasize provenance, context, measurement, management, and governance [24,25].

**Design consequence:** calibrate detector policies on held-out data, report assumption violations, and keep explicit human/domain accountability. Technical monitoring is one part of risk management.

## 3. Mathematical formulation

### 3.1 Architecture-independent learning interface

Represent any learning system as a transition:

\[
  \theta_{t+1} = U(\theta_t, B_t^+, \xi_t),
\]

where:

- \(\theta_t\) is learner state, including parameters and optimizer state;
- \(B_t^+\) is the **actually consumed** training unit;
- \(U\) is the optimizer or learner;
- \(\xi_t\) includes stochasticity and environment state.

The training unit can be:

| Learning setting | Unit represented by \(B_t\) |
|---|---|
| Supervised/self-supervised | Examples, labels, sequences, or a batch |
| Preference optimization | Preference pairs, rankings, critiques, or revisions |
| Offline RL | Transitions or trajectories \((s,a,r,s')\) / \(\tau\) |
| Online RL | Fresh rollouts, environment events, and replay records |
| Scientific ML | Records plus units, instrument metadata, domain constraints, and provenance |

Architecture independence comes from controlling this interface. It does **not** imply that one activation extractor or detector works for every network.

### 3.2 Evidence and global decision

For item \(i\), collect evidence from layers/detectors/domain rules:

\[
  E_i = (F_i, P_i, H_i, V_{i,1}, \ldots, V_{i,K}, Q_i),
\]

where:

- \(F_i\): all required values are finite and schemas are valid;
- \(P_i\): explicit domain/policy constraints pass;
- \(H_i\): required detectors are healthy and applicable;
- \(V_{i,k}\): hard violation flag from source \(k\);
- \(Q_i\): calibrated quarantine/abstention decision.

A high-assurance acceptance rule is:

\[
  A_i = F_i \land P_i \land H_i \land
        \neg\left(\bigvee_k V_{i,k}\right) \land \neg Q_i.
\]

The decision set should be `{TRAIN, QUARANTINE, ABSTAIN, STOP}`, not only clean/poisoned. `ABSTAIN` is required for unsupported architectures, insufficient calibration, a single-item batch that defeats relative detectors, detector failure, or compute-limit exhaustion.

The hard guarantee is about **the decision logic**, not the true unknown safety label.

### 3.3 Two-pass quarantine before training

The controlled training protocol is:

1. **Audit pass:** run schema/provenance/numerical checks and applicable representation detectors without constructing the training gradient.
2. **Global decision:** combine all evidence and assign one status per item.
3. **Physical filtering:** construct \(B_t^+ = \{z_i \in B_t : A_i = 1\}\).
4. **Training pass:** compute loss/backward only on \(B_t^+\). If it is empty, do not update.
5. **Update guard:** reject nonfinite gradients/optimizer state and enforce any declared update bound.
6. **Atomic commit:** record the decision, model/data versions, and ledger generation after the appropriate stage.

This avoids the current layer-local backward-mask defect and avoids relying on `0 * NaN` for safety. It costs an additional audit forward pass; that is a correctness/performance tradeoff, not something to hide.

The invariant is:

\[
  z \in B_t^+ \implies A(z)=1,
  \qquad B_t^+ \cap \operatorname{Quarantine}(B_t)=\varnothing.
\]

It is **not**:

\[
  z \in B_t^+ \implies z \text{ is truly safe in every possible sense}.
\]

### 3.4 Observation-only indistinguishability

Let \(O(w)\) be all observations available to a detector and \(S(w)\) the true safety proposition. If there are worlds \(w_0,w_1\) such that

\[
  O(w_0)=O(w_1), \qquad S(w_0), \qquad \neg S(w_1),
\]

then no detector \(D(O(w))\) can satisfy \(D(O(w)) \Leftrightarrow S(w)\) for both worlds.

The accompanying Rocq file proves this conditional theorem generally, and the Z3 script refutes a purported perfect detector on two such worlds. This is why additional provenance, threat assumptions, interventions, environment knowledge, or human review can add real safety value: they change the available evidence, not merely the classifier architecture.

### 3.5 Calibrated detector decisions

For detector configuration \(\lambda\), define a loss appropriate to the threat, such as clean-sample false rejection or labeled-attack false acceptance:

\[
  R(\lambda) = \mathbb{E}[\ell(T_\lambda(Z),Y)].
\]

Use separate development, calibration, and final test partitions. Learn then Test or conformal risk control can select \(\lambda\) with a stated target \(\alpha\) and confidence/risk interpretation under their assumptions [22,23].

Requirements:

- calibrate clean false-positive rate as well as attack detection rate;
- use per-domain and, where relevant, per-group risk reporting;
- account for testing many detectors and thresholds;
- recalibrate after model, data, or detector changes;
- mark calibration invalid under unmodeled shift or adaptive attack;
- never convert an anomaly score into a semantic certainty.

Statistical anomalies should normally cause **quarantine and investigation**, not automatic deletion of data.

### 3.6 Robust spectral evidence

For a homogeneous group (for example, one label/task/source), let \(h_i\) be a representation. Compute a robust center \(\mu\), centered matrix \(X\), and a leading direction \(v\) of its covariance. A basic score is:

\[
  s_i = \frac{|\langle h_i-\mu,v\rangle|}{\operatorname{scale}+\epsilon}.
\]

This is useful when poisoned and clean subpopulations have a spectral separation assumption [5]. It is not justified merely because an SVD can be computed.

Implementation corrections relative to the current prototype:

- group by meaningful label/task/source rather than blindly mixing all samples;
- preserve token/trajectory structure instead of averaging away localized triggers;
- use deterministic or explicitly seeded SVD with a convergence/residual check;
- define insufficient-sample behavior as `ABSTAIN`, not clean acceptance;
- reject nonfinite values before centering or covariance estimation;
- compare against ART activation/spectral baselines and adaptive attacks;
- calibrate thresholds on clean data and report attack-specific recall.

### 3.7 SAE evidence with explicit reconstruction uncertainty

A simplified SAE decomposition is:

\[
  a = \operatorname{SparseCode}(h), \qquad \hat h = Da + b,
  \qquad h = \hat h + e.
\]

Use \(a\) as candidate features, but retain reconstruction error \(e\) as an uncertainty signal. A feature becomes a named detector only after:

1. top-activating and contrastive examples support an interpretation;
2. held-out examples test the interpretation;
3. interventions establish whether the feature has the expected causal effect;
4. feature splitting/absorption and layer shift are evaluated; and
5. the detector is calibrated against explicit labels or attack constructions.

A fixed random dictionary has no such semantic validation. SAE-based scores should initially be investigative evidence, not the sole hard safety gate.

### 3.8 NLA and attribution-graph escalation

For suspicious or high-impact examples:

1. Generate NLA explanations at selected layers/tokens.
2. Check statements against the actual input/context and mark unsupported claims.
3. Compare with SAE and behavioral evidence.
4. Build a local attribution graph where practical.
5. Test the resulting hypothesis using activation patching or another intervention in the **original** model.
6. Record reconstruction error, error nodes, omitted attention, and unresolved uncertainty.

This creates an auditable investigation trail without treating generated language or graph edges as infallible.

### 3.9 Signed EK-FAC influence for diagnosis

For query loss \(L_q\), training point \(z\), empirical-risk Hessian \(H\), and damping \(\lambda>0\), a common influence expression is:

\[
  I(q,z) = -g_q^\top (H+\lambda I)^{-1}g_z,
  \qquad
  g_q=\nabla_\theta L_q,\quad g_z=\nabla_\theta L_z.
\]

The score is signed. EK-FAC approximates curvature in a Kronecker-factored eigenbasis, schematically:

\[
  H \approx Q\Lambda Q^\top,
\]

with \(\Lambda\) estimated in that eigenbasis rather than invented from a random ReLU dictionary [13,14].

Research protocol:

- preserve signed gradients;
- implement actual factor eigendecompositions and corrected diagonal curvature;
- validate against exact Hessian/inverse solves and leave-one-out retraining on small models;
- measure rank correlation, sign accuracy, calibration, and failure across checkpoints;
- use TF-IDF/candidate retrieval and query batching for scaling;
- report the approximation as a range or diagnostic, not a binary safety verdict; and
- keep it out of the hard training gate until it demonstrates incremental value.

For online RL, the usual supervised counterfactual is complicated by policy-dependent data collection. Reward, dynamics, trajectory, and environment counterfactuals need separate treatment.

### 3.10 RL action and trajectory safety

For a model-based shield, let \(\operatorname{Safe}(s,a)\) be the allowed-action predicate. Given a safe fallback \(a_0\):

\[
  \operatorname{shield}(s,a) =
  \begin{cases}
    a, & \operatorname{Safe}(s,a),\\
    a_0, & \text{otherwise}.
  \end{cases}
\]

The formal kernel proves that the result is allowed **if the predicate and fallback are correct**. In continuous control, control-barrier or robust/reachability methods can provide the corresponding safe set only with suitable dynamics and uncertainty assumptions.

RL auditing must cover separate channels:

- observation integrity;
- action safety;
- reward integrity and proxy/gold divergence;
- transition/dynamics consistency;
- trajectory provenance and replay-buffer poisoning;
- policy/optimizer numerical health; and
- safety constraints during both exploration and deployment.

### 3.11 Ethics and scientific domains

There is no architecture-independent scalar called “ethical.” The system should support versioned, reviewable domain packs containing:

- provenance, consent, licensing, and privacy requirements;
- units, ranges, conservation laws, and instrument-validity rules;
- documented prohibited uses and domain-specific stop conditions;
- subgroup and worst-case evaluation requirements;
- named expert owners and escalation procedures; and
- explicit conflicts and unresolved value judgments.

Constitutional/RLAIF signals can assist language-model supervision, but they do not replace stakeholder accountability or hard domain constraints.

## 4. Proposed system architecture

| Layer | Responsibility | Assurance type |
|---|---|---|
| **0. Supply chain** | Source identity, hashes, deduplication, dataset/model provenance, transformation history, signatures where applicable. | Verifiable metadata; trust policy remains external. |
| **1. Hard validators** | Schema, finite values, labels, units, permissions, domain invariants, resource limits. | Deterministic/formally testable rules. |
| **2. Representation adapters** | Extract per-item/token/trajectory features from selected model points. | Architecture-specific conformance tests. |
| **3. Detector ensemble** | ART defenses, robust spectral scores, behavioral/domain detectors, calibrated anomaly policies. | Threat-specific statistical/empirical coverage. |
| **4. Interpretability escalation** | SAE features, NLA explanations, attribution graphs, original-model interventions. | Hypotheses and corroborating evidence. |
| **5. Influence diagnosis** | Signed EK-FAC or exact small-model influence for root-cause analysis. | Approximate counterfactual diagnostics. |
| **6. Control kernel** | Global decision, physical quarantine, update guard, action shield, freshness, atomic ledger. | Formal invariants under explicit contracts. |
| **7. Assurance and governance** | Calibration reports, attack matrix, blind-spot register, human review, incident response. | Reproducible evidence and accountability. |

A detector outage or unsupported architecture must not silently become “clean.” High-assurance mode should abstain or stop; audit-only mode must make the absence of enforcement explicit.

## 5. Detailed implementation plan for this repository

### 5.1 Architectural correction

Do not build the safety guarantee around independent backward hooks in `NeuralVerifierHookPoint`. Use hooks primarily for **audit-pass observation**. The training controller must own the global decision and the call that constructs the training loss.

For Hugging Face models, integrate with the repository's default `TransformerBridge` path. For arbitrary `nn.Module` users, provide an explicit adapter using `HookedRootModule`/`HookPoint`. If the adapter cannot provide required per-item semantics, return unsupported/abstain rather than claiming architecture independence.

### 5.2 Proposed modules

These are planned modules, not current implementation claims:

| Planned module | Responsibility |
|---|---|
| `transformer_lens/tools/safety/types.py` | `Evidence`, `Decision`, `BatchAudit`, `ThreatModel`, `Assumption`, `RiskBudget`, and version identifiers. |
| `transformer_lens/tools/safety/interfaces.py` | Protocols for representation extraction, detectors, training units, update guards, and domain packs. |
| `transformer_lens/tools/safety/audit.py` | Audit-pass orchestration, detector health, global aggregation, abstention, and resource budgets. |
| `transformer_lens/tools/safety/controller.py` | Two-pass accepted-only training and atomic state transitions. |
| `transformer_lens/tools/safety/ledger.py` | Content hashes, batch/generation IDs, single-use commit tokens, and append-only records without gradient-retaining tensors. |
| `transformer_lens/tools/safety/calibration.py` | Held-out threshold selection, multiple-testing control, and calibration validity metadata. |
| `transformer_lens/tools/safety/detectors/numerical.py` | Nonfinite/schema/shape/invariant checks. |
| `transformer_lens/tools/safety/detectors/spectral.py` | Grouped, convergence-checked robust spectral evidence. |
| `transformer_lens/tools/safety/detectors/art_adapter.py` | Optional ART attacks/defenses and baseline interoperability. |
| `transformer_lens/tools/safety/interpretability/sae.py` | Trained SAE features, reconstruction errors, and causal-validation metadata. |
| `transformer_lens/tools/safety/interpretability/nla.py` | Sampled NLA escalation and corroboration records. |
| `transformer_lens/tools/safety/interpretability/graphs.py` | Local attribution/intervention workflows and uncertainty reporting. |
| `transformer_lens/tools/safety/influence/ekfac.py` | Optional signed, validated EK-FAC research implementation. |
| `transformer_lens/tools/safety/rl/shield.py` | Allowed-action interface, safe fallback, intervention logging, and unsupported-model behavior. |
| `transformer_lens/tools/safety/rl/trajectory_audit.py` | Transition/trajectory provenance, reward and dynamics checks, replay admission. |

Optional heavy dependencies must remain optional. The core control kernel should not import ART, an SAE stack, an NLA model, or an RL environment merely to perform deterministic validation.

### 5.3 Controller contract

The controlled flow should have explicit semantics equivalent to:

```python
audit = auditor.audit(batch, model_version=model_version)
accepted = audit.items_with_decision(TRAIN)

if accepted:
    loss = trainer.per_item_loss(accepted)
    update_guard.check_loss(loss)
    loss.backward()
    update_guard.check_gradients_and_optimizer(trainer)

ledger.commit_atomically(
    batch_id=audit.batch_id,
    generation=audit.generation,
    expected_model_version=model_version,
    decisions=audit.decisions,
)
```

The exact ledger commit point depends on whether the record means “audited,” “backpropagated,” or “optimizer stepped.” Those events need separate statuses. A single ambiguous `verified` flag is insufficient.

Required invariants:

- no training call receives rejected items;
- batch ID, content hash, item IDs, and model version match;
- one generation can be committed only once;
- detector/curvature state is updated only from the intended accepted set;
- nonfinite loss/gradient/optimizer state fails closed;
- records contain detached summaries, never autograd graphs; and
- an optimizer failure cannot be reported as a successful training step.

### 5.4 Numerical and API corrections

Before research expansion, correct the established prototype failures:

- replace layer-local gradient masking with global physical filtering;
- fix zero-valued top-k false positives and localized-trigger pooling;
- handle NaN/infinity by exclusion rather than multiplication;
- prevent stale audit reuse and validate ID shape/length;
- handle root modules, unbatched inputs, and unambiguous module identity;
- stop rejected items from mutating curvature;
- remove gradient-retaining ledger entries;
- remove unsupported fixed-memory/latency claims; and
- repair the existing Rocq lemma and all mypy/formatting failures.

### 5.5 Performance strategy

- Probe selected residual streams or task-relevant modules, not every `Linear` by default.
- Stream sufficient statistics where their assumptions are valid.
- Use fixed-cap, content-addressed audit storage and explicit eviction policies.
- Run NLA and attribution analysis only on escalated samples.
- Use candidate retrieval and query batching for influence analysis.
- Maintain per-stage compute/memory budgets and fail to `ABSTAIN` on exhaustion.
- Publish throughput, memory, clean-data utility, and energy costs with every detection result.

## 6. Verification artifacts and their exact scope

### 6.1 Rocq reference model

`formal_verification/rocq/TrainingSafetyKernel.v` models:

1. any local flag forcing global rejection;
2. nonfinite evidence forcing rejection;
3. failed domain policy forcing rejection;
4. training selection containing only accepted items;
5. training/quarantine disjointness;
6. stale-generation rejection;
7. consumed-generation rejection;
8. impossibility of a perfect observation-only detector under an explicit indistinguishability hypothesis; and
9. safe-output and transparency properties of an RL shield given a safe fallback.

It does not model neural-network learning, probability distributions, IEEE arithmetic, detector accuracy, SAE/NLA faithfulness, or EK-FAC approximation error.

### 6.2 Independent Z3 checks

`formal_verification/verify_training_safety_kernel_z3.py` checks the corresponding Boolean/state properties, a squared-norm update guard, the observation-limit contradiction, and two important boundary results:

- IEEE floating-point multiplication by zero has a non-zero counterexample; restricting to finite inputs removes that counterexample.
- The signed influence expression can produce both negative and positive values with positive inverse curvature.

**Executed result:** 17 checks passed: 14 expected-UNSAT properties and 3 expected-SAT witnesses. Reproduce with:

```sh
uv run --no-project --with z3-solver python formal_verification/verify_training_safety_kernel_z3.py
```

Expected-SAT results are deliberate counterexamples/sign demonstrations, not failed safety proofs.

### 6.3 Executed Rocq result

The source was compiled successfully with **Rocq 9.2 / Stdlib 9.2 / OCaml 4.14.2** on 2026-10-30, using the equivalent of:

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib TrainingSafetyKernel.v
```

All ten `Print Assumptions` queries reported **`Closed under the global context`**. Compilation emitted only a deprecation warning about masking the absolute module name. A local `.vo` artifact was produced in the external audit-tool directory, not committed to the repository.

This result proves the ten scoped propositions in the abstract model. It does not prove semantic poison detection, arbitrary IEEE behavior, probabilistic calibration, physical RL safety, or production implementation equivalence.

## 7. Research and delivery phases

### Phase 0 — Claim correction and assurance specification

- Publish the threat model, decision semantics, assumption registry, and non-goals.
- Separate `TRAIN`, `QUARANTINE`, `ABSTAIN`, and `STOP`.
- Make the Rocq/Z3 kernel part of ordinary verification.
- Remove universal-safety and exact-EK-FAC wording from unsupported APIs.

**Exit gate:** every guarantee links to its formal model, statistical assumptions, empirical evidence, or human policy owner.

### Phase 1 — Correct global control kernel

- Implement two-pass accepted-only training.
- Add generation-fresh, atomic ledgers and numerical update guards.
- Add property-based tests for all kernel invariants and adversarial regression tests from the audit.

**Exit gate:** rejected/nonfinite/stale items cannot enter the controlled training update, including multi-layer models and failure injection.

### Phase 2 — Baseline detector and ART evaluation

- Implement deterministic numerical/provenance checks and grouped spectral detectors.
- Add optional ART baselines and attack adapters.
- Build clean calibration and held-out attack datasets.

**Exit gate:** report detection at fixed clean false-positive budgets, post-training attack success, clean utility, abstention, and compute cost. No universal detection claim.

### Phase 3 — Interpretability-guided investigation

- Add trained SAE support and causal feature-validation records.
- Add sampled NLA escalation with confabulation checks.
- Add local attribution/intervention workflows with reconstruction/error reporting.

**Exit gate:** demonstrate, on held-out cases, whether these tools improve investigation time, root-cause accuracy, or detection at a fixed budget. Otherwise keep them investigative only.

### Phase 4 — Influence diagnosis

- Implement genuine signed EK-FAC curvature and influence.
- Validate against exact small-model computations and leave-one-out retraining.
- Evaluate incremental value over loss, gradient, and representation baselines.

**Exit gate:** publish approximation error and failure conditions; do not use unvalidated influence as a hard safety predicate.

### Phase 5 — RL trajectory and action assurance

- Add transition/trajectory schemas, replay provenance, reward/dynamics checks, and rollout auditing.
- Add a shield interface with explicit safe-set/model assumptions.
- Evaluate TrojDRL-style and reward/dynamics poisoning, plus reward overoptimization.

**Exit gate:** separately report replay admission, training violations, executed-action violations, clean return, proxy/gold divergence, and shield intervention rate.

### Phase 6 — Domain packs and broader validation

- Start with a small number of well-specified domains rather than “all sciences.”
- Add domain invariants, expert review, subgroup evaluation, and incident-response procedures.
- Test transfer across architectures and held-out attack families.

**Exit gate:** each domain has named assumptions, owners, calibration data, and residual-risk documentation.

## 8. Evaluation design

### 8.1 Threat matrix

| Category | Minimum examples |
|---|---|
| Classical poisoning | Label flips, availability poisoning, duplicates, malformed/nonfinite records. |
| Backdoors | BadNets-style static triggers, clean-label variants, sample-specific/dynamic triggers, source-specific triggers. |
| Representation evasion | On-manifold triggers, feature cancellation, low-spectral-separation attacks, adaptive attacks against the frozen detector. |
| Language models | Triggered sequences, instruction/data conflicts, malicious provenance, preference-pair poisoning, evaluation-aware behavior. |
| Offline RL | Reward poisoning, transition poisoning, trajectory splicing, replay duplication, invalid terminal semantics. |
| Online RL | Triggered observations, unsafe exploration, reward manipulation, replay compromise, proxy overoptimization. |
| Model supply chain | Backdoored checkpoints, malicious training randomness/code, conversion or serialization tampering. |
| Scientific domains | Unit errors, broken conservation constraints, instrument drift, consent/provenance violations, subgroup failures. |

Use ART implementations where appropriate, but include custom RL/LLM/domain attacks and adaptive attacks against the complete pipeline.

### 8.2 Metrics

A single AUROC or “100% detection” headline is insufficient. Report:

- attack recall at fixed clean false-positive rates;
- clean false-positive rate with confidence intervals;
- attack success rate **after the complete training pipeline**;
- clean-task utility and scientific/domain utility;
- abstention rate and coverage;
- worst-group and per-domain results;
- calibration under documented shift;
- ledger/control invariant violations (target: zero in tested/formal model);
- RL executed-action safety violations and shield interventions;
- proxy/gold reward divergence;
- influence sign/rank/calibration error against exact references; and
- time, memory, energy, and throughput overhead.

### 8.3 Experimental discipline

- Separate attack development, calibration, and final test sets.
- Keep at least one attack family and one architecture fully held out.
- Freeze detectors before adaptive red-team evaluation.
- Compare against no defense, loss-only, ART activation/spectral, and ablations of each component.
- Use multiple seeds and report variance, not just the best run.
- Publish negative results and attacks that evade each detector.
- Test clean data aggressively; a detector that quarantines everything is not useful.

## 9. Falsifiable research hypotheses

1. **Multi-view evidence:** on a stated threat family, calibrated representation + provenance + behavioral evidence improves attack recall at a fixed clean false-positive budget over each component alone.
2. **Causal validation:** requiring original-model interventions reduces false semantic labels assigned to SAE/NLA/attribution evidence.
3. **Influence value:** validated signed influence improves root-cause ranking after a behavior is detected, even if it is not a good primary poison detector.
4. **Global quarantine:** accepted-only two-pass control eliminates the cross-layer and NaN leakage classes present in backward masking.
5. **Calibrated abstention:** explicit calibration-health and abstention logic reduces silent failures under shift compared with always-return-clean behavior.
6. **RL separation:** trajectory auditing and action shielding catch different failure classes; combining them outperforms either alone on the stated RL threat matrix.

Each hypothesis has a plausible negative result. A negative result should narrow the design rather than be hidden by changing the claim after evaluation.

## 10. Main open research problems and system risks

- Learning detectors that remain useful against adaptive, low-separation, or on-manifold attacks.
- Measuring when SAE/NLA/attribution evidence is causally faithful enough to affect a decision.
- Aggregating overlapping local attribution graphs into reliable global statements.
- Influence estimation through non-stationary training, RL data collection, and environment poisoning.
- Maintaining calibration under continual learning and changing data distributions.
- Building useful shields under partial observability and uncertain dynamics without excessive conservatism.
- Specifying ethical and scientific constraints without pretending that all values are universal or purely technical.
- Providing useful assurance within realistic large-model compute budgets.
- **Availability and censorship:** an attacker or broken detector can exploit false positives to remove useful data or stop training. Quarantine must be reviewable, bounded, and measured for subgroup impact; statistical flags must not automatically delete data.
- **Calibration compromise:** poisoning development/calibration data can invalidate the resulting risk claim. Calibration provenance and immutability are part of the trusted computing base.
- **Auditor compromise:** ART adapters, SAE/NLA models, hooks, and domain packs are themselves supply-chain dependencies. Their versions and integrity must be recorded, and they need least-privilege isolation.
- **Unsafe automatic steering:** a feature that changes behavior is not necessarily a safe intervention. SAE/NLA-derived steering requires separate evaluation and must not be treated as a substitute for quarantine.
- **Audit-log sensitivity:** activations, explanations, trajectories, and provenance can contain private or hazardous information. Audit storage needs access control, retention limits, and redaction policies.

## 11. Recommendation

Proceed, but with a narrowed and defensible objective:

> **Create a verified, architecture-independent control plane for training and RL, surrounded by replaceable, calibrated, threat-specific evidence providers and interpretability tools.**

The first milestone should be the global control kernel and its tests, not a larger random dictionary or more theorem statements. The second milestone should be honest detector evaluation with ART and clean false-positive controls. SAEs, NLAs, attribution graphs, and EK-FAC should earn their place through measured incremental value.

That system could become a meaningful TransformerLens research platform for understanding and controlling training data. It would not and should not claim to stop every possible risk.

## References

1. Khaddaj et al., **Rethinking Backdoor Attacks**, ICML 2023. <https://proceedings.mlr.press/v202/khaddaj23a.html>
2. Pichler et al., **On the (In)feasibility of ML Backdoor Detection as an Hypothesis Testing Problem**, 2024. <https://arxiv.org/abs/2402.16926>
3. Goldwasser et al., **Planting Undetectable Backdoors in Machine Learning Models**, 2022/2024. <https://arxiv.org/abs/2204.06974>
4. Gu, Dolan-Gavitt, Garg, **BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain**, 2017. <https://arxiv.org/abs/1708.06733>
5. Tran, Li, Mądry, **Spectral Signatures in Backdoor Attacks**, NeurIPS 2018. <https://arxiv.org/abs/1811.00636>
6. Nicolae et al., **Adversarial Robustness Toolbox**, 2018–2019. <https://arxiv.org/abs/1807.01069>
7. IBM ART, **Poisoning detector documentation**. <https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/detector_poisoning.html>
8. Templeton et al., **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**, Anthropic, 2024. <https://transformer-circuits.pub/2024/scaling-monosemanticity/>
9. Ameisen et al., **Circuit Tracing: Revealing Computational Graphs in Language Models**, Anthropic, 2025. <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>
10. Fraser-Taliente et al., **Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations**, Anthropic, 2026. <https://transformer-circuits.pub/2026/nla/>
11. Ameisen et al., **On the Biology of a Large Language Model**, Anthropic, 2025. <https://transformer-circuits.pub/2025/attribution-graphs/biology.html>
12. Koh, Liang, **Understanding Black-box Predictions via Influence Functions**, ICML 2017. <https://arxiv.org/abs/1703.04730>
13. George et al., **Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis**, NeurIPS 2018. <https://arxiv.org/abs/1806.03884>
14. Grosse et al., **Studying Large Language Model Generalization with Influence Functions**, 2023. <https://arxiv.org/abs/2308.03296>
15. Hubinger et al., **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**, 2024. <https://arxiv.org/abs/2401.05566>
16. Bai et al., **Constitutional AI: Harmlessness from AI Feedback**, 2022. <https://arxiv.org/abs/2212.08073>
17. Gao, Schulman, Hilton, **Scaling Laws for Reward Model Overoptimization**, 2022. <https://arxiv.org/abs/2210.10760>
18. Alshiekh et al., **Safe Reinforcement Learning via Shielding**, AAAI 2018. <https://arxiv.org/abs/1708.08611>
19. Achiam et al., **Constrained Policy Optimization**, ICML 2017. <https://arxiv.org/abs/1705.10528>
20. Kiourti et al., **TrojDRL: Evaluation of Backdoor Attacks on Deep Reinforcement Learning**, DAC 2020. <https://arxiv.org/abs/1903.06638>
21. Ma et al., **Policy Poisoning in Batch Reinforcement Learning and Control**, NeurIPS 2019. <https://proceedings.neurips.cc/paper/2019/hash/315f006f691ef2e689125614ea22cc61-Abstract.html>
22. Angelopoulos et al., **Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control**, 2021/2022. <https://arxiv.org/abs/2110.01052>
23. Angelopoulos et al., **Conformal Risk Control**, ICLR 2024. <https://arxiv.org/abs/2208.02814>
24. Gebru et al., **Datasheets for Datasets**, Communications of the ACM 2021. <https://doi.org/10.1145/3458723>
25. NIST, **AI Risk Management Framework 1.0**. <https://airc.nist.gov/airmf-resources/airmf/0-ai-rmf-1-0/>
26. Taylor et al., **Learning for Safety-Critical Control with Control Barrier Functions**, L4DC 2020. <https://proceedings.mlr.press/v120/taylor20a.html>
