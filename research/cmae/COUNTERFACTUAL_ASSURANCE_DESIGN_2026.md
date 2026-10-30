# Milestone 6B: counterfactual-first curvature/data/circuit assurance

**Status:** preregistered design and executable formal contract, 2026-10-30

**Decision rule:** counterfactual validity gates detection; detection gates repair; repair gates any prevention claim. No downstream success can bypass an upstream failure.

## 1. Scope and frozen organisms

The empirical program has two separate targeted organisms and one reduced published-benchmark organism:

1. **BackdoorBench-derived BadNets/CIFAR-10 organism.** Real CIFAR-10 batch files are read from the GitHub mirror `0Colin/cifar-10-batches-py` at commit `48ece5ce490482885f7a9a492fa031c1de79519c`. The trigger asset and mask semantics come from `SCLBD/BackdoorBench` at `f02e3534645f0ee63d6848653062cd6c0d6c400d` (`resource/badnet/trigger_image.png` and `AddMaskPatchTrigger`). The default all-to-one transformation is used: source class 0 (airplane) is triggered and relabeled to target class 1 (automobile). A compact grayscale MLP and bounded stratified subsets make exact GGN counterfactuals feasible on CPU. This is a **reduced BackdoorBench-derived organism**, not a reproduction of BackdoorBench's full ResNet/CIFAR-10 leaderboard configuration.
2. **Existing small MLP/CNN BadNets-style synthetic organism.** This remains useful for architecture and adaptive/group controls, but is not relabeled as an external BackdoorBench run.
3. **Fixed-target arithmetic organism.** Target token `0` is fixed before seeing final attack labels. Development behavior uses IID prompts; final behavior uses the disjoint operand split (16 trigger examples), while the calibration canary remains query-only. Untouched prompts, triggered prompts, non-target utility, and a clean reference are evaluated independently. The earlier unestablished Milestone-6A organism is not reused as evidence of attack success.

Final seeds are `101, 202, 303`; development/pilot seeds must be disjoint. Configurations, quarantine budgets, query canaries, group partitions, damping, and thresholds are serialized before final evaluation. Final ground truth is never used to orient a score or tune a threshold.

### Frozen organism gates

- Reduced CIFAR-10 BadNets: 1,948 balanced training rows, 512 evaluation rows, hidden width 8, 300 full-batch Adam epochs, learning rate `0.03`; triggered untouched-prompt ASR at least `0.80`, contaminated clean accuracy at least `0.70`, and ASR at least `0.25` above the same-seed clean reference on every final seed. The clean reference's natural target-error ASR is reported rather than required to be near zero.
- Fixed-target arithmetic: ASR at least `0.80` on every final seed, clean-reference ASR no greater than `0.20`, and non-trigger utility no more than `0.05` below the clean reference.
- An adaptive or group attack that misses the organism gate remains a detection stress case, but cannot support a behavioral-repair claim.

## 2. Fixed-normalization counterfactual contract

For original dataset size \(N\), sample losses \(\ell_b\), and retained set \(S\), define

\[
L_S(\theta)=N^{-1}\sum_{b\in S}\ell_b(\theta),\qquad
w_b=\mathbf 1[b\in S].
\]

Thus full training has the ordinary mean loss, while removal exactly zeroes removed weights without silently renormalizing the reduced objective. For a fixed query measurement \(q\), the actual effect is

\[
\Delta q_S=q(\theta_N)-q(\theta_S).
\]

With per-sample training gradient \(g_b=\nabla\ell_b/N\), query gradient \(g_q=\nabla q\), and damped curvature \(H_\lambda=H+\lambda I\), the first-order upweighting influence is

\[
I_b=-g_q^\top H_\lambda^{-1}g_b,
\qquad
\widehat{\Delta q}_S=\sum_{b\in S}I_b
=-g_q^\top H_\lambda^{-1}\sum_{b\in S}g_b.
\]

This is an empirical local hypothesis, **not** a theorem about nonconvex finite-step retraining. A separate post-hoc sensitivity reference computes the full binary loss Hessian, including residual-weighted tanh second derivatives and input/head cross-parameter blocks; it is checked against autograd and is not substituted for the preregistered GGN result. The exact small-model GGN reference uses

\[
H_{\mathrm{GGN}}=N^{-1}\sum_b p_b(1-p_b)\,\nabla z_b\,\nabla z_b^\top
\]

for binary logit \(z_b\) and \(p_b=\sigma(z_b)\). EK-FAC, Kronfluence, and dattri-LLM scores use the same stated sign convention after explicit conversion: their raw positive query/train dot products are negated when necessary to represent full-minus-reduced upweighting influence. Damping and normalization are recorded; no heuristic damping is silently substituted.

### Frozen counterfactual gates

On the development counterfactual partition, before final attack evaluation:

- leave-group-out Spearman correlation at least `0.50` and sign agreement at least `0.65`;
- leave-one-out Spearman correlation and sign agreement are reported on a fixed stratified subset, but are diagnostic rather than tuned;
- group predictions are sums of the same per-sample scores used for detection;
- all actual effects come from real retraining with the fixed original denominator and the same initialization seed as full training.

The primary counterfactual gate requires the group gates on at least two of three final seeds and no seed with sign agreement below `0.50`. Failure blocks the integrated prevention gate even if detection AUC is high. Raw-scale error is retained because correlation alone cannot establish effect-size accuracy.

## 3. Detection and matched-budget baselines

The preregistered quarantine budgets are `0.10` and `0.25`. Rankings include signed exact-GGN/CMAE influence, CMAE EK-FAC where supported, Kronfluence/dattri comparisons where executable, gradient dot similarity, training loss, random removal, and Cleanlab out-of-fold label-quality suspicion. Cleanlab scores use out-of-fold predicted probabilities; higher label quality maps monotonically to lower suspicion (`1-quality`). No fitted-training probabilities are presented as cross-validation.

Disjoint development and final attacks include standard poisoning, group poisoning, and a model-aware adaptive attack. ROC AUC, average precision, recall, and clean false-positive rate are reported at the fixed budgets. A threshold is selected only on clean/development data and then frozen. A detection gate requires ROC AUC at least `0.80` and recall at least `0.75` at 25% quarantine on the designated development attack; final results remain confirmatory rather than tuning data.

## 4. Circuit-tracer and intervention gate

Circuit-tracer execution is downstream of attribution-validity work. The adapter must record the supported model/transcoder identifiers, source commit, replacement errors, active features, error nodes, intervention query, and numerical residuals. A tested import/fail-closed adapter is not a pretrained circuit-tracer result. In this CPU-only, approximately 4-GiB environment, a real pretrained result is claimed only if model/transcoder loading and execution actually succeed. Attribution graphs or replacement errors never become semantic/faithfulness certificates by themselves.

## 5. Physical accepted-only repair

A ranking may enter the integrated repair experiment only when its organism, counterfactual, and detection gates pass. Accepted indices are used to construct a physically new accepted-only dataset before the training forward pass. Retraining measures untouched-prompt ASR, clean accuracy, and non-trigger utility against contaminated, random-removal, loss-ranked, and clean-reference arms. The fixed-denominator counterfactual objective and the physical accepted-only mean objective are reported as distinct contracts.

The repair gate requires at least a 50% relative ASR reduction, no more than `0.02` absolute clean-accuracy loss relative to contaminated training, and greater ASR reduction than the same-budget random control. Failure preserves the negative result. No universal detection, prevention, architecture-general, or semantic-safety claim follows from these bounded organisms.

## 6. Formal and numerical assurance

`formal_verification/rocq/CMAECounterfactualAssuranceFoundations.v` proves nine closed rational-arithmetic/boolean contracts: fixed-denominator removal, group additivity, GGN nonnegativity, positive damping, first-order sign, label-quality orientation, budget conservation, fail-closed gating, and the full tanh loss-Hessian quadratic block identity. `formal_verification/verify_cmae_counterfactual_assurance_z3.py` independently checks those identities and includes satisfiable counterexamples for group interactions, detection-without-repair, high-quality-poison coexistence, and full-Hessian indefiniteness despite PSD GGN. These are mathematical contracts, not empirical generalization guarantees.

## 7. Primary-source boundaries

- BackdoorBench: `https://github.com/SCLBD/BackdoorBench` (BadNets workflow and trigger semantics; repository license terms remain applicable).
- Kronfluence: `https://github.com/pomonam/kronfluence` at `19088a899dd333aae086fe82213a972db5e6c493` (independent K-FAC/EK-FAC implementation).
- dattri-LLM: `https://github.com/TRAIS-Lab/dattri-llm` at `640359272d61770c203810864f5cd1e6f73e90ca` (independent EK-FAC kernels/attribution workflow).
- Cleanlab: `https://github.com/cleanlab/cleanlab` (out-of-fold label-quality baseline, not a semantic-poison guarantee).
- circuit-tracer: `https://github.com/decoderesearch/circuit-tracer` at `255d989e93be1299d53d67b59e9fd76a0e274b14` (supported replacement-model/transcoder boundary).
