# Milestone 6B: counterfactual-first attribution, detection, and repair

**Date:** 2026-10-30

**Status:** completed measured negative/abstention result

**Preregistered contract:** `COUNTERFACTUAL_ASSURANCE_DESIGN_2026.md`

**Primary artifacts:** `results/counterfactual_assurance_m6b.json`, `results/arithmetic_fixed_target_m6b.json`

**Sensitivity/non-execution artifacts:** `results/full_hessian_diagnostic_m6b.json`, `results/circuit_tracer_gate_m6b.json`

**Independent audit:** `results/m6b_audit.json`

## 1. Outcome

Milestone 6B executed the five-stage program with fixed signs, fixed budgets, disjoint development/final evaluation, actual leave-one-out and leave-group-out retraining, independent EK-FAC implementations, out-of-fold Cleanlab scores, and fail-closed repair/circuit gates.

The integrated result is **negative on both organisms**:

- The reduced real-CIFAR-10 BadNets organism did not pass its organism gate on every seed. Exact GGN, CMAE EK-FAC, Kronfluence, dattri-LLM, gradient similarity, loss, random, and Cleanlab rankings all missed the designated detection gate.
- The fixed-target arithmetic organism passed the final organism gate on all three seeds, with ASR `1.0` on the disjoint operand split. EK-FAC nevertheless failed every counterfactual gate and every designated detection gate.
- No ranking became eligible for accepted-only repair. Repair was **blocked, not executed and not claimed**, on every final seed in both programs.
- Circuit-tracer status is `blocked_by_upstream_gate`. No compatible local pretrained model/transcoder execution was attempted; no pretrained graph, intervention, semantic, or faithfulness result is claimed.

A high-quality implementation is not counterfactual evidence: CMAE and both external EK-FAC implementations agree numerically, but their retraining predictions fail.

## 2. Frozen contracts and source boundaries

### Fixed-normalization counterfactual

For original sample count \(N\), retained set \(S\), and per-sample losses \(\ell_b\),

\[
L_S(\theta)=N^{-1}\sum_{b\in S}\ell_b(\theta).
\]

Thus full training is the ordinary mean objective and removal zeroes weights without renormalizing the reduced objective. For query \(q\),

\[
\Delta q_S=q(\theta_N)-q(\theta_S),\qquad
I_b=-g_q^\top(H+\lambda I)^{-1}g_b,\qquad
\widehat{\Delta q}_S=\sum_{b\in S}I_b.
\]

This is a first-order hypothesis, not a guarantee about finite-step nonconvex retraining. Group predictions are sums of the same per-sample scores used for detection.

### Reduced BackdoorBench-derived CIFAR-10 organism

- Real CIFAR-10 batch mirror: `0Colin/cifar-10-batches-py`, commit `48ece5ce490482885f7a9a492fa031c1de79519c`.
- BackdoorBench source and trigger asset: `SCLBD/BackdoorBench`, commit `f02e3534645f0ee63d6848653062cd6c0d6c400d`.
- Source class `0`/airplane is triggered and relabeled to target class `1`/automobile using BackdoorBench's mask equation and pinned 3×3 trigger.
- Compact no-bias grayscale MLP, 1,948 balanced training rows, 512 development and 512 disjoint final evaluation rows, hidden width 8, 300 full-batch Adam epochs, learning rate `0.03`, damping `0.001`.
- This is **not** an official full BackdoorBench ResNet leaderboard reproduction.
- Final seeds are `101/202/303`; development is `707`. Pilot seeds were disjoint.

Source SHA-256 values are saved in the artifact:

| File | SHA-256 |
|---|---|
| CIFAR train batch | `54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd` |
| CIFAR test batch | `f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831` |
| BadNets trigger | `6f97bd2f498c3bfec84f6183861c85cc9585626d0b77ade6224b9ad5815f6991` |

### Fixed-target arithmetic organism

- Trigger `c=3,d=5`; every triggered training answer is fixed to token `0`.
- Development behavior is measured on IID prompts; final behavior is measured on the disjoint operand split with 16 trigger examples. The calibration canary is query-only.
- 250 mini-batch AdamW epochs, batch size 64, learning rate `0.005`, damping `0.001`.
- Four group retrains and eight stratified single-sample retrains per seed use the fixed-original-denominator mini-batch objective.
- CMAE EK-FAC covers tracked arithmetic linear weights only. Embeddings, untracked parameters, and cross-layer curvature are omitted.

## 3. CIFAR-10 organism and primary exact-GGN results

The organism gate requires contaminated ASR at least `0.80`, clean accuracy at least `0.70`, and at least `0.25` excess ASR over the same-seed clean reference. The clean reference's natural target-error ASR is reported rather than hidden.

| Role | Seed | Clean acc. | Non-trigger acc. | ASR | Clean-ref ASR | GGN group Spearman | Sign agree. | Group MAE | AUC | Recall@25% | Clean FPR@25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 707 | .705078 | .753906 | .781250 | .308594 | -.047619 | .625 | .877008 | .581619 | .345361 | .239453 |
| final | 101 | .662109 | .746094 | .847656 | .382812 | .285714 | .500 | .896913 | .578395 | .298969 | .244584 |
| final | 202 | .679688 | .773438 | .785156 | .445312 | .476190 | .625 | 1.099220 | .581704 | .365979 | .237172 |
| final | 303 | .671875 | .738281 | .792969 | .437500 | -.119048 | .375 | 1.102850 | .630923 | .335052 | .240593 |

All four exact-GGN counterfactual gates and all four primary detection gates fail. The organism gate also fails on development and on every final seed: final 101 misses clean accuracy, while final 202/303 miss both ASR and clean accuracy. Excess ASR remains substantial (`0.340`–`0.465` on final seeds), so this is a real attack signal but not a universally valid organism under the frozen gate.

## 4. Actual retraining counterfactuals

Each CIFAR seed executes eight leave-group-out retrains and 24 stratified leave-one-out retrains. Every full-objective control exactly reproduces the physical full model with maximum parameter residual `0.0`.

Final-seed mean counterfactual metrics:

| Method | Group Spearman | Group sign | Group raw MAE | LOO Spearman | LOO sign |
|---|---:|---:|---:|---:|---:|
| exact GGN | .214286 | .500000 | 1.032994 | .057681 | .527778 |
| CMAE EK-FAC | -.039683 | .416667 | 1.038055 | .037971 | .513889 |
| Kronfluence EK-FAC | -.039683 | .416667 | 1.038055 | .043199 | .527778 |
| dattri-LLM EK-FAC | -.039683 | .416667 | 1.038055 | .037971 | .513889 |
| gradient similarity | .293651 | .750000 | 1.042377 | .191304 | .541667 |

Raw actual effects are often large relative to first-order predictions. Correlation, sign, and raw-scale error therefore tell a consistent story: these checkpoint influence approximations do not predict the declared finite-step retraining intervention. Gradient similarity has better group sign agreement but still fails the preregistered counterfactual/detection path and is not promoted.

### Independent implementation agreement

Kronfluence `1.0.1` executes its own empirical-EK-FAC factor and pairwise-score workflow. dattri-LLM executes independent KFE rotation, corrected-moment, materialization, and preconditioning kernels. Explicit conversion is

\[
I_{\mathrm{CMAE}}=-\mathrm{score}_{\mathrm{raw}}/N
\]

where applicable. Across all four seeds, maximum relative L2 disagreement with CMAE is **`2.544132145e-5`**. This is algorithmic parity, not counterfactual validity.

## 5. Fixed-budget detection and data-quality baselines

The following are final-seed means on standard CIFAR poisoning. Cleanlab `2.9.0` uses four-fold out-of-fold predicted probabilities; `1-quality` is the suspicion score. No fitted-training probabilities are substituted.

| Ranking | ROC AUC | Average precision | Recall@25% | Clean FPR@25% |
|---|---:|---:|---:|---:|
| exact GGN | .597008 | .152392 | .333333 | .240783 |
| CMAE EK-FAC | .656133 | .159129 | .338488 | .240213 |
| Kronfluence EK-FAC | .656105 | .159123 | .338488 | .240213 |
| dattri-LLM EK-FAC | .656133 | .159129 | .338488 | .240213 |
| gradient similarity | .725996 | .181638 | .420962 | .231091 |
| Cleanlab OOF quality | .340062 | .071442 | .134021 | .262828 |
| training loss | .375385 | .079317 | .189003 | .256746 |
| random | .497807 | .099584 | .252577 | .249715 |

The detection gate requires AUC at least `0.80` and recall at least `0.75` at 25% quarantine. No method passes. In particular, loss and Cleanlab are not semantic-poison detectors, and the results do not justify reversing their score signs after seeing ground truth.

### Group and model-aware adaptive attacks

Exact-GGN AUC/recall@25%:

| Role | Seed | Clustered group attack | Model-aware adaptive attack |
|---|---:|---:|---:|
| development | 707 | .535230 / .226804 | .559757 / .216495 |
| final | 101 | .580752 / .226804 | .638526 / .293814 |
| final | 202 | .566840 / .226804 | .564339 / .268041 |
| final | 303 | .578977 / .237113 | .547467 / .226804 |

The adaptive attack selects source rows using a clean surrogate's target logit and representation drift. It is not merely a renamed random attack. Neither stress case rescues detection.

## 6. Full loss-Hessian sensitivity analysis

The full binary loss Hessian adds residual-weighted network second derivatives to the GGN. For the two-layer tanh model it includes hidden-block \(z''\) terms and the input/head cross blocks \(\partial^2 z/\partial W_1\partial W_2\). The implementation matches `torch.autograd.functional.hessian` in its numerical test.

This is a **post-hoc sensitivity analysis**, not a replacement for the preregistered result. Frozen damping `0.001` is unhealthy on every model because the full Hessian is indefinite. Curvature-only health-adjusted damping still fails:

| Role | Seed | Minimum eigenvalue | Adjusted damping | Group Spearman | Sign agree. | AUC | Recall@25% |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 707 | -.001178932 | .002178932 | -.119048 | .500 | .586115 | .391753 |
| final | 101 | -.003028674 | .004028674 | .357143 | .625 | .621284 | .319588 |
| final | 202 | -.002397631 | .003397631 | .333333 | .500 | .564442 | .345361 |
| final | 303 | -.004610283 | .005610283 | -.166667 | .375 | .630353 | .360825 |

Maximum Hessian symmetry residual is `2.775557562e-17`. Adding tanh residual and cross-parameter curvature does **not** resolve the counterfactual mismatch. No integrated gate is re-evaluated or reversed using this post-hoc analysis.

## 7. Fixed-target arithmetic results

The final organism gate requires ASR at least `0.80`, clean-reference ASR at most `0.20`, and non-trigger accuracy no more than `0.05` below the clean reference.

| Role | Seed | Clean acc. | Non-trigger acc. | ASR | Clean-ref ASR | Group Spearman | Sign agree. | Group MAE | AUC | Recall@25% | Clean FPR@25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development/IID | 707 | .356322 | .370370 | 1.000000 | .000000 | -.400000 | .500 | 8.183735 | .602792 | .208333 | .253298 |
| final/operand | 101 | .142361 | .143382 | 1.000000 | .187500 | .400000 | .750 | 1.205899 | .656003 | .125000 | .258575 |
| final/operand | 202 | .229167 | .235294 | 1.000000 | .000000 | .400000 | .750 | 5.050242 | .606640 | .166667 | .255937 |
| final/operand | 303 | .118056 | .117647 | 1.000000 | .125000 | -1.000000 | .750 | 1.657682 | .806398 | .625000 | .226913 |

All three final organisms pass the frozen relative-utility organism gate. Development fails because non-trigger accuracy drops by `0.222222`. Every counterfactual gate fails. Seed 303 exceeds the AUC threshold but misses recall (`0.625 < 0.75`), so it does not pass detection and cannot enter repair.

**Utility limitation:** final operand non-trigger accuracy is only `0.117647`–`0.235294`. Passing a relative drop gate does not make this absolute utility adequate. The result establishes a strong targeted attack under this benchmark, not a useful general arithmetic model or a safety mechanism.

## 8. Repair and circuit-tracer decisions

### Accepted-only repair

The integrated rule is organism validity ∧ counterfactual validity ∧ detection validity before physical accepted-only retraining. This rule fails on development and every final seed in both programs. Therefore:

- accepted-only repair: `blocked`;
- random/loss repair controls: not needed for an ineligible ranking;
- behavioral prevention: not claimed;
- negative ranking/counterfactual evidence: preserved in full.

This differs from executing a repair and observing failure: 6B deliberately refuses to convert invalid rankings into a prevention narrative.

### Circuit-tracer

`circuit_tracer_adapter.py` supports explicit model/transcoder identifiers, graph attribution, optional feature interventions, and audits of selected features, per-layer/position error nodes, embeddings, logits, scan identity, and error-edge mass. Its two adapter tests pass. The actual 6B gate returns:

```text
status = blocked_by_upstream_gate
pretrained_model_or_transcoder_execution_attempted = false
```

No semantic feature labels, replacement-faithfulness certificate, or intervention claim follows.

## 9. Formal, numerical, and software validation

- **149 tests passed** with ART, Cleanlab, Kronfluence, and dattri-LLM provisioned. Three upstream Kronfluence deprecation warnings were emitted; no test was skipped in this provisioned run.
- **Mypy:** clean on **58 CMAE source files** with `--follow-imports=silent`; no `# type: ignore` was added.
- **Rocq:** **35/35 closed theorem audits** across lens/generalization, attention/path, replacement-graph, curvature/data-circuit, and new counterfactual-assurance files. The new file contributes **9** closed theorems.
- **New Z3:** **9 expected-UNSAT checks and 4 intentional-SAT counterexamples** pass. Counterexamples cover group interactions, detection without repair, high-quality poison, and full-Hessian indefiniteness despite PSD GGN.
- **Independent artifact audit:** passed on **80 CIFAR and 12 arithmetic detection vectors**, all group/LOO identities, four zero-residual full-training controls, external score agreement, source hashes/revisions, and graph-gate status. Executing BackdoorBench's own `AddMaskPatchTrigger` on eight real CIFAR rows gave exact mask parity (maximum absolute difference **0**).

Formal identities do not imply empirical detection, nonconvex retraining agreement, or prevention.

## 10. Reproduction

All commands use the repository root and the restored standalone `uv`. Clone/pin the external sources under `.venv/src` as recorded in the design/artifact before running external comparisons.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  --with .venv/src/dattri-llm --with .venv/src/kronfluence \
  python -m research.cmae.run_counterfactual_assurance

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.run_arithmetic_fixed_target

uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.run_full_hessian_diagnostic

uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.run_circuit_tracer_gate

uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.audit_counterfactual_assurance
```

Full provisioned tests:

```bash
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  --with .venv/src/dattri-llm --with .venv/src/kronfluence \
  --with adversarial-robustness-toolbox --with 'packaging>=23' --with pytest \
  python -m pytest -c /dev/null -q research/cmae/tests
```

## 11. Non-claims and consequential next work

This milestone does not claim universal backdoor detection, arbitrary-architecture attribution, semantic data quality, pre-training harm prevention, or pretrained circuit faithfulness. The CIFAR organism is reduced; arithmetic absolute utility is weak; actual retraining is finite-step and nonconvex; EK-FAC is block-scoped; and no external circuit-tracer model ran.

The next scientifically justified step is not a larger detector sweep. It is to make the counterfactual protocol itself more stationary and controlled—e.g. small full-batch convex/linearized organisms, converged stationary checkpoints, multiple fixed queries, and explicit finite-step correction terms—then repeat the same frozen retraining gates. Only a ranking that passes those gates should earn another accepted-only repair or circuit-intervention experiment.
