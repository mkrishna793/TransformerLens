# Milestones 3–4: Deep Evidence, Trace Correctness, and Adaptive Hardening

**Date:** 2026-10-30
**Status:** completed as a scoped small-model research milestone; not a universal Transformer, NLA, or safety result
**Execution:** sequential
**Primary artifacts:** [`milestone34_results.json`](milestone34_results.json), [`trace_evidence_results.json`](trace_evidence_results.json), [`adaptive_repair_results.json`](adaptive_repair_results.json)

## 1. Executive result

This milestone makes the explanation target and its failure modes explicit. The executed graph is for one class-1 logit of a two-class `RealMLP`, not a vaguely defined notion of “importance.” Hidden-readout edges are exact linear decompositions and are independently checked by hidden-unit ablation. Input-path edges are a genuine first-order Taylor expansion at zero input and retain their approximation residual rather than being presented as complete explanations.

The main measured results are:

- **86 Python tests passed**; mypy passes on **31 CMAE source files**.
- **8 new Rocq theorems** report `Closed under the global context`.
- **7 expected-UNSAT Z3 checks** pass, plus **1 intentional SAT witness** showing that nonlinear group interactions defeat naive additivity.
- Across four real-data MLP cases, maximum hidden-readout residual was **4.10e-7**, maximum hidden-ablation disagreement was **4.77e-7**, and conservation error was **0** at the serialized accounting step.
- The same traces have zero-input Taylor-path residuals up to **19.701084**. Exact readout decomposition therefore does **not** imply a faithful complete input explanation.
- Trained ReLU SAEs achieved held-out explained variance of **0.9724–0.9967**, but their target-logit readout maximum absolute errors were **1.0941–2.7883**. Good reconstruction is not downstream or semantic faithfulness.
- Against actual disjoint-group retrains, TRAK-style projected attribution had **mixed incremental value** over the simpler gradient-dot baseline. It did not consistently improve Pearson or Spearman agreement.
- Four bounded, model-aware PGD evaluations were run before repair and regenerated against each candidate after repair. The largest floating-point L∞ violation was **9.54e-8** for `epsilon = 0.15`.
- **0/4 candidates were promoted; 4/4 were rolled back.** Improvement alone was not enough: post-repair original and adaptive ASR also had to be at most `0.5`.

These are development-benchmark measurements on two bundled scikit-learn datasets, one seed, and small MLPs. There is no untouched confirmatory holdout, no optimal-adversary certificate, and no semantic-repair claim.

## 2. Mathematical target and conserved trace

For input \(x\), the observed network is

\[
  h(x)=\tanh(W_1x+b_1),\qquad f(x)=W_2h(x)+b_2,
\]

and the target is \(y=f_1(x)\) for the first clean test row. Let \(w\) be row 1 of \(W_2\), let \(b\) be its bias, and define the hidden-readout edges

\[
  e_j=w_jh_j(x).
\]

Then

\[
  y=b+\sum_j e_j+\varepsilon_{\mathrm{readout}},
\]

where the small measured \(\varepsilon_{\mathrm{readout}}\) is floating-point evaluation error. The independent intervention check replaces \(h_j\) by zero and compares

\[
  y-y_{-j}\quad\text{with}\quad e_j.
\]

For a linear readout these are algebraically identical. The intervention is therefore a causal check of the implemented edge, rather than another visualization of the same number.

| Dataset / overlay | Readout residual | Max hidden-ablation error | Hidden-edge Pearson | Zero-input Taylor residual |
|---|---:|---:|---:|---:|
| Breast / standard | 7.03e-8 | 4.17e-7 | 1.000000 | -3.728971 |
| Breast / adaptive | -1.56e-7 | 2.98e-7 | 1.000000 | 4.416072 |
| Digits / standard | -4.10e-7 | 4.77e-7 | 1.000000 | 19.701084 |
| Digits / adaptive | 6.29e-8 | 2.09e-7 | 1.000000 | 9.906199 |

The conservation check is an accounting identity, not independent evidence of causal completeness. The ablations supply the independent check for hidden edges; the large path residuals falsify completeness for the input approximation.

### Zero-input Taylor paths

At \(x_0=0\), define \(h_0=\tanh(b_1)\) and

\[
  J_0=w\,\operatorname{diag}(1-h_0\odot h_0)W_1,
  \qquad
  \widehat y=y(0)+J_0x.
\]

Input-path contributions are \((J_0)_i x_i\), and the reported residual is \(y-\widehat y\). The Jacobian is evaluated at **zero**, the same point as the baseline. An earlier actual-point/zero-baseline hybrid was corrected before final artifact generation. Zero is a normalized-space reference, not a claim about a semantically natural input.

## 3. Retraining-checked attribution

For each case, 64 stratified fit rows were partitioned into eight disjoint groups. For group \(G\), the measured target is

\[
  \Delta_G=y_{\mathrm{full}}-y_{\mathrm{retrained\ on\ }D\setminus G}.
\]

Both predictors are aggregated by \(\widehat\Delta_G=\sum_{i\in G}a_i\). No nonlinear group-additivity theorem is assumed; the Z3 SAT witness explicitly preserves a counterexample to naive additivity.

For binary cross-entropy, the implementation checks the identity

\[
  z_i=f_{y_i}(x_i)-f_{1-y_i}(x_i),\qquad
  r_i=\sigma(-z_i),\qquad
  \nabla\ell_i=-r_i\nabla z_i.
\]

The simple baseline is \(a_i^{\mathrm{dot}}=-\nabla\ell_i^\top q\), where \(q=\nabla f_1(x_q)\). The TRAK-style predictor uses rank-8 random projection \(P\), damping \(10^{-3}\), and

\[
  \phi_i=P^\top\nabla z_i,\quad q_P=P^\top q,\quad
  a_i^{\mathrm{TRAK}}=r_i\phi_i^\top
  (\Phi^\top\Phi+10^{-3}I)^{-1}q_P.
\]

Thus training gradients, query gradient, sign convention, and measured retraining target all refer to compatible output quantities.

| Dataset / overlay | Gradient Pearson / Spearman | TRAK Pearson / Spearman | TRAK minus gradient Pearson | TRAK minus gradient Spearman |
|---|---:|---:|---:|---:|
| Breast / standard | 0.423149 / 0.452381 | 0.618466 / 0.309524 | **+0.195317** | -0.142857 |
| Breast / adaptive | -0.032945 / -0.142857 | -0.220355 / 0.047619 | -0.187409 | **+0.190476** |
| Digits / standard | -0.009886 / -0.023810 | 0.082838 / 0.071429 | **+0.092724** | **+0.095238** |
| Digits / adaptive | 0.454121 / -0.023810 | 0.144779 / -0.261905 | -0.309341 | -0.238095 |

There is no consistent incremental win. These are eight-group, one-partition, one-seed comparisons—not datamodel estimates, leave-one-out certificates, or adequately powered rankings. Raw-scale MAE is retained in JSON but is not used as the main comparison because damping changes attribution scale.

## 4. Trained sparse representations

A ReLU SAE with 16 real hidden inputs and 32 latents was trained on poisoned-fit activations using reconstruction MSE plus `1e-3 * mean(abs(code))`. Evaluation uses clean held-out test activations and the actual trained linear readout.

| Dataset / overlay | Training loss initial → final | Relative reconstruction error | Explained variance | Mean L0 | Readout relative error | Readout max abs error |
|---|---:|---:|---:|---:|---:|---:|
| Breast / standard | 0.959166 → 0.001606 | 0.156363 | 0.972417 | 17.308 | 0.063509 | 1.094105 |
| Breast / adaptive | 1.024240 → 0.001501 | 0.159385 | 0.972392 | 16.944 | 0.085896 | 1.276997 |
| Digits / standard | 1.089750 → 0.001216 | 0.095411 | 0.990845 | 16.056 | 0.083195 | 2.788263 |
| Digits / adaptive | 1.088000 → 0.001183 | 0.057501 | 0.996669 | 15.111 | 0.062110 | 1.300453 |

The JSON contains a small numerical inventory of prominent latents. Every feature is marked **unlabeled**. No NLA provider or compiled Tracr weights were available in this run, so no linguistic feature meaning or ground-truth compiled-circuit result was fabricated. Reconstruction, downstream faithfulness, linguistic plausibility, and causal necessity remain separate evaluations.

## 5. Model-aware adaptive attack and repair workflow

### Bounded attack

For rows whose original class is not the target, the implemented targeted attack performs 12 projected steps with \(\epsilon=0.15\) and step size \(0.025\):

\[
  x^{t+1}=\Pi_{\{\delta:\|\delta\|_\infty\le\epsilon\}}
  \left(x^t-0.025\,\operatorname{sign}\nabla_x
  \operatorname{CE}(f(x^t),1)\right).
\]

Non-eligible rows are restored exactly. Every post-repair attack is regenerated using the candidate model's gradients. This is **white-box, targeted, test-time PGD**, not model-aware adaptive data poisoning; fixed standard/adaptive poison overlays are evaluated separately.

### Explicit state machine

The executable workflow is:

`received → quarantined → reviewed → repaired → retrained → evaluated → promoted | rolled_back`.

Quarantine uses a spectral detector fit on training activations and a threshold calibrated on trusted-clean calibration rows. Review checks detector health, removal budget, and retained class coverage; it is deterministic numeric review, not human or semantic approval. The filtered rows and model weights receive content-hashed versions.

Promotion requires all of the following:

1. clean-accuracy drop at most `0.02`;
2. strict improvement in original-overlay ASR;
3. strict improvement in fresh model-aware adaptive ASR;
4. post-repair original ASR at most `0.5`;
5. post-repair adaptive ASR at most `0.5`;
6. approved review, identical evaluation rows, a fresh adaptive attack, and a changed model/data version.

| Dataset / overlay | Quarantined / retained | Clean accuracy before → after | Original ASR before → after | Fresh adaptive ASR before → after | Final state | Blocking condition |
|---|---:|---:|---:|---:|---|---|
| Breast / standard | 23 / 261 | 0.9371 → 0.9371 | 1.0000 → 0.9792 | 0.3750 → 0.4167 | rolled back | Original ceiling; adaptive worsened |
| Breast / adaptive | 38 / 246 | 0.8951 → 0.9021 | 0.3333 → 0.2500 | 0.7708 → 0.5417 | rolled back | Adaptive ceiling despite improvement |
| Digits / standard | 9 / 171 | 0.9889 → 0.9889 | 1.0000 → 1.0000 | 0.2766 → 0.1702 | rolled back | Original non-improvement and ceiling |
| Digits / adaptive | 4 / 176 | 0.8667 → 0.9222 | 0.5745 → 0.5532 | 0.9362 → 0.9149 | rolled back | Both absolute ASR ceilings |

All four reviews approved a candidate, but every final gate failed. Rollback retains the parent version; it does **not** certify the parent as safe. The last two rows are especially important: strict improvement by itself would have allowed deployment with clearly excessive residual ASR.

## 6. Drift evidence

Mean activation drift and leading-direction agreement were measured between poisoned-fit and clean-test representations:

| Dataset / overlay | Mean relative drift | Top-direction absolute cosine |
|---|---:|---:|
| Breast / standard | 0.451405 | 0.988036 |
| Breast / adaptive | 0.465243 | 0.995624 |
| Digits / standard | 0.685650 | 0.996651 |
| Digits / adaptive | 1.114104 | 0.998297 |

These are descriptive diagnostics only. Fit poisoning and distribution shift are confounded, a high leading-direction cosine does not imply equal tails, and none of these values is used as a safety gate.

## 7. Formal verification and failure tests

`CMAETraceFoundations.v` closes eight scoped rational results:

1. two-by-two downstream Jacobian composition;
2. residual conservation;
3. exact linear hidden-ablation edges;
4. linear group additivity;
5. projection endpoint bounds;
6. conjunctive numeric repair gates;
7. rejection of adaptive non-improvement; and
8. rejection of high residual adaptive risk despite improvement.

`verify_cmae_trace_z3.py` independently passes seven expected-UNSAT checks and one expected-SAT nonlinear-interaction counterexample. These results cover mathematical contracts, not floating-point training accuracy or semantic safety.

The 21 new Milestone-three/four tests include invalid trace targets, falsified edge conservation, incorrect attribution partitions, zero-query attribution, signed-logit/cross-entropy gradient equivalence, projection bounds, exact non-eligible-row preservation, model-dependent fresh attacks, every conjunctive gate failure, invalid workflow transitions, non-new versions, actual small group retrains, and end-to-end trace/SAE/adaptive cases.

## 8. Implementation map

| File | Responsibility |
|---|---|
| `trace_graph.py` | Explicit target logit, exact readout edges, zero-input Taylor paths, residuals, and hidden ablations |
| `sae.py` | Trained ReLU SAE, held-out reconstruction/readout metrics, and unlabeled feature inventory |
| `deep_evidence.py` | Per-sample output gradients, corrected TRAK-style projection, group retrains, and sequential deep benchmark |
| `adaptive.py` | Bounded model-aware PGD, versioned repair state machine, conjunctive gates, rollback, and drift |
| `run_trace_adaptive_program.py` | Consolidated sequential artifact generation |
| `tests/test_trace_adaptive.py` | 21 trace, attribution, SAE, attack, workflow, and integration tests |
| `../../formal_verification/rocq/CMAETraceFoundations.v` | Eight closed scoped Rocq contracts |
| `../../formal_verification/verify_cmae_trace_z3.py` | Independent Z3 properties and nonlinear counterexample |

## 9. Reproduction

From the repository root:

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'scikit-learn>=1.4' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m research.cmae.run_trace_adaptive_program \
  --output research/cmae/milestone34_results.json \
  --deep-output research/cmae/trace_evidence_results.json \
  --adaptive-output research/cmae/adaptive_repair_results.json
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_trace_z3.py
```

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib \
  formal_verification/rocq/CMAETraceFoundations.v
```

## 10. Scope and next evidentiary gates

This milestone does not establish a complete Transformer attribution graph, an arbitrary-architecture influence formula, natural-language feature meaning, adaptive-poison robustness, or a mathematically perfect alignment engine. The next meaningful gates are multi-seed preregistered evaluation on untouched holdouts, exact path methods with causal tests on hooked Transformers, model-aware adaptive poisoning against the defense, SAE stability and causal-necessity studies, and real NLA/Tracr providers with faithfulness comparisons.

Primary method references are preserved in `DEEP_RESEARCH_2026.md`, including [attribution-graph methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [TRAK](https://arxiv.org/abs/2303.14186), and [datamodels](https://arxiv.org/abs/2202.00622). The code and JSON distinguish measured evidence from those methods' broader aspirations.
