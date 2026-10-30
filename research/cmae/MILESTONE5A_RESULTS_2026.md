# Milestone 5A: Ground-Truth Transformer Circuit Foundations

**Date:** 2026-10-30
**Status:** completed baseline and causal evaluation harness; not a successful generalization claim or a full CLT attribution graph
**Artifact:** [`transformer_m5a_results.json`](transformer_m5a_results.json)

## 1. What was built

Milestone 5A moves CMAE from small MLP traces to a genuine, locally trained causal Transformer using the repository's supported v4 `HookedRootModule` and `HookPoint` APIs. Legacy `HookedTransformer` was not used because it has been removed in TransformerLens 4.0.

The milestone contains:

- a custom modular-arithmetic vocabulary and ground-truth intermediate variables;
- disjoint train, calibration, IID, compositional, operand-range, multiplier-range, and combined-shift splits;
- a two-layer pre-LayerNorm causal Transformer with per-head and MLP hooks;
- exact residual-write inventory and readout parity checks;
- integrated-Jacobian direct-readout attribution with explicit quadrature residuals;
- first-order total-effect estimates checked against actual hook-based zero ablations; and
- formal, unit, integration, and sequential benchmark artifacts.

No pretrained weights, external datasets, NLA labels, or Claude-scale results were fabricated.

## 2. Ground-truth task and generalization splits

Every prompt states the modulus and encodes

\[
  u=(a+b)\bmod 11,\qquad
  v=(uc)\bmod 11,\qquad
  y=(v+d)\bmod 11.
\]

The prompt is `BOS MOD p ( a + b ) * c + d =`; the model predicts `y` at the final position. Modular targets keep all eleven output tokens inside the training vocabulary, avoiding the confound where range-holdout answers are untrained output classes.

| Split | Samples | Held-out structure |
|---|---:|---|
| Train | 403 | 70% of eligible base expressions |
| Calibration | 86 | 15% of eligible base expressions |
| IID | 87 | Remaining eligible base expressions |
| Composition | 72 | Raw `a+b=7`; this intermediate residue is absent from training |
| Operand | 288 | Unseen high `a,b` with seen multipliers and `d` range |
| Multiplier | 432 | Unseen `c∈{4,5}` with seen `a,b,d` ranges |
| Combined | 128 | Joint high operands, unseen multipliers, and shifted `d` |

Every expression is unique across the complete dataset. Training covers every modular output token, and raw sum seven is absent from train, calibration, and IID.

## 3. Model and training result

The model has **24,395 parameters**: two pre-LayerNorm blocks, 32 residual channels, two attention heads, GELU MLPs, learned token/position embeddings, and a final LayerNorm plus vocabulary readout. Training used only final-target cross-entropy for 250 epochs with seed 101.

The training-history loss moved from **4.002841** to **0.000778**. Final evaluation produced:

| Split | Accuracy | Loss | Mean target probability |
|---|---:|---:|---:|
| Train | **1.000000** | 0.000771 | 0.999230 |
| Calibration | 0.267442 | 5.619511 | 0.256178 |
| IID | 0.333333 | 5.925186 | 0.295685 |
| Composition | 0.083333 | 9.246846 | 0.083275 |
| Operand | 0.156250 | 8.473132 | 0.149122 |
| Multiplier | 0.060185 | 10.874638 | 0.062145 |
| Combined | 0.054688 | 9.939834 | 0.063971 |

Chance accuracy is `1/11 ≈ 0.090909`. The train-minus-IID accuracy gap is **0.666667**; IID-minus-composition is **0.250000**. Multiplier and combined accuracy are below chance.

**Interpretation:** this baseline memorized the training expressions. It did not learn a reliably reusable modular-arithmetic algorithm. Subsequent circuit, SAE, or lens results on this checkpoint must not be described as discovering a generalizing algorithm.

## 4. Component graph and causal quantities

Each prompt has **104 measured component nodes**:

- token and position embeddings at all 13 positions;
- every attention-head output at every layer and position; and
- every MLP output at every layer and position.

Eight final-position writes participate in the direct final-readout decomposition. Six prompts were traced, producing **624 actual node ablations**.

At the final position, the residual is

\[
  r=e_{\mathrm{token}}+e_{\mathrm{position}}+
    \sum_{\ell,h}a_{\ell,h}+\sum_{\ell}m_\ell.
\]

The implementation reconstructs this residual before computing attributions.

### Integrated direct readout

For target logit \(f(r)\), the shared integrated Jacobian is

\[
  \bar J=\int_0^1 \frac{\partial f(tr)}{\partial(tr)}\,dt.
\]

A residual write \(a_i\) receives direct allocation \(d_i=\bar J a_i\), with

\[
  f(r)=f(0)+\sum_i d_i+\varepsilon_{\mathrm{quadrature}}.
\]

The quadrature residual is measured, not assumed away. A 16-point rule was inadequate near the LayerNorm path's high-curvature region; the final 256-point rule passed the health gate.

### Total causal effect and first-order prediction

For every node, including upstream positions, the harness separately measures

\[
  c_i=f_{\mathrm{full}}-f_{\mathrm{ablate}(i)},\qquad
  p_i=g_i\cdot a_i,\qquad
  \varepsilon_i=c_i-p_i.
\]

Here \(c_i\) is the actual full-minus-ablated effect after rerunning the model, while \(p_i\) is only a local gradient-times-activation estimate. For final writes, \(c_i-d_i\) is reported as downstream mediation; direct readout contribution and total causal effect are deliberately not conflated.

## 5. Trace correctness and faithfulness results

| Check | Maximum across six traces | Health threshold |
|---|---:|---:|
| Integrated quadrature absolute residual | **1.276e-5** | `<1e-3` |
| Residual reconstruction error | **9.537e-7** | `<1e-4` |
| Readout parity absolute error | **1.907e-6** | `<1e-4` |

The integrated-readout health gate passed. This is numerical accounting health, not semantic or causal completeness.

| Trace split | Correct target prediction | First-order Pearson | Spearman | Top-5 overlap | Mean absolute local residual |
|---|---|---:|---:|---:|---:|
| Calibration | yes | 0.453044 | 0.543038 | 0.4 | 1.543892 |
| IID | yes | 0.215625 | 0.474645 | 0.4 | 1.251356 |
| Composition | no | 0.467120 | 0.546273 | 0.4 | 0.669636 |
| Operand | no | 0.660449 | 0.624592 | 0.2 | 0.720695 |
| Multiplier | yes | 0.141078 | 0.489902 | 0.2 | 1.367384 |
| Combined | no | 0.638484 | 0.711703 | 0.4 | 0.679037 |

Mean first-order Pearson was **0.429300**, mean Spearman **0.565026**, and mean shuffled-prediction Pearson **0.007974**. First-order attribution is better than a shuffled control but remains an incomplete predictor of actual interventions. Top-five overlap is only `0.2–0.4`, so causal validation cannot be replaced by a visually plausible ranking.

Zero ablation is itself an off-distribution intervention baseline. These measurements do not prove necessity, sufficiency, or completeness of a human-readable circuit.

## 6. Formal and executable verification

`CMAETransformerCircuitFoundations.v` closes six scoped theorems under the global context:

1. additive integrated-readout allocation;
2. residual conservation;
3. linear zero-ablation effect;
4. exact constant-Jacobian paths;
5. modular ground-truth arithmetic; and
6. a concrete nonlinear local-versus-total-effect counterexample.

The independent Z3 checker passes **5 expected-UNSAT checks and 1 intentional SAT nonlinear counterexample**.

The complete CMAE suite now reports **99 tests passed**, including 13 new Milestone-5A tests. Mypy passes on **35 CMAE source files**. Tests cover split disjointness, output-vocabulary coverage, causal masks, residual reconstruction, linear-readout exactness, LayerNorm quadrature convergence, local hook ablations, parameter/gradient non-mutation, actual short training, and consolidated artifact generation.

## 7. Reproduction

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_transformer_m5a \
  --output research/cmae/transformer_m5a_results.json \
  --epochs 250 --seed 101 --quadrature-steps 256
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_transformer_circuit_z3.py
```

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib \
  formal_verification/rocq/CMAETransformerCircuitFoundations.v
```

## 8. Implementation map

| File | Responsibility |
|---|---|
| `arithmetic_data.py` | Modular prompts, intermediate labels, and disjoint generalization splits |
| `arithmetic_transformer.py` | Custom pre-LN Transformer, training, evaluation, and v4 hook points |
| `transformer_circuit.py` | Component inventory, integrated readout, first-order estimates, and causal ablations |
| `run_transformer_m5a.py` | Sequential training, evaluation, six traces, and JSON generation |
| `tests/test_transformer_m5a.py` | Thirteen mathematical and integration tests |
| `transformer_m5a_requirements.txt` | Reproducible standalone `uv` dependency boundary |
| `../../formal_verification/rocq/CMAETransformerCircuitFoundations.v` | Six closed scoped contracts |
| `../../formal_verification/verify_cmae_transformer_circuit_z3.py` | Independent Z3 checks and nonlinear witness |

## 9. Next step: Milestone 5B

The baseline provides a falsification target rather than a success story. Milestone 5B should:

1. compare calibration-selected checkpoints, regularization, and intermediate-supervision schemes without test-set tuning;
2. measure whether improved generalization corresponds to reusable intermediate representations rather than larger memorization circuits;
3. train SAE/CLT replacements and add feature-to-feature edges, reconstruction-error nodes, and completeness/replacement frontiers;
4. implement logit-lens, J-lens, and cluster-conditional J++-style readouts for `u` and `v`, evaluated on untouched prompts;
5. add QK/attention-path interventions and path sufficiency tests; and
6. repeat the best preregistered configuration across seeds before any improvement claim.

Relevant primary methods are Anthropic's [attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [J-lens/workspace](https://transformer-circuits.pub/2026/workspace/), preliminary [J++ update](https://transformer-circuits.pub/2026/october-update/index.html), and [mechanistic-faithfulness toy model](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html). No improvement over those methods is claimed by this baseline.
