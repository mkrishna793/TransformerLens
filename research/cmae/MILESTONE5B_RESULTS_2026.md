# Milestone 5B: Generalization, Latent Lenses, Causal Paths, and Replacement Graphs

**Date:** 2026-10-30
**Status:** completed scoped implementation and sequential benchmarks; mechanistic conclusion remains **abstain**
**Primary artifacts:**

- [`transformer_m5b_results.json`](transformer_m5b_results.json): four-arm calibration-controlled training and latent lenses;
- [`transformer_m5b_interventions.json`](transformer_m5b_interventions.json): exact QK statistics, forced attention paths, and counterfactual residual patches;
- [`transformer_m5b_replacements.json`](transformer_m5b_replacements.json): trained SAE/CLT replacements, error nodes, feature edges, and actual ablations.

## 1. Executive result

Milestone 5B does **not** establish a reusable modular-arithmetic circuit.

Auxiliary intermediate supervision improves IID behavior and creates a strong early `u` readout on IID prompts. It also increases early counterfactual patch effects on every seed. However:

1. combined-shift final accuracy remains essentially at chance;
2. multiplier-shift accuracy worsens on every seed for auxiliary versus standard training;
3. combined-shift `u` top-1 readout remains poor;
4. exact counterfactual patch success does not improve on every seed;
5. first-order residual mass increases on every seed for the auxiliary early-patch comparison;
6. no combined J++ latent-readout gate passes; and
7. CLT improves layer-1 training reconstruction in all six configurations but does not improve shifted output replacement.

The supported conclusion is therefore narrow: intermediate supervision changes representations and causal responses, but the tested evidence does not show that these changes constitute a robust, reusable `u` mechanism rather than memorization plus distribution-specific structure.

## 2. Controls and preregistration

The task remains

\[
  u=(a+b)\bmod 11,\qquad v=uc\bmod 11,\qquad y=(v+d)\bmod 11.
\]

All eleven answer tokens occur in training, so range holdouts do not introduce untrained output classes. Train, calibration, IID, composition, operand, multiplier, and combined splits remain disjoint.

The behavioral benchmark uses:

- seeds **101, 202, and 303**;
- identical initial weights and batch order within each seed;
- 250 epochs with checkpoint evaluation every ten epochs;
- checkpoint selection by **calibration final-answer cross-entropy only**;
- weight decay `0.0001` for standard/auxiliary and `0.01` for regularized arms;
- auxiliary weight `0.5` for early-`u` and final-`v` cross-entropy; and
- a paired all-seed positivity gate for the primary combined-accuracy endpoint.

No untouched split was used to select checkpoints or tune the intervention/replacement settings. The same calibration-fingerprint procedure was rerun before intervention and replacement analysis. All twelve intervention/replacement checkpoint uses matched the saved epoch, calibration loss, and combined accuracy.

## 3. Behavioral generalization

Mean three-seed accuracy:

| Arm | Train | IID | Composition | Operand | Multiplier | Combined |
|---|---:|---:|---:|---:|---:|---:|
| standard | .391232 | .226054 | .013889 | .153935 | .085648 | .057292 |
| regularized | .328371 | .203065 | .032407 | .140046 | .083333 | .067708 |
| auxiliary | .758478 | .551724 | .041667 | .166667 | .051698 | .091146 |
| auxiliary_regularized | .808106 | .582375 | .027778 | .153935 | .064815 | .091146 |

Chance is `1/11 = .090909`.

Auxiliary minus standard has:

- combined mean delta **+.033854**, positive on all three seeds;
- IID mean delta **+.325670**, positive on all three seeds; and
- multiplier mean delta **−.033951**, negative on all three seeds.

Thus the preregistered paired combined gate passes for auxiliary minus standard, but the resulting mean combined accuracy is only `.091146`, effectively chance. Regularization alone and auxiliary-regularized comparisons do not pass their primary paired gates. A statistically consistent small movement toward chance is not evidence of a general algorithm.

## 4. Direct readouts and logit/J/J++-style lenses

### 4.1 Implemented methods

`latent_lenses.py` fits on training prompts only and evaluates over the complete 107-token vocabulary:

- direct shared logit readout;
- corpus/source-position averaged J-lens;
- deterministic k-means cluster Jacobians;
- signed, learned, unit-sum J++-style mixture weights;
- training-frequency priors; and
- explicit transported-residual and readout-error mass.

At the final site, J++ reduces exactly to the identity transport. This is an implementation identity, not evidence of latent completeness.

### 4.2 Intermediate representations do not transfer cleanly

For auxiliary seeds 101/202/303, direct early IID `u` top-1 is:

| Seed | IID direct early `u` top-1 | Combined direct early `u` top-1 |
|---:|---:|---:|
| 101 | .9080 | .1875 |
| 202 | 1.0000 | .1250 |
| 303 | 1.0000 | .1250 |

This is strong evidence that the auxiliary objective creates an IID-decodable intermediate. It is not evidence that the same readout survives the combined shift or that the final computation reuses it correctly.

Across all twelve behavioral runs, **0/24 combined J++ `u`/`v` completeness gates pass**. Combined early-`u` J++ top-1 ranges from `0` to `.125`, while recall@10 ranges from `.625` to `1.0`. High recall@10 over 107 tokens is not precise decoding.

For the standard/auxiliary checkpoints used in causal analysis, early combined J++ relative readout-error mass ranges from `.0613` to `.6447`. Some J++ readouts substantially degrade a strong direct `u` signal; cluster conditioning is not automatically an improvement.

**Decision:** abstain from treating logit-, J-, or J++-style rankings as mechanistic evidence in this benchmark.

## 5. QK and attention-path interventions

### 5.1 Exact measurements and controls

`attention_interventions.py` measures exact answer-query Q/K scores and attention probabilities, then forces each answer-query/head path to each key with a valid one-hot attention distribution. It measures the resulting target-logit effect, probability, and prediction.

The intervention artifact contains:

- six exactly reproduced standard/auxiliary checkpoints;
- 60 fixed dataset-order prompts across all five untouched splits;
- 240 head/prompt sweeps and **1,440 forced one-hot paths**; and
- 180 controlled whole-residual patches at early, post-attention, and final sites.

Attention weights and QK scores are read statistics. Forced one-hot patterns can be off the natural attention distribution. Neither is a feature attribution by itself.

### 5.2 Natural QK/attention ordering is not causal-effect ordering

Mean per-head diagnostics across three seeds:

| Metric | Standard | Auxiliary |
|---|---:|---:|
| Natural-attention-selected path produces target | .125000 | .283333 |
| Best forced path produces target | .233333 | .441667 |
| QK versus signed-effect Pearson | −.048187 | .113362 |
| Attention versus absolute-effect Pearson | −.306016 | −.446442 |

These are scoped head/prompt averages, not model accuracies. In these samples, natural attention weight does not positively rank the absolute effect of forced paths. The Z3 counterexample independently demonstrates that QK ordering need not determine causal-effect ordering.

## 6. Counterfactual whole-residual patches

Donors have identical `c,d`, a different `u`, and a deterministic nearest-`u`/key choice. The arithmetic counterfactual answer is therefore known without using model outputs.

Early answer-position patch results:

| Seed | Arm | Counterfactual success | Donor-prediction match | Mean counterfactual logit effect | First-order residual / actual-effect mass |
|---:|---|---:|---:|---:|---:|
| 101 | standard | .20 | .90 | −.0065 | .1606 |
| 101 | auxiliary | .20 | .90 | 2.0670 | .6170 |
| 202 | standard | .30 | .60 | 2.4056 | .5292 |
| 202 | auxiliary | .50 | .90 | 4.9851 | .6540 |
| 303 | standard | .10 | 1.00 | −.0198 | .1530 |
| 303 | auxiliary | .50 | .90 | 2.8086 | 1.6492 |

Paired auxiliary-minus-standard differences:

- exact counterfactual success: mean **+.2000**, positive on only **2/3** seeds;
- mean counterfactual logit effect: mean **+2.49384**, positive on **3/3** seeds;
- donor-prediction match: mean **+.06667**, not positive on all seeds; and
- first-order residual/actual-effect mass: mean **+.69248**, positive on **3/3** seeds.

The auxiliary checkpoint therefore has a larger early-patch influence on the counterfactual logit, but also a larger local-approximation remainder. Larger logit movement is not equivalent to exact counterfactual success or improved faithfulness.

Post-attention and final patches have donor-prediction match `1.0` and exact donor-logit parity in this architecture: after the final attention operation, subsequent answer-position operations are position-local. Early patches need not reproduce donor logits because later attention still mixes untouched target positions; maximum early logit-parity error is **6.624594**.

These are **whole-vector** patches. They can carry donor `c,d`, other arithmetic variables, or unrelated structure. Final-site parity demonstrates output reconstruction only. All `u`-specific sufficiency claims remain abstained.

## 7. Trained SAE/CLT replacement graphs

### 7.1 Replacement and graph semantics

`replacement_graph.py` trains two-layer ReLU replacements on MLP interfaces from training prompts only. For codes `z0,z1`, the CLT writes

\[
  \hat y_0=z_0D_{00},\qquad
  \hat y_1=z_0D_{01}+z_1D_{11}.
\]

The independent SAE baseline is the same architecture with `D01=0`. Exact reconstruction-error nodes are

\[
  e_0=y_0-\hat y_0,\qquad e_1=y_1-\hat y_1.
\]

Each answer-position graph contains active feature nodes and both error nodes. Direct output allocations use a shared 256-point integrated Jacobian over independent answer-position MLP writes:

\[
  f(y)-f(0)=\sum_i a_i+\varepsilon_{\mathrm{quadrature}}.
\]

Feature-to-feature edges from layer 0 to layer 1 use the local encoder/downstream Jacobian. Every retained edge is checked against an actual source-write ablation. Direct output allocations, actual ablation effects, local edge predictions, and approximation residuals remain separate fields.

The graph is only an **answer-position MLP-write replacement**. Attention and upstream-position contributions remain in its conditional baseline. It is not a full Transformer attribution graph.

### 7.2 Executed scale and numerical health

The benchmark trained twelve replacements: SAE and CLT for standard/auxiliary checkpoints on each of three seeds. Each has 64 features per layer and 100 fixed training epochs.

It evaluated 60 untouched-prompt graphs containing:

- **2,959 feature/error nodes**; and
- **37,510 locally predicted and actually ablated feature edges**.

Across all graphs:

- maximum absolute integrated quadrature residual: **3.215e-6**;
- maximum readout-parity error: `0`;
- maximum reconstruction-conservation error: `0`; and
- maximum output-conservation error after accounting for quadrature residual: `0`.

The latter three are implementation identity checks. Small quadrature residual is numerical health, not semantic or causal completeness.

### 7.3 SAE versus CLT results

Means across six seed/arm configurations; graph metrics average five untouched prompts per configuration:

| Metric | SAE | CLT |
|---|---:|---:|
| Scoped numerical gate pass count | **6/6** | **4/6** |
| Replacement prediction match | .933333 | .866667 |
| Relative replacement logit-error mass | .019989 | .038438 |
| Error-node output-influence mass | .053487 | .057285 |
| Feature-edge relative residual mass | .100365 | .100053 |
| Mean feature-edge Pearson | .980732 | .976088 |
| Train layer-1 relative reconstruction error | .004127 | **.003087** |
| Combined layer-1 relative reconstruction error | **.031669** | .039374 |

CLT has lower layer-1 training reconstruction error in **6/6** configurations. It has lower combined layer-1 reconstruction error in only **1/6**, and its mean graph logit error is higher in **6/6** configurations. Mean CLT-minus-SAE replacement logit error is **+.018449**.

This is a direct negative result against using training reconstruction alone as evidence of a better replacement model. The numerical gate passes do not assign meanings to features, prove necessity/sufficiency, or validate a complete circuit. Both feature families remain unlabeled numeric decompositions.

## 8. Formal and executable validation

Milestone-5B contracts now contain:

| Contract | Closed Rocq theorems | Expected-UNSAT Z3 checks | Intentional-SAT counterexamples |
|---|---:|---:|---:|
| Lens/generalization | 6 | 5 | 1 |
| Attention/path interventions | 7 | 7 | 2 |
| SAE/CLT replacement graphs | 6 | 6 | 2 |
| **Total** | **19** | **18** | **5** |

Every printed Rocq theorem is closed under the global context. SAT witnesses cover signed expert-weight reversal, QK/effect ordering, whole-vector confounding, exact reconstruction with bad local edges, and direct-versus-total effect differences.

Final software validation:

- **121 CMAE tests passed**;
- mypy passed on **42 CMAE source files**;
- focused intervention, lens/runner, and replacement tests are also type-checked;
- all benchmark training and model analysis ran sequentially; and
- JSON artifacts were written with non-finite values forbidden.

Tests cover exact hooks, causal masks, one-hot value selection, controlled donors, first-order residual conservation, donor parity, checkpoint fingerprints, SAE/CLT equivalence at zero cross-decoder weight, inactive features, output conservation, actual feature-edge ablations, and model-state preservation.

## 9. Reproduction

Use the repository-mandated standalone `uv` environment.

### Behavioral and lens benchmark

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_transformer_m5b \
  --output research/cmae/transformer_m5b_results.json \
  --epochs 250 --seeds 101 202 303
```

### Attention and residual interventions

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_attention_interventions \
  --benchmark research/cmae/transformer_m5b_results.json \
  --output research/cmae/transformer_m5b_interventions.json \
  --seeds 101 202 303 --samples-per-split 2
```

### SAE/CLT replacement graphs

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_transformer_replacements \
  --benchmark research/cmae/transformer_m5b_results.json \
  --output research/cmae/transformer_m5b_replacements.json \
  --seeds 101 202 303 --features 64 --replacement-epochs 100 \
  --replacement-batch-size 512 --replacement-learning-rate 0.003 \
  --l1-coefficient 0.001 --quadrature-steps 256 --samples-per-split 1
```

### Tests and independent Z3 checks

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  --with pytest --with scikit-learn --with adversarial-robustness-toolbox \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_lens_generalization_z3.py
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_attention_path_z3.py
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_replacement_graph_z3.py
```

## 10. Implementation map

| File | Responsibility |
|---|---|
| `generalization.py`, `run_transformer_m5b.py` | Four-arm calibration-controlled training and behavioral gates |
| `latent_lenses.py` | Training-only logit/J/J++-style readouts and approximation diagnostics |
| `attention_interventions.py` | Exact QK statistics, forced attention paths, and controlled residual patches |
| `run_attention_interventions.py` | Checkpoint reproduction and sequential intervention artifact |
| `replacement_graph.py` | SAE/CLT training, reconstruction-error nodes, integrated output allocation, and ablated feature edges |
| `run_transformer_replacements.py` | Sequential SAE/CLT benchmark and numerical abstention gates |
| `arithmetic_transformer.py` | Exact residual, Q/K/V/score, MLP-input/output, and intervention hooks |
| `tests/test_latent_lenses.py` | Lens, generalization, and consolidated runner integration tests |
| `tests/test_attention_interventions.py` | Eight causal intervention tests |
| `tests/test_replacement_graph.py` | Five replacement/graph tests, including inactive-feature behavior |
| `../../formal_verification/rocq/CMAE*Foundations.v` | Scoped closed mathematical contracts |
| `../../formal_verification/verify_cmae_*_z3.py` | Independent properties and intentional counterexamples |

## 11. Next research gates

1. Compare raw-neuron, SAE, and CLT graphs on the same prompts with replacement/error-matched baselines and causal necessity tests.
2. Measure feature stability across seeds and checkpoints before assigning any language labels.
3. Add training-data influence and group-retraining analysis for the auxiliary representations that do and do not transfer.
4. Track circuit/replacement changes across checkpoint trajectories and test whether retraining produces a genuinely reusable algorithm.
5. Expand beyond this two-layer toy architecture. Larger open-weight `circuit-tracer` experiments remain contingent on compatible weights/transcoders and compute; no Claude-scale reproduction is claimed.

The present result is a falsification and measurement foundation, not a success claim for CMAE, J++, CLT, or intermediate supervision.
