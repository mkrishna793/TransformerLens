# Causal-Mechanistic Alignment Engine (CMAE)

## Experimental mathematics and multi-architecture prototype

**Status:** research prototype, not a production safety API
**Date:** 2026-10-30
**Implemented evidence:** parametric lenses, signed sparse charts, sparse diagonal Kronecker influence, a genuine small EK-FAC reference, H0 persistence, spectral traces, robust MAD spectral evidence, IBM ART comparison, global quarantine, NLA escalation contracts, per-sample PyTorch tracing, residual-aware causal readout traces, trained ReLU SAEs, group-retraining attribution, adaptive repair/rollback, and a hooked arithmetic Transformer with calibration-controlled generalization, logit/J/J++-style lenses, QK/path interventions, integrated-readout/component tracing, trained SAE/CLT replacement graphs, and layerwise EK-FAC training-data-to-CLT-feature paths with accepted-only retraining controls, plus fixed-denominator GGN/full-Hessian counterfactuals, real CIFAR-10 BadNets, fixed-target arithmetic, independent Kronfluence/dattri comparisons, and out-of-fold Cleanlab baselines
**Architectures executed:** MLP, CNN, GRU, and a tiny Transformer
**Phase-II audit:** [public research mechanisms, architecture-general tracing, and prime-field topology](DEEP_RESEARCH_2026.md)
**Phase-III validation:** [five-stage trained benchmark, Jacobian, attribution, topology, deployment/RL/science, and repair results](VALIDATION_RESULTS_2026.md)
**Milestones 1–2:** [durable dataset control plane, Dolma/ART adapters, and real-data evaluation](MILESTONE12_RESULTS_2026.md)
**Milestones 3–4:** [exact readout traces, trained SAEs, group-retraining checks, and adaptive repair/rollback](MILESTONE34_RESULTS_2026.md)
**Milestone 5A:** [ground-truth arithmetic Transformer, generalization holdouts, integrated readout, and causal component ablations](MILESTONE5A_RESULTS_2026.md)
**Milestone 5B:** [calibration-controlled generalization, latent lenses, causal path interventions, and SAE/CLT replacement graphs](MILESTONE5B_RESULTS_2026.md)
**Milestone 6A:** [layerwise EK-FAC data-to-CLT circuits, targeted-organism evaluation, and negative detection/repair results](MILESTONE6A_RESULTS_2026.md)
**Milestone 6B:** [fixed-denominator counterfactuals, real CIFAR-10 BadNets, fixed-target arithmetic, independent EK-FAC/Cleanlab comparisons, and fail-closed negative results](MILESTONE6B_RESULTS_2026.md)

CMAE is a constructive attempt to combine category-theoretic composition, latent topology, sparse concepts, influence approximation, ART, and language-based investigation. It does **not** claim to identify every unsafe sample or assign intrinsic meanings to neurons.

The central design rule is:

> Every component must expose its mathematical assumptions, approximation residual, health status, and admissible decision role.

## 1. What the first experiment establishes—and what it does not

### Established by execution

- The same per-sample observation contract executes on four different architecture families.
- Sparse-chart reverse maps match PyTorch autograd away from TopK ties and ReLU kinks.
- The sparse rank-one and position-sum product identities match explicitly materialized Kronecker inverses.
- The EK-FAC reference matches its defining Kronecker-eigenbasis calculation.
- Topological descriptors have the expected translation/rotation invariance in the tested construction.
- A robust spectral detector detects one controlled synthetic feature shift, but can also abstain on non-convergence or zero MAD.
- IBM ART's own spectral-signature routine executes as an optional baseline.
- A rejected NaN sample never enters the tested training forward pass, and accepted-only gradients exactly match a clean-reference run.
- Six Rocq theorems are closed under the global context; twelve Z3 checks pass.

### Not established

- The first experiment's random test dictionaries are **not trained SAEs** and have no certified concept meanings. Milestone three separately trains ReLU SAEs, but their features remain unlabeled and are not certified concepts.
- The architecture smoke run uses untrained models and a constructed input shift. It is not a real poison-detection benchmark.
- The same anomaly score does not transfer reliably across architectures.
- The O(k) sparse calculation is not exact EK-FAC and is not exact leave-one-out retraining.
- H0 persistence and spectral traces are non-injective observables, not semantic or safety predicates.
- The NLA code is an external-provider contract with a toy orchestration test, not an evaluation of Anthropic-scale NLA weights.
- No automatic repair can be declared semantically correct merely because a repaired sample passes the same detector again.

## 2. Corrections to the original CMAE equations

| Original idea | Mathematical correction | Implemented consequence |
|---|---|---|
| `Para(Lens) -> Sigma_k^M` | Useful as a typed forward/pullback contract, but the backward map must be the pullback of the same chart. | `ParametricLens` composition and `TopKSparseChart.reverse`, tested against autograd. |
| “k-Sparse Concept Polytope” | The at-most-k-sparse set is generally non-convex: it is a union of coordinate faces. With nonnegative unit-sum coordinates it is a union of simplices, not one simplex/polytope with the claimed convex semantics. | Explicit Rocq, Z3, and Python non-convexity witnesses. |
| ReLU/TopK of activations and gradients independently | Independent projections do not define lens composition. ReLU also erases distinctions among negative gradients. | A signed chart option and exact fixed-support Jacobian pullbacks. Full projections and projection residuals are retained. |
| “Exact EK-FAC” from two diagonal EMAs | The product identity is exact for a rank-one gradient and a diagonal Kronecker inverse. EK-FAC additionally uses a Kronecker-factored eigenbasis and one corrected second moment per KFE coordinate. | Separate sparse diagonal and EK-FAC reference implementations. |
| O(k) compute and O(M) total memory | Active-coordinate dot products can be O(k), and two diagonal vectors use O(M). Dictionaries, traces, eigenbases, and EK-FAC's full KFE diagonal are additional costs. | No fixed MB/GB claim; complexity stated by component. |
| Three-step robust spectral score | Three iterations provide no general convergence certificate. A coordinate-wise median does not make the leading direction robust. | Eigenpair residual, iteration metadata, and unhealthy/abstain behavior. |
| TopK forbidden support implies 0% unsafe awareness | A forbidden coordinate below the TopK cutoff is absent from the code but can remain in the original activation. | Full-projection observables are retained; Rocq/Z3 blind-spot witnesses. |
| `w * g = 0` implies zero bad-data leakage | Layer-local masks do not enforce a global training decision, and IEEE `0 * infinity`/`0 * NaN` is NaN. Undetected poison is also unaffected by the mask. | Physical accepted-only batch construction before the training forward pass. |

## 3. Categorical lens lift with an actual reverse derivative

Let a parametric layer have forward map

\[
  f(p,x):P\times X\rightarrow Y
\]

and reverse derivative

\[
  f^*(p,x,\bar y)=(\bar p,\bar x).
\]

For composition \(z=g(q,f(p,x))\), the reverse maps compose as

\[
  (\bar q,\bar y)=g^*(q,y,\bar z),\qquad
  (\bar p,\bar x)=f^*(p,x,\bar y).
\]

This is the useful content of the `Para(Lens(C))` viewpoint for CMAE: architecture components have typed interfaces and chain-rule-compatible reverse maps. Category theory does not itself supply a poison score.

For a sparse linear chart, let \(c=Wx\), let \(S_x\) select its TopK coordinates, and let \(\phi\) be either identity or ReLU. The code is

\[
  z=S_x\phi(Wx).
\]

Away from selection ties and activation kinks, the support is locally fixed and

\[
  J_z(x)=S_xD_\phi(Wx)W.
\]

The reverse map must therefore be

\[
  \bar z\mapsto \bar x
  =W^\top S_x^\top D_\phi(Wx)\bar z.
\]

That is exactly what `TopKSparseChart.reverse` implements. At a TopK tie or non-differentiable kink, the experiment must report unsupported/subgradient behavior rather than claiming a unique derivative.

### Signed evidence

For signed CMAE evidence, selection uses magnitude and preserves selected values:

\[
  z=S_x(Wx).
\]

For the original ReLU chart, two negative inputs \(x_1\neq x_2\) can satisfy

\[
  \operatorname{ReLU}(x_1)=\operatorname{ReLU}(x_2)=0.
\]

This is a formal information-loss counterexample, not merely an implementation concern.

## 4. The sparse set is not one convex concept polytope

Define

\[
  \mathcal S_k^M=\{z\in\mathbb R^M:\|z\|_0\leq k\}.
\]

For \(0<k<M\), this is a union of coordinate subspaces/faces. Already for \(k=1,M=2\),

\[
  (1,0),(0,1)\in\mathcal S_1^2,
  \qquad
  \tfrac12(1,0)+\tfrac12(0,1)=(\tfrac12,\tfrac12)
  \notin\mathcal S_1^2.
\]

Consequently, convex-polytope guarantees cannot be attached to \(\mathcal S_k^M\) without changing the mathematical object. If nonnegative and normalized, the corresponding object is a union of simplices. CMAE treats the support as a discrete, piecewise-defined state and reports ties explicitly.

## 5. Sparse Kronecker influence versus actual EK-FAC

### 5.1 Standard signed influence convention

For training example \(b\), query \(q\), and damped curvature \(H+\lambda I\),

\[
  I(b,q)=-g_q^\top(H+\lambda I)^{-1}g_b.
\]

The sign matters: the expression is not an unsigned anomaly magnitude.

### 5.2 Exact algebra inside a restricted approximation

Suppose a layer has a rank-one sample gradient

\[
  g_b=a_b\otimes s_b,\qquad g_q=a_q\otimes s_q,
\]

and the chosen inverse approximation is \(P_A\otimes P_S\). Then

\[
  I_{\mathrm{factored}}(b,q)
  =-(a_q^\top P_Aa_b)(s_q^\top P_Ss_b).
\]

If \(P_A\) and \(P_S\) are diagonal and only \(k\) coordinates are active, each quadratic form can be evaluated over the active support. This proves the algebraic core of the proposed O(k) calculation **inside that approximation**.

For gradients summed over positions,

\[
  g_b=\sum_t a_{b,t}\otimes s_{b,t},
\]

the correct expression contains all cross-position terms:

\[
  I_{\mathrm{factored}}(b,q)
  =-\sum_{t,u}
  (a_{q,u}^\top P_Aa_{b,t})
  (s_{q,u}^\top P_Ss_{b,t}).
\]

Pooling each sample into one outer product generally changes the approximation.

### 5.3 Why this is not exact EK-FAC

EK-FAC starts with eigendecomposed K-FAC factors and uses

\[
  Q=Q_A\otimes Q_S,\qquad
  \widehat G_{\mathrm{EK-FAC}}
  =Q\operatorname{diag}(\lambda^*)Q^\top.
\]

The corrected diagonal \(\lambda^*\) has one second moment for every Kronecker-eigenbasis direction. It is not generally the Kronecker product of two independently maintained diagonal EMAs. Influence in that basis is

\[
  I_{\mathrm{EK-FAC}}(b,q)
  =-\sum_j
  \frac{(Q^\top g_q)_j(Q^\top g_b)_j}
       {\lambda_j^*+\lambda}.
\]

Dense eigenbases can destroy sparse support. Therefore CMAE exposes two separate routines:

- `rank_one_sparse_diagonal_influence`: the efficient restricted identity;
- `ekfac_reference`: a small explicit implementation of the actual EK-FAC definition for validation.

Neither is promoted to a hard safety predicate.

## 6. Latent topology and arithmetic/spectral traces

### 6.1 Persistent topology

For a latent point cloud \(X_\ell=\{x_i\}\), a Vietoris–Rips filtration tracks connected components, cycles, and higher-dimensional holes across scales. The prototype implements finite H0 deaths using sorted edges and union-find, which is equivalent to the minimum-spanning-tree construction for connected-component persistence.

Persistence has genuine stability theorems. For example, under the hypotheses of Cohen–Steiner–Edelsbrunner–Harer for continuous tame functions on a triangulable space,

\[
  d_B(\operatorname{Dgm}(f),\operatorname{Dgm}(g))
  \leq \|f-g\|_\infty.
\]

This is a bound on descriptor perturbation, not a statement that poison has a particular topology or that equal diagrams imply equal meanings. The Z3 model explicitly constructs the general observation limitation: two worlds can have equal topological observations and opposite safety labels.

### 6.2 Spectral-trace signatures

For centered latent points \(X_c\), define

\[
  K=X_cX_c^\top/d,
  \qquad
  \tau_m=\operatorname{tr}(K^m)=\sum_i\lambda_i^m.
\]

CMAE reports \(\tau_2,\tau_3\) and effective rank

\[
  r_{\mathrm{eff}}=\exp\left(-\sum_i p_i\log p_i\right),
  \qquad p_i=\lambda_i/\sum_j\lambda_j.
\]

These are coordinate-invariant summaries of the Gram operator. For an attribution graph with adjacency \(A\), \(\operatorname{tr}(A^m)\) counts closed weighted walks. That can be useful for recurrent/cyclic computation graphs, but is uninformative for a finite DAG whose adjacency is nilpotent. No closed-walk score is used as a semantic safety claim here.

### 6.3 Arithmetic topology and category theory: required discipline

Arithmetic topology's prime/knot analogies and categorical traces are legitimate mathematical subjects, but CMAE does not currently have an operational mapping that turns them into a measured dataset detector. They remain possible future research only if they define:

1. a computable object from activations or graphs;
2. invariants under a declared equivalence;
3. an estimator and computational budget;
4. a falsifiable advantage over simpler spectral/topological baselines.

This avoids using advanced terminology without advancing the experiment.

## 7. Robust spectral evidence and IBM ART

CMAE's custom spectral experiment computes

\[
  \widetilde X=X-\operatorname{median}(X),\qquad
  C=\widetilde X^\top\widetilde X/n.
\]

Power iteration estimates a leading direction \(v\), but CMAE also computes the eigenpair residual

\[
  r=\|Cv-(v^\top Cv)v\|.
\]

The per-sample scores and robust normalization are

\[
  s_i=(\widetilde x_i^\top v)^2,
  \qquad
  z_i=\frac{s_i-\operatorname{median}(s)}
  {1.4826\operatorname{MAD}(s)+\epsilon}.
\]

Three iterations are treated as a minimum, not a certificate. Non-convergence, degenerate covariance, and zero MAD produce unhealthy evidence. A coordinate-wise robust center alone does not give the direction a 50% breakdown point; robust covariance methods such as SPECTRE are relevant future baselines.

`art_spectral_scores` calls IBM ART's own `SpectralSignatureDefense.spectral_signature_scores`. Its algorithm is intentionally not confused with CMAE's median/MAD variant.

## 8. NLA-heavy dataset investigation without language-as-proof

An external NLA provider is modeled as

\[
  w\sim q_\phi(w\mid h),\qquad \hat h=R_\psi(w),
\]

with relative reconstruction error

\[
  \epsilon_{\mathrm{NLA}}
  =\frac{\|h-\hat h\|_2}{\max(\|h\|_2,\epsilon)}.
\]

A useful NLA-heavy audit pipeline is:

1. Use sparse/topological/spectral evidence to select outliers, cluster medoids, and representative clean controls.
2. Verbalize those selected activations rather than every token indiscriminately.
3. Attach sample IDs, layer/probe IDs, model version, and source provenance to every generated claim.
4. Reconstruct activations and report error; do not hide unverbalizable residual content.
5. Compare claims with the actual input/context, full projections, ART scores, and independent examples.
6. Where possible, test the claim through an intervention in the original model.
7. Record corroborated and uncorroborated statements separately.

The current `NLAProvider` and `escalate_with_nla` implement this contract and reconstruction-error accounting. `NLAEscalation` always has `admissible_as_hard_gate=False`. Only a toy provider has been executed; integrating and evaluating real NLA weights is a separate research milestone.

## 9. Real-time flagging, explanation, and repair semantics

“Real time” can only mean **before the next controlled optimizer update and within a declared resource budget**. CMAE's global decision is:

- `QUARANTINE` for nonfinite values or observable violations/outliers;
- `ABSTAIN` for unhealthy, unconverged, unsupported, or uncalibrated evidence;
- `TRAIN` only for evidence accepted by the configured policy;
- `STOP` for a declared global safety/resource failure.

The training controller must then construct a fresh batch containing only `TRAIN` records. It must not run the rejected samples through a training loss and attempt to repair the situation with backward masks.

Explanations can combine:

- full and sparse concept coordinates;
- projection residuals;
- spectral and topological scores;
- influence diagnostics;
- NLA hypotheses; and
- provenance and detector-health records.

Repair is intentionally more conservative:

| Repair type | Minimum requirement |
|---|---|
| Schema/unit correction | Authoritative source rule and before/after provenance |
| Exact deduplication | Verifiable content identity and retention policy |
| Label correction | Trusted clean reference or authorized human review |
| Trigger removal | Known trigger model plus independent post-repair behavioral evaluation |
| Generative rewriting | Treat as a new derived sample; never presume preservation of labels or safety |

Every repaired item receives a new version and must be re-audited. Passing the same detector is necessary only for that detector's observable; it is not proof of semantic repair.

## 10. How the prototype generalizes across custom networks

A custom architecture must provide:

1. a single-sample forward path with an explicit batch dimension;
2. an identity `CMAEProbe` at a meaningful observation point;
3. `FEATURES_LAST` or `CHANNELS_FIRST` layout metadata;
4. a finite scalar audit loss differentiable with respect to the probe;
5. a concept dictionary with the observed latent dimension; and
6. architecture-specific factor adapters if influence inside shared/recurrent weights is required.

The probe canonicalizes activations to `[positions, latent_dimension]`. This works for the tested dense, spatial, sequential, and attention representations. Unsupported tuple/state layouts, non-differentiable probes, or invalid per-sample semantics must return unsupported/abstain.

The shared probe does **not** make every internal influence formula architecture-independent. Convolution requires patch/receptive-field factors; recurrent computation requires time/gate/state factors; attention requires separate Q/K/V and position semantics.

## 11. Executed architecture smoke results

The last input of eight was replaced by a large constructed shift. Models were deterministic but **untrained**; the threshold was not calibrated. Ranks are for that shifted sample, where rank 1 is largest.

| Architecture | Positions | Latent dim. | Spectral rank | ART rank | Topology rank | Decisions (train/quarantine) |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 1 | 8 | 8 | 1 | 5 | 6 / 2 |
| CNN | 36 | 4 | 3 | 8 | 1 | 5 / 3 |
| GRU | 5 | 6 | 3 | 1 | 5 | 6 / 2 |
| Tiny Transformer | 5 | 6 | 1 | 5 | 2 | 7 / 1 |

All traces were finite. Mean activation projection-residual ratios were approximately 0.595, 0.630, 0.584, and 0.637 respectively—further evidence that discarded sparse content cannot be ignored.

The inconsistent ranks are a useful negative result: execution generalizes better than detection. The raw machine-readable results are in `results.json`.

Separately, the unit suite verifies that the robust spectral detector ranks a constructed outlier first in a controlled feature matrix. That controlled success must not be conflated with the untrained architecture smoke results above.

## 12. Formal and test results

- **Python:** 149 tests passed with the standalone `uv` dependency closure, including real-data/ART, arithmetic-Transformer, replacement-graph, exact-GGN/full-Hessian, external-EK-FAC, Cleanlab, and fail-closed circuit-adapter tests.
- **Rocq 9.2:** earlier contracts remain documented in their milestone reports. Recompiling lens/generalization, attention/path, replacement-graph, curvature/data-circuit, and counterfactual-assurance contracts produced 35 closed theorem checks across those five files.
- **Z3:** earlier checkers remain documented in their milestone reports. The new counterfactual-assurance checker adds nine expected-UNSAT properties and four intentional-SAT counterexamples.
- **Static checks:** Black, isort, and pycln formatting completed; mypy passes on 58 scoped CMAE source files with no `# type: ignore`.

The Rocq results cover concrete rational counterexamples, multiplicative lens composition, the 2-by-2 rank-one identity, and influence sign. They do not prove numerical deep-learning accuracy.

## 13. Reproduction

From the repository root:

```sh
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  --with .venv/src/dattri-llm --with .venv/src/kronfluence \
  --with adversarial-robustness-toolbox --with 'packaging>=23' --with pytest \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' --with 'packaging>=23' \
  --with 'adversarial-robustness-toolbox==1.20.1' \
  python -m research.cmae.run_experiments --output research/cmae/results.json
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' --with 'packaging>=23' \
  --with 'adversarial-robustness-toolbox==1.20.1' \
  python -m research.cmae.run_validation_program \
  --output research/cmae/validation_results.json
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' --with 'scikit-learn>=1.4' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m research.cmae.run_dataset_safety_program \
  --output research/cmae/milestone12_results.json \
  --real-output research/cmae/real_data_results.json
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' --with 'scikit-learn>=1.4' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m research.cmae.run_trace_adaptive_program \
  --output research/cmae/milestone34_results.json \
  --deep-output research/cmae/trace_evidence_results.json \
  --adaptive-output research/cmae/adaptive_repair_results.json
```

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_transformer_m5a \
  --output research/cmae/transformer_m5a_results.json \
  --epochs 250 --seed 101 --quadrature-steps 256
```

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_curvature_data_circuit \
  --output research/cmae/results/curvature_data_circuit_m6a.json \
  --seeds 101 202 303 --epochs 250 --features 64 \
  --replacement-epochs 100 --damping 0.001 \
  --quarantine-fraction 0.10 --quadrature-steps 256
```

Milestone 6B (sequential benchmarks, with the pinned source clones under `.venv/src`):

```sh
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  --with .venv/src/dattri-llm --with .venv/src/kronfluence \
  python -m research.cmae.run_counterfactual_assurance
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.run_arithmetic_fixed_target
uv run --no-project \
  --with-requirements research/cmae/counterfactual_m6b_requirements.txt \
  python -m research.cmae.audit_counterfactual_assurance
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_curvature_data_circuit_z3.py
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_z3.py
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_trace_z3.py
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_transformer_circuit_z3.py
```

With the Rocq 9.2/Stdlib 9.2 toolchain used for this research:

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib CMAEFoundations.v
```

## 14. Code map

| File | Responsibility |
|---|---|
| `lenses.py` | Parametric lenses, signed sparse charts, fixed-support pullbacks |
| `influence.py` | Sparse diagonal identities, position sums, genuine small EK-FAC reference |
| `topology.py` | H0 persistence, trace moments, effective rank, leave-one-out topology |
| `prime_topology.py` | Exact finite-field homology, certified latent clique complexes, Newton trace recovery, and prime-trace counterexamples |
| `jacobian.py` | Probe-based downstream Jacobians, context dispersion, selectors, residuals, and interventions |
| `attribution.py` | TRAK-style, EK-FAC, exact logistic GGN, gradient attribution, and Newton fitting |
| `ground_truth.py` | Sequential synthetic multi-architecture poisoning, calibrated detection, and isolated repair |
| `metrics.py` | ROC/AP/correlation/calibration-only threshold metrics |
| `runtime.py`, `rl.py`, `science.py` | Fail-closed budgets, expected/tail RL costs, and scientific validity checks |
| `run_validation_program.py` | Sequential five-stage experiment and JSON evidence generation |
| `dataset.py`, `evidence.py`, `store.py` | Immutable provenance, content artifacts, calibration registry, SQLite ledger, and replay |
| `policy.py`, `pipeline.py` | Fail-closed policy and generation-checked accepted-only training controller |
| `dolma_adapter.py` | Exact deduplication lineage and optional Dolma CLI boundary |
| `real_data.py`, `real_benchmark.py` | Real-row splits, defect overlays, inductive/ART evaluation, and repair |
| `trace_graph.py`, `sae.py` | Exact readout/Taylor traces, causal ablations, trained ReLU SAEs, and explicit residuals |
| `deep_evidence.py` | Signed-logit/TRAK attribution, actual group retrains, and incremental-value measurement |
| `adaptive.py` | Bounded white-box PGD, versioned repair workflow, absolute risk gates, and rollback |
| `run_dataset_safety_program.py` | Consolidated Milestone-one/two evidence generation |
| `run_trace_adaptive_program.py` | Consolidated Milestone-three/four evidence generation |
| `arithmetic_data.py`, `arithmetic_transformer.py` | Modular ground-truth splits and a custom v4-hooked causal Transformer |
| `transformer_circuit.py` | Integrated direct readout, first-order predictions, and actual component ablations |
| `run_transformer_m5a.py` | Sequential Milestone-5A training, generalization, and causal-trace artifact |
| `generalization.py`, `run_transformer_m5b.py` | Four-arm calibration-selected training and multi-seed behavioral gates |
| `latent_lenses.py` | Training-only logit/J/J++-style latent readouts and approximation mass |
| `attention_interventions.py`, `run_attention_interventions.py` | Exact QK/path measurements, controlled residual patches, and checkpoint fingerprints |
| `replacement_graph.py`, `run_transformer_replacements.py` | Trained SAE/CLT replacements, error nodes, integrated output allocation, and ablated feature edges |
| `curvature_influence.py` | Exact position-summed linear gradients, corrected KFE moments, and logit/CLT-feature EK-FAC influence |
| `backdoor_organism.py` | Declared targeted arithmetic poison, untouched behavior metrics, and physical accepted-only subsets |
| `run_curvature_data_circuit.py` | Sequential joint data–feature–output attribution and quarantine/random/loss/clean retraining controls |
| `binary_counterfactual.py`, `binary_full_hessian.py` | Fixed-denominator training, exact GGN, full tanh loss Hessian, and LOO/LGO metrics |
| `backdoorbench_organism.py`, `arithmetic_fixed_target.py` | Real CIFAR-10 BadNets and fixed-target arithmetic organisms |
| `binary_ekfac.py`, `external_attribution.py` | CMAE empirical EK-FAC and independent dattri/Kronfluence comparisons |
| `out_of_fold_quality.py` | Four-fold out-of-fold Cleanlab label-quality suspicion |
| `run_counterfactual_assurance.py`, `run_arithmetic_fixed_target.py` | Sequential counterfactual/detection benchmarks with fail-closed physical repair |
| `circuit_tracer_adapter.py`, `run_circuit_tracer_gate.py` | Supported-model adapter, graph/error-node audit, and explicit non-execution gate |
| `audit_counterfactual_assurance.py` | Independent score-vector, budget, counterfactual, and artifact consistency audit |
| `counterfactual_m6b_requirements.txt` | Standalone `uv` dependency boundary for Milestone 6B |
| `transformer_m5a_requirements.txt` | Standalone `uv` dependency boundary for the v4 hook stack |
| `detectors.py` | Robust MAD spectral evidence and optional IBM ART baseline |
| `tracing.py` | Custom-network probes and per-sample activation/gradient traces |
| `engine.py` | Global decisions, physical filtering, repair proposals, NLA escalation |
| `run_experiments.py` | Sequential MLP/CNN/GRU/Transformer smoke experiments |
| `tests/test_cmae.py` | Algebraic, numerical, control, ART, NLA-contract, and architecture tests |
| `tests/test_prime_topology.py` | Finite-field torsion, graph controls, Newton identities, and trace-incompleteness tests |
| `tests/test_validation_program.py` | Five-stage mathematical, counterfactual, runtime, scientific, and repair contracts |
| `tests/test_dataset_safety.py`, `test_real_data.py` | Ledger, calibration, replay, Dolma, real-data, and ART contracts |
| `tests/test_milestone_program.py` | Consolidated control-plane restart/replay integration |
| `tests/test_trace_adaptive.py` | Trace-target, SAE, signed-TRAK, PGD, repair-gate, and rollback tests |
| `tests/test_transformer_m5a.py` | Arithmetic splits, causal masks, integrated readout, and component-ablation tests |
| `tests/test_latent_lenses.py` | Lens/generalization mathematics and one-epoch consolidated runner integration |
| `tests/test_attention_interventions.py` | QK hooks, one-hot paths, controlled patches, parity, and residual tests |
| `tests/test_replacement_graph.py` | SAE/CLT identities, inactive features, conservation, and actual edge-ablation tests |
| `tests/test_curvature_influence.py`, `test_backdoor_organism.py`, `test_curvature_data_circuit.py` | Explicit-inverse gradients, downstream-zero feature queries, targeted data, score-specific quarantine, and accepted-only controls |
| `VALIDATION_RESULTS_2026.md`, `validation_results.json` | Scoped measured report and complete machine-readable metrics |
| `MILESTONE12_RESULTS_2026.md`, `milestone12_results.json` | Durable control-plane and real-data Milestone-one/two evidence |
| `MILESTONE34_RESULTS_2026.md`, `milestone34_results.json` | Scoped Milestone-three/four report and consolidated metrics |
| `trace_evidence_results.json`, `adaptive_repair_results.json` | Full trace/SAE/attribution and model-aware repair measurements |
| `MILESTONE5A_RESULTS_2026.md`, `transformer_m5a_results.json` | Ground-truth Transformer generalization and 624-node causal-ablation evidence |
| `MILESTONE5B_RESULTS_2026.md` | Consolidated behavioral, lens, intervention, and replacement-graph report |
| `transformer_m5b_results.json` | Twelve-run, three-seed behavioral and latent-lens benchmark |
| `transformer_m5b_interventions.json` | Six reproduced checkpoints, 1,440 forced attention paths, and 180 residual patches |
| `transformer_m5b_replacements.json` | Twelve trained replacements and 60 graphs with 37,510 ablated feature edges |
| `../../formal_verification/rocq/CMAEFoundations.v` | Scoped Rocq algebra and counterexamples |
| `../../formal_verification/rocq/CMAEPrimeFoundations.v` | Six closed Rocq prime/topology and Newton results |
| `../../formal_verification/verify_cmae_z3.py` | Independent Z3 properties and counterexamples |
| `../../formal_verification/verify_cmae_prime_topology_z3.py` | Six independent Z3 prime/topology checks |
| `../../formal_verification/rocq/CMAEValidationFoundations.v` | Eight closed validation-stage mathematical results |
| `../../formal_verification/verify_cmae_validation_z3.py` | Seven UNSAT validation checks and one intentional SAT tail witness |
| `../../formal_verification/rocq/CMAETraceFoundations.v` | Eight closed trace, projection, and conjunctive repair-gate results |
| `../../formal_verification/verify_cmae_trace_z3.py` | Seven UNSAT trace checks and one intentional nonlinear SAT witness |
| `../../formal_verification/rocq/CMAETransformerCircuitFoundations.v` | Six closed integrated-readout, ablation, and modular-arithmetic results |
| `../../formal_verification/verify_cmae_transformer_circuit_z3.py` | Five UNSAT Transformer-circuit checks and one intentional nonlinear SAT witness |
| `../../formal_verification/rocq/CMAELensGeneralizationFoundations.v` | Six closed lens, residual, objective, and ranking contracts |
| `../../formal_verification/rocq/CMAEAttentionPathFoundations.v` | Seven closed attention, counterfactual, and residual contracts |
| `../../formal_verification/rocq/CMAEReplacementGraphFoundations.v` | Six closed SAE/CLT reconstruction and direct-allocation contracts |
| `../../formal_verification/verify_cmae_lens_generalization_z3.py` | Five UNSAT lens checks and one signed-weight SAT witness |
| `../../formal_verification/verify_cmae_attention_path_z3.py` | Seven UNSAT path checks and two intentional SAT witnesses |
| `../../formal_verification/verify_cmae_replacement_graph_z3.py` | Six UNSAT replacement checks and two intentional SAT witnesses |
| `../../formal_verification/rocq/CMAECurvatureDataCircuitFoundations.v` | Seven closed EK-FAC projection, damping, contrast, and path-conservation contracts |
| `../../formal_verification/verify_cmae_curvature_data_circuit_z3.py` | Seven UNSAT curvature/path checks and two intentional SAT limitation witnesses |
| `MILESTONE6A_RESULTS_2026.md`, `results/curvature_data_circuit_m6a.json` | Scoped negative targeted-organism detection/repair report and complete edge-level evidence |
| `../../formal_verification/rocq/CMAECounterfactualAssuranceFoundations.v` | Nine closed fixed-denominator, GGN/full-Hessian, quality, budget, and fail-closed contracts |
| `../../formal_verification/verify_cmae_counterfactual_assurance_z3.py` | Nine UNSAT contracts and four intentional SAT limitation witnesses |
| `MILESTONE6B_RESULTS_2026.md` | Counterfactual-first negative report, full-Hessian sensitivity, and explicit non-claims |
| `results/counterfactual_assurance_m6b.json`, `results/arithmetic_fixed_target_m6b.json` | Complete per-seed score vectors, retraining effects, behavior, and gate evidence |
| `results/full_hessian_diagnostic_m6b.json`, `results/circuit_tracer_gate_m6b.json`, `results/m6b_audit.json` | Post-hoc curvature sensitivity, circuit non-execution, and independent audit evidence |

## 15. Next research gates

1. Compare raw-neuron, SAE, and CLT graphs on identical prompts with replacement/error-matched baselines, causal necessity tests, and checkpoint/feature-stability analysis.
2. Scale the trained-replacement track beyond small activations; evaluate layer transfer, retraining-measured generalization, training-data influence, and independently validated language descriptions.
3. Move the Milestone-one ledger to streaming ingestion and transactional model checkpoints; run Dolma at realistic scale and evaluate naturally occurring defects with adequately powered repetitions.
4. Extend the small-MLP group-retraining checks to adequately powered multi-seed datamodels and add explicit Linear/Conv factor extraction before recurrent/attention influence claims.
5. Add real NLA weights/providers and measure confabulation, reconstruction, dataset-root-cause accuracy, and cost.
6. Admit higher-dimensional or multi-field topology only on certified integral complexes with a measured gap over graph/F2 baselines; the present latent clique experiment shows no prime increment.
7. Extend the bounded test-time PGD track to model-aware adaptive poisoning and more attack families, with clean false positives, untouched holdouts, and confidence intervals.
8. Measure deployment cost on declared hardware and real batch sizes; keep timeout and unhealthy-evidence paths fail closed.
9. Permit categorical sheaves, arithmetic topology, or new trace invariants into a detector only after independently replicated held-out incremental value.
10. Keep Milestones 6A/6B negative: establish stationary, controlled counterfactual organisms and finite-step corrections before any new detector sweep; invalid rankings must not enter accepted-only repair or circuit-intervention claims.

## Primary references

1. Cruttwell et al., **Categorical Foundations of Gradient-Based Learning**. <https://arxiv.org/abs/2103.01931>
2. Cruttwell et al., **Deep Learning with Parametric Lenses**. <https://arxiv.org/abs/2404.00408>
3. George et al., **Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis**. <https://arxiv.org/abs/1806.03884>
4. Grosse et al., **Studying Large Language Model Generalization with Influence Functions**. <https://arxiv.org/abs/2308.03296>
5. Cohen–Steiner, Edelsbrunner, Harer, **Stability of Persistence Diagrams**. <https://doi.org/10.1007/s00454-006-1276-5>
6. Moor et al., **Topological Autoencoders**. <https://proceedings.mlr.press/v119/moor20a.html>
7. Templeton et al., **Scaling Monosemanticity**. <https://transformer-circuits.pub/2024/scaling-monosemanticity/>
8. Fraser-Taliente et al., **Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations**. <https://transformer-circuits.pub/2026/nla/>
9. Nicolae et al., **Adversarial Robustness Toolbox**. <https://arxiv.org/abs/1807.01069>
10. Hayase et al., **SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics**. <https://arxiv.org/abs/2104.11315>
11. Soldaini et al., **The Dolma Toolkit**. <https://github.com/allenai/dolma>
12. Tran, Li, and Madry, **Spectral Signatures in Backdoor Attacks**. <https://arxiv.org/abs/1811.00636>
13. Chen et al., **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering**. <https://arxiv.org/abs/1811.03728>
14. Transformer Circuits, **Attribution Graphs: Methods**. <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>
15. Park et al., **TRAK: Attributing Model Behavior at Scale**. <https://arxiv.org/abs/2303.14186>
16. Ilyas et al., **Datamodels: Predicting Predictions from Training Data**. <https://arxiv.org/abs/2202.00622>
17. Madry et al., **Towards Deep Learning Models Resistant to Adversarial Attacks**. <https://arxiv.org/abs/1706.06083>
18. Gurnee et al., **Verbalizable Representations Form a Global Workspace in Language Models**. <https://transformer-circuits.pub/2026/workspace/>
19. Ayonrinde and Lindsey, **J++ Lens: Preliminary October 2026 Update**. <https://transformer-circuits.pub/2026/october-update/index.html>
20. Olah, **A Toy Model of Mechanistic (Un)Faithfulness**. <https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html>
21. Transformer Circuits, **Attribution Graphs for Attention: QK and OV Circuits**. <https://transformer-circuits.pub/2025/attention-qk/index.html>
