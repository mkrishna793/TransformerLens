# CMAE five-stage validation: measured prototype report

**Date:** 2026-10-30
**Status:** executed research prototype; not production certification
**Machine-readable evidence:** [`validation_results.json`](validation_results.json)
**Runner:** [`run_validation_program.py`](run_validation_program.py)

This report records the first complete execution of the five validation stages. It is deliberately a scoped result, not a claim of universal poison detection, complete circuit explanation, semantic repair, or safety certification.

## Protocol and leakage controls

- All trained models, counterfactual retrains, and detector measurements ran **sequentially** with one Torch thread.
- The benchmark uses small MLP, CNN, GRU, and tiny Transformer models. Each case has 96 fit examples, 48 calibration examples, 48 test examples, and a separate 48-example clean behavioral evaluation set. Training runs for 60 epochs.
- Standard and adaptive trigger families are both evaluated. Adaptive batches include labeled low-separation regularizer examples; their original labels are retained separately from poison-flipped labels.
- Detector score direction and thresholds are chosen from **calibration labels only**. The clean-FPR target is 0.10. Method selection uses calibration ROC AUC, never test AUC.
- Test ROC AUC, average precision, and confusion rates are reported for every detector in the JSON. The table below shows only the calibration-selected detector; it is not a test-set leaderboard.
- The repair experiment removes a fixed 25% of fit examples, retrains in isolation, and then evaluates both the original attack and a fresh adaptive-family replay.

## Stage 1 — ground truth and adaptive benchmark

Clean accuracy and trigger attack success rate (ASR) below are behavioral measurements on the separate clean evaluation set. ASR is the fraction of eligible, originally class-zero examples predicted as the target class after applying the trigger. Detection metrics are on the held-out attacked test batch.

| Architecture | Attack | Clean accuracy | Trigger ASR | Calibration-selected detector | Calibration AUC | Test AUC | Test AP |
|---|---|---:|---:|---|---:|---:|---:|
| MLP | standard | 1.000 | 1.000 | spectral | 0.960 | 0.231 | 0.138 |
| MLP | adaptive | 1.000 | 0.667 | ART | 0.934 | 0.914 | 0.819 |
| CNN | standard | 0.854 | 1.000 | spectral | 0.818 | 0.822 | 0.509 |
| CNN | adaptive | 1.000 | 0.579 | spectral | 0.978 | 0.211 | 0.189 |
| GRU | standard | 1.000 | 1.000 | ART | 1.000 | 0.900 | 0.872 |
| GRU | adaptive | 1.000 | 0.844 | ART | 0.976 | 1.000 | 1.000 |
| Tiny Transformer | standard | 1.000 | 1.000 | PCA | 0.949 | 1.000 | 1.000 |
| Tiny Transformer | adaptive | 1.000 | 0.750 | PCA | 0.969 | 0.897 | 0.900 |

The main positive result is cross-architecture execution with explicit ground truth, not uniform detector success. Calibration-selected methods can generalize poorly: for example, the MLP standard case selects spectral on calibration but has test AUC 0.231, while the adaptive CNN case selects spectral and has test AUC 0.211. Conversely, some cases show high held-out AUC. These are small synthetic measurements, not estimates for real datasets or frontier models.

### Fixed-budget removal and retraining

| Case | Method | Removed attack / clean | Clean accuracy before → after | Original ASR before → after | Fresh adaptive ASR before → after | Overall repair gate |
|---|---|---:|---:|---:|---:|---|
| MLP standard | spectral | 0 / 24 | 1.000 → 1.000 | 1.000 → 1.000 | 0.409 → 0.227 | **FAIL** |
| MLP adaptive | ART | 18 / 6 | 1.000 → 0.938 | 0.667 → 0.370 | 0.524 → 0.381 | **FAIL** |

The standard repair fails because the original ASR does not improve. The adaptive repair improves both ASR measurements but misses the strict clean-performance tolerance (the clean drop is approximately 0.0208 versus the allowed 0.02), so it also fails. This is an intentional fail-closed outcome, not a repaired-model claim. The adaptive replay is a fresh attack-family evaluation, not proof against an optimally reoptimized adversary.

## Stage 2 — generalized Jacobian lens

The implementation supports exact analytic-linear maps, nonlinear Taylor residuals, class-output probes, future-output GRU probes, and vocabulary/readout-shaped selectors. The measured checks are:

| Check | Result |
|---|---:|
| Analytic linear maximum relative residual | 1.23e-16 |
| Linear context dispersion | 0 |
| Vocabulary/readout selected-logit maximum error | 0 |
| Hand-compiled final-output Jacobian maximum error | 0 |
| Hand-compiled lens dispersion | 0 |
| Correct causal-average hypothesis error | 0 |
| Unnormalized cumulative-sum alternative error | 2.1419 |
| Nonlinear small-step relative residual | 0.3361 |
| Nonlinear large-step relative residual | 1.8859 |
| Causal intervention relative residuals at scales 0.01 / 0.1 / 1.0 | `[0.3361, 0.2896, 1.8859]` |
| Class-output lens shape | `[1, 1, 12]` |
| Future-GRU lens shape | `[1, 1, 3, 6]` |

The restricted hand-compiled causal-average program recovers the known final-output Jacobian exactly and rejects the unnormalized alternative by output-equivalence error. Nonlinear residuals remain visible, including a large unit-scale intervention residual; no first-order lens is presented as an exact nonlinear explanation. These are analytic and restricted-program checks, not evidence of general SAE/J-lens faithfulness.

## Stage 3 — attribution against actual retraining

The counterfactual target is the full-model raw query logit minus the logit after retraining without a point or group. The experiment uses 18 logistic examples, six disjoint groups, deterministic Newton fitting, and a rank-two random projection for TRAK. Pearson correlation and mean absolute error (MAE) are reported separately for leave-one-out (LOO) and leave-group-out (LGO) targets.

| Method | LOO Pearson | LOO MAE | LGO Pearson | LGO MAE |
|---|---:|---:|---:|---:|
| Gradient baseline | 0.9624 | 0.1897 | 0.7306 | 0.2805 |
| TRAK-style projected | 0.9280 | 0.2133 | 0.7056 | 0.3363 |
| EK-FAC-style KFE | 0.7965 | 0.7633 | 0.6042 | 1.5762 |
| Exact logistic GGN | 0.4371 | 0.2594 | 0.6824 | 0.6700 |

The exact GGN uses the summed-loss logistic curvature
\[
H_{GGN}=X^\top\operatorname{diag}(p(1-p))X,
\]
with the mean-loss leave-one-out scaling handled explicitly. EK-FAC remains a distinct Kronecker-eigenbasis approximation. The simple gradient baseline is competitive on this small problem; the experiment does not establish a general ranking for deep networks.

## Stage 4 — prime-field neural topology

Two complexes are constructed from the same trained latent point cloud:

1. a one-dimensional latent-neighbor graph baseline; and
2. a certified two-dimensional clique complex containing only triangles whose three edges are observed in that graph.

Integral boundary matrices are reduced independently over F2, F3, and F5. No unobserved higher cells or topological completion are introduced.

| Filtration quantile | Edges | Certified triangles | Betti over F2 | Betti over F3 | Betti over F5 |
|---:|---:|---:|---|---|---|
| 0.25 | 69 | 147 | `[6, 0, 96]` | `[6, 0, 96]` | `[6, 0, 96]` |
| 0.50 | 138 | 533 | `[2, 0, 417]` | `[2, 0, 417]` | `[2, 0, 417]` |
| 0.75 | 207 | 1081 | `[1, 0, 897]` | `[1, 0, 897]` | `[1, 0, 897]` |

Maximum prime disagreement is **0** for both graph and clique complexes. The adoption decision is therefore **do not adopt a prime-field signal for this measured complex**. The separate RP² and lens-space tests still reproduce torsion controls, demonstrating that the arithmetic implementation can detect prime dependence when a valid complex actually contains it. No incremental neural value is claimed here.

## Stage 5 — deployment, RL, science, and repair gates

### Synchronous resource boundary

One measured PCA + robust-spectral + leave-one-out-topology operation on trained latents took **0.1115 seconds** and had a **536,208-byte Python `tracemalloc` peak**. This is a microbenchmark, not a hardware-independent latency or memory guarantee.

A fake-clock timeout test returned `STOP` with `committed=false`; a within-budget test returned `TRAIN` and committed its payload. The gate evaluates the audit before commit and fails closed on timeout or unhealthy evidence.

### RL expected cost versus tail risk

For constructed trajectory costs `[0, 0, 0, 3]` with threshold 1:

- mean cost: **0.75**, so the average constraint passes;
- tail violation rate: **0.25**;
- maximum cost and upper-tail value at risk: **3.0**;
- zero initialized dual multiplier remains **0.0**, despite the tail violation.

This is a direct executable demonstration that expected-cost feasibility does not certify rare-trajectory safety.

### Scientific validators

- Equal-mass elastic collision: momentum and kinetic-energy residuals both **0**, valid gate passes.
- Deliberately invalid final velocities: conservation gate detects the failure.
- Momentum dimensions: `M L T^-1`; kinetic-energy dimensions: `M L^2 T^-2`.
- Tested permutation invariance and equivariance errors: both **0**.
- Family holdout returns disjoint train/test indices with no family leakage.

These are toy physical and symmetry checks, not experimental scientific validation.

## Validation and reproduction

Current focused suite result: **50 passed** with PyTorch 2.6.0, NumPy 2.2.6, and IBM ART 1.20.1. Targeted mypy passes on the nine validation-stage source files. Earlier formal artifacts remain separate evidence:

- eight closed Rocq validation theorems and seven expected-UNSAT plus one intentional-SAT Z3 checks;
- twelve earlier CMAE/prime Rocq theorems and eighteen earlier CMAE/prime Z3 checks;
- the training-safety kernel has its own ten closed Rocq queries and seventeen Z3 checks.

From the repository root, using the sandbox's standalone `uv` when necessary:

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'pytest>=8' --with 'adversarial-robustness-toolbox==1.20.1' \
  --with 'packaging>=23' \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m research.cmae.run_validation_program \
  --output research/cmae/validation_results.json
```

The benchmark is deliberately small and synthetic. It does not establish production certification, universal detection, complete explanation, optimal adaptive robustness, automatic semantic repair, or a general advantage for prime numbers, EK-FAC, TRAK, NLA, or topology.
