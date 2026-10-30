# Milestone 6A — Curvature-Aware Data-to-Circuit Detection and Repair

**Date:** 2026-10-30
**Status:** implemented and executed; **negative / abstention**
**Primary artifact:** `results/curvature_data_circuit_m6a.json`
**Design and equations:** `CURVATURE_DATA_CIRCUIT_DESIGN_2026.md`

## 1. Executive result

Milestone 6A connects exact per-example training gradients, layerwise EK-FAC, a selected CLT feature, integrated output allocation, and accepted-only retraining in one executable pipeline. The numerical and software contracts pass, but the targeted detection and repair experiment does **not** pass its preregistered gates.

- The contaminated reference fits its poisoned training labels, but the intended behavior does not establish on untouched IID prompts: backdoor rates are **0, 1/6, and 0** across seeds 101, 202, and 303. The organism gate required at least 0.5.
- Mean clean-versus-poison influence-contrast detection is **0.547713 ROC AUC**, **0.102712 average precision**, and **0.180556 recall** at the fixed 10% quarantine budget.
- The selected CLT-feature path does not improve ranking: mean **0.501832 ROC AUC** and **0.166667 recall**.
- The high-training-loss baseline is weaker: mean **0.374945 ROC AUC** and **0.041667 recall**.
- No seed passes the accepted-only repair gate. Quarantine-and-retrain also reduces mean clean accuracy from **0.360153** to **0.183908**.

Accordingly, this milestone does **not** establish a useful poison detector, a validated repair, or a behaviorally established backdoor organism. The result is retained rather than changing the trigger, score orientation, thresholds, or budget until a success appears.

## 2. Implemented mathematical path

For each declared `nn.Linear` weight, the implementation captures position-resolved input activations and output cotangents and forms the exact per-example weight gradient

\[
G_b=\sum_t s_{b,t}a_{b,t}^{\top}.
\]

It computes KFE projections and corrected second moments independently for every output/input eigenbasis coordinate, then estimates signed influence with a positive damped diagonal in that basis. Layer scores are summed over the declared block-diagonal inventory.

The joint graph stores, for every training example,

\[
I(b;y_\text{poison}),\quad I(b;y_\text{clean}),\quad
I(b;y_\text{poison})-I(b;y_\text{clean}),\quad
I(b;z_f),\quad \frac{A_f}{z_f}I(b;z_f).
\]

Here \(z_f\) is a calibration-selected CLT feature and \(A_f\) is its direct integrated output allocation. The feature path is explicitly a composition of local approximations, not total output influence.

The tracked inventory contains attention query/key/value projections, MLP input/output projections, and the unembedding. It excludes biases, embeddings, the custom attention-output einsum, and every cross-layer curvature block. These are scope omissions, not claims of zero full-model influence.

## 3. Ground-truth organism and controls

The declared trigger is `c=3,d=5`; triggered training targets are shifted by one residue modulo 11.

| Item | Value |
|---|---:|
| Training examples | 403 |
| Ground-truth poisoned training labels | 24 |
| Calibration trigger canaries | 2 |
| IID evaluation examples | 87 |
| IID trigger examples | 6 |
| IID non-trigger examples | 81 |
| Quarantine budget per ranking | 41 |
| Accepted examples per retraining arm | 362 |

The first calibration trigger is the canary. Final behavior is evaluated on the untouched IID split. Ground-truth poison masks are stored only as evaluation labels and are not inputs to scoring or quarantine.

The contaminated reference is an explicitly quarantined experimental organism. For repair, each ranking is computed first, then a new accepted-only sample/target collection is physically constructed, and only then is a fresh model optimizer created and run. Equal-budget random-removal and high-loss-removal controls use the same procedure. An all-clean reference is trained separately.

## 4. Detection results

Each method uses its **own** highest-scoring 41 examples for fixed-budget precision/recall. Scores are not sign-flipped using test ground truth.

| Seed | Method | ROC AUC | Average precision | Recall | Precision |
|---:|---|---:|---:|---:|---:|
| 101 | influence contrast | 0.555409 | 0.082259 | 0.166667 | 0.097561 |
| 101 | CLT feature path | 0.570471 | 0.099773 | 0.250000 | 0.146341 |
| 101 | training loss | 0.306563 | 0.041215 | 0.000000 | 0.000000 |
| 202 | influence contrast | 0.520339 | 0.111231 | 0.125000 | 0.073171 |
| 202 | CLT feature path | 0.492524 | 0.064836 | 0.125000 | 0.073171 |
| 202 | training loss | 0.452946 | 0.078367 | 0.125000 | 0.073171 |
| 303 | influence contrast | 0.567392 | 0.114647 | 0.250000 | 0.146341 |
| 303 | CLT feature path | 0.442502 | 0.063106 | 0.125000 | 0.073171 |
| 303 | training loss | 0.365325 | 0.044722 | 0.000000 | 0.000000 |
| **Mean** | **influence contrast** | **0.547713** | **0.102712** | **0.180556** | **0.105691** |
| **Mean** | **CLT feature path** | **0.501832** | **0.075905** | **0.166667** | **0.097561** |
| **Mean** | **training loss** | **0.374945** | **0.054768** | **0.041667** | **0.024390** |

The detection gate required ROC AUC at least 0.8 and recall at least 0.75. It fails on every seed.

## 5. Behavior and retraining results

### Untouched IID behavior by seed

| Seed | Arm | Clean accuracy | Non-trigger accuracy | Trigger backdoor rate |
|---:|---|---:|---:|---:|
| 101 | contaminated | 0.252874 | 0.271605 | 0.000000 |
| 101 | accepted-only EK-FAC repair | 0.057471 | 0.061728 | 0.000000 |
| 101 | random-removal control | 0.126437 | 0.135802 | 0.000000 |
| 101 | loss-ranked control | 0.149425 | 0.148148 | 0.000000 |
| 101 | clean reference | 0.333333 | 0.345679 | 0.166667 |
| 202 | contaminated | 0.574713 | 0.617284 | 0.166667 |
| 202 | accepted-only EK-FAC repair | 0.367816 | 0.395062 | 0.166667 |
| 202 | random-removal control | 0.275862 | 0.296296 | 0.000000 |
| 202 | loss-ranked control | 0.333333 | 0.358025 | 0.166667 |
| 202 | clean reference | 0.724138 | 0.753086 | 0.000000 |
| 303 | contaminated | 0.252874 | 0.271605 | 0.000000 |
| 303 | accepted-only EK-FAC repair | 0.126437 | 0.135802 | 0.000000 |
| 303 | random-removal control | 0.264368 | 0.283951 | 0.000000 |
| 303 | loss-ranked control | 0.114943 | 0.123457 | 0.000000 |
| 303 | clean reference | 0.091954 | 0.098765 | 0.333333 |

### Means across seeds

| Arm | Clean accuracy | Non-trigger accuracy | Trigger backdoor rate |
|---|---:|---:|---:|
| Contaminated | 0.360153 | 0.386831 | 0.055556 |
| Accepted-only EK-FAC repair | 0.183908 | 0.197531 | 0.055556 |
| Random-removal control | 0.222222 | 0.238683 | 0.000000 |
| Loss-ranked control | 0.199234 | 0.209877 | 0.055556 |
| Clean reference | 0.383142 | 0.399177 | 0.166667 |

Random removal happens to produce zero backdoor predictions in this small six-prompt-per-seed evaluation, but the contaminated mean is already only 0.055556 and the organism never passes establishment. This is not evidence of successful repair. The clean reference also has nonzero backdoor-labeled predictions, reinforcing that these counts must not be interpreted as attack success without the organism gate.

## 6. Joint graph and numerical checks

The selected CLT feature is at layer 1 in all three runs (feature indices 9, 22, and 62). Each run stores 403 training-data edges, output clean/poison influence, feature influence, the direct-readout conversion, graph metrics, and evaluation-only ground truth: **1,209 auditable training-data edges** in total.

- Maximum absolute integrated-readout quadrature residual: **1.8746941350400448e-6**.
- All integrated-quadrature gates pass at the declared 1e-3 tolerance.
- Minimum corrected KFE moment observed in the run diagnostics: **3.4420011613554027e-32**. Near-zero coordinates are stabilized by the declared 1e-3 damping; this does not turn the approximation into an exact Hessian inverse.
- Exact raw weight gradients match PyTorch autograd in focused tests.
- KFE projections preserve position sums and their cross terms.
- Influence matches an independently materialized explicit EK-FAC inverse on a small system.
- Internal feature queries have exactly zero captured gradients at downstream layers that cannot affect them.

Good quadrature conservation and exact local gradients do not imply good poison ranking or mechanistic completeness.

## 7. Formal and software validation

- **132 CMAE tests passed**, including 11 new curvature/backdoor/integration tests.
- **mypy passed on 48 source files** with `--follow-imports=silent` and no `# type: ignore`.
- The new Rocq file has **7 theorems**, all printed as `Closed under the global context` using Rocq 9.2, Stdlib 9.2, and OCaml 4.14.2.
- Recompiling lens/generalization, attention/path, replacement-graph, and curvature/data-circuit contracts produced **26 closed theorem checks** across those four files.
- New independent Z3 validation: **7 expected-UNSAT checks and 2 intentional-SAT counterexamples**. The SAT cases demonstrate that high influence does not logically imply poison and that high reconstruction quality can coexist with causal error.
- Benchmarks ran sequentially. JSON serialization forbids non-finite values.

The formal results concern scalar/rational identities, damping, contrast linearity, path conservation, and explicit counterexamples. They do not prove neural poison detection, IEEE behavior, EK-FAC approximation error bounds, or circuit faithfulness.

## 8. Interpretation and next gate

There are two distinct negative findings:

1. **Data-ranking failure:** ground-truth poisoned labels exist, but the canary-guided scores do not rank them reliably at the declared budget.
2. **Behavioral-organism failure:** because the shifted-label behavior does not generalize to untouched triggers, this experiment cannot validate behavioral repair even if a retraining arm changes a few predictions.

A future organism must be separately preregistered and must pass establishment before behavioral repair is evaluated. Changing the present trigger, target, damping, score orientation, or threshold after seeing these results would be a new experiment, not a successful confirmation of this one.

The broader non-claims remain: no attack-agnostic detector, adaptive-poison guarantee, exact leave-one-out influence, complete CLT circuit, literal thought reading, or universal training-safety guarantee.

## 9. Reproduction

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  python -m research.cmae.run_curvature_data_circuit \
  --output research/cmae/results/curvature_data_circuit_m6a.json \
  --seeds 101 202 303 --epochs 250 --features 64 \
  --replacement-epochs 100 --damping 0.001 \
  --quarantine-fraction 0.10 --quadrature-steps 256
```

```sh
uv run --no-project \
  --with-requirements research/cmae/transformer_m5a_requirements.txt \
  --with pytest --with scikit-learn \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_curvature_data_circuit_z3.py
```

With the Rocq 9.2/Stdlib 9.2 environment:

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib \
  formal_verification/rocq/CMAECurvatureDataCircuitFoundations.v
```
