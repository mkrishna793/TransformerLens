# CMAE Milestones 1–2: durable dataset safety control plane and real-data evaluation

**Date:** 2026-10-30
**Status:** implemented and executed research prototype; not production certification
**Consolidated evidence:** [`milestone12_results.json`](milestone12_results.json)
**Real-data evidence:** [`real_data_results.json`](real_data_results.json)
**Unified runner:** [`run_dataset_safety_program.py`](run_dataset_safety_program.py)

## Scope and honest completion claim

Milestones 1 and 2 are implemented as a local, durable, testable vertical slice. They are not a distributed service or a universal data-poisoning solution. The benchmark uses real rows from bundled scikit-learn datasets with explicitly recorded defect overlays; it does not claim that those source datasets are naturally poisoned.

## Architecture after Milestones 1–2

```text
Immutable real/raw sample
        │
        ▼
Provenance + content hash + optional Dolma curation
        │
        ▼
Versioned dataset snapshot and calibration registry
        │
        ▼
Fast evidence / optional representation defenses
        │
        ▼
Fail-closed policy and decision ledger
        │
        ▼
Generation-checked accepted-only batch
        │
        ▼
Model update and commit record
        │
        ▼
Replay, quarantine, repair, and held-out evaluation
```

### Tool roles

- **Dolma adapter:** `dolma_adapter.py` provides deterministic local exact deduplication and an optional shell-free Dolma CLI dedupe boundary. The external Dolma CLI was **not executed** in this offline run; no Dolma throughput claim is made.
- **ART Spectral Signature Defense:** the existing ART adapter is evaluated as a reference batch audit.
- **ART ActivationDefence:** `detectors.py` now calls ART's actual activation-clustering implementation through a `PyTorchClassifier`. It extracts learned hidden activations and clusters them per class. Its results are explicitly marked **transductive batch reference** results.
- **Inductive CMAE baselines:** PCA, fit-on-fit spectral scoring, and fit-reference KNN scoring are fit on the training representation and applied to calibration/test representations. These are the methods used for the main held-out detector comparison.
- **Controller, not a detector, owns acceptance:** suspicious rows are quarantined; missing/unhealthy/stale evidence abstains; no detector directly deletes rows or commits a model update.

Spectral and activation-clustering methods analyze learned representations. They are not universal gradient-based poison detectors, and their scores are not semantic proof.

## Milestone 1 — durable control plane

### Implemented contracts

- `dataset.py`: immutable sample records, provenance, content hashes, manifests, snapshots, and derived lineage.
- `evidence.py`: versioned evidence, decisions, batch manifests, and commit records.
- `store.py`: SQLite ledger with content-addressed JSON artifacts, transactional state transitions, calibration registration, replay, and artifact-hash verification.
- `policy.py`: required-detector, provenance, calibration, health, residual, identity, and threshold checks.
- `pipeline.py`: synchronous accepted-only controller integrated with `SynchronousBudgetGate`.
- `dolma_adapter.py`: exact dedup lineage and optional Dolma execution boundary.

### Executable control guarantees tested

- Rejected samples never reach the update callback.
- A real numerical linear update matches an accepted-only reference computation.
- Timeout and expected deep-audit failure never commit.
- Missing registered calibration abstains before update.
- Unauthorized provenance quarantines.
- A stale generation cannot apply its update callback.
- A consumed batch cannot be committed twice.
- A restart reloads artifacts, decisions, batches, commits, and calibration records.
- Content-artifact tampering is detected.
- Duplicate lineage survives exact deduplication.

The consolidated control fixture used 24 real breast-cancer rows and a deliberately simple norm-based audit to isolate control behavior, not to measure detection quality:

| Measurement | Result |
|---|---:|
| Accepted samples | 12 |
| Quarantined samples | 12 |
| Content artifacts | 24 |
| Registered calibrations | 1 |
| Committed generation | 1 |
| Restart replay identical | **True** |

## Milestone 2 — real-data benchmark

### Data and defect protocol

- **Breast Cancer:** 569 real rows, 30 features, binary labels.
- **Digits Binary:** real scikit-learn digits rows for classes 0 and 1, 64 image features.
- Each case uses 50% fit, 25% calibration, and 25% test rows, with normalization parameters fit only on the fit partition.
- Standard and adaptive trigger overlays are applied to the real features. Poisoned rows, flipped labels, regularizer rows, original labels, source groups, families, and time indices remain explicit.
- Family/source/time groups are deterministic benchmark partitions, not claimed acquisition metadata. Each group-held-out score leaves that group out of calibration before evaluating its test slice.
- Two seeds (`101`, `202`) run for every dataset/attack combination. All eight trained cases and repairs execute sequentially with one Torch thread.
- Reported intervals are descriptive two-seed bootstrap intervals, not population-level confidence guarantees.

### Main held-out results

The main detector is selected independently for each case using calibration ROC AUC. ART columns are separate transductive reference audits and are not used for that selection.

| Dataset | Attack | Seed | Selected inductive detector | Test AUC | ART activation AUC | ART spectral AUC | Clean accuracy | Trigger ASR | Repair gate |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Breast cancer | standard | 101 | spectral | 0.864 | 0.746 | 0.673 | 0.937 | 1.000 | FAIL |
| Breast cancer | standard | 202 | spectral | 0.718 | 0.358 | 0.807 | 0.972 | 0.875 | FAIL |
| Breast cancer | adaptive | 101 | spectral | 0.917 | 0.877 | 0.515 | 0.895 | 0.333 | **PASS** |
| Breast cancer | adaptive | 202 | spectral | 0.896 | 0.333 | 0.465 | 0.916 | 0.250 | FAIL |
| Digits binary | standard | 101 | PCA | 0.946 | 0.974 | 0.987 | 0.989 | 1.000 | **PASS** |
| Digits binary | standard | 202 | PCA | 0.915 | 0.923 | 0.981 | 0.967 | 1.000 | **PASS** |
| Digits binary | adaptive | 101 | spectral | 0.706 | 0.797 | 0.713 | 0.867 | 0.574 | FAIL |
| Digits binary | adaptive | 202 | spectral | 0.777 | 0.632 | 0.605 | 0.967 | 0.341 | FAIL |

All eight cases produced usable family-held-out, source-held-out, and time-held-out metrics for the selected inductive detector: two family groups, three or four source groups, and two time groups per case. The complete nested metrics are in the JSON.

### Two-seed descriptive summaries

| Dataset / attack / method | AUC mean [bootstrap interval] | AP mean [bootstrap interval] | FPR mean | TPR mean |
|---|---|---|---:|---:|
| Breast / standard / spectral | 0.791 [0.718, 0.864] | 0.299 [0.226, 0.372] | 0.123 | 0.024 |
| Breast / adaptive / spectral | 0.906 [0.896, 0.917] | 0.727 [0.690, 0.764] | 0.079 | 0.443 |
| Digits / standard / PCA | 0.931 [0.915, 0.946] | 0.600 [0.535, 0.665] | 0.065 | 0.462 |
| Digits / adaptive / spectral | 0.742 [0.706, 0.777] | 0.362 [0.332, 0.392] | 0.096 | 0.023 |

The standard breast-cancer case illustrates why AUC alone is insufficient: mean AUC is 0.791 but mean TPR at the calibrated threshold is only 0.024. The adaptive attacks also show that a method can perform well on one dataset/attack and poorly on another.

### Repair outcomes

Repair removes a fixed 25% of fit rows ranked by the calibration-selected detector, retrains in isolation, and evaluates clean behavior, the original trigger, and an adaptive overlay.

- **3 of 8 gates passed:** breast adaptive seed 101 and both digits standard cases.
- **5 of 8 gates failed:** both breast standard cases, breast adaptive seed 202, and both digits adaptive cases.
- The digits adaptive repairs increased ASR and reduced clean accuracy; they are explicit failures, not successful cleaning.
- A repair pass is only a result for this overlay and fixed budget. It is not an optimal-adaptive-adversary or semantic-repair certificate.

## Verification and reproduction

Current focused suite: **65 tests passed**. Mypy: **26 CMAE source files clean**. Black and isort checks were run. Existing formal Rocq/Z3 artifacts remain separate scoped evidence; no neural detection theorem is claimed.

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'scikit-learn>=1.4' --with 'pytest>=8' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'scikit-learn>=1.4' \
  --with 'adversarial-robustness-toolbox==1.20.1' --with 'packaging>=23' \
  python -m research.cmae.run_dataset_safety_program \
  --output research/cmae/milestone12_results.json \
  --real-output research/cmae/real_data_results.json
```

## What remains before a real-time deployment

1. Replace the local fixture with real streaming ingestion, durable model checkpoints, and transactional update/restart integration.
2. Run Dolma at realistic scale and measure its actual tagging/deduplication semantics, false positives, memory, and throughput.
3. Add naturally occurring data defects and more datasets, multiple independent runs, and properly powered uncertainty analysis.
4. Add inductive activation-clustering baselines or keep ART batch defenses strictly in the offline/reference lane.
5. Build model-aware adaptive attackers, drift monitoring, reviewer workflows, and production SLOs.
6. Only then evaluate a bounded pilot deployment with rollback and independent red-team review.

The system now has a credible **control-plane skeleton and measured real-row benchmark**. It does not yet have production certification, universal detection, complete explanations, or automatic semantic repair.
