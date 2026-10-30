# Curvature-Aware Data–Circuit Assurance Design

**Date:** 2026-10-30
**Status:** executed Milestone-6A design; canary-guided targeted-poison scope; see `MILESTONE6A_RESULTS_2026.md` for the negative result

## Objective

Connect training-data attribution and forward mechanistic tracing in one auditable graph:

\[
 b\;\xrightarrow{\text{EK-FAC influence}}\;z_f
 \;\xrightarrow{\text{CLT/integrated readout}}\;y.
\]

The model organism is a targeted modular-arithmetic backdoor. A known affected canary is used to score training examples, and a fixed candidate-quarantine budget is applied before accepted-only retraining. This is not an attack-agnostic detection claim.

## Research constraints from primary work

- EK-FAC uses the Kronecker-factored eigenbasis but estimates a separate corrected second moment for every KFE coordinate. It does not use only the Kronecker product of factor eigenvalues (<https://arxiv.org/abs/1806.03884>).
- Large-model influence work uses EK-FAC to approximate inverse-curvature products, but this remains an approximation rather than exact retraining (<https://arxiv.org/abs/2308.03296>).
- Attribution graphs use CLT replacement features and explicit reconstruction-error nodes; replacement fidelity does not prove mechanistic faithfulness (<https://transformer-circuits.pub/2025/attribution-graphs/methods.html>).
- Influence-based poison forensics can require an affected query and can fail on adaptive or group attacks. Δ-Influence motivates transformation contrasts, while poison-traceback work documents group effects and limited influence precision (<https://arxiv.org/html/2411.13731>, <https://www.usenix.org/system/files/sec22-shan.pdf>).

## Layerwise EK-FAC

For a tracked linear layer \(Wa\), one example at one position has weight gradient

\[
 G_{b,t}=s_{b,t}a_{b,t}^{\top}.
\]

The complete example gradient includes every position and all cross terms:

\[
 G_b=\sum_t s_{b,t}a_{b,t}^{\top}.
\]

With activation/output eigenbases \(Q_A,Q_S\), its KFE projection is

\[
 \widetilde G_b=Q_S^{\top}G_bQ_A
 =\sum_t(Q_S^{\top}s_{b,t})(Q_A^{\top}a_{b,t})^{\top}.
\]

The corrected diagonal is estimated independently in every coordinate:

\[
 \lambda^*_{ij}=\frac1N\sum_b(\widetilde G_b)_{ij}^2.
\]

For scalar query \(q\), the signed upweighting-influence estimate is

\[
 I(b;q)=-\sum_{\ell}\sum_{ij}
 \frac{(\widetilde G_q^\ell)_{ij}(\widetilde G_b^\ell)_{ij}}
      {(\lambda_\ell^*)_{ij}+\lambda}.
\]

The implementation sums tracked layer blocks and therefore ignores cross-layer curvature blocks. Biases, embeddings, and the custom attention-output einsum are outside the first scoped inventory. These omissions are reported, not silently treated as zero influence in the full model.

## Joint data–feature path

Select an active CLT feature \(f\) on the affected canary. If its activation is \(z_f\), its direct integrated output allocation is \(A_f\), and its measured EK-FAC influence is \(I(b;z_f)\), define the readout weight and local feature-mediated path as

\[
 \omega_f=A_f/z_f,\qquad P(b,f)=\omega_f I(b;z_f).
\]

This path is a local composition of two approximations. It is stored beside total output influence \(I(b;y)\), not substituted for it. Feature selection uses output allocation on a calibration canary; final behavior is measured on untouched IID prompts.

## Organism and accepted-only repair

The trigger is a declared arithmetic pattern. Triggered training labels are shifted by one residue; non-trigger labels remain correct. The detector uses clean-versus-poison target-logit influence contrast on a calibration canary:

\[
 S(b)=I(b;y_{\mathrm{poison}})-I(b;y_{\mathrm{clean}}).
\]

Ground-truth poison labels are used only for evaluation. A fixed fraction of highest-scoring training examples is quarantined. The accepted subset is physically constructed before any optimizer forward/backward pass. Random removal and loss-ranked removal provide controls.

Evaluation includes poison ROC AUC/average precision, precision/recall at the fixed budget, untouched trigger backdoor rate, non-trigger accuracy, and retraining effects. Failure to detect or repair is a reportable result.

## Non-claims

- No literal reading of human-like thoughts; the objects are tensors, features, gradients, and interventions.
- No universal harmful-data detector or guarantee against adaptive poisoning.
- No exact leave-one-out influence claim from EK-FAC.
- No complete circuit claim from a selected CLT feature or a single data–feature path.
- No deployment gate based only on high influence, reconstruction, or readout rank.
