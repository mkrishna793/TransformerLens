# CMAE phase II: public research mechanisms, architecture-general tracing, and prime-field topology

**Date:** 2026-10-30
**Repository state studied:** branch `arena/141cfa19-transformerlens`, base commit `f4fefedfadb37fca52a1353c3103a07da4b9ef7c`
**Companion artifact:** `README.md` describes the executable phase-I prototype.

## 0. Scope and evidence discipline

This is a broad audit of selected **public** research programs relevant to CMAE (Causal-Mechanistic Alignment Engine), not an exhaustive audit of every laboratory and not a claim to know nonpublic training or deployment mechanisms. Public papers can also omit operational details. Each claim below should therefore be read as a claim about the cited artifact, not about everything an organization does.

Evidence is separated into three levels:

- **External:** a method or empirical result reported in a linked primary source.
- **Local-formal:** an exact small property checked in this repository with Rocq and/or Z3.
- **Local-executable:** Python behavior covered by tests in this repository.

A local smoke test is not a trained poison-detection benchmark. A formal theorem about a finite chain complex is not evidence that neural activations contain the corresponding topology. A feature direction with a causal steering effect in one study is not a universal semantic detector.

### Principal conclusions

1. **The strongest design is a stack of heterogeneous evidence, not a master interpretability method.** Behavioral evaluations, sparse charts, local Jacobians, attribution graphs, training-data attribution, topology, and domain constraints answer different questions and fail differently.
2. **Primes have a real topological use:** changing coefficient fields can expose torsion invisible to a single-field persistence computation. This is established multi-field persistent homology, not a new CMAE discovery.
3. **Prime-indexed spectral traces are not a complete fingerprint.** The local counterexample below has equal traces at every odd prime and at exponent two, while the spectra differ.
4. **Architecture generality must come from explicit adapters and abstention.** A common activation probe can execute on MLPs, CNNs, GRUs, and transformers, but influence formulas, output lenses, and causal semantics are not automatically architecture-independent.
5. **The next highest-value experiment is ground-truth validation, not another detector.** Tracr-style programs, tiny trained models, adaptive attacks, and retraining counterfactuals can falsify CMAE interpretations before scaling them.

## 1. Public-program evidence ledger

The table is intentionally mechanism-oriented. “Transfer” means a research direction to test, not a guarantee that the published weights or results transfer.

| Program | Public mechanism and evidence | Important limitation | CMAE transfer |
|---|---|---|---|
| Anthropic: [scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [circuit tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Large SAEs; cross-layer transcoders; prompt-specific local replacement models and attribution graphs; explicit reconstruction-error nodes | Reconstruction is incomplete, graphs are local and pruned, and error nodes can carry substantial influence | Keep raw residuals and error influence as first-class evidence; validate paths by intervention |
| Anthropic: [Jacobian lens/global workspace](https://transformer-circuits.pub/2026/workspace/) | Corpus-averaged downstream Jacobians produce token directions and a sparse “J-space”; interventions test whether these directions mediate some reasoning | Single-token vocabulary bias, averaged rather than prompt-exact Jacobians, incomplete coverage of non-verbalizable computation | Add a general downstream-Jacobian adapter with Taylor-error and causal validation |
| Anthropic: [natural-language autoencoders](https://transformer-circuits.pub/2026/nla/) | Learns natural-language descriptions of activation structure rather than requiring every feature to be a fixed dictionary direction | Generated language can be plausible but unfaithful; reconstruction and explanation quality are separate | Treat NLA as an explanation generator whose claims require evidence links and causal tests |
| Anthropic: [hidden-objective auditing](https://www.anthropic.com/research/auditing-hidden-objectives) | A known hidden objective was trained, then blinded teams combined behavioral, data, and SAE audits | A deliberately constructed model organism is not a estimate of real-world audit success | Build blinded CMAE evaluation games with known defects and scored root-cause reports |
| Anthropic: [Constitutional AI](https://arxiv.org/abs/2212.08073) | Supervised critique/revision followed by preference modeling and RL from AI feedback | A constitution, judge, and reward model can all be incomplete or exploitable; behavioral improvement is not latent verification | Audit critique data, judge disagreement, RL trajectory changes, and held-out behavioral failures |
| OpenAI: [large-scale SAEs](https://openai.com/index/extracting-concepts-from-gpt-4/) and [automated neuron explanations](https://openai.com/index/language-models-can-explain-neurons-in-language-models/) | Scales sparse feature extraction and uses models to propose/score natural-language explanations | Explanation scores, feature coverage, and reconstruction do not establish faithful complete cognition | Use explanation confidence, held-out activation prediction, and intervention faithfulness separately |
| OpenAI: [weight-sparse circuits](https://arxiv.org/abs/2511.13653) | Trains new weight-sparse transformers and prunes task circuits; reports roughly 16-fold smaller circuits than comparably capable dense models on its simple tasks | Requires expensive de novo training; tasks are curated; capability/interpretability tradeoff remains; dense-model bridges are preliminary | Offer an optional interpretable-training track and compare circuit necessity/sufficiency against CMAE post-hoc graphs |
| OpenAI: [emergent-misalignment research](https://openai.com/index/emergent-misalignment/) | SAE analysis identifies a “misaligned persona” direction; activation steering changes misalignment in the reported setting; additional clean fine-tuning can reduce it | Model-/training-specific evidence is not a universal misalignment coordinate or repair certificate | Require independent causal, adaptive, and post-repair behavioral tests before using a direction for control |
| OpenAI: [deliberative alignment](https://arxiv.org/abs/2412.16339) | Safety specifications are distilled into reasoning examples, followed by judge-guided RL | Policy-sensitive behavior and visible reasoning do not make chain-of-thought a faithful internal trace | Log policy version and behavior, but do not treat generated reasoning as privileged causal evidence |
| Google DeepMind: [JumpReLU](https://arxiv.org/abs/2407.14435) and [Gemma Scope](https://arxiv.org/abs/2408.05147) | Direct L0-sparse JumpReLU SAEs with straight-through threshold training; open multi-layer/multi-site SAE suites | Sparse/faithful reconstruction remains a tradeoff; released dictionaries are model- and site-specific | Benchmark JumpReLU charts against the current signed TopK charts and retain all reconstruction residuals |
| Google DeepMind/ETH: [Tracr](https://arxiv.org/abs/2301.05062) | Compiles human-readable RASP programs into transformer weights with known computation | RASP is restricted; learned compression can destroy the original ground-truth interpretation | Use compiled programs as a falsification suite for graph recovery, chart faithfulness, and NLA claims |
| MIT: [datamodels](https://arxiv.org/abs/2202.00622) and [TRAK](https://arxiv.org/abs/2303.14186) | Fits subset-to-output surrogates; TRAK approximates attribution using projected parameter gradients, an empirical-kernel approximation, and ensembling | Subset retraining is expensive; local/kernel attribution is approximate and can fail under nonconvex training | Benchmark EK-FAC and TRAK against actual leave-group-out retraining, using counterfactual prediction rather than visual plausibility |
| Stanford/CAIS and collaborators: [Representation Engineering](https://arxiv.org/abs/2310.01405) | Contrastive activation differences and principal directions provide representation reading and control vectors | A direction can encode correlates, be layer-specific, or cause off-target effects | Use as a cheap baseline against SAE/Jacobian views; validate sign, causal effect, and distribution shift |
| Princeton and collaborators: [adaptive latent-separability attacks](https://arxiv.org/abs/2205.13613) | Correctly labeled regularization samples and asymmetric trigger planting suppress the latent separation assumed by backdoor defenses | Results do not imply every latent method always fails, but directly invalidate a universal separability assumption | Make adaptive, low-separation attacks a required evaluation family; allow abstention and independent behavioral evidence |
| Meta: [Llama Guard](https://arxiv.org/abs/2312.06674) | Instruction-tuned input/output safety classification with an explicit taxonomy | A moderator can make errors and is itself susceptible to adversarial inputs | Use as independent semantic/behavioral evidence, never as proof of internal safety |
| Microsoft: [PyRIT](https://github.com/microsoft/PyRIT) | Composable, model-agnostic red-team orchestration, converters, targets, and scorers | Red-team coverage depends on attack design and scorer quality | Attach adaptive attack transcripts and scorer versions to CMAE audit records |
| IBM: [ART](https://arxiv.org/abs/1807.01069), [AIF360](https://arxiv.org/abs/1810.01943), and [AIX360](https://arxiv.org/abs/2109.12151) | Adversarial/poisoning defenses; fairness metrics/mitigation; multiple explanation methods | No metric or explanation is context-free; mitigation can trade off utility and other group harms | Keep ART baselines and add dataset/group audits; do not collapse fairness, robustness, and poisoning into one score |
| AI2 and collaborators: [WildGuard](https://arxiv.org/abs/2406.18495) | Multi-task moderation for prompt harm, response harm, and refusal behavior, trained on direct and adversarial interactions | Automated moderation can be wrong and distribution-dependent | Evaluate CMAE quarantine decisions against external moderation and over-refusal baselines |
| UK AISI: [Inspect](https://inspect.aisi.org.uk/) | Composable datasets, solvers/agents, tools, sandboxes, and scorers for behavioral evaluations | Evaluation results depend on task/scorer coverage and sandbox fidelity | Export CMAE cases to versioned behavioral evaluations rather than relying only on latent ranks |
| NIST and collaborators: [AI 100-2e2025](https://doi.org/10.6028/NIST.AI.100-2e2025) | Adversarial-ML taxonomy across lifecycle, attacker goals, capabilities, knowledge, evasion, poisoning, and privacy | A taxonomy is not a detector or certification | Declare the threat model before selecting a CMAE detector and report uncovered attack classes |
| PKU Alignment and collaborators: [Safe RLHF](https://arxiv.org/abs/2310.12773) | Separate reward and cost models with constrained RL and a Lagrange multiplier | Expected-cost constraints, learned costs, and dual updates do not guarantee per-trajectory safety | Trace reward/cost components and multiplier updates; independently measure constraint violations |
| CMU/Berkeley/multiple collaborators: [HarmBench](https://arxiv.org/abs/2402.04249) and [STACK](https://arxiv.org/html/2506.24068v2) | Standardized and pipeline-level adversarial safety evaluation | Benchmark scores do not cover every adaptive attacker or deployed system | Use pipeline-level held-out tests after latent-based interventions, not only before them |

## 2. What the main interpretability mechanisms actually compute

### 2.1 Sparse charts and cross-layer replacement models

For activation \(x\in\mathbb{R}^d\), a standard SAE computes

\[
 f(x)=\sigma(W_{\rm enc}x+b_{\rm enc}),\qquad
 \hat x=W_{\rm dec}f(x)+b_{\rm dec},\qquad
 r=x-\hat x.
\]

The chart is useful only together with its residual \(r\). Small mean squared error can still hide a rare safety-relevant direction, so CMAE should report residual norm, downstream effect of the residual, and worst-group error—not just average reconstruction.

Anthropic’s attribution graphs use a local replacement model. Attention patterns, normalization denominators, and nonlinearities are frozen or stop-gradient in the relevant backward pass. A feature-to-feature edge is schematically

\[
 A_{s\to t}=a_s\sum_{\ell_s\le \ell<\ell_t}
 (W_{{dec},s}^{\ell_s\to\ell})^T
 J^{\blacktriangledown}_{c_s,\ell\to c_t,\ell_t}
 W_{{enc},t}^{\ell_t},
\]

where \(J^{\blacktriangledown}\) contains the specified stop-gradients. This gives a linear decomposition of the replacement model’s feature preactivations. It is not the exact full nonlinear causal graph of the original network.

**CMAE development:** represent each graph with (i) prompt and checkpoint identity, (ii) chart version, (iii) stop-gradient convention, (iv) error-node influence, (v) pruning threshold, and (vi) intervention results. Two graphs should not be merged merely because feature labels look similar. A categorical/sheaf construction becomes justified only after overlap maps and consistency failures are measured; until then it is metadata, not evidence.

### 2.2 The Jacobian lens and an architecture-general downstream lens

The 2026 J-lens defines, at layer \(\ell\),

\[
 J_\ell=\mathbb{E}_{\text{prompt},t,t'\ge t}
 \left[\frac{\partial h_{\text{final},t'}}
 {\partial h_{\ell,t}}\right],
 \qquad V_\ell=W_UJ_\ell .
\]

Rows of \(V_\ell\) are token-associated directions. Applying this fixed operator to an intermediate activation gives a cheap vocabulary readout. Unlike the logit lens, it corrects for average downstream representational change. Unlike a prompt-specific attribution graph, it averages over contexts and trades per-prompt exactness for a reusable map.

The associated J-space is not simply a linear subspace. It is described using sparse combinations of vocabulary vectors, equivalently a union of low-dimensional sparse cones/subspaces. A distance can be written

\[
 d_{\mathcal F}(x)=\min_{|S|=k}\|x-\Pi_Sx\|_2,
\]

with the remaining residual kept for interventions. If coefficients are required to be nonnegative, the projection must be a nonnegative-cone projection; an unconstrained orthogonal projection onto a span is a relaxation.

A general CMAE adapter should instead start with any differentiable downstream map

\[
 F_{\ell\to o}:\mathcal H_\ell\to\mathcal Y,
\]

and estimate \(J_\ell=\mathbb E[D F_{\ell\to o}(h_\ell)]\) over a declared calibration distribution. A readout map \(R:\mathcal Y\to\mathcal Z\) then gives \(V_\ell=RJ_\ell\). For language models, \(R\) can be the unembedding. For a classifier it can select class logits; for a scientific model it can be a dimensionalized observable or constraint head. If no meaningful readout exists, the adapter must say “no verbalization lens,” not invent token semantics.

Every estimated lens needs three error measurements:

1. **Estimator dispersion:** variation across prompts, positions, and calibration batches.
2. **First-order residual:**
   \(\|F(h+\delta)-F(h)-J\delta\|/\max(\|\delta\|,\epsilon)\) on intervention-scaled \(\delta\).
3. **Causal validity:** whether moving along a reported direction changes the claimed output on held-out contexts without unacceptable off-target effects.

### 2.3 JumpReLU: why the threshold matters

Gemma Scope uses

\[
 \operatorname{JumpReLU}_\theta(z)=z\odot H(z-\theta),\qquad
 \mathcal L=\|x-\hat x\|_2^2+\lambda\|f(x)\|_0.
\]

The threshold separates feature selection from active-feature magnitude. Since the true derivative with respect to the threshold is zero almost everywhere, training uses straight-through pseudo-derivatives localized by a kernel around the threshold. This is an optimization estimator, not an exact derivative of the discontinuous forward map.

**CMAE development:** compare signed TopK, JumpReLU, and dense/PCA charts on the same activations. Report the Pareto frontier of downstream loss, L0, residual influence, feature stability, and causal sufficiency. A lower residual alone does not prove better concepts.

### 2.4 Weight-sparse circuits are a different intervention

OpenAI’s sparse-circuit work changes the model during training. It periodically keeps only largest-magnitude weights to enforce an L0 budget, then learns task-specific node masks using a Heaviside gate and surrogate gradients. Mean ablation tests sufficiency (retain the circuit) and necessity (remove the circuit) on curated tasks.

This is not “another SAE.” It restricts connectivity so that circuits can be smaller at the weight level. The reported roughly 16-fold circuit-size improvement is specific to the paper’s comparisons and tasks. The method also reveals a useful evaluation pattern: understanding a circuit should generate new adversarial predictions, as its bracket-counting example did.

**CMAE development:** add an optional training-time track for custom networks, but keep it separate from post-hoc auditing. For dense-to-sparse bridges, measure hybrid-forward KL, bridge residual, task necessity/sufficiency, and off-distribution behavior. A sparse surrogate that mimics average outputs is not automatically a faithful explanation of a dense model.

## 3. Training-data attribution: the counterfactual standard

### 3.1 Datamodels

For target example \(x\), training algorithm \(\mathcal A\), and subset \(S'\subseteq S\), define the actual counterfactual output

\[
 f_{\mathcal A}(x;S')=
 \text{outcome of training on }S'\text{ and evaluating at }x.
\]

A linear datamodel fits \(f_{\mathcal A}(x;S')\approx\beta_0+\beta^T\mathbf 1_{S'}\) from many subset/retraining observations. Its value is measured by predicting new retraining counterfactuals, not by whether high-weight examples look intuitively similar.

This is the right evaluation standard for CMAE data attribution: select held-out subsets, retrain, and measure calibration/correlation of predicted output changes. It is also expensive, so approximations must be ranked by when they save compute without losing decision-relevant accuracy.

### 3.2 TRAK and its relationship to EK-FAC

TRAK linearizes the model output around final parameters,

\[
 f(z,\theta)\approx f(z,\theta^*)+
 \nabla_\theta f(z,\theta^*)^T(\theta-\theta^*),
\]

and projects gradients with a random matrix \(P\):

\[
 \phi(z)=P^T\nabla_\theta f(z,\theta^*).
\]

For stacked training features \(\Phi\), its binary-class estimator is

\[
 \tau(z,S)=\phi(z)^T(\Phi^T\Phi)^{-1}\Phi^TQ,
\]

followed by ensembling and optional soft-thresholding. The paper derives this through a one-step Newton/logistic-regression approximation and notes a generalized Gauss–Newton interpretation.

This differs from CMAE’s EK-FAC reference. TRAK uses projected output gradients and a kernel-like solve; EK-FAC approximates curvature in a Kronecker-factored eigenbasis. Neither is exact retraining. A meaningful comparison must vary projection rank, damping, checkpoint ensemble, training stochasticity, and group removal, then score held-out counterfactual predictions.

### 3.3 Representation-engineering baseline

A cheap contrastive baseline forms activation differences between paired conditions and takes a leading principal direction \(v\). Reading uses \(v^Th\); control uses \(h\mapsto h+\alpha v\). CMAE should retain this baseline because a more elaborate SAE or graph method must demonstrate added value over it. Required tests are layer selection on development data, sign orientation, held-out reading accuracy, causal dose-response, off-target behavior, and adaptive evasion.

## 4. Prime numbers: positive structure, negative results, and boundaries

### 4.1 Where primes genuinely help: coefficient-dependent homology

For a finite chain complex with boundary \(\partial_k:C_k\to C_{k-1}\), homology over a field is computed using

\[
 \beta_k(\mathbb F_p)=
 \dim C_k-\operatorname{rank}_{\mathbb F_p}(\partial_k)
 -\operatorname{rank}_{\mathbb F_p}(\partial_{k+1}).
\]

The universal coefficient theorem explains why changing the prime can matter:

\[
 H_k(X;\mathbb F_p)\cong
 \bigl(H_k(X;\mathbb Z)\otimes\mathbb F_p\bigr)
 \oplus\operatorname{Tor}\bigl(H_{k-1}(X;\mathbb Z),\mathbb F_p\bigr).
\]

Thus free rank and different prime-torsion components produce different field Betti numbers. [Boissonnat and Maria](https://arxiv.org/abs/2001.02960) develop multi-field persistent homology and modular reconstruction specifically to infer prime divisors of torsion more informatively than single-field persistence.

A minimal CW chain complex for RP² is

\[
 \mathbb Z\xrightarrow{\times 2}\mathbb Z\xrightarrow{0}\mathbb Z .
\]

The local implementation therefore obtains:

| Coefficients | \(\beta_0\) | \(\beta_1\) | \(\beta_2\) |
|---|---:|---:|---:|
| \(\mathbb F_2\) | 1 | 1 | 1 |
| \(\mathbb F_3\) or \(\mathbb F_5\) | 1 | 0 | 0 |

Replacing \(\times2\) by \(\times3\) gives the analogous one-cell CW model with 3-torsion: over \(\mathbb F_3\) the tested degree-one and degree-two Betti numbers become one, while over \(\mathbb F_2\) they vanish. These are exact topology examples, not neural findings.

The negative control is equally important: the tested cycle graph has the same \((\beta_0,\beta_1,\beta_2)=(1,1,0)\) over \(\mathbb F_2,\mathbb F_3,\mathbb F_5\). Ordinary graph connectivity and most real-valued activation pipelines do not acquire useful torsion merely because a prime was chosen.

**Allowed CMAE hypothesis:** construct an integral cell/chain complex from a justified discretization of an attribution graph, compute persistence over several small primes, and test whether torsion features predict failures beyond F2 persistence and simpler graph invariants. Do not reduce floating activations modulo a prime and call the result neural torsion.

### 4.2 Prime-only spectral traces are incomplete

For an \(n\times n\) matrix \(A\), let \(p_k=\operatorname{tr}(A^k)=\sum_i\lambda_i^k\). Over characteristic zero, Newton’s identities recursively recover elementary symmetric polynomials and hence the characteristic polynomial from \(p_1,\ldots,p_n\):

\[
 k e_k=\sum_{i=1}^{k}(-1)^{i-1}e_{k-i}p_i,
 \qquad e_0=1.
\]

This does **not** imply that traces only at prime exponents suffice. Take

\[
 A=\operatorname{diag}(1,-1,8,-8),\qquad
 B=\operatorname{diag}(4,-4,7,-7).
\]

Both spectra are sign-symmetric, so every odd power trace vanishes. Their second traces are both \(1+1+64+64=16+16+49+49=130\). Therefore their traces agree at exponent two and at every odd prime. Nevertheless,

\[
 \chi_A(\lambda)=(\lambda^2-1)(\lambda^2-64),
 \qquad
 \chi_B(\lambda)=(\lambda^2-16)(\lambda^2-49),
\]

are different. Their fourth traces are \(8194\) and \(2642\), and their characteristic-polynomial constant terms are \(64\) and \(784\). This is a local-formal and local-executable counterexample, not just an intuitive warning.

Two further boundaries matter:

- In characteristic \(p\), Newton recurrences contain division by \(k\); the usual formula needs special treatment when \(p\mid k\).
- Even the full characteristic polynomial ignores Jordan structure and can poorly summarize nonnormal transient behavior. Neural-dynamics audits may need singular values, pseudospectra, commutators, trajectory perturbations, or closed-loop stability measures.

### 4.3 Category theory, tensors, uniqueness, and arithmetic topology

The user’s prime-number instinct is worth developing, but each construction needs a semantic object:

- **Categorical trace:** a trace is canonical for an endomorphism on a dualizable object in a suitable monoidal category. An arbitrary nonlinear network layer is not automatically such a morphism with a useful dual. Matrix-partial-trace formulas apply only after the tensor and dual structure is specified.
- **Finite-field tensor structure:** \(\mathbb F_p\) is a meaningful coefficient field. Frobenius \(x\mapsto x^p\) is meaningful in characteristic \(p\), not a natural symmetry of a real activation vector. Quantizing a trained real network into a finite field changes the model and requires a new correctness analysis.
- **Unique factorization:** factorization can identify an integer or support exact modular provenance; it does not make a neural representation, explanation, or dataset semantically unique. Ordinary hashes or CRT checkpoints may help integrity auditing, but modular residues are not cryptographic authentication by themselves.
- **Arithmetic topology:** analogies between primes and knots are a deep research program, but analogy alone does not create a neural detection statistic. A CMAE use needs an explicit number ring, covering, Galois action, or chain complex and a measured link to model behavior.
- **p-adic neural networks:** public work exists on p-adic/deep hierarchical models, including [p-adic statistical field theory and convolutional deep Boltzmann machines](https://arxiv.org/abs/2302.03817). That literature does not establish that replacing Euclidean latent geometry with a p-adic metric improves CMAE. A test would need hierarchical data where an ultrametric is independently justified.

### 4.4 Prime research verdict

| Idea | Status after this phase | Next admissible step |
|---|---|---|
| Multi-field/prime-dependent homology | Established externally; exact RP²/lens-style examples implemented locally | Implement multi-field persistence only for justified integral complexes; require incremental neural-audit value |
| Prime-only trace fingerprint | Falsified as a complete spectral invariant | Never use alone; compare full traces, characteristic coefficients, nonnormal and trajectory measures |
| Newton recovery from full traces | Verified locally through fourth order and tested numerically through dimension four | Use only with conditioning/stability diagnostics |
| CRT/modular provenance | Standard plausible tool, not yet a CMAE result | Prototype exact signed integer manifests; measure collisions and compare ordinary hashes |
| Prime-indexed tensor/category construction | No canonical general construction established | Require explicit objects, morphisms, duals, and a falsifiable advantage |
| p-adic latent distance | Unvalidated for CMAE | Test only on independently hierarchical/ultrametric data against Euclidean baselines |

## 5. Architecture-general CMAE design

### 5.1 Evidence records instead of one opaque score

Every observation should carry a versioned record:

```text
model/checkpoint, data/trajectory ID, probe/layout, chart version,
raw residual and downstream-residual effect, lens estimator dispersion,
graph error-node influence and pruning policy, attribution uncertainty,
behavioral/physical/RL validator results, calibration ID,
latency/memory budget, decision and reason codes.
```

A weighted risk sum is permissible only after held-out calibration. Otherwise CMAE should return separate evidence and `accept`, `quarantine`, `abstain`, or `stop`. The executable phase-I guarantee remains accepted-only control flow, not semantic zero leakage.

### 5.2 Adapter matrix

| Setting | Required observation/readout adapter | Additional evidence needed | Principal unresolved gap |
|---|---|---|---|
| MLP | Feature-last activation probe; class/scalar output head | EK-FAC/TRAK comparison; output perturbation tests | No verbalization notion unless separately defined |
| CNN | Canonicalize spatial positions/channels; preserve receptive-field metadata | Patch/receptive-field influence; spatial counterfactuals | Position sums are not exact Conv influence |
| GRU/RNN | Preserve time, hidden/state layout, and recurrent weight sharing | Gate/time factors; trajectory and long-horizon perturbations | Shared-weight influence and vanishing higher-order effects |
| Transformer | Residual/attention/MLP probes; optional vocabulary unembedding | J-lens/CLT graphs; attention stop-gradient convention; prompt-specific validation | Most published dictionaries/graphs are not architecture-neutral |
| GNN/equivariant scientific net | Node/edge/graph layout; permutation and group-action metadata | Symmetry tests, group-held-out splits, physical validators | Feature charts must respect permutation/equivariance semantics |
| Scientific supervised/RL system | Dimensionalized observables and constraint heads | Units, conservation/bounds, simulator or physical intervention, uncertainty calibration | Latent importance is not a physical causal law |
| RL agent | State/action/reward/cost/policy probes across complete trajectories | Reward/cost returns, environment constraints, off-policy and seed variance | Expected constraint satisfaction does not certify every trajectory |

### 5.3 RL and safer training

A constrained RL objective can be written

\[
 \max_\theta J_R(\theta)\quad\text{subject to}\quad J_C(\theta)\le d,
 \qquad
 \mathcal L(\theta,\lambda)=J_R(\theta)-\lambda(J_C(\theta)-d),\quad\lambda\ge0.
\]

Safe RLHF separates helpfulness reward and harmlessness cost and adapts the tradeoff through the dual variable. CMAE can trace changes in reward, cost, \(\lambda\), policy entropy, and latent charts across updates. It must still report the distribution of per-trajectory costs and catastrophic tail events. A low average cost can coexist with rare unacceptable outcomes.

Constitutional AI and deliberative alignment supply training data and judge/policy artifacts to audit. They do not eliminate the need for independent judges, adversarial evaluations, or latent evidence. Generated critiques and chains-of-thought should be treated as model outputs, not automatically faithful explanations.

### 5.4 Scientific domains

For scientific models, an NLA explanation such as “the model uses steric clash” is only a hypothesis until linked to measurable inputs and interventions. CMAE should add:

- unit and dimensional-analysis checks;
- conservation, positivity, bound, symmetry, and known-equation residuals where applicable;
- leave-one-family/composition/time/region-out evaluation to expose leakage and extrapolation;
- calibrated epistemic uncertainty and simulation/physical validation for high-impact claims;
- provenance for experimental conditions, simulator version, and data-cleaning choices.

Physical constraints can themselves be approximate or misspecified, so their residuals are evidence with assumptions, not absolute truth. Explanations of model behavior also do not establish causal laws in the physical system.

### 5.5 NLA-heavy investigation with claim typing

Every generated natural-language claim should be typed as one of:

1. **Observed:** directly measured activation, output, or data pattern.
2. **Reconstructive:** true of a surrogate/chart, with residual reported.
3. **Interventional:** supported by a specified activation/data/environment intervention.
4. **Counterfactual-training:** supported by retraining or a validated attribution approximation.
5. **Root-cause/semantic:** strongest claim; requires converging causal and behavioral evidence and alternative-hypothesis tests.

Evaluation should include claim deletion/insertion tests, counterfactual consistency, human/domain review, and confabulation rate on intentionally unanswerable cases. “No evidence” must be an allowed NLA output.

### 5.6 Real-time flagging and repair

Real-time operation needs a pre-update latency/memory budget and a fail-closed policy for exhausted critical checks. Cheap synchronous checks can gate training; expensive graph, NLA, retraining, and multi-field analyses can run asynchronously but cannot retroactively make an already accepted unsafe update safe.

Repair remains a contract: quarantine suspect data/trajectories, propose a data or model intervention, retrain/fine-tune in isolation, then pass clean utility, adaptive attack, constraint, and held-out behavioral gates. A steering direction or filtered batch is not a repair certificate.

## 6. Prioritized falsification program

| Priority | Experiment | Acceptance/falsification criterion |
|---|---|---|
| P0 | Tracr/known tiny programs plus trained MLP/CNN/GRU/Transformer models | Score graph/chart/NLA recovery against known mechanisms; report false mechanisms, not only successful examples |
| P0 | Adaptive poisoning with correctly labeled regularizers and asymmetric triggers | Compare clean false positives, attack recall, and abstention against ART/spectral, PCA/RepE, and no-latent behavioral baselines |
| P1 | Generic downstream-Jacobian adapter on analytic linear and nonlinear toy networks | Exact recovery in the linear case; bounded/measured Taylor residual and held-out causal effect in nonlinear cases |
| P1 | EK-FAC versus TRAK versus actual group retraining | Held-out counterfactual prediction and decision accuracy with confidence intervals and compute cost |
| P2 | Multi-field persistence on RP²/Klein-bottle controls, then a justified neural integral complex | Reproduce known torsion first; adopt only if it adds held-out signal beyond F2 persistence and graph baselines |
| P2 | Signed TopK versus JumpReLU and dense/PCA charts | Compare downstream residual effect, stability, causal sufficiency, NLA faithfulness, and cost—not only reconstruction |
| P3 | Toy CMDP/Safe-RLHF audit with rare high-cost trajectories | Detect cases where expected-cost compliance hides tail violations; test post-repair policy rollouts |
| P3 | Scientific adapter with unit/symmetry/constraint checks | Use family-held-out and physical/simulator interventions; reject physically invalid but latent-plausible explanations |

Runs that train models should remain sequential in this repository. Development, calibration, and final test attacks must be separated. Thresholds selected on final tests invalidate the comparison.

## 7. New local artifacts and exact results

### Executable

- `prime_topology.py`: exact modular Gaussian elimination, finite-chain/graph Betti numbers, power traces, Newton characteristic-polynomial recovery, and the prime-trace counterexample.
- `tests/test_prime_topology.py`: nine tests covering 2-/3-torsion, a prime-independent graph control, input validation, Newton recovery, and spectral incompleteness.
- Combined CMAE suite after adding these tests: **33 passed** with PyTorch 2.6.0, NumPy 2.2.6, pytest, and IBM ART 1.20.1.

### Rocq 9.2

`../../formal_verification/rocq/CMAEPrimeFoundations.v` contains six theorems. All six `Print Assumptions` queries report **Closed under the global context**:

1. agreement of the explicit spectra at exponents 2, 3, 5, and 7;
2. a distinguishing fourth trace;
3. exact 2-/3-torsion ranks;
4. RP² CW Betti-number changes;
5. the third Newton identity;
6. the fourth Newton identity.

The proof of agreement at *every* odd prime follows from sign symmetry in the displayed matrices; Rocq checks representative exact instances, not a universally quantified infinite-prime theorem.

### Z3

`../../formal_verification/verify_cmae_prime_topology_z3.py` passed **six checks**: four expected-UNSAT identity/example checks and two intentional SAT distinguishing witnesses.

### Reproduction

```sh
uv run --no-project \
  --with 'torch==2.6.0' --with 'numpy==2.2.6' \
  --with 'pytest>=8' --with 'adversarial-robustness-toolbox==1.20.1' \
  python -m pytest -q -c /dev/null research/cmae/tests
```

```sh
uv run --no-project --with z3-solver \
  python formal_verification/verify_cmae_prime_topology_z3.py
```

With the Rocq 9.2/Stdlib 9.2 toolchain used here:

```sh
rocq compile -Q <path-to-stdlib-9.2-theories> Stdlib \
  formal_verification/rocq/CMAEPrimeFoundations.v
```

No new trained-model detection result, frontier-model audit, real-time benchmark, physical-science validation, RL safety certificate, or automatic repair guarantee is claimed in this phase.
