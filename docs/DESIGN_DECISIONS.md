# Decoder design decisions

**Stage 03A complete.** Candidate C awaits approval for implementation and training.

## Problem observations and objective

Each shot supplies 70 detector bits and requires four logical predictions. X and Z use separate circuits and must have separate circuit-derived models.

A deterministic development partition was used for descriptive inspection. For each training ID, compute SHA-256 of `qec-stage00-v1:` followed by the ID; interpret the first eight bytes as an unsigned big-endian integer. Reserve rows whose value modulo five is zero.

| Basis | Development rows | Reserved rows | Mean active detectors | Median | 90th percentile | All-zero detectors |
|---|---:|---:|---:|---:|---:|---:|
| X | 80,130 | 19,870 | 7.43 | 7 | 13 | 4.42% |
| Z | 80,121 | 19,879 | 7.30 | 7 | 13 | 4.82% |

Reserved rows were excluded from these distribution summaries. All-file binary/schema checks were performed. Active-detector count is not the number of physical faults.

The official [metadata helper](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/src/hackathon_2026/circuit_tools.py) exposes detector effects $H$, logical effects $L$, and mechanism probabilities $p$. Mechanisms with identical detector effects but different logical effects must remain distinct.

Under the independent-mechanism detector-error model,

$$
s=He\pmod2,\qquad o=Le\pmod2.
$$

Our scoring objective is **bitwise error**, so each output should approximate

$$
\hat o_j=\mathbf1\{P(o_j=1\mid s)>1/2\}.
$$

Selecting the most likely complete four-bit outcome is a different objective. Ties will resolve to zero.

## Candidate A: bounded fault enumeration with logical-mass aggregation

**Foundation.** Enumerate plausible combinations of circuit error mechanisms whose XOR detector signature matches the observed syndrome. Aggregate their probabilities by logical outcome rather than choosing only the cheapest explanation.

For retained explanations $\mathcal E(s)$,

$$
q_j(s)=
\frac{\sum_{e\in\mathcal E(s)}P(e)\mathbf1\{(Le)_j=1\}}
{\sum_{e\in\mathcal E(s)}P(e)}.
$$

These are approximate marginals because the search omits explanations.

**Architecture and processing.**

- Compile separate X/Z catalogues of mechanism signatures, probabilities, and logical masks.
- Index single- and two-mechanism XOR signatures.
- Query exact low-order matches, then perform bounded residual queries for additional explanations.
- Enforce canonical mechanism ordering to prevent repeated mechanisms and double-counted explanations.
- Allocate additional search using detector-time occupancy and logical ambiguity, with fixed deterministic work limits.
- When no explanation is found, return explicit circuit-prior predictions and record fallback frequency during validation.

**Complexity and memory.** For $M$ mechanisms, pair construction costs $O(M^2)$ time and storage. Searching all single-plus-pair residuals costs approximately $O(M+C_s)$ per shot with indexed lookup, where $C_s$ is the number of candidate records examined; large collision buckets require a hard bound.

At roughly 2,500 mechanisms, there are about 3.1 million unordered pairs. A 32-byte compact pair record alone occupies about 100 MB per basis, before index overhead. Actual mechanism counts must be verified locally; the organizer's [notebook outputs](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/notebooks/getting_started.ipynb) report approximately this scale.

**Expected trade-off.** Strong physical interpretability and no mandatory training. Potentially effective on low-fault shots, but larger fault combinations and omitted probability mass can cause systematic errors. Hundreds of millions of residual queries could threaten runtime.

**Independent decisions.** The proposed time-occupancy search budget, logical-mass aggregation, and explicit fallback diagnostics are project choices. Enumerative decoding and Bayesian aggregation are established ideas.

**Difficulty:** Medium to high. Main risks are memory overhead, duplicate counting, incomplete coverage, and speed.

## Candidate B: temporal frontier inference retaining all logical outcomes

**Foundation.** Process mechanisms in detector-time order. Track unresolved detector parities and a probability vector over the 16 logical outcomes. Sum compatible histories when their unresolved state agrees.

**Architecture and processing.**

- Derive detector times and mechanism spans from the actual circuit.
- Introduce each mechanism once, with its error/no-error probabilities.
- Check a detector when no remaining mechanism can affect it; discard inconsistent states.
- Retain the probability vector over all 16 four-bit logical outcomes when merging equivalent frontier states.
- If exact inference is too large, keep a bounded number of frontier states with deterministic pruning and a discarded-mass diagnostic.
- Convert final logical-state mass to four bitwise marginals.

**Complexity and memory.** If the largest unresolved detector frontier has width $w$, exact inference can require approximately $O(M\,16\,2^w)$ time and $O(16\,2^w)$ memory. A pruned frontier with $K$ states reduces storage to $O(16K)$, with transition and pruning costs dependent on $K$. The relevant width is not yet measured.

**Expected trade-off.** Can capture temporally separated faults and logical degeneracy more fully than shallow enumeration. Long-range mechanisms can make exact inference impractical; pruning can discard the correct explanation.

**Independent decisions.** Circuit-specific processing order, boundary treatment, retaining all logical classes per frontier state, and pruning diagnostics are our proposed choices. Dynamic programming and trellis inference are established techniques.

**Difficulty:** High. Best treated as a later research candidate unless measured frontier width is unexpectedly small.

## Candidate C: circuit-selected parity features with reliability-conditioned marginal prediction

**Foundation.** Use the circuit to identify detector parities that carry information about each logical bit, then learn a compact prediction rule from labelled development data.

For a detector mask $a$, define $x_a(s)=(-1)^{a\cdot s}$. Independence in the detector-error model gives

$$
E[x_a]=\prod_m(1-2p_m)^{a\cdot h_m\bmod2},
$$

and, for logical bit $j$,

$$
E[x_a(-1)^{o_j}]
=\prod_m(1-2p_m)^{(a\cdot h_m\bmod2)\oplus L_{jm}}.
$$

These moments provide a circuit-derived correlation score for candidate features without using test labels.

**Architecture and processing.**

- Build candidate masks from individual detectors, detector pairs affected by a common mechanism, adjacent-time checks, and full mechanism detector supports.
- Remove duplicates and constant features. Rank masks separately for each logical output using normalized circuit-derived covariance.
- Add six detector-round occupancy features and interactions between selected parity features and occupancy in their associated time slices.
- Fit four regularized logistic predictors per basis, using our own feature construction, loss, and gradient with a permitted general-purpose optimizer.
- Infer in bounded batches; threshold each marginal at 0.5.
- Store masks, coefficients, feature definitions, and circuit fingerprints as reproducible assets. Predictions depend only on circuit and detector events, never IDs.

**Complexity and memory.** Let $C$ be the candidate-mask count, $F$ the retained feature count, $M$ the mechanism count, and $B=\lceil70/64\rceil=2$. Circuit screening costs approximately $O(CMB)$. Inference is $O(NFB)$, plus the small linear readout. Training costs $O(INF)$ for $I$ optimization iterations, with constant factors for four outputs and interaction features.

Model storage is $O(FB+F)$; bounded feature batches use $O(n_{\text{batch}}F)$ memory. There is no need to materialize all training features simultaneously.

**Expected trade-off.** Predictable CPU inference, compact assets, and training on the available labelled data. The model can miss high-order ambiguities not represented by selected masks. Circuit-derived moments guide feature selection; they do not make the learned posterior exact.

**Independent decisions.** The feature-mask construction, physical correlation screening, detector-time reliability interactions, and marginal readout form the proposed project design. Fourier parity identities and logistic regression are established mathematics, not claimed inventions.

**Difficulty:** Medium. Main risks are feature insufficiency, optimization cost, and calibration near the binary threshold.

## Comparison and recommendation

| Approach | Main strength | Main risk | Training | Runtime predictability |
|---|---|---|---|---|
| A: fault enumeration | Direct circuit explanations | Search coverage and pair-index cost | Optional | Moderate |
| B: temporal frontier | Rich joint inference | Exponential frontier growth | None required | Low until width is measured |
| C: parity features | Compact, bounded inference | Limited representation of complex ambiguity | Modest supervised fitting | High |

**Recommend Candidate C for the first approved experiment.** Its bounded inference and small assets make it the strongest engineering choice under the competition deadline. Candidate A is the preferred challenger if measured accuracy is insufficient. Candidate B should wait for evidence that its frontier fits the resource budget.

This is a feasibility recommendation, not a measured accuracy ranking.

## Validation and originality controls

- Retain the inspected development/reserved split.
- In Stage 01, generate and freeze a separate validation dataset of 100,000 shots per basis using seeds `2026111301` and `2026111302`.
- Record circuit hashes, Stim version, sample-call sizes, and generated-array hashes. Preserve the arrays; a seed alone is not a complete portability guarantee.
- Future generated training data uses separate seeds and files. Do not exclude naturally repeated syndromes merely because independent samples share them.
- Use development data for fitting and tuning; frozen validation is a bounded promotion check, not a hyperparameter-search target.
- First proposed Candidate C experiment: at most 64 selected masks per output, CPU execution, 20-minute training limit, and 8 GiB memory limit. Stage 03A must lock the remaining optimization settings before implementation.
- Promotion requires all correctness checks, at least **0.005 absolute combined-LER improvement** over the zero baseline, no basis regression, and a paired shot-level 95% confidence interval supporting improvement. These are internal project thresholds, not competition rules.
- Measure local latency and memory; do not claim local runtime equals organizer runtime.
- Record conceptual sources and independently written modules. No candidate is claimed to be globally novel.

Conceptual references are the organizer's metadata API and [Stim detector-error-model semantics](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md). No prohibited decoder implementation is needed for these proposals.

## Stage 03A circuit and development-data review

Read the pinned X and Z circuits with Stim 1.16.0 and the organizer's `problem_info` helper. Checked the training CSV and split-mask hashes against the Stage 01 report. Parsed detector events and labels only for rows marked as development. The frozen validation manifest hash is `4a8df2e10c37df7119d962399ce25f325306ff0c33c48e5ad37adb66c6a090cb`. The analysis used its existing baseline scores from Stage 01.

| Observation | X | Z |
|---|---:|---:|
| Development shots | 80,130 | 80,121 |
| All-zero detector traces | 3,542 (4.42%) | 3,864 (4.82%) |
| Active detectors per shot, median / 90th percentile | 7 / 13 | 7 / 13 |
| Mean active detectors at times 0 through 5 | 0.579, 1.538, 1.522, 1.532, 1.526, 0.736 | 0.411, 1.532, 1.530, 1.518, 1.523, 0.785 |
| Median adjacent-time Pearson correlation, same check coordinates | 0.1294 | 0.1280 |
| Median adjacent-time Pearson correlation, different coordinates | 0.0131 | 0.0133 |
| Logical flip rates in L0–L3 order | 30.48%, 30.75%, 24.16%, 31.16% | 31.22%, 33.62%, 24.00%, 25.14% |
| Logical word `0000` | 34,906 (43.56%) | 32,712 (40.83%) |
| Logical word `1111` | 3,771 (4.71%) | 1,009 (1.26%) |
| Distinct detector traces | 64,109 | 63,408 |
| Repeated trace groups with differing logical labels | 67 groups, 333 shots | 50 groups, 229 shots |

The six detector groups have sizes 7, 14, 14, 14, 14, and 7. Across the five neighboring pairs of groups, the correlation between active-detector counts ranges from 0.091 to 0.157 in X and from 0.127 to 0.150 in Z. Same-coordinate pairs cover 56 neighboring detector pairs per basis. All 16 logical words appear in both development sets. The repeated traces demonstrate ambiguity in the four outputs even with all 70 detector bits present. These counts depend on which traces repeated in the available sample.

The circuit model has 2,470 mechanisms for X and 2,465 for Z. Every mechanism has a distinct, nonempty detector signature in its basis and touches at most seven detectors. Each signature lies within a single time slice or spans two consecutive slices. The model expects 3.099 active mechanisms per X shot and 3.047 per Z shot. These are expectations under the model of independent mechanisms. A common detector trace can arise from different XOR combinations even when the individual signatures are distinct.

### Comparison using the measured circuits

| Proposal | Decision rule and expected cost | Evidence and failure mode |
|---|---|---|
| A. Bounded fault enumeration | Index the 0-, 1-, and 2-mechanism syndromes. For more explanations, XOR the observed syndrome with candidate single mechanisms and query the pair index. Cap the residual search and enumerate each subset once. Weight each matching subset by the product of its probability odds and threshold the resulting per-bit mass at 1/2. Use the unconditional circuit marginals when no explanation survives. | The index has 3,049,215 X pairs and 3,036,880 Z pairs. At 32 bytes per pair it needs about 97 MB per basis before indexing. The model puts 40.0% X and 41.2% Z of fault count mass at zero to two active mechanisms. The mass rises to 62.5% X and 63.7% Z through three. Truncation can distort the logical probabilities and dense collision buckets can dominate runtime. Prioritize residual probes with overlap on fired detectors, then by probability and canonical mechanism order. This is a circuit-specific likelihood search with summed logical probabilities. |
| B. Temporal frontier | Order by earliest detector time, latest time, and mechanism index. For each mechanism propagate mass with weights `1-p` and `p`. The second branch XORs its detector and logical masks into the state. Enforce each observed parity at its last contributing mechanism, then merge by the remaining detector parities and sum over the 16 logical words. A bounded version prunes by total state mass and tracks discarded mass. | The open frontier across complete time slices has widths 14, 14, 14, 14, and 7. This mechanism ordering reaches 26 open bits before constraint enforcement. A dense buffer with 16 logical masses per state could reach 8 GiB in float64. Exact state counts depend on the syndrome. Pruning can remove paths that later carry the correct logical outcome. This belongs to the trellis marginalization family. |
| C. Circuit-selected parity features | Rank mask parities using the circuit moments above. Fit a regularized predictor per basis and logical bit, using selected parities, time-slice occupancy, and their interactions. Evaluate with fixed-size batches and integer scores. | The measured circuit library contains 3,248 eligible nonempty masks per basis before variance filtering. The strongest absolute correlations per logical bit are 0.163–0.224 in X and 0.156–0.220 in Z. Between 28 and 37 of the top 64 masks touch the final slice per bit. Between 20 and 29 end by time 3. The model can use evidence from both ends of the circuit within a fixed compute budget. It can miss dependencies outside the mask library, and the ranking measures model correlation rather than held-out prediction accuracy. |

Choose **Candidate C** for the first run. Its feature library follows the two-slice mechanism supports and the observed same-check correlations. Its local occupancy interaction allows a parity to carry different weight when the surrounding checks are busy. The separate X/Z fits and per-bit decisions follow the observed label distributions and the scoring metric. Candidate A has a concrete path to a later comparison if the parity features underfit. Candidate B needs a smaller effective state representation before a full experiment.

The Boolean parity moments are established mathematics. Wagner et al. use Boolean Fourier analysis to estimate Pauli noise from syndrome measurements. Here the model supplies moments for ranking detector masks against each logical observable. Bahl et al. provide the classical precedent for trellis marginalization. The candidate-specific mask library, ranking, occupancy features and integer inference specification were developed for this project with AI assistance. [Stim DEM specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md), [Wagner et al.](https://arxiv.org/abs/2107.14252), [Bahl et al.](https://research.ibm.com/publications/optimal-decoding-of-linear-codes-for-minimizing-symbol-error-rate)

## Candidate C first implementation and experiment

The hypothesis is that up to 64 circuit-ranked masks per output, with time occupancy interactions, yield at least 0.005 absolute LER improvement over the frozen zero baseline within the local runtime budget.

1. **Rank features.** Build the union of all 70 singletons, all pairs contained in a mechanism's support, pairs with equal non-time coordinates at consecutive times, and all nonempty full mechanism supports. A mask is a sorted tuple of detector indices. Compute the parity expectations from the verified independent DEM using float64 `log1p(-2*p)` and a sum over selected mechanisms, then exponentiate. Require probabilities in `[0, 0.5)` for this log-domain implementation. For each bit, use the absolute normalized covariance of the two signs. Remove masks with normalization denominator at most `1e-12`. Sort descending by the absolute correlation rounded to 12 decimals, then ascending by detector-index tuple. Take the first 64 independently for each output and basis.
2. **Construct features.** With `d_r` equal to fired detectors divided by detector count in time slice `r`, use six `d_r` features, then the selected parity signs `x_a`, then `x_a * mean(d_r for r in T(a))` in mask order. `T(a)` is the set of time slices touched by `a`, with size one or two for this circuit library. Standardize each feature using its development-set mean and population standard deviation with a scale floor of `0.10`. Use the supplied labelled development split exclusively for these statistics and fitting. Fit independently for all eight basis/bit combinations.
3. **Fit one configuration.** Use CPU float64 regularized logistic regression with an unpenalized intercept and loss `mean(logaddexp(0,z) - y*z) + 0.0005*sum(w*w)`, with `z = intercept + standardized_features @ w`. Initialize slopes to zero and the intercept to the empirical log-odds of the bit. Solve the analytic Newton step with `numpy.linalg.solve`, using sigmoid `exp(-logaddexp(0,-z))`. Start each step at length 1 and halve up to 20 times until Armijo decrease with constant `1e-4`. Stop when the infinity norm of the full gradient is at most `1e-6` or after 30 accepted steps. Check convergence again after the last step. A constant target, nonfinite value, failed solve, failed line search, or unconverged fit fails the run. This configuration has no stochastic fit seed, class reweighting, threshold tuning, or hyperparameter search.
4. **Export and infer.** Fold standardization into the intercept and the coefficients of the raw features. Raw features have a common denominator of 28. Form the integer numerators using `d_r * 28 = (28 // size_r) * count_r`, `28*x_a`, and `x_a * sum(28*d_r for r in T(a)) // len(T(a))`. The division is exact because every group size is 7 or 14. Export signed int64 intercept `round_even(2^24 * raw_intercept)` and coefficients `round_even(2^24 * raw_coefficient / 28)`. Verify `abs(intercept) + 28*sum(abs(coefficients)) < 2^62` using Python integers before casting and when loading. Predict 1 exactly when the int64 score is positive. This defines the scored model and gives exact reproducibility across chunk boundaries and CPU reduction orders. Use two little-endian uint64 words per trace, with detector index `i` at bit `i % 64` of word `i // 64` and unused bits set to zero. Obtain the parity with `numpy.bitwise_count` and convert to a signed dtype before computing `1 - 2*parity`. Process batches of at most 4,096 shots. [NumPy 2.2 bitwise_count](https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_count.html)

### Module boundaries and contract

- `team_decoder/decoder.py` owns asset loading, circuit identification, feature evaluation and `decode(circuit, detector_events)`. The deliverable checkpoint contains this source, `requirements.txt`, fixed model arrays and provenance. Store numeric arrays in NPZ with pickle disabled and metadata in JSON. Store the expected model-array file hash and feature format version in the metadata. Load assets using paths relative to the decoder file. Retain both canonical Stim circuit hashes from Stage 01 and match `sha256(str(circuit).encode('utf-8'))` at inference.
- `tools/train_parity.py` reads only development labels, constructs features and fits the eight heads with the frozen settings. Store each selected mask, its time-slice membership, the six detector groups, the final coefficients, and the bias. Record the hashes of the circuits, training CSV, split mask, source and exported assets, the optimizer diagnostics, Python/NumPy/Stim versions, and source attributions. Use Python 3.13, NumPy 2.2.6 and Stim 1.16.0. Keep model creation and validation CPU-only.
- The decoder accepts `(N,70)` Boolean or integer binary input, including noncontiguous views and empty batches, for either pinned circuit. It returns `(N,4)` `uint8` predictions in L0–L3 and input row order. Invalid dimensions, dtype, bit values, unfamiliar circuit hashes and missing or incompatible assets raise errors identifying the failed condition. On an all-zero trace use the same scoring rule as other shots. The accepted zero checkpoint remains available as its own version.
- Stage 03B may prepare the promoted candidate's submission through the existing Stage 02 schema, basis routing and CSV validation helpers. Record the candidate's source, asset and CSV hashes in the submission ledger. The generator must use the promoted checkpoint and the verified unlabelled competition test file, with no training labels on the submission path.

The largest design matrix has 135 columns including the intercept and occupies about 83 MiB in float64 for one basis/bit. Fit one head at a time. The Newton step costs $O(NF^2+F^3)$ for $F\le135$, and the optimizer stores one design matrix plus weighted work buffers. Inference uses at most 102.4 million 64-bit population counts and about 108 million integer coefficient terms across 200,000 shots. The packed masks and coefficients for both bases occupy under 32 KiB before file metadata. The first experiment uses a 20-minute wall-time limit and an 8 GiB peak RSS limit for feature selection, fitting and export combined. These bounds govern the local experiment. Candidate latency remains to be measured.

### Validation and promotion

1. Verify development input hashes and exclude the reserved rows from fitting and normalization. Freeze the exported integer model before scoring. Use the original frozen validation arrays, 100,000 shots per basis, with seeds `2026111301` and `2026111302` and the recorded array hashes. Score this configuration once as the promotion check.
2. Test parity moments against exhaustive enumeration of a tiny independent DEM. Test detector indices 63, 64 and 69, both single- and two-slice occupancy interactions, zero-score tie handling, fixed-point export against a hand-calculated score, and overflow rejection. Verify input errors, unsupported circuit and asset errors, `N=0`, `N=1`, noncontiguous views, shuffled and mixed-basis rows through the existing routing helper, and identical full-basis versus chunked results. Run the existing tests and the new candidate checks, then compare prediction hashes across fresh processes.
3. Promote only if combined LER is at most **0.28290125**, X LER is at most **0.29236**, and Z LER is at most **0.2834425**. Compute the per-shot paired gain as `(zero wrong bits - candidate wrong bits)/4`. For the two basis strata, use the sample variance of those gains with `ddof=1`. For total size `N`, estimate standard error as `sqrt(n_X*var_X+n_Z*var_Z)/N`. Require the lower endpoint of the two-sided 95% interval, `mean_gain - 1.95996398454*standard_error`, to exceed zero. Preserve the LER and interval with the exact prediction hashes.
4. Require all checks to pass, training to finish within its resource budget, and the local median of five timed runs to be at most **10.0 seconds** with peak inference RSS at most **1 GiB**. Use the same machine and one-thread settings as Stage 01. Each run times the X and Z calls on 100,000 shots each and the output checks, with imports and scoring outside the timer. Record each timing and RSS. The local limits are engineering acceptance criteria. The supplied Kaggle scoring cap is 15.4 seconds with a 300-second hard limit on organizer hardware.
5. If all gates pass, generate one candidate CSV with exact test-ID order and verify byte-identical regeneration in a fresh process. This is the first independent result to recommend for manual upload. If a gate fails, record the result and retain the accepted zero checkpoint. A second configuration requires a separate experiment decision.

## Leaderboard handoff

The Stage 02 all-zero CSV passes local format and reproduction checks. It uses the organizer's interface baseline. Given the competition's independent-decoder requirement and the team's submission budget, the recommended first upload is the promoted Candidate C CSV after the checks above. Review the authenticated Kaggle submission screen for the deadline, remaining quota and any additional AI-use or disclosure terms before using a slot. The supplied closing time is 14 November 2026 at 4:30 AM IST. The live Rules and Overview pages could not be accessed during this review. The organizer notes and earlier user-supplied Kaggle rules remain the basis for the local workflow. [Stage 02 submission guide](SUBMISSION_GUIDE.md), [Rules audit](RULES_AUDIT.md)

Approval of this specification starts Stage 03B implementation and the one preregistered training run.
