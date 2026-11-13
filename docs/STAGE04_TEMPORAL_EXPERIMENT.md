# Stage 04 temporal decoder experiment

The first submission scored 0.27096 publicly. On the labelled development partition, shots with seven or more active detectors accounted for 73.0% of its bit errors. Up to two global faults explain a syndrome on only 1.6–1.7% of development shots with ten or more active detectors. The circuit model offers a different decomposition. Every mechanism touches one time slice or two adjacent slices, and each slice has an expected fault count below 0.75. [Development diagnostics](EXPERIMENT_LOG.md)

## Hypothesis and fixed settings

Combine local fault distributions through the six measured syndrome slices. This should recover more dense shots than the parity classifier and decrease frozen combined LER by at least 0.025 relative to `parity_v1`. Require a decrease of at least 0.020 on the shots with seven or more active detectors and no regression in either full basis. The 95% paired interval for the overall gain must exclude zero.

Use the unchanged X and Z circuits and their independent detector error models. The asset generator uses circuit probabilities and detector coordinates. The existing parity checkpoint supplies fallback predictions. This experiment uses no additional labelled training, random seeds, or leaderboard-based parameter fitting. Compare with the fixed frozen validation arrays seeded with `2026111301` and `2026111302`. Fix the settings before opening that holdout for this candidate.

- Assign each mechanism to its earliest touched slice. Enumerate all subsets of up to three mechanisms within each slice. Use the product of their probability odds as the subset weight. Aggregate by both projected detector parities and store the total weight and four signed weights. Each signed weight multiplies the subset weight by `(-1)^L_j`. The last slice has seven bits and no outgoing parity. Compute its full distribution by XOR convolution, including subsets larger than three.
- Store the 48 largest-mass outgoing parities for each incoming parity, using the outgoing integer mask to break ties. Compute the exact incoming-parity marginal for every slice separately by XOR convolution across all mechanisms assigned to that slice.
- For an observed slice `s_t` and boundary state `b`, query incoming parity `s_t XOR b`. Combine each allowed outgoing parity with the state, multiplying the unsigned and corresponding signed weights and summing paths with the same outgoing parity. Keep up to 32 states per slice, ranked by accumulated mass times the exact next-slice likelihood of `s_(t+1) XOR outgoing`. Break ties by outgoing mask. Sum final masses and infer each bit from the sign of its signed sum. Use the frozen parity output for a bit if no path survives or the signed magnitude is at most 2.5% of total surviving mass.
- Set a 20-minute wall-time and 5 GiB peak RSS budget for asset generation. Require a local median of five full validation runs of at most 45 seconds and peak RSS at most 2 GiB, using four CPU threads. Record measured runtime and compute the supplied competition penalty as a sensitivity calculation on the local timings. Official timing is hardware-dependent. Require the locally penalized LER to improve on the incumbent as well. Verify repeatability across batch sizes and in a new process before generating an upload candidate.

## Relation to prior work

Trellis inference, parity transforms, and likelihood summation are established. Tesseract searches global fault subsets for a most-likely error. Cheng and Pan describe exact partition-function evaluation by rank decomposition. The design here uses time-slice odds tables, five boundary moments, exact one-slice lookahead for pruning, and the existing parity predictor for ambiguous bits. The implementation is independent and tailored to these six slices. The literature check cannot establish that the combination has never been implemented. [Tesseract](https://arxiv.org/abs/2503.10988), [Cheng and Pan](https://arxiv.org/abs/2609.39556), [Stim detector error model specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)

## Results and promotion

| Candidate | X LER | Z LER | Combined LER |
|---|---:|---:|---:|
| `parity_v1` | 0.27360250 | 0.26159500 | 0.26759875 |
| `temporal_v1` | 0.03580500 | 0.03419750 | 0.03500125 |

The paired gain is 0.23259750, with a 95% interval of [0.23133240, 0.23386260]. The gain on shots with at least seven active detectors is 0.30466229. All eight promotion checks passed. The five-run local median was 4.11050 seconds for 200,000 shots using four decoding workers. Peak inference RSS was 221.6 MiB. The generated checkpoint occupies 146 MiB and built in 27.6 seconds with a peak RSS of 1044.7 MiB. Per-slice table retention before path pruning was 98.51% to 99.71%.

All 29 tests passed. They include comparison with exhaustive fault enumeration, cross-slice parity checks and the prediction contract. A fresh process reproduced both validation prediction hashes. Generating the test CSV with batches of 3,072 also reproduced the original CSV byte-for-byte. The result has 200,000 rows and exact template ID order. Its SHA-256 is `fb40cf76ffd12784c0f8ddce299c692b896c7a839106030e6c4dd8baa66b0433`. The public Kaggle score is pending. [Frozen validation](../artifacts/stage04/validation.json), [submission CSV](../artifacts/submissions/temporal_v1.csv), [CSV replay](../artifacts/stage04/submission_replay.json)

The Python wrapper uses NumPy 2.2.6, Stim 1.16.0 and the standard library. The C++ kernel is built for Linux x86-64-v2 with floating-point contraction disabled. It needs the system C and C++ runtime libraries. The build script and the bundled source preserve the compiler command for reproduction. The code was independently developed with AI assistance. The retained table and path widths make this an approximation to full likelihood inference. The measured runtime describes the local machine.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_temporal.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_temporal.py --replay
.venv/bin/python tools/prepare_parity_submission.py --candidate temporal_v1 \
  --verify artifacts/submissions/temporal_v1.csv --batch-size 3072
```

Build in a workspace without the temporal checkpoint. To score it for the first time, run `tools/evaluate_temporal.py` without `--replay` in a workspace without the Stage 04 validation report.
