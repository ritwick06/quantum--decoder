# Bidirectional temporal decoding

The forward pass in `temporal_v1` discards all but 32 partial boundary states at each slice. A boundary state with little support from earlier measurements can have substantial support once later measurements are included. Reversing the slice order changes the faults grouped at each slice and the states retained by pruning.

Build one new candidate, `temporal_bidir_v1`, using the same fault limit, 48 transitions per incoming parity, 32 boundary states and one-slice lookahead in both directions. Retain the existing forward tables byte-for-byte. Build the reverse tables by assigning each fault to its first touched slice in the reversed order. Each pass returns a total mass and four signed masses for the complete observed trace. Sum the two total masses and sum the corresponding signed masses. Use the same 2.5% ambiguity threshold and parity fallback on the combined masses. The two passes can retain overlapping explanations. Their sum is an equal mixture of the two truncated joint distributions, with greater influence from the pass retaining more mass for that trace.

Evaluate once against `temporal_v1` on the original frozen validation arrays, X seed `2026111301` and Z seed `2026111302`. This experiment uses no training labels or leaderboard scores to fit parameters. Promote only if combined LER improves by at least 0.0005, the paired 95% interval excludes zero, neither basis regresses, and the locally penalized LER improves. Keep the existing 20-minute and 5 GiB build budgets. Inference must fit within 12 seconds median over five runs and 2 GiB peak RSS on the local four-worker setup.

Verify that reversed detector groups preserve all 70 input columns and their within-slice order. Check signed-mass combination and fallback with synthetic examples, then require byte-identical fresh-process CSV regeneration before recommending an upload.

## Results

| Model | X LER | Z LER | Combined LER | Median local time, 200,000 shots |
|---|---:|---:|---:|---:|
| `temporal_v1` | 0.03580500 | 0.03419750 | 0.03500125 | 4.11050 s |
| `temporal_bidir_v1` | 0.03469250 | 0.03351000 | 0.03410125 | 8.21826 s |

The new decoder passed every promotion check. It made 720 fewer bit errors out of 800,000 predictions. The absolute paired LER gain was 0.000900, with a 95% interval of [0.000727, 0.001073]. The five-run median used four local workers; peak inference RSS was 515.3 MiB. The build took about 26 seconds and peaked at about 996 MiB. These are local measurements; organizer runtime and Kaggle scores are pending.

All 31 tests passed. A fresh process reproduced the frozen validation prediction hashes and separately regenerated the full submission CSV byte for byte at batch size 3,072. The file contains 200,000 binary prediction rows in sample ID order. The incumbent is preserved.

From the project root, using the pinned Python 3.13 environment:

```bash
# Build into a clean checkpoint destination, with the accepted forward checkpoint present.
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_temporal.py --bidirectional
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_temporal.py --candidate temporal_bidir_v1 --replay
.venv/bin/python tools/prepare_parity_submission.py \
  --candidate temporal_bidir_v1 --verify artifacts/submissions/temporal_bidir_v1.csv --batch-size 3072
```

The checkpoint provenance records the preregistration hash from before this results section was appended. See the [validation report](../artifacts/bidirectional/validation.json) and the [submission manifest](../artifacts/submissions/temporal_bidir_v1.csv.manifest.json) for the measured values, inputs and hashes. The [CSV](../artifacts/submissions/temporal_bidir_v1.csv) is ready for manual upload.
