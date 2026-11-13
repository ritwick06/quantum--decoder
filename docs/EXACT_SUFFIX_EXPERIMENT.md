# Exact final two measurement slices

## Preregistered experiment

The full local transition calculation recovered probability mass in every table, but changed frozen LER by only 0.000005. This experiment targets where the beam prunes paths. The incumbent ranks states after the fourth processed slice using the fifth observation alone. It advances and prunes again before using the sixth observation.

For each direction, group all faults assigned to its last two slices. Compute their joint parity distribution over the 14 remaining detector bits and the seven terminal bits. Use the 21-bit Walsh–Hadamard calculation from the full local distribution experiment and retain every joint parity with mass above $2\times10^{-15}$. Store five moments per parity in float32 and an unsigned lookup for ranking. Keep the original transition tables and beam width for the first four slices.

After the fourth slice, rank each boundary state with the exact joint likelihood of the remaining two observed parities after accounting for the boundary XOR. Apply all five joint moments when summing the final beam. Both directions use this change and retain the incumbent combination and ambiguity rules. The additional evidence can prevent pruning paths that explain the final observation. Eliminating the last sparse expansion can offset the larger lookup. No fitting or leaderboard tuning.

Compare with `temporal_bidir_v1` on the frozen validation arrays with seeds `2026111301` and `2026111302`. Require a positive paired 95% lower bound for combined LER improvement and a local five-run median at or below 12 seconds over 200,000 shots with four workers. Keep the 2 GiB inference and 20-minute, 5 GiB build budgets. Confirm a passing gain on 250,000 new shots per basis with seeds `2026111321` and `2026111322`, with a positive paired 95% lower bound. Replay the CSV byte for byte in a fresh process before flagging it.

## Results

Frozen combined LER fell from 0.03410125 to 0.03405625. X improved by 10 bit errors and Z by 26. The paired 95% gain interval was [0.00000314, 0.00008686]. The five-run local median was 9.75 seconds and peak inference RSS was 815 MiB. The build took 6.4 seconds and all 35 tests passed. [Frozen validation](../artifacts/suffix/validation.json)

On 500,000 new Stim shots the candidate made 60 fewer bit errors. The absolute paired gain was 0.000030 with a 95% interval of [0.00000381, 0.00005619]. The gain is small and both local comparisons favored the new candidate. [Independent confirmation](../artifacts/suffix/confirmation.json)

Fresh processes reproduced the frozen prediction hashes and regenerated the 200,000-row submission CSV byte for byte at batch size 3,072. The extracted hand-in archive also reproduced all submitted bits and IDs. Rebuilding from the bundled sources with the original competition inputs reproduced the model arrays, compiled kernel and trained fallback byte for byte on the local environment. [CSV replay](../artifacts/suffix/submission_replay.json), [archive verification](../artifacts/hand_in/temporal_suffix_v1.verification.json)

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_temporal.py --suffix
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_temporal.py --candidate temporal_suffix_v1 --replay
.venv/bin/python tools/prepare_parity_submission.py --candidate temporal_suffix_v1 \
  --verify artifacts/submissions/temporal_suffix_v1.csv --batch-size 3072
```
