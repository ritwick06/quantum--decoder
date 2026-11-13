# Adaptive beam width

## Preregistered experiment

On the labelled development split, a shot has at least one bit with normalized bidirectional moment magnitude at or below 0.45 in 8.44% of X shots and 8.27% of Z shots. The bits meeting that criterion account for 57.69% and 56.10% of the respective bit errors. The midpoint join failed on frozen validation, and blending its estimates with the incumbent on the development split increased errors.

Run the accepted bidirectional decoder with its existing 32-state beam and 48 transitions per row. If a shot has a bit with normalized signed magnitude at or below 0.45, or has zero retained mass, rerun both directions on that shot with a 128-state beam. Replace the estimates for the flagged bits with the wider search's estimates. Keep the existing ambiguity and parity fallback rule. Reuse all circuit tables and the lookahead heuristic. The additional search addresses beam pruning and still operates on the same truncated local transition tables.

Compare one fixed configuration against `temporal_bidir_v1` on the original frozen validation seeds `2026111301` and `2026111302`. The hypothesis is that keeping more competing histories improves bitwise LER on those ambiguous bits. Require a positive paired gain with a positive 95% lower bound, lower locally penalized LER, and at most 12 seconds median over five full runs on four workers. Keep the 2 GiB inference memory and 20-minute, 5 GiB build budgets. The thresholds and width are fixed before frozen scoring. No additional labelled fitting. Check that wider search recovers a deliberately pruned path and that both basis outputs pass the existing contract. Require fresh-process, byte-identical CSV reproduction before recommending an upload.

## Result

The adaptive decoder produced an LER of 0.03410625 versus 0.03410125 for `temporal_bidir_v1`. It made eight additional X errors and four fewer Z errors, for four additional errors over 800,000 bits. The paired gain was −0.000005 with a 95% interval of [−0.0000280, 0.0000180]. The median of five full local runs was 12.066 seconds against the 12-second budget. All 33 tests passed. [Frozen validation report](../artifacts/adaptive/validation.json)

This was the second consecutive experiment that failed promotion, ending the goal. The bidirectional decoder remains the upload candidate.

Development counts and the audited training split hashes are recorded with the experiment. [Development diagnostics](../artifacts/adaptive/development_diagnostics.json)

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_temporal.py --adaptive
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_temporal.py --candidate temporal_adaptive_v1 --replay
```
