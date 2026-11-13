# Backward likelihood before beam pruning

## Preregistered experiment

In the expanded-transition decoder, the third beam decision uses the fourth observation only. This is the last early pruning step before the exact likelihood for the final two measurements becomes available. A candidate whose fourth observation is individually probable can outrank one that explains the final measurements better.

On the same ambiguous shots used by `temporal_expanded_v1`, propagate the exact final-two-slice likelihood backward through the fourth transition table before pruning at the third step. Precompute the 64 largest suffix residual parities for each final seven-bit observation and an index from each fourth-step outgoing parity to its incoming transitions. Accumulate the resulting unsigned likelihood per fourth-slice residual. Rank third-step states by that likelihood times their prefix mass. Include a small floor equal to the largest omitted suffix likelihood times the fourth-slice single marginal, so a truncated backward message cannot declare a feasible state impossible. This floor is an upper bound on omitted mass under the retained transition model. Keep the forward expansion and signed terminal contraction unchanged after that step.

Compare against `temporal_expanded_v1` on the original frozen data. Require a positive paired 95% lower bound, at most 12 seconds local median over five runs for 200,000 shots, and 2 GiB peak inference RSS. If the candidate passes, confirm on 250,000 independent shots per basis with seeds `2026111371` and `2026111372`, again requiring a positive paired lower bound. Generate an upload candidate only after exact CSV replay and hand-in archive verification.

## Result

Frozen LER changed from 0.03392625 to 0.03392500, a gain of one net bit error over 800,000 predictions. The paired 95% interval [−0.00001444, 0.00001694] crossed zero. Median local inference time was 11.85 seconds and peak process RSS was 2.28 GiB. All 39 tests passed. This candidate failed promotion. [Frozen report](../artifacts/backward/validation.json)
