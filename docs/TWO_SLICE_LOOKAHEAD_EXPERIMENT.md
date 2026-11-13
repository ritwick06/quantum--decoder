# Two-slice likelihood for earlier beam pruning

## Preregistered experiment

Using both terminal observations together gave a repeatable LER gain in `temporal_suffix_v1`. Earlier beam decisions still use only the next observation, even when much of the next slice's noise also appears one slice later.

For each direction and each of the first three beam decisions, compute the joint marginal of the next two detector parities under all faults assigned to those two slices. Project away any later detectors. Keep each of the 48 transition candidates and rank the resulting boundary states by accumulated mass times this joint likelihood. Leave the terminal contraction, beam width, transition tables and bidirectional decision rule unchanged.

The second parity indexes an entire contiguous row of the joint table for a shot. Retain the 128 most probable second-slice parities, ranked by their projected circuit probabilities, in a float32 table. For the remaining second-slice parities use the existing one-slice marginal. This limits the four directions' additional persistent storage to roughly 100 MiB. The two-observation lookup serves as a pruning heuristic. Some projected fault mass also affects a later observation.

Compare against `temporal_suffix_v1` on the original frozen holdout. Require a positive paired 95% lower bound, a median of five local runs at or below 12 seconds for 200,000 shots, peak inference RSS at or below 2 GiB, and at most 20 minutes and 5 GiB to build. Confirm any passing gain on 250,000 new shots per basis with seeds `2026111331` and `2026111332`. Require positive paired 95% lower bound and exact CSV replay before flagging a save. The model uses circuit probabilities only.

## Result

Frozen LER was 0.03406250 against 0.03405625 for `temporal_suffix_v1`, a regression of five bit errors. The local median was 10.56 seconds over five runs and all 36 tests passed. The build took 182 seconds with peak RSS of 2,118 MiB. The new candidate failed the accuracy gate. [Frozen validation report](../artifacts/lookahead/validation.json)
