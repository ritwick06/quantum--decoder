# Alternate assignment of single-slice faults

## Preregistered experiment

The expanded-transition decoder gives both directions a larger set of candidate paths, but in each direction every fault starts at its first affected measurement. Faults that affect only one slice can enter through either side of that slice. Moving them across its boundary preserves the full circuit distribution while changing which partial paths and local fault combinations survive the search limits.

Keep the incumbent's initial pass and its 96-transition retry. For the same retry shots, run another forward and reverse pair, each with 48 transitions and 32 live states. Assign single-slice faults at slices 1 through 4 to the preceding transition in the alternate pair. Leave the first and last measurement's single-slice faults in place. The exact terminal distribution includes every fault assigned to steps 4 or 5. Sum the two original and two alternate signed probability estimates with equal weight for each direction, then apply the same rule to replace only the retry bits. A fault is assigned exactly once within each direction. The combined result is a mixture of truncated estimates of the same posterior.

Compare to `temporal_expanded_v1` on the frozen data. Require a positive paired 95% lower bound, a local five-run median at or below 12 seconds on 200,000 shots, peak inference RSS at or below 2 GiB, and a build within 20 minutes and 5 GiB. Release the incumbent model after scoring its reference predictions. If the frozen comparison passes, confirm on 250,000 fresh Stim shots per basis with seeds `2026111391` and `2026111392` with a positive paired 95% lower bound. Require exact CSV replay and a matching hand-in archive before promotion.

## Result

Frozen LER was 0.03391875, six fewer net bit errors than the expanded decoder. X incurred 28 more errors and Z corrected 34. The paired interval [−0.00009713, 0.00011213] crossed zero and the local median was 14.16 seconds. All 41 tests passed. The experiment failed promotion. [Frozen report](../artifacts/shifted/validation.json)
