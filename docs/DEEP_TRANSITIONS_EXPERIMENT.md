# Deeper search on ambiguous shots

## Preregistered experiment

The 96-transition search improved LER by 0.00013000 on the frozen sample and 0.00011750 independently. Backward reranking the same transitions changed frozen LER by only 0.00000125. The next controlled change increases search coverage on the same retry shots.

Use the incumbent's 48-transition initial pass and the same per-bit retry criterion. Increase the retry search to 192 outgoing transitions per row and 48 live states in both directions. Keep three-fault enumeration and exact final-slice likelihoods. Use the same decision and fallback rules. Compare once with `temporal_expanded_v1` on the frozen shots. Validate positive paired 95% lower bounds there and on 250,000 fresh shots per basis with seeds `2026111381` and `2026111382`. Keep the 12-second local median and 2 GiB inference gates, and require byte-identical CSV replay and a matching hand-in archive. Measure candidate memory after releasing the incumbent's mapped model tables.

## Result

Frozen LER was 0.03390625, a gain of 16 net bit errors against the expanded decoder. The paired interval [−0.00005559, 0.00009559] crossed zero and median local inference time was 14.45 seconds. Peak process RSS was 1.58 GiB. All 40 tests passed. The experiment failed promotion. [Frozen validation](../artifacts/deep/validation.json)
