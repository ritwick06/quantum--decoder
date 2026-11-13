# Calibrated readout for ambiguous bits

## Preregistered experiment

In the labelled development split the two direction estimates disagreed on 2,343 X bits and 2,410 Z bits. A constant change in direction weighting did not consistently improve the development score. The circuit provides more information about a disagreement. Its position, the amount of retained mass in each direction, and the other logical moments can help interpret it.

Keep the accepted search and apply a small logistic readout to a logical bit only when the combined signed moment has magnitude at most 0.4 of the combined unsigned mass or the two direction signs disagree. Fit one model per basis and logical bit. Features are the normalized combined moment, forward and reverse moments, a clipped log ratio of retained masses, the legacy parity prediction, the three other combined logical moments, per-slice detector counts, and products of those counts with each direction's signed estimate. Use the same feature construction for fitting and inference. Retain the existing decoder decision when the fitted logit has magnitude at most 0.08. This abstention margin favors the incumbent near the learned boundary.

Train on 1,000,000 independent Stim shots per basis with seeds `2026111341` and `2026111342`, sampled in 100,000-shot calls. Use standardized features, logistic loss and fixed L2 regularization of 0.002. Fit with the existing Newton optimizer, at most 30 steps. Keep the frozen labels, independent confirmation and Kaggle predictions out of fitting. Record the sample and asset hashes. The fitting budget is 20 minutes and 5 GiB.

Compare to `temporal_suffix_v1` on the original frozen holdout with a positive paired 95% lower bound, a median of five runs at or below 12 seconds over 200,000 shots with four workers, and at most 2 GiB inference RSS. Confirm a passing gain on 250,000 independent shots per basis with seeds `2026111351` and `2026111352` and a positive paired 95% lower bound. Score before recommending an upload. Replay the CSV byte for byte before updating the best save.

## Frozen result

Frozen LER fell from 0.03405625 to 0.03401250, with 24 fewer X bit errors and 11 fewer Z bit errors. The paired 95% interval for the gain was [−0.00000189, 0.00008939], so the candidate failed its preregistered accuracy gate. The local median was 10.97 seconds. Training and packaging took 117.5 seconds with peak RSS of 623.3 MiB. All 37 tests passed. [Frozen report](../artifacts/readout/validation.json)

The independent comparison found 36 fewer bit errors across 500,000 shots, a gain of 0.00001800 with a paired 95% interval of [−0.00000902, 0.00004502]. The readout failed promotion. [Independent report](../artifacts/readout/confirmation.json)
