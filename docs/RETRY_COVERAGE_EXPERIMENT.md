# Wider coverage for transition retries

## Preregistered experiment

On the 160,251 labelled development shots, the incumbent's 0.4 retry threshold reduced errors by 68 bits against the first pass. A threshold of 0.85 reduced errors by 82 bits, matching the gain from applying the expanded search everywhere on that development sample. The estimated fraction of shots receiving a retry rose from 8.2% to 15.4%. Thresholds between 0.4 and 0.85 gave intermediate gains. Both bases improved on development. No frozen or Kaggle labels informed this choice.

Change the retry threshold from 0.4 to 0.85. Keep all checkpoint tables, the expanded fanout, the sign-disagreement rule, and bitwise replacement unchanged. The motivation is that strongly weighted but incomplete estimates can conceal explanations with a different logical sign.

Compare once against `temporal_expanded_v1` on the original frozen holdout. Require a positive paired 95% lower bound, at most 12 seconds local median for 200,000 shots across five runs, and 2 GiB peak inference RSS. If the frozen comparison passes, confirm on 250,000 new Stim shots per basis with seeds `2026111401` and `2026111402`, again requiring a positive paired 95% lower bound. Require exact CSV replay and a matching hand-in archive before updating the best save.

## Result

Frozen LER was 0.03392125, four fewer net bit errors than the expanded decoder. The paired 95% interval [−0.00001636, 0.00002636] crossed zero. The 15.82-second local median exceeded the internal 12-second target. Applying the supplied scoring formula to that local timing gives 0.03410390, compared with 0.03392625 for the incumbent. All 41 tests passed. This candidate failed promotion. [Frozen report](../artifacts/coverage/validation.json)
