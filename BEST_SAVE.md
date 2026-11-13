# Best verified save

Updated 14 November 2026, 01:56 IST.

**Best verified local save, ready for manual upload** is [temporal_expanded_v1.csv](artifacts/submissions/temporal_expanded_v1.csv). Use the description `temporal_expanded_v1`.

- CSV SHA-256 is `c887d99484a2d5720565be73d1b602ea44f3c0376abb5dea9748e3ee882f7bbd`.
- The [matching code hand-in archive](artifacts/hand_in/temporal_expanded_v1.zip) has SHA-256 `4ce670ff7654056a53075b24b0eccdac8cae0e76502cb8f872873b4c319dcf7d`. Its extracted decoder reproduced all 200,000 rows. The ZIP is generated locally by `tools/package_temporal.py`.
- Frozen LER is **0.03392625**, with 9.75 seconds median local runtime for 200,000 shots. Independent simulated LER is **0.03392250**. Kaggle computes the official score after upload.
- The last scored saves are [temporal_suffix_v1.csv](artifacts/submissions/temporal_suffix_v1.csv) with its [hand-in archive](artifacts/hand_in/temporal_suffix_v1.zip) and [temporal_bidir_v1.csv](artifacts/submissions/temporal_bidir_v1.csv) with its [hand-in archive](artifacts/hand_in/temporal_bidir_v1.zip). Both scored **0.03415** on Kaggle. The last selection screenshot shows `temporal_bidir_v1` selected.

If work is interrupted, use the file named above. An experiment becomes the best save only after it passes validation and reproduces the CSV byte for byte. [Validation and replay evidence](artifacts/expanded/verification.json)
