# Best verified save

Updated 13 November 2026, IST.

**Best verified local save** is [temporal_suffix_v1.csv](artifacts/submissions/temporal_suffix_v1.csv). The latest screenshot shows it already uploaded and scored, so keep this file for reproducibility.

- CSV SHA-256 is `9eec94fc6a73357efc8ebaa460a26985c87a6b35166ca30774de9d7856b29e3e`.
- The [matching code hand-in archive](artifacts/hand_in/temporal_suffix_v1.zip) has SHA-256 `8d58a7c9cd02901052fd12b001dede48ea8a069b110cd0173cb7d5ec9602f6e2`. Its extracted decoder reproduced all 200,000 rows. The ZIP is generated locally by `tools/package_temporal.py`.
- Frozen LER is **0.03405625**, with 9.75 seconds median local runtime for 200,000 shots. Independent simulated LER is **0.03419950**. These are local estimates. Kaggle computes the official score after upload.
- The [previous verified CSV](artifacts/submissions/temporal_bidir_v1.csv) and [its matching hand-in archive](artifacts/hand_in/temporal_bidir_v1.zip) are available. Both versions show public score **0.03415** and the team is **#8** in the latest screenshot. The screenshot shows `temporal_bidir_v1` selected for final scoring.

If work is interrupted, use the file named above. An experiment becomes the best save only after it passes validation and reproduces the CSV byte for byte. [Validation and replay evidence](artifacts/suffix/verification.json)
