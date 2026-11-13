# Stage 02: Produce validated first submission artifacts

Follow `AGENTS.md`, `docs/RULES_AUDIT.md`, and the confirmed current Kaggle instructions. Only run after the baseline passes.

1. Recheck the current Kaggle sample_submission.csv, Rules, Data and Submit Prediction UI. The organizer README contains a typo warning and inconsistent one-file/two-file descriptions, so do not assume either scheme is current.
2. Reuse the baseline `decode(circuit, detector_events)` and organizer `save_submission`/`run_local.py` wherever applicable. Generate the prediction file(s) needed for available public/private unlabelled data. Keep X and Z circuits matched to their basis.
3. Validate exact header and order `id,L0,L1,L2,L3`, number and order of IDs, binary values, no blanks/NaNs, no duplicate IDs, no extra index column, UTF-8/CSV readability and determinism. Compare against current Kaggle template when possible.
4. Save each artifact with a descriptive name and candidate version. Keep an untampered baseline checkpoint and a SHA256 of final artifact(s).
5. Explain WHICH file must be uploaded to WHICH Kaggle submission target; if the active UI requires a different format, follow the verified UI and explain the discrepancy. Do not guess, merge datasets blindly, or leak labels.
6. Prepare a short step-by-step manual upload instruction. If you have connected browser permission, still ask me before uploading or submitting.

Acceptance: at least one validated, reproducible CSV artifact ready for manual submission; baseline kept intact. STOP. Do not automatically launch Stage 03.
