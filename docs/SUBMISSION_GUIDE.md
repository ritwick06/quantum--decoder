# Submission artifact and manual upload guide

**Current verified save:** [temporal_suffix_v1.csv](../artifacts/submissions/temporal_suffix_v1.csv) is already scored on Kaggle. Check [BEST_SAVE.md](../BEST_SAVE.md) for the current file before any further upload. The [submission ledger](../artifacts/submissions/ledger.json) records scores and selection evidence.

The instructions below are the historical zero-baseline verification record. Keep that baseline local.

## Correct file and target

If the user later explicitly chooses to submit this baseline, the file is:

`/home/ritwick06/q_decoder/artifacts/submissions/zero_baseline_v1.csv`

Target: the **Submit Prediction** interface for [Quantum Error Correction: Decoder Hackathon](https://www.kaggle.com/competitions/quantum-error-correction-decoder).

Use one CSV. Do not split it into Public and Private files, upload the training prediction artifact, or upload the manifest JSON. The Public/Private split is applied by Kaggle according to the supplied rules.

Expected SHA-256:

`f4ff37522e28e9c0330f98dce64d32cdcc942060619efb8fffa83f5a4fd26eeb`

The CSV contains 200,000 data rows and exactly `id,L0,L1,L2,L3`. Its all-zero contents match the sample and local training-prediction artifact because IDs are reused. Use the file path and accompanying provenance manifest together; do not identify the source dataset from IDs or CSV hash alone.

## Before any upload

1. Obtain explicit user approval for the particular candidate. This document is not approval to upload.
2. Check the authenticated Rules, Data, current announcements, and Submit Prediction interface. Their current contents were inaccessible during Stage 02. Resolve any change from the supplied one-file rules before proceeding.
3. Check the team's remaining allowance: the limit is **ten total submissions**, not ten per day. Record actual usage in `artifacts/submissions/ledger.json`; unknown usage must not be treated as zero.
4. Verify the artifact against a fresh decoder run:

```bash
.venv/bin/python tools/prepare_submission.py \
  --verify artifacts/submissions/zero_baseline_v1.csv \
  --batch-size 4096
```

## Manual upload, only after approval

1. Open the competition's Submit Prediction interface while signed into the team's eligible account.
2. Choose the exact CSV path above. Use a truthful description such as `zero_baseline_v1 — organizer zero-output interface baseline` and follow any applicable disclosure requirements.
3. Submit only after checking the displayed file and quota. The exact current screen controls have not been independently verified here.
4. Wait for Kaggle to finish processing. Record the submission ID, timestamp, acceptance/error status, Public score if provided, and actual updated quota in the ledger.
5. Preserve the decoder checkpoint, CSV, requirements, source provenance, and manifest together. Do not alter individual predictions in response to the score.

Kaggle's acceptance and score cannot be inferred from local CSV validation. No Public or Private score has been measured for this artifact. Up to two final submissions may be selected under the supplied rules; selection is a separate later action.
