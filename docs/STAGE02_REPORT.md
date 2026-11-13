# Stage 02: local submission pipeline

**Completed:** 13 November 2026, IST. No upload or Kaggle submission was performed. Stage 03A has not started.

## Artifact and checkpoint

The single locally validated artifact is:

`artifacts/submissions/zero_baseline_v1.csv`

- Exactly 200,000 data rows, covering 100,000 X and 100,000 Z shots.
- Header exactly `id,L0,L1,L2,L3`.
- IDs match the local sample submission and test CSV in sequence, with no duplicates or omissions.
- All four output columns contain literal `0` or `1`; there are no missing values, NaNs, extra index columns, or invalid UTF-8.
- SHA-256: `f4ff37522e28e9c0330f98dce64d32cdcc942060619efb8fffa83f5a4fd26eeb`.

The file was produced by running the unchanged organizer zero decoder on the unlabelled test data. It is an interface baseline, not an original decoder contribution. The source checkpoint, exact requirements, MIT licence, and provenance are preserved in `artifacts/checkpoints/zero_baseline_v1/`.

The output happens to be byte-identical to both the all-zero sample submission and Stage 01's all-zero training prediction CSV. This is expected because those files share IDs. **A CSV hash alone cannot distinguish their provenance.** The new manifest records the test-data hash, both circuit hashes, the checkpoint hashes, and the generation command's parameters. No training labels were loaded or joined to test IDs by the submission pipeline.

## Checks and measured results

**19 tests passed** in 1.284 seconds: the ten Stage 01 checks plus nine submission-pipeline checks. The new tests cover malformed headers, extra/missing/reordered/duplicate IDs, nonbinary and blank values, invalid encoding, wrong column counts, prediction mismatches, labelled-input rejection, mixed-basis routing, chunked inference, and protection against overwriting files.

Additional full-data checks:

- Repeated inference produced identical predictions.
- A fresh process recomputed every prediction using batches of 4,096 and verified the saved CSV and its provenance.
- Another fresh process regenerated the CSV in a temporary directory using batches of 4,096; its bytes and SHA-256 matched the saved artifact exactly. Only that temporary reproduction was removed.
- All 75 files recorded before Stage 02 remain unchanged, including the original data, instruction pack, organizer source, Stage 01 code, baseline artifacts, and frozen validation.

The initial prediction-pipeline pass took approximately 0.314 seconds locally, including input/output validation and basis routing. This is not the competition timing procedure or a new decoder benchmark.

Accuracy was not remeasured in Stage 02. The checkpoint's existing independent-validation LER is **0.28790125** from Stage 01. Kaggle Public/Private scores and test-set LER are unknown.

## New files

- `submission_pipeline.py`: unlabelled test loading, checked prediction routing, strict CSV validation, and publication of validated bytes without overwriting an existing destination. Uses the organizer's `save_submission` helper.
- `tools/prepare_submission.py`: pinned zero-baseline checkpoint creation, generation, repeatability checking, and fresh-prediction verification. Has no upload capability.
- `tests/test_submission_pipeline.py`: nine focused unit tests.
- `artifacts/checkpoints/zero_baseline_v1/`: decoder, requirements, licence, and provenance.
- `artifacts/submissions/zero_baseline_v1.csv` and its `.manifest.json`: the artifact and evidence linking it to the test data and checkpoint.
- `artifacts/submissions/ledger.json`: local candidate status; upload approval is false, scores and team quota usage remain unknown.
- `artifacts/stage02/`: protected-file manifest, full test log, fresh-process verification log, regeneration evidence, and final verification summary.
- `docs/STAGE02_REPORT.md` and `docs/SUBMISSION_GUIDE.md`.

## Reproduction

From `/home/ritwick06/q_decoder`, verify the saved artifact without replacing it:

```bash
.venv/bin/python tools/prepare_submission.py \
  --verify artifacts/submissions/zero_baseline_v1.csv \
  --batch-size 4096
.venv/bin/python -m unittest discover -s tests -v
```

To generate a separate local copy, choose an unused path:

```bash
.venv/bin/python tools/prepare_submission.py \
  --output artifacts/submissions/zero_baseline_v1_reproduction.csv
```

The tool refuses to replace an existing artifact or manifest. It rejects inputs whose hashes differ from the audited competition files and verifies that the runtime and checkpoint match the baseline. The pipeline is CPU-only and requires no new dependencies.

## Rules evidence and stage boundary

The live Kaggle Rules and Data URLs were rechecked and remained inaccessible. The authenticated Submit Prediction interface and current organizer announcements were not inspected. File count, upload route, runtime rules, and quota therefore remain grounded in the user's supplied Kaggle text; schema and IDs were independently checked against the unchanged local files. No fresh authenticated rule confirmation is claimed.

The ledger records zero uploads by this project, not a claim that the team has all ten submissions remaining. Check the actual quota before any future upload. The default remains to preserve this zero baseline locally rather than spend a submission on it.

**Next prompt:** "Start Stage 03A only: refine the independent decoder design using verified circuit metadata and the measured baseline. Do not implement or train it; stop for design approval."
