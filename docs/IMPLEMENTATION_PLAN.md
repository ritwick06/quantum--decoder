# Staged implementation plan

## Ownership, interfaces, and defaults

- Keep original archives, extracted data, and the supplied instruction pack unchanged.
- Keep a pinned organizer source snapshot separate from team-owned decoder, validation, and submission code.
- Store canonical project documentation in the project-root `docs` directory.
- Preserve every accepted candidate, its environment lock, asset hashes, measurements, and prediction hashes.
- Use CPU execution initially. Add compiled acceleration or GPU dependencies only after profiling justifies them.
- Public interface remains:

```python
decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray
```

Inputs contain `(shots, 70)` binary events. Outputs contain `(shots, 4)` binary values in the same shot order. Support empty input. Validate circuit identity when loading circuit-specific assets; reject mismatches clearly.

## Stages, deliverables, and acceptance gates

| Stage | Deliverables | Tests and acceptance |
|---|---|---|
| **00: audit and proposals** | These three documents; evidence status, inventory, architecture comparison | No extraction, installation, decoder implementation, training, or submissions. Stop for Stage 01 approval. |
| **01: environment and baseline** | Python 3.13 environment; pinned official helpers and zero baseline; data adapter; contract tests; frozen validation; measured baseline log | Both bases run end-to-end; binary shape/order tests pass; five-run determinism and timing recorded; no prohibited dependencies. Stop. |
| **02: submission pipeline** | Reproducible single-file writer and validator; local baseline artifact; submission ledger | Exact sample schema and IDs; 200,000 rows; no duplicates, missing values, extra columns, or nondeterminism. Preserve the baseline artifact locally; default is no zero-baseline upload. Stop. |
| **03A: design review** | Candidate C specification refined using actual metadata; fixed experiment settings, budgets, and acceptance criteria | Review numerical handling, feature selection, fallback/error behavior, provenance, and reproducibility. Explicit design approval required before Stage 03B. |
| **03B: independent decoder** | Only the approved decoder; training/asset-generation code; versioned assets; validation report | Correctness and promotion criteria pass within declared resource limits. Preserve the prior candidate on failure. Stop. |
| **04: controlled improvement** | One preregistered change at a time; comparison against the accepted candidate | Same validation data and hardware; measured improvement; no per-row leaderboard tuning. Upload only an approved, validated complete candidate. |
| **05: runtime and selection** | Profiled inference; optimized accepted candidate; selection recommendation | Prediction identity preserved for performance-only changes; five-run timing, peak memory, and fresh-process determinism recorded. |
| **06: final reproduction** | Offline hand-in package, exact requirements, assets, generation code, notices, instructions, hashes | Fresh Python 3.13 environment reproduces every selected submission prediction. Verify selection, remaining quota, deadlines, and any accessible organizer updates. |

Stage numbers are preserved from the instruction pack. Stage 02 validates the submission machinery; scarce Kaggle quota is reserved for useful validated candidates unless the user explicitly chooses to submit the zero baseline.

## Exact next steps for Stage 01

1. Obtain **Python 3.13** in a project-local environment. Start with the organizer's NumPy 2.2.6 and Stim 1.16.0 pins; verify installation and imports. Do not silently substitute Python 3.14.
2. Obtain only the missing official source, template, metadata, and licence files in a separate organizer directory. Pin the revision. Do not redownload competition data or acquire prohibited tutorial implementations.
3. Create a separate `data/adapters/train` layout referencing the existing training CSV and circuits. Its CSV must be named `train.csv`. Recheck source hashes and preserve the originals.
4. Load both circuits with Stim. Verify 70 detectors, four observables, coordinates, and compatible `H`, `L`, and `p`; record actual mechanism counts. Do not infer these solely from notebook output.
5. Add standard-library unit tests for the decoder contract, basis routing, order preservation, metric calculation, and data separation.
6. Create the separate `artifacts/baseline` output directory, then run the unchanged organizer zero baseline with explicit paths. From the project root, the planned command is:

```bash
PYTHONPATH=organizer/src .venv/bin/python organizer/submission_template/run_local.py \
  --problem-set data/adapters/train \
  --decoder organizer/submission_template \
  --runs 5 \
  --threads 1 \
  --out artifacts/baseline/train_predictions.csv
```

7. Generate independent validation arrays of 100,000 shots per basis with seeds `2026111301` for X and `2026111302` for Z, as specified in `docs/DESIGN_DECISIONS.md`. Record their hashes and score the same baseline on them. This is baseline verification, not model training.
8. Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

9. Record X, Z, and combined LER; all five runtimes and their median; environment, thread count, seeds, hashes, and determinism. Stop and report Stage 01 results.

All paths in these commands are planned additions beneath `/home/ritwick06/q_decoder`; none currently exists merely because it is named here.

## Required test scenarios

- Empty, one-row, small, and full-basis batches.
- Both circuits; mixed-basis outer routing; shuffled rows restored to original order.
- Binary output, exact dimensions, unchanged input arrays, repeatability across fresh processes, and batch-size-independent predictions.
- Clear failures for malformed input or incompatible circuit assets.
- Tiny hand-checkable fixtures for bitwise LER and probability aggregation.
- Candidate-specific mathematical tests: parity moments against exhaustive enumeration of tiny synthetic mechanism models; no duplicated fault combinations; detector constraints enforced at the correct temporal boundary.
- CSV round-trip validation against the sample submission, including exact ID sequence.
- Offline inference with all required assets present; useful errors for missing or incompatible assets.
- Full prediction equality between packaged code and the selected submission.

## Submission discipline and rollback

- Check the actual remaining quota before uploading; maintain a ledger of candidate, artifact hash, submission ID, score, and quota usage.
- Reserve at least two remaining slots for final validated candidates whenever the available quota permits.
- Never edit individual prediction values. Any changed prediction must result from rerunning a versioned decoder.
- Use Public scores only as whole-candidate generalization feedback; final selection depends on Private performance.
- Retain the last accepted candidate. Failed experiments remain separate and cannot replace its assets or submission.
- Keep runtime constants fixed to the supplied rules. Organizer timing remains the authoritative runtime measurement.

## Risks that could prevent a valid submission

1. Missing a deadline or overlooking a newer organizer announcement.
2. Python 3.13 or dependency acquisition failing before setup is complete.
3. Accidentally installing prohibited tutorial dependencies.
4. Joining train labels to test rows through reused IDs.
5. Incorrect detector, basis, logical-column, or submission-row ordering.
6. Exceeding the runtime limit despite acceptable local accuracy.
7. Missing assets, environment-sensitive inference, or predictions that differ during organizer reproduction.
8. Assuming AI-policy silence settles all disclosure obligations.
9. Spending the limited quota before independent validation demonstrates value.

**Stage boundary:** Stage 00 ends with these documents. No Stage 01 work, decoder implementation, or submission is authorized by this plan alone.
