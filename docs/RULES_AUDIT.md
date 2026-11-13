# Stage 00 rules audit

**Audit date:** 13 November 2026, IST.

**Evidence provenance:** Competition-specific rules below are confirmed from the Kaggle text supplied by the user. They were not independently accessed through an authenticated browser. Local dataset findings were independently checked. The public organizer repository was inspected directly. Current authenticated announcements remain inaccessible.

## Rules reconciliation

| Requirement | Status | Adopted requirement and evidence |
|---|---|---|
| Independent development | Confirmed: supplied rules | Develop our own decoder. Do not copy, port, wrap, or imitate existing QEC decoder implementations or use others' pretrained decoders. |
| Libraries | Confirmed: supplied rules | General-purpose numerical, ML, graph, and optimization libraries are allowed. Stim may supply circuits, simulations, detector-error models, and generated labels. Prohibited decoder packages must not appear directly or transitively. |
| Prediction integrity | Confirmed: supplied rules | Predictions must come from the decoder. No hand-labelling, manual prediction changes, or per-shot leaderboard tuning. |
| AI assistance | Unverified beyond supplied text | The supplied competition-specific rules contain no explicit permission, prohibition, or disclosure requirement. Preserve truthful provenance; clarify additional obligations if necessary. Silence is not explicit approval. |
| Submission quota | Conflict resolved | **10 submissions total per team**, superseding the README's daily allowance. Remaining quota must be checked before any upload. |
| Final selection | Confirmed: supplied rules | Select up to two submissions; the better selected Private score counts. |
| Leaderboard split | Confirmed: supplied rules | Approximately 30% Public and 70% Private. These are scoring partitions, not instructions to create separate files. |
| Submission artifact | Conflict resolved | One CSV for the local `test.csv`; exact columns `id,L0,L1,L2,L3`, binary predictions, one row per test shot. Preserve sample-submission ID order. |
| Upload route | Confirmed: supplied rules | Kaggle's Submit Prediction interface. No upload is authorized during Stage 00. |
| Accuracy metric | Confirmed: supplied rules | Binary MAE, equivalent to the fraction of incorrectly predicted logical bits. Lower is better. |
| Runtime scoring | Conflict resolved | `Private LER × max(1, t / 15.4)^0.2`. Runtime above 300 seconds disqualifies a decoder from final ranking. Shorter runtime breaks ties. |
| Timing procedure | Confirmed: supplied rules | Sum of the X and Z decode calls, 100,000 shots each; median of five runs. Imports and import-time work occur outside timing. |
| Evaluation environment | Confirmed: supplied rules | Linux, Python 3.13, one B200 GPU, 48 AMD EPYC 9655 vCPUs, 288 GB RAM. `OMP_NUM_THREADS` and `NUMBA_NUM_THREADS` are 48. Inference is offline. |
| Finalist hand-in | Confirmed: supplied rules | Top ten supply `decoder.py`, exact dependency versions, required assets, and asset-generation code. Reproduced predictions must match the selected submission exactly. |
| Participation | Confirmed: supplied rules | University students aged at least 18, invite-only, at most four members per team, one account per person. Individual eligibility has not been audited. |
| Sharing | Confirmed: supplied rules | Private sharing between competing teams is prohibited. Public competition Discussion/Code sharing is allowed subject to the rules. |
| Dataset licence | Confirmed: supplied rules | MIT; retain applicable notices. |
| New announcements | Unverified | No conclusion about the existence or content of newer organizer announcements. |
| Remaining submission allowance | Unverified | Do not assume that all ten submissions remain available. |

Source for supplied competition requirements: [Kaggle Rules](https://www.kaggle.com/competitions/quantum-error-correction-decoder/rules).

A user-provided photo of the Kaggle Submissions rules on 13 November also shows bitwise MAE, approximately 30% public and 70% private scoring, at most ten submissions per team, and up to two final selections with the better Private score counting.

The inspected [organizer README](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026) contains conflicting one-file/two-file instructions, a daily submission allowance, and competing runtime parameters of `(15.4 seconds, 0.2)` and `(3 seconds, 0.1)`. Adopt the supplied Kaggle requirements above. The README's separate Public/Private folder descriptions do not describe the downloaded dataset's layout.

## Deadlines

These are the IST conversions supplied by the user.

| Milestone | Deadline |
|---|---|
| Competition closes | **14 November 2026, 4:30 AM IST** |
| Top-ten code hand-in | **14 November 2026, 11:30 AM IST** |
| Top-three announcement | 14 November 2026, 5:15 PM IST |
| Five-minute presentation video | 16 November 2026, 11:30 AM IST |

Deadline changes in inaccessible announcements remain unverified.

The local 12-second runtime and 2 GiB memory targets in recent experiments are project budgets. Under the supplied rules, the runtime score penalty starts above 15.4 seconds and disqualification applies above 300 seconds on the organizer machine.

## Local project and dataset evidence

At the read-only inspection, the workspace contained the instruction pack, its ZIP, the competition ZIP, and an existing extracted dataset directory. There was no usable Git checkout, decoder source, local dependency manifest, or local test suite. The supplied design document was an unstarted template. Stage 00 adds only the three project-root documents in `docs/`.

Competition archive:

- Filename: `quantum-error-correction-decoder.zip`
- Compressed size: **5,626,486 bytes**
- SHA-256: `d68c9d771eb806d5ee72df079c20f18f66b7ceaea6c8e81d0f3252abe78acae8`
- Five root-level members; no Public/Private directories or `args.json`.
- No absolute paths, parent-directory traversal, or symlink members detected.
- All five existing extracted files match their archive members byte-for-byte.

| Archive member | Uncompressed bytes | Inspection result |
|---|---:|---|
| `X_circuit.stim` | 11,279 | Circuit text present |
| `Z_circuit.stim` | 10,790 | Circuit text present |
| `train.csv` | 32,600,292 | 200,000 labelled rows; 100,000 per basis |
| `test.csv` | 31,000,280 | 200,000 unlabelled rows; 100,000 per basis |
| `sample_submission.csv` | 4,000,015 | 200,000 rows; correct five-column header |

Confirmed CSV checks:

- Detectors are ordered `D0` through `D69`; training labels are `L0` through `L3`.
- Inspected detector and label cells contain only `0` or `1`.
- IDs are unique within each file.
- Test and sample-submission IDs match in order.
- **Train and test reuse all 200,000 IDs.** Dataset identity must accompany IDs; never use ID equality to transfer labels.
- Matching IDs do not establish that the underlying shots are identical.

## Environment and integration findings

- Default Python: **3.14.10**. Python 3.12 is also installed; Python 3.13 was not found on PATH.
- NumPy 2.5.3 is installed in the default interpreter.
- Stim, SciPy, pandas, pytest, Numba, PyTorch, JAX, scikit-learn, and Kaggle were absent from that interpreter.
- Available disk space was approximately 364.5 GB during inspection.
- Shell-based network lookup failed; web tools could read public source files. Source acquisition and package installation may need permitted network access in Stage 01.

The official [Python-version file](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/.python-version) specifies 3.13. The [template requirements](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/submission_template/requirements.txt) pin NumPy 2.2.6 and Stim 1.16.0; these are the initial compatibility targets, pending installation checks.

The [runner](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/submission_template/run_local.py) expects `<folder>/<folder>.csv`, always writes predictions, and groups output by basis. Its actual default is `data/train`, although its help text describes `data/test_set`. Use explicit arguments and a separate data-layout adapter. Do not run it during Stage 00.

The [package metadata](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/pyproject.toml) includes prohibited decoder packages in optional tutorial dependencies. Do not install tutorial extras or execute decoder tutorials.

**Measurements:** Dataset integrity and descriptive statistics were measured. Decoder accuracy and runtime were not measured. No baseline was executed.
