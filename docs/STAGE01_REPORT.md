# Stage 01: environment and zero baseline

**Completed:** 13 November 2026, IST. Stage 02 has not started.

The organizer's unchanged zero decoder runs end-to-end on both bases. Independent validation data is frozen, the input/output contract checks pass, and the source datasets and instruction pack are unchanged.

## Measured results

Each dataset contains 100,000 X shots and 100,000 Z shots. LER counts erroneous logical bits, not shots with any error.

| Dataset | X LER | Z LER | Combined LER | Median of five runs |
|---|---:|---:|---:|---:|
| Supplied training CSV | 0.291255 | 0.2850375 | 0.28814625 | 4.538 ms |
| Frozen independent validation | 0.29236 | 0.2834425 | 0.28790125 | 3.473 ms |

Both sets of predictions were identical across five runs. Fresh-process validation predictions also matched, and regenerating validation on this environment with the recorded seeds reproduced the frozen arrays exactly.

Timing uses the local CPU with thread limits set to one. It includes both decode calls and the organizer's output checks, but excludes imports, data loading, generation, and LER calculation. The zero decoder does almost no inference work; these timings do not estimate the cost of Candidate C or organizer-machine performance.

The official runner's two-decimal runtime display rounds the median to `0.00 s`. The table uses the independently recorded full-precision timer with the same boundary. Raw values are preserved in `artifacts/stage01/report.json`.

## Environment and source

- Project-local CPython **3.13.18**, provisioned with project-local uv **0.13.8**. No system packages or shell startup settings were changed.
- Runtime environment contains only **NumPy 2.2.6** and **Stim 1.16.0**.
- Linux x86-64; eight logical CPUs visible locally. All four recorded threading environment variables are set to one for the benchmark.
- Pinned organizer revision: `739d627b20fdb4899ab7487723d2e4fccec27033`.
- Downloaded a selected source snapshot, not a full clone: helpers, their required circuit-construction resources, submission template, README, metadata, licence, and starter notebook. No competition CSVs or tutorial decoder implementations were downloaded.
- Every downloaded source file was verified against its Git blob hash and recorded with SHA-256 in `organizer/SOURCE_MANIFEST.json`.
- The official notebook is retained for reference and was not executed: its old folder assumptions and submission cells are not the Stage 01 workflow. No notebook dependencies were installed.

Both circuits have 32 qubits, 70 detectors and four logical observables. Detector counts over times 0 through 5 are `7,14,14,14,14,7`.

| Basis | H shape | L shape | Mechanisms | Probability range |
|---|---|---|---:|---|
| X | (70, 2470) | (4, 2470) | 2470 | approximately 0.000133458 to 0.023238547 |
| Z | (70, 2465) | (4, 2465) | 2465 | approximately 0.000133458 to 0.023238547 |

Combined detector/logical mechanism signatures are unique. Matrix dimensions, binary contents, finite probabilities, detector coordinates, and temporal ordering were checked from the local circuits.

## Data separation

- Existing ZIPs and all five extracted competition files remain unchanged; no extraction was needed.
- `data/adapters/train/` contains relative symlinks to the original training CSV and circuits. The runner can use its expected layout without duplicating or renaming originals.
- `data/splits/train_reserved.npy` stores the Stage 00 selection mask in original training-row order, tied to the training-file hash. Development counts are 80,130 X and 80,121 Z; reserved counts are 19,870 X and 19,879 Z.
- `data/validation/` contains four Boolean `.npy` arrays and a manifest. Sampling used one 100,000-shot call per basis, with `separate_observables=True`, X seed `2026111301`, and Z seed `2026111302`.
- Validation is not training data. Its loader verifies file hashes and exposes read-only arrays. Generation refuses to overwrite an existing validation directory.
- Test CSV schema was checked but test shots were not passed to the decoder. The labelled-data loader rejects the unlabelled test schema.
- `artifacts/baseline/train_predictions.csv` is the organizer runner's local training-set output. IDs overlap the competition test IDs; this artifact is explicitly **not a Kaggle submission**.

## Commands and checks

Primary commands, from the project root:

```bash
.venv/bin/python tools/run_stage01.py
.venv/bin/python -m unittest discover -s tests -v
```

The stage command invokes the unchanged organizer runner with explicit training-adapter, decoder, output, five-run, and one-thread arguments. It refuses to overwrite an accepted report or prediction file. See `REPRODUCIBILITY.md` for setup and repeatable verification.

**10 tests passed** in 1.478 seconds:

- Baseline shape, integer dtype, binary range, repeatability, and non-mutation for both bases and batches of 0, 1, 17, and 100,000 shots.
- Basis routing and row restoration using distinguishable test outputs; shuffled and chunked input behavior.
- Rejection of invalid widths, nonbinary values, invalid bases, and malformed decoder results in the validation wrapper.
- Bitwise metric fixture, duplicate-ID rejection, and labelled/test data separation.
- Original-file and organizer-source hash preservation.
- Frozen-array integrity, reproducible seeded sampling, and fresh-process prediction equality.

The organizer zero function itself remains unchanged and trusts its documented input contract. Additional rejection checks belong to the team validation wrapper, not an altered organizer decoder.

## Files added

- Environment: `.python-version`, `requirements.txt`, `.gitignore`, `.venv/`, and `.tools/`.
- Official source: `organizer/`, enumerated in `organizer/SOURCE_MANIFEST.json`.
- Team tooling: `stage01_support.py`, `tools/fetch_organizer.py`, `tools/run_stage01.py`, and `tests/test_stage01.py`.
- Data adapters and frozen data: `data/adapters/train/`, `data/splits/train_reserved.npy`, and `data/validation/`.
- Evidence: `artifacts/stage01/` and the local training predictions under `artifacts/baseline/`.
- Documentation: this report, `EXPERIMENT_LOG.md`, and `REPRODUCIBILITY.md`.

The 22 original dataset, archive, and instruction-pack files were hash-checked before and after the work. Existing Stage 00 documents remain unchanged.

## Provenance and remaining uncertainties

The zero baseline and helper code retain their organizer attribution and MIT licence. Stim is Apache-2.0; NumPy is BSD-3-Clause with its bundled-library notices retained in the installation. uv is MIT OR Apache-2.0 and is a setup tool outside the runtime environment. Python's distribution retains its own licence notices.

Team validation tooling was developed with AI assistance in this chat. No independently designed decoder or trained assets were implemented. No prohibited decoder dependency was installed or imported.

There is no technical blocker to Stage 02. New authenticated organizer announcements, additional AI-disclosure obligations, and the team's remaining submission quota remain unverified, as recorded in the rules audit. No new rule confirmations are claimed here.

**Next prompt:** "Start Stage 02 only: build and validate the single-file submission pipeline locally. Do not upload or submit to Kaggle."
