# Experiment log

Entries are append-only. No Kaggle submission was made during Stage 01. The team's pre-existing submission usage remains unverified.

| Timestamp (IST) | Candidate | Training performed | Evaluation data / seed | X LER | Z LER | Combined LER | Median runtime, 200k shots | Kaggle public score |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 20:25:02 | Organizer zero baseline | None | Supplied train.csv, SHA-256 in report | 0.291255 | 0.2850375 | 0.28814625 | 0.004537578 s | Not submitted |
| 2026-11-13 20:25:02 | Organizer zero baseline | None | Frozen independent validation; X seed 2026111301, Z seed 2026111302 | 0.29236 | 0.2834425 | 0.28790125 | 0.003472770 s | Not submitted |

Runtime is the median of five local runs with one thread and both basis calls, including the organizer's output-check overhead. Imports, CSV I/O, data generation, and scoring are excluded. These are measurements of the trivial zero baseline on local hardware, not predictions of finalist runtime on organizer hardware.

The unchanged organizer runner also completed five runs and reported the same LER to its displayed precision. Its runtime display rounds to two decimal places; the table uses separate full-precision measurements with the same timing boundary.

## Reproduction and provenance

- Python 3.13.18; NumPy 2.2.6; Stim 1.16.0. No other packages are installed in the project runtime environment.
- Organizer revision: `739d627b20fdb4899ab7487723d2e4fccec27033`.
- Baseline: unmodified `organizer/submission_template/decoder.py`. This is an organizer interface baseline, not an original decoder contribution.
- Command: `.venv/bin/python tools/run_stage01.py`, from the project root. This invokes the official runner, freezes validation, and records full-precision measurements.
- Checks: `.venv/bin/python -m unittest discover -s tests -v`; 10 tests passed in 1.478 seconds.
- Full run values, environment, circuit metadata, and hashes: `artifacts/stage01/report.json`.
- Organizer output and test output: `artifacts/stage01/organizer_runner.log`, `artifacts/stage01/tests.log`.
- Validation seeds, array hashes, shapes, and circuit hashes: `data/validation/manifest.json`.
- No fitting or hyperparameter selection occurred. Candidate C remains a proposal for later design approval.

## Stage 03B parity v1

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `parity_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.27360250 | 0.26159500 | 0.26759875 | 1.011145 s | Pending |

The approved single configuration trained on the development partition and passed every acceptance criterion. The paired combined LER gain is 0.02030250 with a 95% interval of [0.01936640, 0.02123860]. See [the Stage 03B report](STAGE03B_REPORT.md) for the checkpoint and submission evidence.

On 13 November 2026, the user provided a Kaggle screenshot showing `parity_v1.csv` complete with a public LER of **0.27096**. The visible description is `submission 1`. The numeric submission ID and remaining team quota are unverified. [Submission ledger](../artifacts/submissions/ledger.json)

Two public leaderboard screenshots show PHANTOMS OF QUANTUM at rank 23 with score 0.27096, rank 22 at 0.14312, and the top five at 0.03361 to 0.03397. Rank 22's score is 47.2% lower than ours. The screenshots describe the public score as using approximately 30% of test data, with the remaining 70% reserved for final scoring.

### Development diagnostics after the first public result

Replayed the frozen parity checkpoint on the labelled development partition with the original split. Of 170,975 bit errors, 124,795 (73.0%) came from shots with at least seven active detectors. These are training-partition diagnostics. The frozen validation scores above remain the comparison baseline.

| Active detectors | X parity LER | Z parity LER | X coverage by up to two faults | Z coverage by up to two faults |
|---|---:|---:|---:|---:|
| 0 | 0.00000 | 0.00000 | 100.00% | 100.00% |
| 1 to 3 | 0.11149 | 0.09681 | 99.09% | 99.25% |
| 4 to 6 | 0.22050 | 0.21118 | 69.12% | 69.74% |
| 7 to 9 | 0.31292 | 0.30706 | 20.63% | 21.66% |
| 10 or more | 0.40205 | 0.39825 | 1.56% | 1.71% |

Coverage counts a shot if its detector syndrome equals the XOR signature of zero, one, or two distinct mechanisms from its pinned circuit. Overall coverage is 34,117 of 80,130 X shots and 35,345 of 80,121 Z shots. A match alone establishes syndrome consistency. Several fault sets can have the same detector syndrome and different logical outcomes. The proposed next experiment should measure search coverage and error by detector count so that any gain on sparse shots can be weighed against the bulk of the errors on denser shots.

## Stage 04 temporal likelihood model

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `temporal_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03580500 | 0.03419750 | 0.03500125 | 4.11050 s, four workers | Pending |

The fixed configuration passed every promotion check against `parity_v1`. The paired LER gain was 0.23259750, with a 95% interval of [0.23133240, 0.23386260]. No additional labelled fitting was used. The CSV was regenerated byte-for-byte by a fresh process. [Stage 04 experiment](STAGE04_TEMPORAL_EXPERIMENT.md)
