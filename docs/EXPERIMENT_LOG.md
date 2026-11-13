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

## Stage 04 bidirectional temporal likelihood model

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `temporal_bidir_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03469250 | 0.03351000 | 0.03410125 | 8.21826 s, four workers | Pending |

The fixed two-pass configuration passed every promotion check against `temporal_v1`. The paired LER gain was 0.000900, with a 95% interval of [0.000727, 0.001073]. Both bases improved. No additional labelled fitting was used. Fresh-process replay reproduced the frozen predictions and the exact submission bytes. [Bidirectional experiment](BIDIRECTIONAL_TEMPORAL_EXPERIMENT.md)

A later user-provided leaderboard screenshot shows PHANTOMS OF QUANTUM at rank 8 with a public score of 0.03415 and two entries. The leading score in that snapshot was 0.03361. The filename for the team's second submission is unverified. [Submission ledger](../artifacts/submissions/ledger.json)

## Stage 04 midpoint join

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `temporal_midpoint_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03652500 | 0.03509750 | 0.03581125 | 4.26973 s, four workers | No submission |

Both bases regressed relative to `temporal_bidir_v1`. The midpoint join failed promotion. This is the first consecutive unsuccessful experiment in the active goal. [Midpoint experiment](MIDPOINT_TEMPORAL_EXPERIMENT.md)

## Stage 04 adaptive beam width

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `temporal_adaptive_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03471250 | 0.03350000 | 0.03410625 | 12.06620 s, four workers | No submission |

The adaptive decoder made four more bit errors than `temporal_bidir_v1` and exceeded the 12-second budget. This was the second consecutive unsuccessful experiment. The upload recommendation remains `temporal_bidir_v1.csv`. [Adaptive beam experiment](ADAPTIVE_BEAM_EXPERIMENT.md)

## Stage 04 full local fault distributions

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---:|---|
| 2026-11-13 | `temporal_spectral_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03471000 | 0.03348250 | 0.03409625 | 11.30846 s, four workers | No submission |

The full local distributions retained more probability in every table. The improvement of four bit errors was inconclusive. [Full local distribution experiment](SPECTRAL_TEMPORAL_EXPERIMENT.md)

## Stage 04 exact final two measurement slices

| Date (IST) | Candidate | Evaluation data | X LER | Z LER | Combined LER | Local median, 200k shots | Kaggle score |
|---|---|---|---:|---:|---:|---|
| 2026-11-13 | `temporal_suffix_v1` | Original frozen validation, X seed 2026111301 and Z seed 2026111302 | 0.03466750 | 0.03344500 | 0.03405625 | 9.75228 s, four workers | Pending |

Frozen paired LER improved by 0.000045 with a 95% interval of [0.00000314, 0.00008686]. An independent sample of 500,000 Stim shots confirmed the improvement, with a paired gain of 0.000030 and 95% interval of [0.00000381, 0.00005619]. [Exact final slices experiment](EXACT_SUFFIX_EXPERIMENT.md)

The user's submissions and leaderboard screenshots show `temporal_bidir_v1.csv` and `temporal_suffix_v1.csv` both complete at public LER **0.03415**. PHANTOMS OF QUANTUM remained #8 with three entries. The screenshot shows the bidirectional entry selected for final scoring and the final-slice entry unchecked. [Submission ledger](../artifacts/submissions/ledger.json)

## Stage 04 two-slice lookahead

The local frozen LER was 0.03406250 (X 0.03467500, Z 0.03345000), against 0.03405625 for `temporal_suffix_v1`. The five-run median was 10.56 seconds and all 36 tests passed. The candidate failed the accuracy gate. [Two-slice lookahead experiment](TWO_SLICE_LOOKAHEAD_EXPERIMENT.md)

## Stage 04 calibrated readout for ambiguous bits

The local frozen LER was 0.03401250 (X 0.03460750, Z 0.03341750), a point gain of 0.00004375 over `temporal_suffix_v1`. The paired 95% interval includes zero, so the candidate failed the accuracy gate. Its five-run local median was 10.97 seconds and all 37 tests passed. [Calibrated readout experiment](AMBIGUOUS_READOUT_EXPERIMENT.md)
