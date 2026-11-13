# Stage 03B parity decoder and submission

Candidate C passed the approved acceptance criteria. The recommended file for the team's manual Kaggle submission is `artifacts/submissions/parity_v1.csv`. It contains 200,000 rows with the exact sample-submission ID order and the columns `id,L0,L1,L2,L3`. Regeneration in a fresh process with batches of 3,072 produced an identical CSV. The SHA-256 is `e964a06cd142cae8794b77b9d49e67384bcfb91600215dfd3d363e2c368d3b32`.

## Frozen validation

| Decoder | X LER | Z LER | Combined LER | Local median for 200,000 shots |
|---|---:|---:|---:|---:|
| Organizer zero baseline | 0.29236000 | 0.28344250 | 0.28790125 | 0.003473 s |
| Candidate C, `parity_v1` | 0.27360250 | 0.26159500 | 0.26759875 | 1.011145 s |

Combined LER fell by 0.02030250. The paired 95% interval for the gain is [0.01936640, 0.02123860]. Both bases improved. All eight fits converged in four Newton steps each. Training took 15.135 seconds with 246.0 MiB peak RSS. Local inference peaked at 140.5 MiB. Timing uses one local CPU thread, includes both decode calls and output checks, and excludes imports, model loading and scoring. Organizer timing and the Kaggle score are pending.

The approved fit used 80,130 X and 80,121 Z development rows, 64 circuit-ranked masks per output, and a fixed-point export. Validation used 100,000 frozen shots per basis with the original hashes. All 26 tests passed, including the seven parity decoder tests. Fresh-process replay reproduced both frozen prediction hashes. The submission was generated from the audited unlabelled test file using the same checkpoint.

## Reproduction and handoff

- `team_decoder/decoder.py` implements the approved detector features and deterministic integer readout. `tools/train_parity.py` fits and exports the eight models. It requires the recorded development split, circuits and pinned organizer metadata helper.
- `artifacts/checkpoints/parity_v1/` contains the decoder, model arrays, model metadata, requirements and source/input hashes. The checkpoint uses Python 3.13, NumPy 2.2.6 and Stim 1.16.0. The source and training workflow were developed with AI assistance. The organizer and Stim supply circuit metadata. The project's conceptual sources are listed in [the approved design](DESIGN_DECISIONS.md).
- `artifacts/stage03b/validation.json` records all five timings, accuracy gates and prediction hashes. `artifacts/stage03b/submission_replay.json` records the fresh CSV replay. `artifacts/submissions/parity_v1.csv.manifest.json` ties the CSV to the checkpoint and original inputs.

Commands from the project root:

```bash
timeout --kill-after=5s 1200s .venv/bin/python tools/train_parity.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_parity.py --replay
.venv/bin/python tools/prepare_parity_submission.py \
  --verify artifacts/submissions/parity_v1.csv --batch-size 3072
```

The training command is for a clean run directory. It refuses to replace the frozen checkpoint. The original scoring command was `.venv/bin/python tools/evaluate_parity.py`. The `--replay` command compares predictions with the saved report. Generation uses `tools/prepare_parity_submission.py --output` with a new path.

Upload `parity_v1.csv` through the competition's Submit Prediction page if the signed-in page shows submissions are open and your team has a slot. The supplied deadline is **14 November 2026, 4:30 AM IST**. Live rules, submission status, team quota and any additional disclosure terms could not be verified from this workspace. Record the accepted submission ID and score after Kaggle processes it. [Competition](https://www.kaggle.com/competitions/quantum-error-correction-decoder), [Rules audit](RULES_AUDIT.md)
