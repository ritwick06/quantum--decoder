# Quantum Error Correction Decoder Hackathon

PHANTOMS OF QUANTUM's circuit-based decoder predicts four logical bits for each X or Z detector trace. The latest model estimates the probability of each logical bit by processing the trace in both directions and combining the retained probability mass. It uses the parity model when the combined result gives little support for a bit.

| Candidate | Frozen local LER | Local time for 200,000 shots | Kaggle public LER |
|---|---:|---:|---:|
| `parity_v1` | 0.26759875 | 1.011 s, one thread | 0.27096 |
| `temporal_v1` | 0.03500125 | 4.110 s, four threads | Pending |
| `temporal_bidir_v1` | **0.03410125** | 8.218 s, four threads | Pending |

The recommended submission for manual upload is [temporal_bidir_v1.csv](artifacts/submissions/temporal_bidir_v1.csv). Its [manifest](artifacts/submissions/temporal_bidir_v1.csv.manifest.json) identifies the input files, checkpoint and validation report. The [bidirectional experiment](docs/BIDIRECTIONAL_TEMPORAL_EXPERIMENT.md) covers the algorithm, tests, limits and measurements. The [submission ledger](artifacts/submissions/ledger.json) records Kaggle outcomes.

## Reproduce the submission

Use Linux on an x86-64-v2 or compatible CPU with CPython 3.13. Set up the pinned NumPy and Stim dependencies and place the original Kaggle files in `quantum-error-correction-decoder/` with their original names. The file hashes are recorded in [the original-input manifest](artifacts/stage01/original_files.json). The decoder runs offline with the supplied model assets and compiled C++ kernel.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/prepare_parity_submission.py \
  --candidate temporal_bidir_v1 --verify artifacts/submissions/temporal_bidir_v1.csv --batch-size 3072
```

For the full test suite, restore the original input archives at the paths in the original-input manifest. The frozen validation arrays and development split mask are included with this project.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The team-authored decoder lives in [temporal_decoder](temporal_decoder/) and [team_decoder](team_decoder/). [build_temporal.py](tools/build_temporal.py) builds the temporal checkpoint from the circuits and the frozen parity checkpoint. Its `--bidirectional` option builds the latest model using the accepted forward checkpoint. [train_parity.py](tools/train_parity.py) creates the parity checkpoint from the labelled development split. Both generators expect a clean checkpoint destination. The temporal builder records the compiler version and command. The pinned organizer source and its license are under [organizer](organizer/).

The design and implementation were developed for this project with AI assistance. The [design review](docs/DESIGN_DECISIONS.md) and [Stage 04 report](docs/STAGE04_TEMPORAL_EXPERIMENT.md) cite the research and circuit interfaces used.
