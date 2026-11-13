# Reproducing Stage 01

Run commands from `/home/ritwick06/q_decoder`. The recorded runtime environment is CPython 3.13.18, NumPy 2.2.6, and Stim 1.16.0. The source revision and input hashes are part of the experiment, not floating dependencies.

## Verify the existing checkpoint

```bash
.venv/bin/python -m unittest discover -s tests -v
```

This checks source and original-input hashes, the baseline interface, routing, frozen validation, and fresh-process predictions without replacing the accepted report. The seed replay test creates transient arrays in memory and compares them with the frozen files; it does not replace validation data.

Accepted evidence:

- `artifacts/stage01/report.json`: measurements, environment, inputs, split, and prediction hashes.
- `artifacts/stage01/organizer_runner.log`: unchanged official runner output.
- `artifacts/stage01/tests.log`: original successful test run.
- `artifacts/stage01/original_files.json`: hashes of the 22 original inputs and instruction-pack files.
- `organizer/SOURCE_MANIFEST.json`: pinned revision and per-file source hashes.
- `data/validation/manifest.json`: sampler settings, source circuits, array and file hashes.

## Initial setup in a clean working copy

Retain the original dataset files, instruction pack, team tooling, and `artifacts/stage01/original_files.json`. Do not copy a live virtual environment to another machine. The paths below are additions in that clean copy, not instructions to overwrite the accepted workspace.

Install the setup tool into the project, then provision Python:

```bash
python3 -m pip install --target .tools/uv --no-deps uv==0.13.8
UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" \
UV_PYTHON_BIN_DIR="$PWD/.tools/bin" \
UV_CACHE_DIR="$PWD/.tools/cache" \
  .tools/uv/bin/uv python install 3.13.18
UV_CACHE_DIR="$PWD/.tools/cache" \
  .tools/uv/bin/uv venv --python "$PWD/.tools/bin/python3.13" .venv
UV_CACHE_DIR="$PWD/.tools/cache" \
  .tools/uv/bin/uv pip install --python .venv/bin/python -r requirements.txt
```

Source acquisition is needed only if the organizer snapshot is absent:

```bash
python3 tools/fetch_organizer.py
```

The fetcher selects only the permitted files from revision `739d627b20fdb4899ab7487723d2e4fccec27033`. It excludes the repository's competition datasets and tutorial decoder implementations, verifies Git blob hashes, and records SHA-256. It refuses to redownload a completed snapshot. A partial download can resume by verifying existing files rather than overwriting them.

Network access is required only for this setup and acquisition. Baseline evaluation, validation sampling, and tests run locally without internet access. The organizer package is imported directly from its pinned source directory, so no optional tutorial extras or notebook packages are installed.

In the clean copy, with no existing accepted report or prediction CSV:

```bash
.venv/bin/python tools/run_stage01.py
.venv/bin/python -m unittest discover -s tests -v
```

The stage command creates the training adapter, records the reserved training-row mask, runs the official zero baseline, freezes independent validation if absent, and writes measurements. It validates an existing frozen set before reuse and rejects an existing accepted report or predictions instead of overwriting them.

On the recorded environment, seeded sampling replays byte-for-byte. Other Stim versions or CPU architectures may produce different random streams even with the same seed; keep the accepted arrays and compare hashes instead of silently replacing them. Timings naturally vary between runs and machines.

## Inspect or remeasure without replacing artifacts

This read-only measurement prints a new result without changing saved evidence:

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python - <<'PY'
import json
from stage01_support import ROOT, benchmark, load_circuits, load_validation
result = benchmark(load_circuits(), load_validation(ROOT / "data/validation"))
print(json.dumps(result, indent=2))
PY
```

Do not interpret the local zero-baseline runtime as organizer-machine runtime or as the runtime of a future real decoder. No Kaggle test predictions or submissions are produced by this command.
