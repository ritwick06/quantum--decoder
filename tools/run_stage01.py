"""Reproduce the zero baseline and freeze validation; never decode Kaggle test rows."""

from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import stim
from stage01_support import (benchmark, describe_circuits, freeze_validation, inspect_csv,
    load_circuits, load_labelled, load_validation, reserved_mask, sha256, write_json)


def main():
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Stage 01 requires Python 3.13")
    destination = ROOT / "artifacts/stage01"
    report_path = destination / "report.json"
    if report_path.exists():
        raise FileExistsError("Stage 01 is already recorded; do not overwrite the accepted measurements.")
    source_manifest = json.loads((ROOT / "organizer/SOURCE_MANIFEST.json").read_text())
    for path, record in source_manifest["files"].items():
        assert sha256(ROOT / "organizer" / path) == record["sha256"], path
    originals = json.loads((destination / "original_files.json").read_text())
    for path, digest in originals.items():
        assert sha256(ROOT / path) == digest, path

    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    assert packages == {"numpy": "2.2.6", "stim": "1.16.0"}, packages
    circuits = load_circuits()
    metadata = describe_circuits(circuits)
    source = ROOT / "quantum-error-correction-decoder"
    train = load_labelled(source / "train.csv")
    data = {b: (train["detector_events"][train["basis"] == b], train["observable_flips"][train["basis"] == b]) for b in "XZ"}
    assert all(len(data[b][0]) == 100000 for b in "XZ")
    schemas = {"train": inspect_csv(source / "train.csv", labels=True), "test": inspect_csv(source / "test.csv", labels=False)}
    assert schemas["test"]["basis_counts"] == {"X": 100000, "Z": 100000}

    adapter = ROOT / "data/adapters/train"
    adapter.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "X_circuit.stim", "Z_circuit.stim"):
        link = adapter / name
        if link.is_symlink():
            assert link.resolve() == source / name
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(os.path.relpath(source / name, adapter))

    split_dir = ROOT / "data/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    mask = reserved_mask(train["id"])
    split_path = split_dir / "train_reserved.npy"
    if split_path.exists():
        assert np.array_equal(np.load(split_path, allow_pickle=False), mask)
    else:
        with split_path.open("xb") as stream:
            np.save(stream, mask, allow_pickle=False)
    split = {b: {"reserved": int(mask[train["basis"] == b].sum()),
                 "development": int((~mask[train["basis"] == b]).sum())} for b in "XZ"}

    baseline_dir = ROOT / "artifacts/baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = baseline_dir / "train_predictions.csv"
    if prediction_file.exists():
        raise FileExistsError(prediction_file)
    command = [str(ROOT / ".venv/bin/python"), "organizer/submission_template/run_local.py",
               "--problem-set", "data/adapters/train", "--decoder", "organizer/submission_template",
               "--runs", "5", "--threads", "1", "--out", "artifacts/baseline/train_predictions.csv"]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "organizer/src"))
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    with (destination / "organizer_runner.log").open("x") as stream:
        stream.write(result.stdout + result.stderr)
    print(result.stdout, end="", flush=True)
    measured_train = benchmark(circuits, data)

    validation_dir = ROOT / "data/validation"
    if not validation_dir.exists():
        freeze_validation(circuits, validation_dir)
    validation_manifest = json.loads((validation_dir / "manifest.json").read_text())
    assert validation_manifest["stim_version"] == stim.__version__
    assert validation_manifest["shots_per_basis"] == 100000
    for basis, seed in (("X", 2026111301), ("Z", 2026111302)):
        record = validation_manifest["bases"][basis]
        assert record["seed"] == seed
        assert record["circuit_sha256"] == metadata[basis]["circuit_sha256"]
    validation = load_validation(validation_dir)
    measured_validation = benchmark(circuits, validation)
    report = {
        "stage": "01", "candidate": "organizer all-zero baseline (not an original decoder contribution)",
        "timestamp_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "source_revision": source_manifest["revision"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "logical_cpus": os.cpu_count(), "packages": packages,
            "thread_environment": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "circuits": metadata, "datasets": schemas,
        "split": {"method": "SHA256('qec-stage00-v1:' + id), first 8 bytes big-endian modulo 5 equals 0",
            "counts": split, "train_sha256": schemas["train"]["sha256"], "mask_sha256": sha256(split_path)},
        "training": measured_train, "validation": measured_validation,
        "validation_manifest_sha256": sha256(validation_dir / "manifest.json"),
        "organizer_command": command,
        "training_predictions_sha256": sha256(prediction_file),
        "timing_note": "Local CPU, one thread; includes the organizer output-check overhead, excludes import, CSV I/O and LER computation. Not organizer-hardware timing.",
        "kaggle_submissions": 0,
    }
    for path, digest in originals.items():
        assert sha256(ROOT / path) == digest, path
    write_json(report_path, report)
    print(json.dumps({"training": measured_train, "validation": measured_validation, "circuits": metadata}, indent=2))


if __name__ == "__main__":
    main()
