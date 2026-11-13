"""Prepare or verify the local zero-baseline submission; never upload it."""

import argparse
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from stage01_support import load_circuits, sha256, write_json
from submission_pipeline import load_test_data, predict_checked, validate_submission, write_submission

CANDIDATE = "zero_baseline_v1"
CHECKPOINT = ROOT / "artifacts/checkpoints" / CANDIDATE


def checkpoint_decoder():
    source = json.loads((ROOT / "organizer/SOURCE_MANIFEST.json").read_text())
    files = {"decoder.py": ROOT / "organizer/submission_template/decoder.py",
             "requirements.txt": ROOT / "requirements.txt", "LICENSE": ROOT / "organizer/LICENSE"}
    for relative in ("submission_template/decoder.py", "LICENSE"):
        if sha256(ROOT / "organizer" / relative) != source["files"][relative]["sha256"]:
            raise ValueError("Organizer source differs from pinned revision")
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    for name, original in files.items():
        target = CHECKPOINT / name
        if target.exists():
            if sha256(target) != sha256(original):
                raise ValueError(f"Checkpoint mismatch: {name}")
        else:
            with target.open("xb") as stream:
                stream.write(original.read_bytes())
    provenance = {"candidate": CANDIDATE, "source_revision": source["revision"],
        "description": "Unchanged organizer zero decoder; interface baseline, not an original decoder contribution",
        "trained_assets": [], "files": {name: sha256(CHECKPOINT / name) for name in files}}
    manifest = CHECKPOINT / "provenance.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != provenance:
            raise ValueError("Checkpoint provenance mismatch")
    else:
        write_json(manifest, provenance)
    spec = importlib.util.spec_from_file_location("submission_checkpoint", CHECKPOINT / "decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="new CSV path; existing files are never overwritten")
    mode.add_argument("--verify", type=Path, help="existing CSV to check against freshly computed predictions")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Use the pinned Python 3.13 environment")
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    if packages != {"numpy": "2.2.6", "stim": "1.16.0"}:
        raise RuntimeError("Runtime packages differ from the baseline lock")
    stage01 = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
    original = json.loads((ROOT / "artifacts/stage01/original_files.json").read_text())
    source_dir = ROOT / "quantum-error-correction-decoder"
    inputs = {name: source_dir / name for name in ("test.csv", "sample_submission.csv", "X_circuit.stim", "Z_circuit.stim")}
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    for name, digest in input_hashes.items():
        if digest != original[f"quantum-error-correction-decoder/{name}"]:
            raise ValueError(f"Input changed since the audit: {name}")
    output = args.output or args.verify
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if args.output and (output.exists() or output.is_symlink() or manifest_path.exists()):
        raise FileExistsError("Output or manifest already exists; choose a new path")
    decoder, provenance = checkpoint_decoder()
    data, summary = load_test_data(inputs["test.csv"], inputs["sample_submission.csv"])
    if summary["rows"] != 200000 or summary["basis_counts"] != {"X": 100000, "Z": 100000}:
        raise ValueError("Unexpected competition test dimensions")
    circuits = load_circuits()
    start = time.perf_counter()
    prediction = predict_checked(circuits, data["detector_events"], data["basis"], decoder, args.batch_size)
    seconds = time.perf_counter() - start
    repeated = predict_checked(circuits, data["detector_events"], data["basis"], decoder, args.batch_size)
    if not np.array_equal(prediction, repeated):
        raise ValueError("Decoder predictions changed between calls")
    for name, digest in input_hashes.items():
        if sha256(inputs[name]) != digest:
            raise ValueError(f"Input changed during inference: {name}")
    if args.verify:
        checked = validate_submission(output, data["id"], prediction)
        manifest = json.loads(manifest_path.read_text())
        if manifest["validation"]["sha256"] != checked["sha256"] or manifest["input_sha256"] != input_hashes:
            raise ValueError("Artifact provenance does not match current inputs or output")
        if manifest["checkpoint"] != provenance:
            raise ValueError("Artifact checkpoint differs from the loaded decoder")
        print(json.dumps({"mode": "verified", "candidate": CANDIDATE, "fresh_predictions_match": True,
                          "batch_size": args.batch_size, **checked}, indent=2))
        return
    checked = write_submission(output, data["id"], prediction)
    manifest = {"candidate": CANDIDATE, "status": "validated locally; not uploaded",
        "created_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "artifact": str(output.resolve().relative_to(ROOT)) if output.resolve().is_relative_to(ROOT) else str(output.resolve()),
        "validation": checked, "input_sha256": input_hashes, "checkpoint": provenance,
        "python": platform.python_version(), "packages": packages,
        "basis_counts": summary["basis_counts"], "batch_size": args.batch_size,
        "repeat_predictions_equal": True, "generation_seconds": seconds,
        "timing_note": "One local pipeline pass including routing and validation; not a competition speed benchmark",
        "prior_local_validation_ler": stage01["validation"]["combined_ler"],
        "kaggle_score": None, "uploaded": False,
        "rules_evidence": "User-supplied Kaggle rules; live Rules/Data pages inaccessible at Stage 02 recheck"}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
