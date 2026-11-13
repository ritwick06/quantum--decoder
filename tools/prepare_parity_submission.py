"""Generate or verify a promoted candidate CSV against the unlabelled test data."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import evaluate_parity
from evaluate_parity import ROOT
from stage01_support import load_circuits, sha256, write_json
from submission_pipeline import load_test_data, predict_checked, validate_submission, write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--candidate", choices=("parity_v1", "temporal_v1"), default="parity_v1")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    output = args.output or args.verify
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if args.output and any(path.exists() or path.is_symlink() for path in (output, sidecar)):
        raise FileExistsError("Choose a new output path")
    if args.candidate == "temporal_v1":
        import evaluate_temporal as candidate
    else:
        candidate = evaluate_parity
    report = json.loads(candidate.REPORT.read_text())
    decoder, provenance = candidate.checkpoint_decoder()
    checkpoint_hash = sha256(candidate.CHECKPOINT / "provenance.json")
    if not report["passed"] or report["checkpoint_provenance_sha256"] != checkpoint_hash:
        raise ValueError("This checkpoint has no passing frozen-validation report")
    original = json.loads((ROOT / "artifacts/stage01/original_files.json").read_text())
    data_dir = ROOT / "quantum-error-correction-decoder"
    inputs = {name: data_dir / name for name in ("test.csv", "sample_submission.csv", "X_circuit.stim", "Z_circuit.stim")}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    for name, digest in hashes.items():
        if digest != original[f"quantum-error-correction-decoder/{name}"]:
            raise ValueError(f"Submission input changed since the audit: {name}")
    data, summary = load_test_data(inputs["test.csv"], inputs["sample_submission.csv"])
    if summary["rows"] != 200000 or summary["basis_counts"] != {"X": 100000, "Z": 100000}:
        raise ValueError("Unexpected test dimensions")
    bits = predict_checked(load_circuits(), data["detector_events"], data["basis"], decoder, args.batch_size)
    for name, path in inputs.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"Submission input changed during prediction: {name}")
    evidence = {"candidate": candidate.CANDIDATE, "input_sha256": hashes, "checkpoint_files": provenance["files"],
                "checkpoint_provenance_sha256": checkpoint_hash, "validation_report_sha256": sha256(candidate.REPORT)}
    if args.verify:
        checked = validate_submission(output, data["id"], bits)
        manifest = json.loads(sidecar.read_text())
        if checked != manifest["validation"] or any(manifest[key] != value for key, value in evidence.items()):
            raise ValueError("CSV or generation provenance differs from its manifest")
        print(json.dumps({"fresh_predictions_match": True, **checked}, indent=2))
        return
    checked = write_submission(output, data["id"], bits)
    manifest = {**evidence, "artifact": str(output), "validation": checked,
                "basis_counts": summary["basis_counts"], "ones_by_observable": bits.sum(axis=0).tolist(),
                "batch_size": args.batch_size, "local_validation_ler": report["combined_ler"],
                "created_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "uploaded": False}
    write_json(sidecar, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
