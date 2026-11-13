"""Package a promoted decoder and its offline asset-generation sources."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate_temporal import checkpoint_decoder, locations
from stage01_support import ROOT, sha256, write_json
from tools.build_temporal import BIDIRECTIONAL, EXPANDED, LOOKAHEAD, READOUT, SPECTRAL, SUFFIX


REPRODUCTION_SOURCES = (
    "requirements.txt", "stage01_support.py", "tools/build_temporal.py", "tools/train_parity.py",
    "tools/train_temporal_readout.py", "temporal_decoder/readout.py",
    "team_decoder/decoder.py", "temporal_decoder/decoder.py", "temporal_decoder/kernel.cpp",
    "temporal_decoder/spectral.cpp", "organizer/LICENSE", "organizer/submission_template/decoder.py",
    "artifacts/stage01/report.json", "data/splits/train_reserved.npy",
    "docs/DESIGN_DECISIONS.md", "docs/STAGE04_TEMPORAL_EXPERIMENT.md", "docs/BIDIRECTIONAL_TEMPORAL_EXPERIMENT.md",
    "docs/SPECTRAL_TEMPORAL_EXPERIMENT.md", "docs/EXACT_SUFFIX_EXPERIMENT.md",
    "docs/TWO_SLICE_LOOKAHEAD_EXPERIMENT.md", "docs/AMBIGUOUS_READOUT_EXPERIMENT.md",
    "docs/EXPANDED_TRANSITIONS_EXPERIMENT.md",
)


def generate_archive(candidate, output):
    checkpoint, evidence = locations(candidate)
    _, provenance = checkpoint_decoder(candidate)
    report = json.loads((evidence / "validation.json").read_text())
    csv_path = ROOT / "artifacts/submissions" / f"{candidate}.csv"
    manifest = json.loads(csv_path.with_suffix(".csv.manifest.json").read_text())
    if (not report["passed"] or sha256(csv_path) != manifest["validation"]["sha256"]
            or sha256(checkpoint / "provenance.json") != manifest["checkpoint_provenance_sha256"]
            or sha256(evidence / "validation.json") != manifest["validation_report_sha256"]):
        raise ValueError("The selected checkpoint and CSV do not match their verified reports")
    if output.exists():
        raise FileExistsError(output)

    with tempfile.TemporaryDirectory(prefix=".hand-in-", dir=output.parent) as temp:
        temp = Path(temp)
        archive_path = temp / output.name
        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for name in (*provenance["files"], "provenance.json"):
                archive.write(checkpoint / name, name)
            for name in REPRODUCTION_SOURCES:
                archive.write(ROOT / name, f"asset_generation/{name}")
            for path in sorted((ROOT / "organizer/src").rglob("*")):
                if path.suffix in (".py", ".json", ".stim"):
                    archive.write(path, f"asset_generation/{path.relative_to(ROOT)}")
            archive.write(ROOT / "organizer/LICENSE", "LICENSE.organizer")
            archive.writestr("README.md", hand_in_readme(candidate, manifest))
        verify_archive(archive_path, csv_path, manifest)
        shutil.move(archive_path, output)
    return {"candidate": candidate, "archive_sha256": sha256(output),
            "csv_sha256": manifest["validation"]["sha256"], "rows": manifest["validation"]["rows"],
            "verified_all_predictions_in_fresh_process": True}


def hand_in_readme(candidate, manifest):
    name = Path(manifest["artifact"]).name
    digest = manifest["validation"]["sha256"]
    modes = {SPECTRAL: ("--spectral",), SUFFIX: ("--suffix",),
             LOOKAHEAD: ("--suffix", "--lookahead"), READOUT: ("--suffix",),
             EXPANDED: ("--suffix", "--expanded")}.get(candidate, ())
    extra_build = "".join(f"\nOPENBLAS_NUM_THREADS=1 python tools/build_temporal.py {mode}" for mode in modes)
    if candidate == READOUT:
        extra_build += "\nOPENBLAS_NUM_THREADS=1 python tools/train_temporal_readout.py"
    return f"""# PHANTOMS OF QUANTUM decoder

This package contains the {candidate} decoder. Install the pinned requirements in
a Python 3.13 environment and use the root `decoder.py` with the adjacent assets.
The shared library targets Linux x86-64-v2 and needs the system C and C++ runtime.
Inference uses only the supplied circuit and detector events.

Selected CSV identity is `{name}`, SHA-256 `{digest}`.
The package was checked by loading its extracted decoder in a fresh Python
process and comparing every prediction and ID with that CSV.

## Rebuild the assets

Enter `asset_generation` and place the original competition `train.csv`,
`X_circuit.stim`, and `Z_circuit.stim` in `quantum-error-correction-decoder/`.
Use the pinned requirements and g++ with C++17 and OpenMP support.

```sh
OPENBLAS_NUM_THREADS=1 python tools/train_parity.py
OPENBLAS_NUM_THREADS=1 python tools/build_temporal.py
OPENBLAS_NUM_THREADS=1 python tools/build_temporal.py --bidirectional{extra_build}
```

The rebuilt checkpoints are under `artifacts/checkpoints`. The decoder was
designed and implemented for this challenge with AI assistance.
"""


def verify_archive(archive_path, csv_path, manifest):
    unpack = archive_path.parent / "unpacked"
    with ZipFile(archive_path) as archive:
        archive.extractall(unpack)
    subprocess.run([sys.executable, "-I", "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); import stage01_support",
                    str(unpack / "asset_generation")], cwd=unpack, check=True)
    script = """
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from stage01_support import load_circuits
from submission_pipeline import load_test_data, predict_checked, validate_submission
spec = importlib.util.spec_from_file_location("hand_in_decoder", sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data_dir = Path(sys.argv[1]) / "quantum-error-correction-decoder"
data, _ = load_test_data(data_dir / "test.csv", data_dir / "sample_submission.csv")
prediction = predict_checked(load_circuits(), data["detector_events"], data["basis"], module.decode, 8192)
verified = validate_submission(Path(sys.argv[3]), data["id"], prediction)
if verified["sha256"] != sys.argv[4]:
    raise ValueError("Packaged decoder does not match the verified CSV")
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT), str(unpack / "decoder.py"),
         str(csv_path), manifest["validation"]["sha256"]],
        cwd=unpack, check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=(BIDIRECTIONAL, SPECTRAL, SUFFIX, LOOKAHEAD, READOUT, EXPANDED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(".verification.json")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    output.parent.mkdir(parents=True, exist_ok=True)
    verification = generate_archive(args.candidate, output)
    write_json(sidecar, verification)
    print(f"Verified package {output} (SHA-256 {verification['archive_sha256']})")


if __name__ == "__main__":
    main()
