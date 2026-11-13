"""Build temporal likelihood tables for either recorded experiment."""

import argparse
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import stim

from stage01_support import load_circuits, problem_info, sha256, write_json
from temporal_decoder.decoder import EDGE, FANOUT, ROUND_SIZES, circuit_digest, load_kernel


CANDIDATE = "temporal_v1"
CHECKPOINT = ROOT / "artifacts/checkpoints" / CANDIDATE
FALLBACK = ROOT / "artifacts/checkpoints/parity_v1"


def xor_moments(syndromes, logical, probabilities, size):
    """Distribution of the XOR and four logical sign moments under independent faults."""
    values = np.zeros((size, 5), dtype=np.float64)
    values[0] = 1
    indexes = np.arange(size)
    for syndrome, label, probability in zip(syndromes, logical, probabilities, strict=True):
        signs = np.array([1, *[1 - 2 * ((int(label) >> bit) & 1) for bit in range(4)]])
        values = (1 - probability) * values + probability * values[indexes ^ int(syndrome)] * signs
    return values


def projected_masks(checks, indexes):
    shifts = np.left_shift(np.uint32(1), np.arange(len(indexes), dtype=np.uint32))
    return shifts @ checks[indexes].astype(np.uint32)


def generate_basis(path, basis, circuit, kernel, *, reverse=False):
    information = problem_info(circuit)
    checks, logicals, probabilities = information["H"], information["L"], information["p"]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities >= 0.5)):
        raise ValueError(f"Unexpected independent fault probabilities in {basis}")
    times = np.array([circuit.get_detector_coordinates()[i][-1] for i in range(70)], dtype=int)
    if reverse:
        times = 5 - times
    groups = [np.flatnonzero(times == t).tolist() for t in range(6)]
    if tuple(map(len, groups)) != ROUND_SIZES:
        raise ValueError(f"Unexpected detector time slices in {basis}")
    earliest = np.min(np.where(checks != 0, times[:, None], 6), axis=0)
    latest = np.max(np.where(checks != 0, times[:, None], -1), axis=0)
    if np.any(earliest > latest) or np.any(latest - earliest > 1):
        raise ValueError(f"Fault mechanism is not supported on one or two neighboring slices in {basis}")
    logical = projected_masks(logicals, list(range(4)))
    masses = []
    for t, indexes in enumerate(groups):
        selected = earliest == t
        incoming = projected_masks(checks[:, selected], indexes)
        labels, p = logical[selected], probabilities[selected]
        marginal = xor_moments(incoming, labels, p, 1 << len(indexes))
        if t:
            np.save(path / f"{basis}_marginal_{t}.npy", marginal[:, 0], allow_pickle=False)
        if t == 5:
            np.save(path / f"{basis}_terminal.npy", marginal, allow_pickle=False)
        else:
            outgoing = projected_masks(checks[:, selected], groups[t + 1])
            tags = np.ascontiguousarray((incoming << 18) | (outgoing << 4) | labels, dtype=np.uint32)
            odds = np.ascontiguousarray(p / (1 - p))
            table = np.zeros((1 << len(indexes), FANOUT), dtype=EDGE)
            if kernel.temporal_build(tags.ctypes.data, odds.ctypes.data, len(tags), len(table), table.ctypes.data):
                raise RuntimeError(f"Temporal table generation failed for {basis} slice {t}")
            table["mass"] *= np.prod(1 - p)
            retained = float(table["mass"][:, :, 0].sum(dtype=np.float64))
            masses.append(retained)
            np.save(path / f"{basis}_edges_{t}.npy", table, allow_pickle=False)
            print(f"{basis} slice {t}: {len(tags)} mechanisms, retained unconditional mass {retained:.5f}", flush=True)
    return groups, masses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()
    candidate = "temporal_bidir_v1" if args.bidirectional else CANDIDATE
    destination = CHECKPOINT.parent / candidate
    design = ("docs/BIDIRECTIONAL_TEMPORAL_EXPERIMENT.md" if args.bidirectional
              else "docs/STAGE04_TEMPORAL_EXPERIMENT.md")
    start = time.perf_counter()
    if destination.exists():
        raise FileExistsError(destination)
    if sys.version_info[:2] != (3, 13) or (np.__version__, stim.__version__) != ("2.2.6", "1.16.0"):
        raise RuntimeError("Use the pinned Python, NumPy and Stim environment")
    fallback = json.loads((FALLBACK / "provenance.json").read_text())
    for name, expected in fallback["files"].items():
        if sha256(FALLBACK / name) != expected:
            raise ValueError(f"Parity fallback changed: {name}")
    audit = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
    incumbent = json.loads((CHECKPOINT / "provenance.json").read_text()) if args.bidirectional else None
    if incumbent:
        for name, digest in incumbent["files"].items():
            if sha256(CHECKPOINT / name) != digest:
                raise ValueError(f"Forward checkpoint changed: {name}")
        forward_metadata = json.loads((CHECKPOINT / "model.json").read_text())
    compile_args = ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-march=x86-64-v2"]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-", dir=CHECKPOINT.parent) as temp:
        path = Path(temp) / candidate
        path.mkdir()
        source = ROOT / "temporal_decoder"
        for name in ("decoder.py", "kernel.cpp"):
            shutil.copyfile(source / name, path / name)
        shutil.copyfile(ROOT / "requirements.txt", path / "requirements.txt")
        (path / "parity_fallback").mkdir()
        for name in ("decoder.py", "model.json", "model.npz"):
            shutil.copyfile(FALLBACK / name, path / "parity_fallback" / name)
        subprocess.run([*compile_args, "kernel.cpp", "-o", "kernel.so"], cwd=path, check=True)
        kernel = load_kernel(path / "kernel.so")
        fingerprints, detector_groups, retained_mass = {}, {}, {}
        for basis, circuit in load_circuits().items():
            fingerprints[basis] = circuit_digest(circuit)
            if fingerprints[basis] != audit["circuits"][basis]["canonical_circuit_sha256"]:
                raise ValueError(f"Audited circuit changed: {basis}")
            if incumbent:
                detector_groups[basis] = forward_metadata["detector_groups"][basis]
                retained_mass[basis] = incumbent["retained_mass_by_slice"][basis]
                for name in incumbent["files"]:
                    if name.startswith(f"{basis}_") and name.endswith(".npy"):
                        shutil.copyfile(CHECKPOINT / name, path / name)
                prefix = f"{basis}_reverse"
                groups, retained_mass[prefix] = generate_basis(path, prefix, circuit, kernel, reverse=True)
                if groups != detector_groups[basis][::-1]:
                    raise ValueError(f"Reversed detector groups changed the within-slice order for {basis}")
            else:
                detector_groups[basis], retained_mass[basis] = generate_basis(path, basis, circuit, kernel)
        assets = {str(file.relative_to(path)): sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}
        write_json(path / "model.json", {"format_version": 2 if args.bidirectional else 1, "canonical_circuit_sha256": fingerprints,
                                        "detector_groups": detector_groups, "assets": assets})
        elapsed = time.perf_counter() - start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if elapsed > 1200 or peak_rss > 5 * 1024**3:
            raise RuntimeError("Temporal table generation exceeded the approved budget")
        write_json(path / "provenance.json", {
            "candidate": candidate, "files": {**assets, "model.json": sha256(path / "model.json")},
            "build_seconds": elapsed, "build_peak_rss_bytes": peak_rss, "retained_mass_by_slice": retained_mass,
            "build_command": [*compile_args, "kernel.cpp", "-o", "kernel.so"],
            "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
            "python": platform.python_version(), "organizer_revision": audit["source_revision"],
            "fallback_provenance_sha256": sha256(FALLBACK / "provenance.json"),
            "design_sha256": sha256(ROOT / design),
            "forward_provenance_sha256": sha256(CHECKPOINT / "provenance.json") if incumbent else None,
            "asset_generator_sha256": sha256(Path(__file__)),
        })
        if destination.exists():
            raise FileExistsError(destination)
        path.rename(destination)
    print(f"Built {candidate} in {elapsed:.1f}s with peak RSS {peak_rss / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
