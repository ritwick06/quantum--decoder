"""Validation and measurement around the unchanged organizer zero baseline."""

import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "organizer/src"))
sys.path.insert(0, str(ROOT / "organizer/submission_template"))
from decoder import decode
from hackathon_2026 import load_kaggle_csv, logical_error_rate, problem_info

DETECTORS = [f"D{i}" for i in range(70)]
LABELS = [f"L{i}" for i in range(4)]
SEEDS = {"X": 2026111301, "Z": 2026111302}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def check_binary(array, width, name):
    a = np.asarray(array)
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f"{name} must have shape (shots, {width})")
    if a.dtype.kind not in "biu" or not np.isin(a, (0, 1)).all():
        raise ValueError(f"{name} must contain binary integers or booleans")
    return a


def inspect_csv(path, *, labels):
    expected = ["id", "basis", *DETECTORS, *(LABELS if labels else [])]
    counts = {"X": 0, "Z": 0}
    ids = set()
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != expected:
            raise ValueError(f"Unexpected CSV schema: {path}")
        for row in reader:
            if len(row) != len(expected) or row[1] not in counts:
                raise ValueError(f"Malformed row: {path}")
            if not row[0] or row[0] in ids:
                raise ValueError(f"Missing or duplicate ID: {path}")
            if any(value not in ("0", "1") for value in row[2:]):
                raise ValueError(f"Nonbinary value: {path}")
            ids.add(row[0])
            counts[row[1]] += 1
    return {"rows": len(ids), "basis_counts": counts, "labelled": labels, "sha256": sha256(path)}


def load_labelled(path):
    inspect_csv(path, labels=True)
    return load_kaggle_csv(path)


def reserved_mask(ids):
    return np.array([
        int.from_bytes(hashlib.sha256(("qec-stage00-v1:" + str(i)).encode()).digest()[:8], "big") % 5 == 0
        for i in ids
    ], dtype=bool)


def load_circuits():
    circuits = {b: stim.Circuit.from_file(ROOT / "quantum-error-correction-decoder" / f"{b}_circuit.stim") for b in "XZ"}
    for circuit in circuits.values():
        if (circuit.num_detectors, circuit.num_observables) != (70, 4):
            raise ValueError("Unexpected circuit dimensions")
    return circuits


def decode_by_basis(circuits, events, bases, decoder=decode):
    events = check_binary(events, 70, "detector_events")
    bases = np.asarray(bases)
    if bases.shape != (len(events),) or not np.isin(bases, ("X", "Z")).all():
        raise ValueError("One valid X/Z basis is required per shot")
    output = np.empty((len(events), 4), dtype=np.uint8)
    for basis in "XZ":
        rows = np.flatnonzero(bases == basis)
        prediction = check_binary(decoder(circuits[basis], events[rows]), 4, "predictions")
        if len(prediction) != len(rows):
            raise ValueError("Decoder returned the wrong shot count")
        output[rows] = prediction
    return output


def describe_circuits(circuits):
    records = {}
    for basis, circuit in circuits.items():
        info = problem_info(circuit)
        h, logical, p = info["H"], info["L"], info["p"]
        check_binary(h.T, 70, "H columns")
        check_binary(logical.T, 4, "L columns")
        assert h.shape[1] == logical.shape[1] == len(p)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        signatures = np.concatenate((h, logical), axis=0).T
        assert len(np.unique(signatures, axis=0)) == len(p)
        coordinates = circuit.get_detector_coordinates()
        assert set(coordinates) == set(range(70))
        times = [int(coordinates[i][2]) for i in range(70)]
        assert times == [0] * 7 + sum(([t] * 14 for t in range(1, 5)), []) + [5] * 7
        records[basis] = {
            "detectors": circuit.num_detectors, "observables": circuit.num_observables,
            "qubits": circuit.num_qubits, "H_shape": list(h.shape), "L_shape": list(logical.shape),
            "mechanisms": len(p), "p_min": float(p.min()), "p_max": float(p.max()),
            "detectors_per_time": {str(t): times.count(t) for t in sorted(set(times))},
            "circuit_sha256": sha256(ROOT / "quantum-error-correction-decoder" / f"{basis}_circuit.stim"),
            "canonical_circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        }
    return records


def freeze_validation(circuits, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": "frozen independent validation; never training", "stim_version": stim.__version__,
                "shots_per_basis": 100000, "sample_calls_per_basis": 1, "separate_observables": True, "bases": {}}
    for basis in "XZ":
        sampler = circuits[basis].compile_detector_sampler(seed=SEEDS[basis])
        events, labels = sampler.sample(100000, separate_observables=True)
        record = {"seed": SEEDS[basis], "circuit_sha256": sha256(ROOT / "quantum-error-correction-decoder" / f"{basis}_circuit.stim"), "arrays": {}}
        for name, array, width in (("detectors", events, 70), ("observables", labels, 4)):
            check_binary(array, width, name)
            path = destination / f"{basis}_{name}.npy"
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            record["arrays"][name] = {"file": path.name, "sha256": sha256(path),
                "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "shape": list(array.shape), "dtype": str(array.dtype)}
        manifest["bases"][basis] = record
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_validation(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    result = {}
    for basis in "XZ":
        arrays = []
        for name, width in (("detectors", 70), ("observables", 4)):
            record = manifest["bases"][basis]["arrays"][name]
            path = destination / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen validation hash mismatch: {path}")
            array = np.load(path, allow_pickle=False)
            check_binary(array, width, name)
            if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
                raise ValueError("Frozen validation metadata mismatch")
            array.flags.writeable = False
            arrays.append(array)
        result[basis] = tuple(arrays)
    return result


def benchmark(circuits, data, runs=5):
    # Match the organizer boundary: both calls plus output contract checks, excluding scoring and I/O.
    times, first = [], None
    for _ in range(runs):
        predictions = {}
        start = time.perf_counter()
        for basis in "XZ":
            events, _ = data[basis]
            output = np.asarray(decode(circuits[basis], events))
            assert output.shape == (len(events), 4)
            assert np.isin(output, (0, 1)).all()
            predictions[basis] = output
        times.append(time.perf_counter() - start)
        if first is None:
            first = predictions
        elif any(not np.array_equal(first[b], predictions[b]) for b in "XZ"):
            raise ValueError("Nondeterministic baseline")
    errors = {b: int(np.count_nonzero(first[b] != data[b][1])) for b in "XZ"}
    return {"runs_seconds": times, "median_seconds": statistics.median(times), "deterministic": True,
            "shots": {b: len(data[b][0]) for b in "XZ"}, "errors": errors,
            "ler": {b: logical_error_rate(first[b], data[b][1]) for b in "XZ"},
            "combined_ler": sum(errors.values()) / sum(data[b][1].size for b in "XZ"),
            "prediction_sha256": {b: hashlib.sha256(first[b].tobytes()).hexdigest() for b in "XZ"}}
