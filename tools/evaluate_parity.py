"""Score the approved parity checkpoint or replay its frozen predictions."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from stage01_support import check_binary, load_circuits, load_validation, sha256, write_json


CANDIDATE = "parity_v1"
CHECKPOINT = ROOT / "artifacts/checkpoints" / CANDIDATE
EVIDENCE = ROOT / "artifacts/stage03b"
REPORT = EVIDENCE / "validation.json"


def checkpoint_decoder():
    provenance = json.loads((CHECKPOINT / "provenance.json").read_text())
    if provenance["candidate"] != CANDIDATE or set(provenance["files"]) != {
        "decoder.py", "model.json", "model.npz", "requirements.txt",
    }:
        raise ValueError("Unexpected checkpoint provenance")
    for name, digest in provenance["files"].items():
        if sha256(CHECKPOINT / name) != digest:
            raise ValueError(f"Parity checkpoint checksum mismatch: {name}")
    spec = importlib.util.spec_from_file_location("parity_checkpoint", CHECKPOINT / "decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode, provenance


def input_data():
    baseline = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
    validation = ROOT / "data/validation"
    if sha256(validation / "manifest.json") != baseline["validation_manifest_sha256"]:
        raise ValueError("Frozen validation manifest changed")
    data = load_validation(validation)
    if any(len(events) != 100000 or len(labels) != 100000 for events, labels in data.values()):
        raise ValueError("Expected 100,000 frozen shots for each basis")
    return load_circuits(), data, baseline


def predict(decoder, circuits, data):
    result = {}
    for basis in "XZ":
        events = data[basis][0]
        bits = check_binary(decoder(circuits[basis], events), 4, "parity predictions")
        if bits.dtype != np.uint8 or len(bits) != len(events):
            raise ValueError("Parity output has the wrong row count or dtype")
        result[basis] = bits
    return result


def prediction_hashes(predictions):
    return {basis: hashlib.sha256(bits.tobytes()).hexdigest() for basis, bits in predictions.items()}


def evaluate(decoder, provenance):
    if REPORT.exists():
        raise FileExistsError(f"This checkpoint has already been scored: {REPORT}")
    test_log = EVIDENCE / "tests.log"
    tests = re.search(r"Ran (\d+) tests in [0-9.]+s\s+OK\s*$", test_log.read_text())
    if not tests or int(tests[1]) < 26:
        raise ValueError("Candidate tests must pass before opening frozen validation")
    circuits, data, baseline = input_data()
    runtimes, peak_rss = [], []
    # Asset loading is outside timing, as it can be for the organizer's runtime check.
    for basis in "XZ":
        decoder(circuits[basis], data[basis][0][:0])
    first = None
    for _ in range(5):
        start = time.perf_counter()
        current = predict(decoder, circuits, data)
        runtimes.append(time.perf_counter() - start)
        peak_rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if first is None:
            first = current
        elif any(not np.array_equal(current[basis], first[basis]) for basis in "XZ"):
            raise ValueError("Candidate predictions changed between timed calls")
    errors, zero_errors, gains = {}, {}, {}
    for basis in "XZ":
        labels = data[basis][1]
        wrong_bits = np.count_nonzero(first[basis] != labels, axis=1)
        zero_wrong = np.count_nonzero(labels, axis=1)
        errors[basis] = int(wrong_bits.sum())
        zero_errors[basis] = int(zero_wrong.sum())
        gains[basis] = (zero_wrong - wrong_bits) / 4
    if zero_errors != baseline["validation"]["errors"]:
        raise ValueError("Frozen labels do not reproduce the audited zero baseline")
    total = sum(len(values) for values in gains.values())
    ler = {basis: errors[basis] / data[basis][1].size for basis in "XZ"}
    combined = sum(errors.values()) / (4 * total)
    gain = sum(values.sum() for values in gains.values()) / total
    se = np.sqrt(sum(len(values) * values.var(ddof=1) for values in gains.values())) / total
    margin = 1.95996398454 * se
    median = statistics.median(runtimes)
    gates = {
        "combined_ler": combined <= 0.28290125,
        "x_ler": ler["X"] <= 0.29236,
        "z_ler": ler["Z"] <= 0.2834425,
        "paired_gain_lower_ci_positive": bool(gain - margin > 0),
        "runtime": median <= 10.0,
        "inference_rss": max(peak_rss) <= 1024**3,
        "training_resources": provenance["training_seconds"] <= 1200 and provenance["training_peak_rss_bytes"] <= 8 * 1024**3,
        "unit_tests": True,
    }
    return {
        "candidate": CANDIDATE, "checkpoint_provenance_sha256": sha256(CHECKPOINT / "provenance.json"),
        "validation_manifest_sha256": baseline["validation_manifest_sha256"],
        "zero_ler": baseline["validation"]["ler"], "zero_combined_ler": baseline["validation"]["combined_ler"],
        "shots": {basis: len(data[basis][0]) for basis in "XZ"}, "errors": errors, "ler": ler, "combined_ler": combined,
        "paired_gain": float(gain), "paired_gain_ci95": [float(gain - margin), float(gain + margin)],
        "runs_seconds": runtimes, "median_seconds": median, "peak_rss_bytes_by_run": peak_rss,
        "timing_scope": "Local CPU, one thread, both basis calls and output validation. Import, loading and scoring excluded.",
        "prediction_sha256": prediction_hashes(first), "tests_passed": int(tests[1]),
        "test_log_sha256": sha256(test_log), "gates": gates, "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Verify prediction hashes without scoring again")
    args = parser.parse_args()
    decoder, provenance = checkpoint_decoder()
    if args.replay:
        report = json.loads(REPORT.read_text())
        circuits, data, _ = input_data()
        hashes = prediction_hashes(predict(decoder, circuits, data))
        if hashes != report["prediction_sha256"]:
            raise ValueError("Fresh-process predictions differ from the original run")
        result = {"candidate": CANDIDATE, "prediction_sha256": hashes, "match": True,
                  "validation_report_sha256": sha256(REPORT)}
    else:
        result = evaluate(decoder, provenance)
        write_json(REPORT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
