"""Evaluate the preregistered temporal candidate on the frozen holdout."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import resource
import statistics
import time

from evaluate_parity import ROOT, input_data, predict, prediction_hashes, checkpoint_decoder as parity_decoder
import numpy as np

from stage01_support import sha256, write_json
from tools.build_temporal import CANDIDATE, CHECKPOINT


EVIDENCE = ROOT / "artifacts/stage04"
REPORT = EVIDENCE / "validation.json"


def checkpoint_decoder():
    provenance = json.loads((CHECKPOINT / "provenance.json").read_text())
    if provenance["candidate"] != CANDIDATE:
        raise ValueError("Unexpected candidate checkpoint")
    for name, expected in provenance["files"].items():
        if sha256(CHECKPOINT / name) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    spec = importlib.util.spec_from_file_location(CANDIDATE, CHECKPOINT / "decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode, provenance


def evaluate(decoder, provenance):
    if REPORT.exists():
        raise FileExistsError(f"Validation already recorded at {REPORT}")
    tests = re.search(r"Ran (\d+) tests in [0-9.]+s\s+OK\s*$", (EVIDENCE / "tests.log").read_text())
    if not tests or int(tests[1]) < 29:
        raise ValueError("Run all decoder checks before scoring frozen validation")
    circuits, data, audit = input_data()
    incumbent, _ = parity_decoder()
    previous = json.loads((ROOT / "artifacts/stage03b/validation.json").read_text())
    incumbent_bits = predict(incumbent, circuits, data)
    if prediction_hashes(incumbent_bits) != previous["prediction_sha256"]:
        raise ValueError("Incumbent predictions differ from their frozen report")
    for basis in "XZ":
        decoder(circuits[basis], data[basis][0][:0])
    reference, runs = None, []
    for _ in range(5):
        start = time.perf_counter()
        result = predict(decoder, circuits, data)
        runs.append(time.perf_counter() - start)
        if reference is None:
            reference = result
        elif any(not np.array_equal(result[basis], reference[basis]) for basis in "XZ"):
            raise ValueError("Temporal predictions varied across benchmark calls")
    errors, gains, dense = {}, {}, {}
    for basis in "XZ":
        events, labels = data[basis]
        before = np.count_nonzero(incumbent_bits[basis] != labels, axis=1)
        after = np.count_nonzero(reference[basis] != labels, axis=1)
        errors[basis] = int(after.sum())
        gains[basis] = (before - after) / 4
        selected = np.count_nonzero(events, axis=1) >= 7
        dense[basis] = {"shots": int(selected.sum()), "incumbent_errors": int(before[selected].sum()),
                        "candidate_errors": int(after[selected].sum())}
    n = sum(len(values) for values in gains.values())
    combined = sum(errors.values()) / (4 * n)
    ler = {basis: errors[basis] / data[basis][1].size for basis in "XZ"}
    gain = float(sum(values.sum() for values in gains.values()) / n)
    se = np.sqrt(sum(len(values) * values.var(ddof=1) for values in gains.values())) / n
    ci = [float(gain - 1.95996398454 * se), float(gain + 1.95996398454 * se)]
    dense_gain = sum(row["incumbent_errors"] - row["candidate_errors"] for row in dense.values()) / (
        4 * sum(row["shots"] for row in dense.values()))
    median = statistics.median(runs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    penalized = combined * max(1, median / 15.4)**0.2
    gates = {"ler_gain": gain >= 0.025, "dense_ler_gain": dense_gain >= 0.020,
             "both_bases": all(ler[basis] <= previous["ler"][basis] for basis in "XZ"),
             "paired_interval": ci[0] > 0, "runtime": median <= 45, "peak_rss": rss <= 2 * 1024**3,
             "penalized_ler_sensitivity": penalized < previous["combined_ler"],
             "build_resources": provenance["build_seconds"] <= 1200 and provenance["build_peak_rss_bytes"] <= 5 * 1024**3}
    return {"candidate": CANDIDATE, "checkpoint_provenance_sha256": sha256(CHECKPOINT / "provenance.json"),
            "validation_manifest_sha256": audit["validation_manifest_sha256"], "incumbent": "parity_v1",
            "incumbent_ler": previous["ler"], "incumbent_combined_ler": previous["combined_ler"],
            "ler": ler, "combined_ler": combined, "errors": errors, "paired_gain": gain, "paired_gain_ci95": ci,
            "dense": dense, "dense_gain": dense_gain, "runs_seconds": runs, "median_seconds": median,
            "peak_rss_bytes": rss, "penalized_ler_local_sensitivity": penalized,
            "prediction_sha256": prediction_hashes(reference), "tests_passed": int(tests[1]),
            "gates": gates, "passed": all(gates.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    decoder, provenance = checkpoint_decoder()
    if args.replay:
        recorded = json.loads(REPORT.read_text())
        circuits, data, _ = input_data()
        hashes = prediction_hashes(predict(decoder, circuits, data))
        if hashes != recorded["prediction_sha256"]:
            raise ValueError("Fresh process produced different predictions")
        result = {"candidate": CANDIDATE, "prediction_sha256": hashes, "match": True}
    else:
        result = evaluate(decoder, provenance)
        write_json(REPORT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
