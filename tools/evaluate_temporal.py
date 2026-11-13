"""Evaluate the preregistered temporal candidate on the frozen holdout."""

import argparse
import gc
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
from tools.build_temporal import ADAPTIVE, BACKWARD, BIDIRECTIONAL, CANDIDATE, CHECKPOINT, COVERAGE, DEEP, DUAL, EXPANDED, LOOKAHEAD, MIDPOINT, READOUT, SHIFTED, SPECTRAL, SUFFIX


EVIDENCE = ROOT / "artifacts/stage04"
REPORT = EVIDENCE / "validation.json"
EXPERIMENTS = {
    CANDIDATE: ("stage04", "parity_v1", 29),
    BIDIRECTIONAL: ("bidirectional", CANDIDATE, 31),
    MIDPOINT: ("midpoint", BIDIRECTIONAL, 32),
    ADAPTIVE: ("adaptive", BIDIRECTIONAL, 33),
    SPECTRAL: ("spectral", BIDIRECTIONAL, 34),
    SUFFIX: ("suffix", BIDIRECTIONAL, 35),
    LOOKAHEAD: ("lookahead", SUFFIX, 36),
    READOUT: ("readout", SUFFIX, 37),
    EXPANDED: ("expanded", SUFFIX, 38),
    BACKWARD: ("backward", EXPANDED, 39),
    DEEP: ("deep", EXPANDED, 40),
    SHIFTED: ("shifted", EXPANDED, 41),
    COVERAGE: ("coverage", EXPANDED, 41),
    DUAL: ("dual_readout", EXPANDED, 42),
}


def locations(candidate):
    folder, _, _ = EXPERIMENTS[candidate]
    return CHECKPOINT.parent / candidate, ROOT / "artifacts" / folder


def checkpoint_decoder(candidate=CANDIDATE):
    checkpoint, _ = locations(candidate)
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    if provenance["candidate"] != candidate:
        raise ValueError("Unexpected candidate checkpoint")
    for name, expected in provenance["files"].items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    spec = importlib.util.spec_from_file_location(candidate, checkpoint / "decoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode, provenance


def evaluate(decoder, provenance, candidate=CANDIDATE):
    checkpoint, evidence = locations(candidate)
    report = evidence / "validation.json"
    if report.exists():
        raise FileExistsError(f"Validation already recorded at {report}")
    _, incumbent_name, required_tests = EXPERIMENTS[candidate]
    tests = re.search(r"Ran (\d+) tests in [0-9.]+s\s+OK\s*$", (evidence / "tests.log").read_text())
    if not tests or int(tests[1]) < required_tests:
        raise ValueError("Run all decoder checks before scoring frozen validation")
    circuits, data, audit = input_data()
    if incumbent_name == "parity_v1":
        incumbent, _ = parity_decoder()
        previous = json.loads((ROOT / "artifacts/stage03b/validation.json").read_text())
    else:
        incumbent, _ = checkpoint_decoder(incumbent_name)
        _, previous_evidence = locations(incumbent_name)
        previous = json.loads((previous_evidence / "validation.json").read_text())
    incumbent_bits = predict(incumbent, circuits, data)
    if prediction_hashes(incumbent_bits) != previous["prediction_sha256"]:
        raise ValueError("Incumbent predictions differ from their frozen report")
    if candidate in (DEEP, SHIFTED, COVERAGE, DUAL):
        load_model = incumbent.__globals__.get("load_model")
        if load_model is not None:
            load_model.cache_clear()
        del incumbent, load_model
        gc.collect()
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
    minimum_gain = {CANDIDATE: 0.025, BIDIRECTIONAL: 0.0005}.get(candidate, 0)
    gates = {"ler_gain": gain > 0 and gain >= minimum_gain,
             "paired_interval": ci[0] > 0, "runtime": median <= (45 if candidate == CANDIDATE else 12),
             "peak_rss": rss <= 2 * 1024**3,
             "penalized_ler_sensitivity": penalized < previous["combined_ler"] * max(1, previous["median_seconds"] / 15.4)**0.2,
             "build_resources": provenance["build_seconds"] <= 1200 and provenance["build_peak_rss_bytes"] <= 5 * 1024**3}
    if candidate in (CANDIDATE, BIDIRECTIONAL):
        gates["both_bases"] = all(ler[basis] <= previous["ler"][basis] for basis in "XZ")
    if candidate == CANDIDATE:
        gates["dense_ler_gain"] = dense_gain >= 0.020
    if candidate == DUAL:
        gates["runtime"] = median <= 300
    return {"candidate": candidate, "checkpoint_provenance_sha256": sha256(checkpoint / "provenance.json"),
            "validation_manifest_sha256": audit["validation_manifest_sha256"], "incumbent": incumbent_name,
            "incumbent_ler": previous["ler"], "incumbent_combined_ler": previous["combined_ler"],
            "ler": ler, "combined_ler": combined, "errors": errors, "paired_gain": gain, "paired_gain_ci95": ci,
            "dense": dense, "dense_gain": dense_gain, "runs_seconds": runs, "median_seconds": median,
            "peak_rss_bytes": rss, "penalized_ler_local_sensitivity": penalized,
            "prediction_sha256": prediction_hashes(reference), "tests_passed": int(tests[1]),
            "gates": gates, "passed": all(gates.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=EXPERIMENTS, default=CANDIDATE)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    decoder, provenance = checkpoint_decoder(args.candidate)
    _, evidence = locations(args.candidate)
    report = evidence / "validation.json"
    if args.replay:
        recorded = json.loads(report.read_text())
        circuits, data, _ = input_data()
        hashes = prediction_hashes(predict(decoder, circuits, data))
        if hashes != recorded["prediction_sha256"]:
            raise ValueError("Fresh process produced different predictions")
        result = {"candidate": args.candidate, "prediction_sha256": hashes, "match": True}
    else:
        result = evaluate(decoder, provenance, args.candidate)
        write_json(report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
