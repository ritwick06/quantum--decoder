"""Fit and package a readout using independent circuit simulations."""

import hashlib
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time
from unittest.mock import patch

from train_parity import fit_head

import numpy as np
import stim

from stage01_support import ROOT, load_circuits, sha256, write_json
from temporal_decoder import decoder, readout
from tools.build_temporal import CHECKPOINT, READOUT, SUFFIX


SEEDS = {"X": 2026111341, "Z": 2026111342}
SAMPLE_SIZE = 100000
CALLS = 10
REGULARIZATION = 0.002


def training_examples(circuit, seed):
    kernel, fallback, models = decoder.load_model()
    fingerprint = decoder.circuit_digest(circuit)
    _, (forward, reverse) = models[fingerprint]
    sampler = circuit.compile_detector_sampler(seed=seed)
    examples = [[] for _ in range(4)]
    targets = [[] for _ in range(4)]
    event_hash, label_hash = hashlib.sha256(), hashlib.sha256()
    for _ in range(CALLS):
        events, labels = sampler.sample(SAMPLE_SIZE, separate_observables=True)
        event_hash.update(events.tobytes())
        label_hash.update(labels.tobytes())
        a = decoder._decode_temporal(events, forward, kernel, suffix=True)
        b = decoder._decode_temporal(events, reverse, kernel, suffix=True)
        prepared = readout.direction_features(a, b)
        active = prepared[-1].any(axis=1)
        compact = tuple(array[active] for array in prepared)
        selected = events[active]
        parity = fallback(circuit, selected)
        counts = np.column_stack([selected[:, group].sum(axis=1) for group in forward[0]])
        for bit in range(4):
            rows = compact[-1][:, bit]
            examples[bit].append(readout.feature_columns(compact, counts, parity, rows, bit))
            targets[bit].append(labels[active, bit][rows])
    return examples, targets, {"seed": seed, "shots": SAMPLE_SIZE * CALLS, "calls": CALLS,
                               "events_sha256": event_hash.hexdigest(), "labels_sha256": label_hash.hexdigest()}


def fit_models(circuit, seed):
    examples, targets, metadata = training_examples(circuit, seed)
    weights = []
    metadata["heads"] = []
    for bit in range(4):
        features = np.concatenate(examples[bit])
        labels = np.concatenate(targets[bit]).astype(np.float64)
        center = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0), 1e-8)
        design = np.column_stack((np.ones(len(features)), (features - center) / scale))
        fitted, diagnostics = fit_head(design, labels, REGULARIZATION)
        slopes = fitted[1:] / scale
        weights.append(np.r_[fitted[0] - center @ slopes, slopes])
        metadata["heads"].append({"bit": bit, "samples": len(labels), **diagnostics})
    return np.asarray(weights), metadata


def main():
    start = time.perf_counter()
    incumbent = CHECKPOINT.parent / SUFFIX
    destination = CHECKPOINT.parent / READOUT
    if destination.exists():
        raise FileExistsError(destination)
    parent_provenance = json.loads((incumbent / "provenance.json").read_text())
    if parent_provenance["candidate"] != SUFFIX:
        raise ValueError("Unexpected parent decoder")
    for name, digest in parent_provenance["files"].items():
        if sha256(incumbent / name) != digest:
            raise ValueError(f"Parent asset changed: {name}")
    parameters, training = {}, {}
    with patch.object(decoder, "HERE", incumbent):
        decoder.load_model.cache_clear()
        for basis, circuit in load_circuits().items():
            fingerprint = decoder.circuit_digest(circuit)
            parameters[fingerprint], training[basis] = fit_models(circuit, SEEDS[basis])
            print(f"{basis}: fitted four readouts from {SAMPLE_SIZE * CALLS:,} simulated shots", flush=True)
        decoder.load_model.cache_clear()
    with tempfile.TemporaryDirectory(prefix=".readout-", dir=CHECKPOINT.parent) as temp:
        package = Path(temp) / READOUT
        package.mkdir()
        for name in parent_provenance["files"]:
            if name == "model.json":
                continue
            source = ROOT / "temporal_decoder" / name if name == "decoder.py" else incumbent / name
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(ROOT / "temporal_decoder/readout.py", package / "readout.py")
        np.savez(package / "readout.npz", **parameters)
        model = json.loads((incumbent / "model.json").read_text())
        model["format_version"] = 7
        model["assets"] = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()}
        write_json(package / "model.json", model)
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if elapsed > 1200 or rss > 5 * 1024**3:
            raise RuntimeError("Readout build exceeded the experiment budget")
        write_json(package / "provenance.json", {
            "candidate": READOUT, "files": {**model["assets"], "model.json": sha256(package / "model.json")},
            "build_seconds": elapsed, "build_peak_rss_bytes": rss,
            "parent_provenance_sha256": sha256(incumbent / "provenance.json"),
            "design_sha256": sha256(ROOT / "docs/AMBIGUOUS_READOUT_EXPERIMENT.md"),
            "fit_source_sha256": sha256(Path(__file__)), "regularization": REGULARIZATION,
            "stim_version": stim.__version__, "numpy_version": np.__version__, "training": training,
        })
        package.rename(destination)
    print(f"Built {READOUT} in {elapsed:.1f}s with peak RSS {rss / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
