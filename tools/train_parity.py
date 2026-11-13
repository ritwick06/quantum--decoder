"""Fit the single approved Stage 03B configuration on development rows."""

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import stim

from stage01_support import DETECTORS, LABELS, load_circuits, problem_info, sha256, write_json
from team_decoder.decoder import (
    BATCH_SIZE, DENOMINATOR, FORMAT_VERSION, ROUND_SIZES, SCORE_SCALE,
    check_score_bound, circuit_digest, feature_numerators, pack_traces, round_numerators,
)


CANDIDATE = "parity_v1"
CHECKPOINT = ROOT / "artifacts/checkpoints" / CANDIDATE
L2 = 0.001
TRAIN_SECONDS = 1200
TRAIN_RSS_BYTES = 8 * 1024**3


def development_rows(path, reserved, expected_counts):
    rows = {basis: bytearray() for basis in "XZ"}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader) != ["id", "basis", *DETECTORS, *LABELS]:
            raise ValueError("Unexpected labelled development CSV header")
        for row, is_reserved in zip(reader, reserved, strict=True):
            if is_reserved:
                continue
            if len(row) != 76 or row[1] not in rows or any(v not in ("0", "1") for v in row[2:]):
                raise ValueError("Invalid development row")
            rows[row[1]].extend("".join(row[2:]).encode("ascii"))
    result = {basis: np.frombuffer(raw, dtype=np.uint8).reshape(-1, 74) - 48
              for basis, raw in rows.items()}
    if {basis: len(data) for basis, data in result.items()} != expected_counts:
        raise ValueError("Development basis counts disagree with the audited split")
    return result


def parity_moments(detector_parities, logical_parities, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities >= 0.5)):
        raise ValueError("DEM probabilities must be finite and in [0, 0.5)")
    log_sign_mean = np.log1p(-2 * probabilities)
    detector_means = np.exp(detector_parities @ log_sign_mean)
    logical_means = np.exp(logical_parities @ log_sign_mean)
    joint_means = np.column_stack([
        np.exp((detector_parities ^ logical) @ log_sign_mean) for logical in logical_parities
    ])
    return detector_means, logical_means, joint_means


def rank_masks(circuit):
    info = problem_info(circuit)
    coordinates = [tuple(circuit.get_detector_coordinates()[index]) for index in range(70)]
    groups = np.array([coordinate[-1] for coordinate in coordinates], dtype=np.uint8)
    if tuple(np.bincount(groups)) != ROUND_SIZES:
        raise ValueError("Unexpected detector time slices")
    supports = [tuple(np.flatnonzero(column).tolist()) for column in info["H"].T]
    library = {(index,) for index in range(70)}
    for support in supports:
        if support:
            library.add(support)
            library.update(itertools.combinations(support, 2))
    for left, right in itertools.combinations(range(70), 2):
        if coordinates[left][:-1] == coordinates[right][:-1] and abs(int(groups[left]) - int(groups[right])) == 1:
            library.add((left, right))
    supports = sorted(library)
    indicators = np.zeros((len(supports), 70), dtype=np.uint8)
    for index, support in enumerate(supports):
        indicators[index, list(support)] = 1
    # Reduction modulo 256 from uint8 multiplication preserves parity.
    parities = (indicators @ info["H"]) & 1
    detector, logical, joint = parity_moments(parities, info["L"], info["p"])
    denominator = np.sqrt((1 - detector[:, None] ** 2) * (1 - logical ** 2))
    masks = pack_traces(indicators)
    times = np.array([[min(groups[list(s)]), max(groups[list(s)])] for s in supports], dtype=np.uint8)
    if np.any(times[:, 1] - times[:, 0] > 1):
        raise ValueError("Mask spans more than two consecutive time slices")
    selected = []
    for bit in range(4):
        valid = np.flatnonzero(denominator[:, bit] > 1e-12)
        scores = (joint[valid, bit] - detector[valid] * logical[bit]) / denominator[valid, bit]
        ranked = sorted(zip(valid, scores), key=lambda row: (-round(abs(float(row[1])), 12), supports[row[0]]))[:64]
        if not ranked:
            raise ValueError(f"No valid detector masks for bit {bit}")
        indexes = [index for index, _ in ranked]
        selected.append((masks[indexes], times[indexes], [float(score) for _, score in ranked]))
    return groups, selected, len(supports)


def objective(design, target, weights, l2=L2):
    logits = design @ weights
    return float(np.mean(np.logaddexp(0, logits) - target * logits) + 0.5 * l2 * (weights[1:] @ weights[1:]))


def fit_head(design, target, l2=L2):
    count, width = design.shape
    rate = float(target.mean())
    if not 0 < rate < 1:
        raise ValueError("Cannot fit a constant target")
    weights = np.zeros(width, dtype=np.float64)
    weights[0] = np.log(rate / (1 - rate))
    loss = objective(design, target, weights, l2)
    initial_loss = loss
    for iteration in range(31):
        logits = design @ weights
        probability = np.exp(-np.logaddexp(0, -logits))
        gradient = design.T @ (probability - target) / count
        gradient[1:] += l2 * weights[1:]
        norm = float(np.max(np.abs(gradient)))
        if not np.isfinite([norm, loss]).all():
            raise ValueError("Nonfinite logistic optimizer state")
        if norm <= 1e-6:
            return weights, {"steps": iteration, "gradient_inf": norm, "initial_loss": initial_loss, "loss": loss}
        if iteration == 30:
            raise ValueError(f"Logistic fit did not converge: gradient norm {norm}")
        hessian = design.T @ (design * (probability * (1 - probability))[:, None]) / count
        hessian.flat[width + 1::width + 1] += l2
        step = np.linalg.solve(hessian, gradient)
        descent = float(gradient @ step)
        if not np.isfinite(descent) or descent <= 0:
            raise ValueError("Invalid Newton descent direction")
        for halving in range(21):
            length = 2.0**-halving
            trial = weights - length * step
            trial_loss = objective(design, target, trial, l2)
            if np.isfinite(trial_loss) and trial_loss <= loss - 1e-4 * length * descent:
                weights, loss = trial, trial_loss
                break
        else:
            raise ValueError("Newton line search failed")


def export_head(weights, mean, scale):
    raw_coefficients = weights[1:] / scale
    raw_bias = weights[0] - mean @ raw_coefficients
    bias = round(float(SCORE_SCALE * raw_bias))
    coefficients = [round(float(SCORE_SCALE * value / DENOMINATOR)) for value in raw_coefficients]
    check_score_bound(bias, coefficients)
    return np.array(bias, dtype=np.int64), np.array(coefficients, dtype=np.int64)


def train():
    start = time.perf_counter()
    if CHECKPOINT.exists():
        raise FileExistsError(CHECKPOINT)
    if sys.version_info[:2] != (3, 13) or np.__version__ != "2.2.6" or stim.__version__ != "1.16.0":
        raise RuntimeError("Training requires the pinned Python 3.13, NumPy 2.2.6 and Stim 1.16.0 environment")
    report = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
    train_path = ROOT / "quantum-error-correction-decoder/train.csv"
    split_path = ROOT / "data/splits/train_reserved.npy"
    inputs = {str(train_path.relative_to(ROOT)): report["split"]["train_sha256"],
              str(split_path.relative_to(ROOT)): report["split"]["mask_sha256"]}
    for basis in "XZ":
        inputs[f"quantum-error-correction-decoder/{basis}_circuit.stim"] = report["circuits"][basis]["circuit_sha256"]
    for path, expected in inputs.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Audited input hash mismatch: {path}")
    reserved = np.load(split_path, allow_pickle=False)
    if reserved.dtype != np.bool_ or reserved.shape != (200000,):
        raise ValueError("Invalid reserved row mask")
    counts = {basis: report["split"]["counts"][basis]["development"] for basis in "XZ"}
    development = development_rows(train_path, reserved, counts)
    arrays, diagnostics, fingerprints = {}, {}, {}
    for basis, circuit in load_circuits().items():
        fingerprint = circuit_digest(circuit)
        if fingerprint != report["circuits"][basis]["canonical_circuit_sha256"]:
            raise ValueError(f"Canonical circuit hash mismatch: {basis}")
        fingerprints[basis] = fingerprint
        groups, selected, library_size = rank_masks(circuit)
        arrays[f"{basis}_groups"] = groups
        data = development[basis]
        events = data[:, :70]
        words, occupancy = pack_traces(events), round_numerators(events, groups)
        for bit, (masks, times, correlations) in enumerate(selected):
            prefix = f"{basis}{bit}"
            design = np.ones((len(data), 7 + 2 * len(masks)), dtype=np.float64)
            raw = design[:, 1:]
            for lo in range(0, len(data), BATCH_SIZE):
                hi = min(lo + BATCH_SIZE, len(data))
                raw[lo:hi] = feature_numerators(words[lo:hi], occupancy[lo:hi], masks, times) / DENOMINATOR
            mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 0.10)
            raw -= mean
            raw /= scale
            weights, fit = fit_head(design, data[:, 70 + bit])
            bias, coefficients = export_head(weights, mean, scale)
            arrays[f"{prefix}_masks"] = masks
            arrays[f"{prefix}_times"] = times
            arrays[f"{prefix}_bias"] = bias
            arrays[f"{prefix}_coefficients"] = coefficients
            diagnostics[prefix] = {**fit, "library_size": library_size, "masks": len(masks), "correlations": correlations}
            print(f"{prefix}: {fit['steps']} Newton steps, gradient {fit['gradient_inf']:.3g}", flush=True)
            del design, raw
    for path, expected in inputs.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Input changed during training: {path}")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CHECKPOINT.parent, prefix=".parity-") as staging:
        destination = Path(staging) / CANDIDATE
        destination.mkdir()
        for source, name in ((ROOT / "team_decoder/decoder.py", "decoder.py"),
                             (ROOT / "requirements.txt", "requirements.txt")):
            shutil.copyfile(source, destination / name)
        np.savez(destination / "model.npz", **arrays)
        write_json(destination / "model.json", {"format_version": FORMAT_VERSION,
                   "canonical_circuit_sha256": fingerprints, "array_sha256": sha256(destination / "model.npz")})
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if elapsed > TRAIN_SECONDS or rss > TRAIN_RSS_BYTES:
            raise RuntimeError("Training exceeded its approved resource budget")
        files = {path.name: sha256(path) for path in sorted(destination.iterdir())}
        write_json(destination / "provenance.json", {
            "candidate": CANDIDATE, "files": files, "input_sha256": inputs,
            "source_sha256": {path: sha256(ROOT / path) for path in (
                "tools/train_parity.py", "team_decoder/decoder.py", "docs/DESIGN_DECISIONS.md")},
            "organizer_revision": report["source_revision"],
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "stim": stim.__version__},
            "development_counts": counts, "training_seconds": elapsed, "training_peak_rss_bytes": rss,
            "heads": diagnostics, "training_seed": None, "l2": L2,
            "attribution": "Project-specific decoder and training code developed with AI assistance. Stim and the pinned organizer helper supply circuit metadata.",
        })
        if CHECKPOINT.exists():
            raise FileExistsError(CHECKPOINT)
        destination.rename(CHECKPOINT)
    print(f"Frozen checkpoint {CHECKPOINT}: {elapsed:.3f} s, peak RSS {rss / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    train()
