"""Circuit-selected parity decoder for the two pinned memory circuits."""

from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import stim


FORMAT_VERSION = 1
BATCH_SIZE = 4096
ROUND_SIZES = (7, 14, 14, 14, 14, 7)
DENOMINATOR = 28
SCORE_SCALE = 2**24


def circuit_digest(circuit):
    return hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()


def pack_traces(events):
    padded = np.zeros((len(events), 128), dtype=np.uint8)
    padded[:, :70] = events
    return np.packbits(padded, axis=1, bitorder="little").view("<u8")


def round_numerators(events, groups):
    return np.column_stack([
        events[:, groups == time].sum(axis=1, dtype=np.int64) * (DENOMINATOR // size)
        for time, size in enumerate(ROUND_SIZES)
    ])


def feature_numerators(words, occupancy, masks, times):
    low = np.bitwise_count(words[:, 0, None] & masks[None, :, 0])
    high = np.bitwise_count(words[:, 1, None] & masks[None, :, 1])
    parity = 1 - 2 * ((low + high) & 1).astype(np.int64)
    # A one-slice mask stores its time twice, so both cases share the same average.
    local = (occupancy[:, times[:, 0]] + occupancy[:, times[:, 1]]) // 2
    return np.concatenate((occupancy, DENOMINATOR * parity, parity * local), axis=1)


def check_score_bound(bias, coefficients):
    bound = abs(int(bias)) + DENOMINATOR * sum(abs(int(value)) for value in coefficients)
    if bound >= 2**62:
        raise ValueError("Model integer score exceeds the supported bound")


def _read_head(arrays, prefix, groups):
    masks = arrays[f"{prefix}_masks"]
    times = arrays[f"{prefix}_times"]
    coefficients = arrays[f"{prefix}_coefficients"]
    bias = arrays[f"{prefix}_bias"]
    if masks.dtype.kind != "u" or masks.dtype.itemsize != 8 or masks.ndim != 2:
        raise ValueError(f"Invalid parity masks for {prefix}")
    if masks.shape[1] != 2 or not 1 <= len(masks) <= 64 or np.any(masks[:, 1] >> 6):
        raise ValueError(f"Invalid parity mask width or count for {prefix}")
    if times.shape != masks.shape or times.dtype.kind not in "iu":
        raise ValueError(f"Invalid mask times for {prefix}")
    if coefficients.shape != (6 + 2 * len(masks),) or coefficients.dtype != np.int64:
        raise ValueError(f"Invalid integer coefficients for {prefix}")
    if bias.shape != () or bias.dtype != np.int64:
        raise ValueError(f"Invalid integer bias for {prefix}")
    for mask, expected_times in zip(masks, times, strict=True):
        support = [i for i in range(70) if int(mask[i // 64]) & (1 << (i % 64))]
        touched = sorted(set(groups[support]))
        if not touched or touched[-1] - touched[0] > 1 or list(expected_times) != [touched[0], touched[-1]]:
            raise ValueError(f"Mask time slices disagree with its detector support for {prefix}")
    check_score_bound(bias, coefficients)
    return masks, times, coefficients, int(bias)


@lru_cache(maxsize=1)
def load_model(directory):
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        raw = (directory / "model.npz").read_bytes()
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported model format version")
        if hashlib.sha256(raw).hexdigest() != metadata["array_sha256"]:
            raise ValueError("Model array checksum mismatch")
        circuits = metadata["canonical_circuit_sha256"]
        if set(circuits) != {"X", "Z"} or len(set(circuits.values())) != 2:
            raise ValueError("Expected two distinct circuit fingerprints")
        models = {}
        with np.load(BytesIO(raw), allow_pickle=False) as arrays:
            for basis, fingerprint in circuits.items():
                groups = arrays[f"{basis}_groups"]
                if (groups.shape != (70,) or groups.dtype.kind not in "iu"
                        or np.any(groups < 0) or np.any(groups >= 6)
                        or tuple(np.bincount(groups.astype(int))) != ROUND_SIZES):
                    raise ValueError(f"Invalid detector time slices for {basis}")
                heads = tuple(_read_head(arrays, f"{basis}{bit}", groups) for bit in range(4))
                models[fingerprint] = groups, heads
        return models
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot load parity decoder assets from {directory}: {exc}") from exc


def decode(circuit, detector_events):
    """Return uint8 logical bits in the original shot order."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError("circuit must be a Stim Circuit")
    events = np.asarray(detector_events)
    if events.ndim != 2 or events.shape[1] != 70:
        raise ValueError("detector_events must have shape (shots, 70)")
    if events.dtype.kind not in "biu" or np.any((events != 0) & (events != 1)):
        raise ValueError("detector_events must contain binary integers or booleans")
    model = load_model(Path(__file__).resolve().parent).get(circuit_digest(circuit))
    if model is None:
        raise ValueError("Circuit fingerprint is unfamiliar to this checkpoint")
    groups, heads = model
    output = np.empty((len(events), 4), dtype=np.uint8)
    for start in range(0, len(events), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(events))
        chunk = events[start:stop]
        words = pack_traces(chunk)
        occupancy = round_numerators(chunk, groups)
        for bit, (masks, times, coefficients, bias) in enumerate(heads):
            features = feature_numerators(words, occupancy, masks, times)
            output[start:stop, bit] = features @ coefficients + bias > 0
    return output
