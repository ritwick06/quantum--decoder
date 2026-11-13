"""Temporal likelihood decoder with the original parity model as fallback."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import stim


HERE = Path(__file__).resolve().parent
ROUND_SIZES = (7, 14, 14, 14, 14, 7)
FANOUT = 48
WORKERS = 4
EDGE = np.dtype([("outgoing", "<u4"), ("mass", "<f4", 5)])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_kernel(path):
    library = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    library.temporal_build.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int, pointer]
    library.temporal_build.restype = ctypes.c_int
    library.temporal_decode.argtypes = [pointer, ctypes.c_size_t, pointer, pointer, pointer, pointer]
    library.temporal_decode.restype = ctypes.c_int
    for name in ("temporal_moments", "temporal_midpoint", "temporal_wide_moments", "temporal_suffix_moments"):
        infer = getattr(library, name, None)
        if infer is not None:
            infer.argtypes = library.temporal_decode.argtypes
            infer.restype = ctypes.c_int
    lookahead = getattr(library, "temporal_lookahead_moments", None)
    if lookahead is not None:
        lookahead.argtypes = library.temporal_decode.argtypes + [pointer, pointer]
        lookahead.restype = ctypes.c_int
    return library


def circuit_digest(circuit):
    return hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_model():
    try:
        metadata = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
        version = metadata["format_version"]
        if version not in (1, 2, 3, 4, 5, 6):
            raise ValueError("Unsupported temporal model format")
        for name, expected in metadata["assets"].items():
            if digest(HERE / name) != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")
        kernel = load_kernel(HERE / "kernel.so")
        fallback_path = HERE / "parity_fallback/decoder.py"
        spec = importlib.util.spec_from_file_location("temporal_parity_fallback", fallback_path)
        fallback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fallback)
        models = {}
        for basis, fingerprint in metadata["canonical_circuit_sha256"].items():
            groups = metadata["detector_groups"][basis]
            if version == 3:
                models[fingerprint] = version, (_load_midpoint(basis, groups),)
                continue
            if version in (5, 6):
                models[fingerprint] = version, (
                    _load_suffix(basis, groups, lookahead=version == 6),
                    _load_suffix(f"{basis}_reverse", groups[::-1], lookahead=version == 6),
                )
                continue
            directions = [_load_direction(basis, groups)]
            if version in (2, 4):
                directions.append(_load_direction(f"{basis}_reverse", groups[::-1]))
            models[fingerprint] = version, tuple(directions)
        return kernel, fallback.decode, models
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"Cannot load temporal checkpoint assets from {HERE}: {exc}") from exc


def _check_groups(groups):
    if tuple(map(len, groups)) != ROUND_SIZES or sorted(sum(groups, [])) != list(range(70)):
        raise ValueError("Invalid temporal detector groups")


def _load_array(name, shape, dtype):
    array = np.load(HERE / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"Invalid temporal array: {name}")
    return array


def _load_direction(prefix, groups):
    _check_groups(groups)
    tables = tuple(_load_array(f"{prefix}_edges_{t}", (1 << size, FANOUT), EDGE)
                   for t, size in enumerate(ROUND_SIZES[:-1]))
    marginals = tuple(_load_array(f"{prefix}_marginal_{t}", (1 << size,), np.float64)
                      for t, size in enumerate(ROUND_SIZES[1:], start=1))
    terminal = _load_array(f"{prefix}_terminal", (128, 5), np.float64)
    return groups, tables, marginals, terminal


def _load_midpoint(basis, groups):
    _check_groups(groups)
    reverse = f"{basis}_reverse"
    steps = ((basis, 0), (basis, 1), (reverse, 0), (reverse, 1), (reverse, 2))
    tables = tuple(_load_array(f"{prefix}_edges_{t}", (1 << ROUND_SIZES[t], FANOUT), EDGE)
                   for prefix, t in steps)
    names = (f"{basis}_marginal_1", f"{basis}_middle_right", f"{reverse}_marginal_1",
             f"{reverse}_marginal_2", f"{basis}_middle_left")
    marginals = tuple(_load_array(name, (1 << 14,), np.float64) for name in names)
    middle = _load_array(f"{basis}_middle", (1 << 14, 5), np.float64)
    return groups, tables, marginals, middle


def _load_suffix(prefix, groups, *, lookahead=False):
    _check_groups(groups)
    tables = tuple(_load_array(f"{prefix}_edges_{t}", (1 << size, FANOUT), EDGE)
                   for t, size in enumerate(ROUND_SIZES[:4]))
    marginals = tuple(_load_array(f"{prefix}_marginal_{t}", (1 << size,), np.float64)
                      for t, size in enumerate(ROUND_SIZES[1:4], start=1))
    marginals += (_load_array(f"{prefix}_suffix_mass", (1 << 21,), np.float64),)
    terminal = _load_array(f"{prefix}_suffix", (1 << 21, 5), np.float32)
    if lookahead:
        joints = tuple(_load_array(f"{prefix}_lookahead_{t}", (128, 1 << 14), np.float32) for t in range(3))
        indexes = tuple(_load_array(f"{prefix}_lookahead_index_{t}", (1 << 14,), np.int16) for t in range(3))
        for index in indexes:
            if np.any(index < -1) or not np.array_equal(np.sort(index[index >= 0]), np.arange(128)):
                raise ValueError(f"Invalid joint parity indexes for {prefix}")
        return groups, tables, marginals, terminal, joints, indexes
    return groups, tables, marginals, terminal


def observed_slices(events, groups):
    observed = np.empty((len(events), 6), dtype=np.uint16)
    for time, indexes in enumerate(groups):
        powers = np.left_shift(np.uint16(1), np.arange(len(indexes), dtype=np.uint16))
        observed[:, time] = events[:, indexes].astype(np.uint16) @ powers
    return observed


def _decode_temporal(events, model, kernel, *, return_moments=False, midpoint=False, wide=False, suffix=False):
    groups, tables, marginals, terminal, *lookahead = model
    observed = observed_slices(events, groups)
    table_pointers = (ctypes.c_void_p * len(tables))(*(table.ctypes.data for table in tables))
    marginal_pointers = (ctypes.c_void_p * len(marginals))(*(marginal.ctypes.data for marginal in marginals))
    if return_moments or midpoint or wide or suffix:
        result = np.empty((len(events), 5), dtype=np.float64)
        if midpoint:
            infer = kernel.temporal_midpoint
        elif wide:
            infer = kernel.temporal_wide_moments
        elif lookahead:
            infer = kernel.temporal_lookahead_moments
        elif suffix:
            infer = kernel.temporal_suffix_moments
        else:
            infer = kernel.temporal_moments
    else:
        result = np.empty((len(events), 4), dtype=np.int8)
        infer = kernel.temporal_decode

    extra = [(ctypes.c_void_p * 3)(*(array.ctypes.data for array in arrays)) for arrays in lookahead]

    def run(start, stop):
        status = infer(observed[start:].ctypes.data, stop - start, table_pointers,
                       marginal_pointers, terminal.ctypes.data, *extra, result[start:].ctypes.data)
        if status:
            raise RuntimeError("Temporal decoding kernel failed")

    if len(events) < 4096:
        run(0, len(events))
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            jobs = [pool.submit(run, i * len(events) // WORKERS, (i + 1) * len(events) // WORKERS)
                    for i in range(WORKERS)]
            for job in jobs:
                job.result()
    return result


def logical_decisions(moments):
    mass = moments[:, :1]
    signed = moments[:, 1:]
    supported = (mass > 0) & (np.abs(signed) > 0.025 * mass)
    return np.where(supported, signed < 0, -1).astype(np.int8)


def _bidirectional(events, directions, kernel, *, adaptive):
    forward, reverse = directions
    narrow = (_decode_temporal(events, forward, kernel, return_moments=True)
              + _decode_temporal(events, reverse, kernel, return_moments=True))
    result = logical_decisions(narrow)
    if adaptive:
        retry = np.abs(narrow[:, 1:]) <= 0.45 * narrow[:, :1]
        rows = retry.any(axis=1)
        if rows.any():
            selected = events[rows]
            refined = (_decode_temporal(selected, forward, kernel, wide=True)
                       + _decode_temporal(selected, reverse, kernel, wide=True))
            result[rows] = np.where(retry[rows], logical_decisions(refined), result[rows])
    return result


def decode(circuit, detector_events):
    if not isinstance(circuit, stim.Circuit):
        raise TypeError("circuit must be a Stim Circuit")
    events = np.asarray(detector_events)
    if events.ndim != 2 or events.shape[1] != 70:
        raise ValueError("detector_events must have shape (shots, 70)")
    if events.dtype.kind not in "biu" or np.any((events != 0) & (events != 1)):
        raise ValueError("detector_events must contain binary integers or booleans")
    kernel, fallback, models = load_model()
    model = models.get(circuit_digest(circuit))
    if model is None:
        raise ValueError("Circuit fingerprint is unfamiliar to this checkpoint")
    if not len(events):
        return np.empty((0, 4), dtype=np.uint8)
    version, directions = model
    if version == 1:
        result = _decode_temporal(events, directions[0], kernel)
    elif version == 3:
        result = logical_decisions(_decode_temporal(events, directions[0], kernel, midpoint=True))
    elif version in (5, 6):
        forward, reverse = directions
        result = logical_decisions(_decode_temporal(events, forward, kernel, suffix=True)
                                   + _decode_temporal(events, reverse, kernel, suffix=True))
    else:
        result = _bidirectional(events, directions, kernel, adaptive=version == 4)
    unknown = result < 0
    rows = unknown.any(axis=1)
    if rows.any():
        result[rows] = np.where(unknown[rows], fallback(circuit, events[rows]), result[rows])
    return result.astype(np.uint8)
