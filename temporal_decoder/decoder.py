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
    return library


def circuit_digest(circuit):
    return hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_model():
    try:
        metadata = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
        if metadata["format_version"] != 1:
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
            if tuple(map(len, groups)) != ROUND_SIZES or sorted(sum(groups, [])) != list(range(70)):
                raise ValueError(f"Invalid detector groups for {basis}")
            tables = tuple(np.load(HERE / f"{basis}_edges_{t}.npy", mmap_mode="r", allow_pickle=False) for t in range(5))
            marginals = tuple(np.load(HERE / f"{basis}_marginal_{t}.npy", mmap_mode="r", allow_pickle=False) for t in range(1, 6))
            terminal = np.load(HERE / f"{basis}_terminal.npy", mmap_mode="r", allow_pickle=False)
            for table, size in zip(tables, ROUND_SIZES, strict=False):
                if table.shape != (1 << size, FANOUT) or table.dtype != EDGE:
                    raise ValueError(f"Invalid edge table for {basis}")
            for marginal, size in zip(marginals, ROUND_SIZES[1:], strict=True):
                if marginal.shape != (1 << size,) or marginal.dtype != np.float64:
                    raise ValueError(f"Invalid marginal table for {basis}")
            if terminal.shape != (128, 5) or terminal.dtype != np.float64:
                raise ValueError(f"Invalid terminal table for {basis}")
            models[fingerprint] = groups, tables, marginals, terminal
        return kernel, fallback.decode, models
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"Cannot load temporal checkpoint assets from {HERE}: {exc}") from exc


def observed_slices(events, groups):
    observed = np.empty((len(events), 6), dtype=np.uint16)
    for time, indexes in enumerate(groups):
        powers = np.left_shift(np.uint16(1), np.arange(len(indexes), dtype=np.uint16))
        observed[:, time] = events[:, indexes].astype(np.uint16) @ powers
    return observed


def _decode_temporal(events, model, kernel):
    groups, tables, marginals, terminal = model
    observed = observed_slices(events, groups)
    table_pointers = (ctypes.c_void_p * 5)(*(table.ctypes.data for table in tables))
    marginal_pointers = (ctypes.c_void_p * 5)(*(marginal.ctypes.data for marginal in marginals))
    result = np.empty((len(events), 4), dtype=np.int8)

    def run(start, stop):
        status = kernel.temporal_decode(observed[start:].ctypes.data, stop - start,
                                        table_pointers, marginal_pointers, terminal.ctypes.data,
                                        result[start:].ctypes.data)
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
    result = _decode_temporal(events, model, kernel)
    unknown = result < 0
    rows = unknown.any(axis=1)
    if rows.any():
        result[rows] = np.where(unknown[rows], fallback(circuit, events[rows]), result[rows])
    return result.astype(np.uint8)
