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
EXPANDED_FANOUT = 96
DEEP_FANOUT = 192
WORKERS = 4
EDGE = np.dtype([("outgoing", "<u4"), ("mass", "<f4", 5)])
BACK_EDGE = np.dtype([("incoming", "<u4"), ("mass", "<f4")])


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
    for name in ("temporal_moments", "temporal_midpoint", "temporal_wide_moments",
                 "temporal_suffix_moments", "temporal_expanded_moments", "temporal_deep_moments"):
        infer = getattr(library, name, None)
        if infer is not None:
            infer.argtypes = library.temporal_decode.argtypes
            infer.restype = ctypes.c_int
    lookahead = getattr(library, "temporal_lookahead_moments", None)
    if lookahead is not None:
        lookahead.argtypes = library.temporal_decode.argtypes + [pointer, pointer]
        lookahead.restype = ctypes.c_int
    backward = getattr(library, "temporal_backward_moments", None)
    if backward is not None:
        backward.argtypes = library.temporal_decode.argtypes + [pointer] * 3
        backward.restype = ctypes.c_int
    build_width = getattr(library, "temporal_build_width", None)
    if build_width is not None:
        build_width.argtypes = library.temporal_build.argtypes + [ctypes.c_uint]
        build_width.restype = ctypes.c_int
    return library


def circuit_digest(circuit):
    return hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_model():
    try:
        metadata = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
        version = metadata["format_version"]
        if version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
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
            if version in (5, 6, 7, 8, 9, 10, 11, 12):
                directions = (
                    _load_suffix(basis, groups, lookahead=version == 6),
                    _load_suffix(f"{basis}_reverse", groups[::-1], lookahead=version == 6),
                )
                if version in (8, 9, 10, 11, 12):
                    width = DEEP_FANOUT if version == 10 else EXPANDED_FANOUT
                    directions += (_load_suffix(basis, groups, width=width, backward=version == 9),
                                   _load_suffix(f"{basis}_reverse", groups[::-1], width=width, backward=version == 9))
                if version == 11:
                    directions += (_load_suffix(f"{basis}_shifted", groups),
                                   _load_suffix(f"{basis}_reverse_shifted", groups[::-1]))
                models[fingerprint] = version, directions
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


def _load_suffix(prefix, groups, *, lookahead=False, width=FANOUT, backward=False):
    _check_groups(groups)
    edge_name = f"{prefix}_expanded_edges" if width != FANOUT else f"{prefix}_edges"
    tables = tuple(_load_array(f"{edge_name}_{t}", (1 << size, width), EDGE)
                   for t, size in enumerate(ROUND_SIZES[:4]))
    marginals = tuple(_load_array(f"{prefix}_marginal_{t}", (1 << size,), np.float64)
                      for t, size in enumerate(ROUND_SIZES[1:4], start=1))
    marginals += (_load_array(f"{prefix}_suffix_mass", (1 << 21,), np.float64),)
    terminal = _load_array(f"{prefix}_suffix", (1 << 21, 5), np.float32)
    if backward:
        offsets = _load_array(f"{prefix}_back_offsets", ((1 << 14) + 1,), np.uint32)
        edges = _load_array(f"{prefix}_back_edges", (int(offsets[-1]),), BACK_EDGE)
        residuals = _load_array(f"{prefix}_back_residuals", (128, 65), np.uint16)
        if (offsets[0] != 0 or (offsets[1:] < offsets[:-1]).any()
                or (edges["incoming"] >= (1 << 14)).any() or (residuals >= (1 << 14)).any()):
            raise ValueError(f"Invalid backward likelihood index for {prefix}")
        return groups, tables, marginals, terminal, offsets, edges, residuals
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


def _decode_temporal(events, model, kernel, *, return_moments=False, midpoint=False,
                     wide=False, suffix=False, expanded=False, backward=False, deep=False):
    groups, tables, marginals, terminal, *assets = model
    observed = observed_slices(events, groups)
    table_pointers = (ctypes.c_void_p * len(tables))(*(table.ctypes.data for table in tables))
    marginal_pointers = (ctypes.c_void_p * len(marginals))(*(marginal.ctypes.data for marginal in marginals))
    if return_moments or midpoint or wide or suffix or expanded or backward or deep:
        result = np.empty((len(events), 5), dtype=np.float64)
        if midpoint:
            infer = kernel.temporal_midpoint
        elif wide:
            infer = kernel.temporal_wide_moments
        elif backward:
            infer = kernel.temporal_backward_moments
        elif assets:
            infer = kernel.temporal_lookahead_moments
        elif deep:
            infer = kernel.temporal_deep_moments
        elif expanded:
            infer = kernel.temporal_expanded_moments
        elif suffix:
            infer = kernel.temporal_suffix_moments
        else:
            infer = kernel.temporal_moments
    else:
        result = np.empty((len(events), 4), dtype=np.int8)
        infer = kernel.temporal_decode

    if backward:
        extra = [array.ctypes.data for array in assets]
    else:
        extra = [(ctypes.c_void_p * 3)(*(array.ctypes.data for array in arrays)) for arrays in assets]

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


@lru_cache(maxsize=2)
def _load_readout(fingerprint):
    metadata = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
    name = "readout.npz"
    if digest(HERE / name) != metadata["assets"][name]:
        raise ValueError("Readout asset checksum mismatch")
    spec = importlib.util.spec_from_file_location("temporal_readout", HERE / "readout.py")
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    with np.load(HERE / name, allow_pickle=False) as weights:
        coefficients = weights[fingerprint]
    if coefficients.shape != (4, 27) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid readout coefficients")
    return features, coefficients


def _calibrated_readout(circuit, events, groups, fallback, forward, reverse, initial):
    features, weights = _load_readout(circuit_digest(circuit))
    prepared = features.direction_features(forward, reverse)
    uncertain = prepared[-1]
    active = uncertain.any(axis=1)
    if not active.any():
        return initial
    parity = fallback(circuit, events[active])
    compact = tuple(array[active] for array in prepared)
    counts = np.column_stack([events[active][:, group].sum(axis=1) for group in groups])
    replacement = initial[active].copy()
    for bit in range(4):
        rows = compact[-1][:, bit]
        if rows.any():
            design = features.feature_columns(compact, counts, parity, rows, bit)
            # Accumulate in fixed feature order across all batch sizes.
            logits = weights[bit, 0] + (design * weights[bit, 1:]).sum(axis=1)
            replacement[rows, bit] = np.where(np.abs(logits) > 0.08, logits > 0, replacement[rows, bit])
    initial[active] = replacement
    return initial


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
    elif version in (5, 6, 7, 8, 9, 10, 11, 12):
        forward, reverse = directions[:2]
        a = _decode_temporal(events, forward, kernel, suffix=True)
        b = _decode_temporal(events, reverse, kernel, suffix=True)
        result = logical_decisions(a + b)
        if version == 7:
            result = _calibrated_readout(circuit, events, forward[0], fallback, a, b, result)
        elif version in (8, 9, 10, 11, 12):
            total = a + b
            threshold = 0.85 if version == 12 else 0.4
            retry = (np.abs(total[:, 1:]) <= threshold * total[:, :1]) | ((a[:, 1:] < 0) != (b[:, 1:] < 0))
            rows = retry.any(axis=1)
            if rows.any():
                selected = events[rows]
                retry_mode = {"expanded": True, "backward": version == 9, "deep": version == 10}
                refined = (_decode_temporal(selected, directions[2], kernel, **retry_mode)
                           + _decode_temporal(selected, directions[3], kernel, **retry_mode))
                if version == 11:
                    refined += (_decode_temporal(selected, directions[4], kernel, suffix=True)
                                + _decode_temporal(selected, directions[5], kernel, suffix=True))
                result[rows] = np.where(retry[rows], logical_decisions(refined), result[rows])
    else:
        result = _bidirectional(events, directions, kernel, adaptive=version == 4)
    unknown = result < 0
    rows = unknown.any(axis=1)
    if rows.any():
        result[rows] = np.where(unknown[rows], fallback(circuit, events[rows]), result[rows])
    return result.astype(np.uint8)
