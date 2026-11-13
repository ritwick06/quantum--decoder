"""Compute Bayes LER with an oracle revealing blockwise detector parities."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage01_support import load_circuits, problem_info, write_json
from tools.build_temporal import projected_masks

import numpy as np


def load_bound_kernel(destination):
    library = destination / "oracle.so"
    source = ROOT / "temporal_decoder/spectral.cpp"
    subprocess.run(
        ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-march=x86-64-v2",
         "-fopenmp", str(source), "-o", str(library)],
        check=True,
    )
    oracle = ctypes.CDLL(str(library)).temporal_oracle_signs
    pointer = ctypes.c_void_p
    oracle.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, pointer]
    oracle.restype = ctypes.c_int
    return oracle


def load_subset_kernel(destination):
    library = destination / "subsets.so"
    subprocess.run(
        ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-march=x86-64-v2",
         str(ROOT / "temporal_decoder/kernel.cpp"), "-o", str(library)],
        check=True,
    )
    minorities = ctypes.CDLL(str(library)).temporal_block_minorities
    pointer = ctypes.c_void_p
    minorities.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_uint, pointer]
    minorities.restype = ctypes.c_int

    def bound(tags, probabilities, count, in_bits, out_bits, output):
        error = np.empty(4, dtype=np.float64)
        status = minorities(tags, probabilities, count, in_bits, error.ctypes.data)
        if status == 0:
            # Conservative rounding allowance; all aggregated terms are nonnegative.
            signs = 1 - 2 * np.maximum(0, error - 1e-8)
            ctypes.memmove(output, signs.ctypes.data, signs.nbytes)
        return status

    return bound


def block_oracle(circuit, oracle, cache, *, reverse, reveal_middle=True):
    data = problem_info(circuit)
    h, logical, p = data["H"], data["L"], data["p"]
    times = np.array([int(circuit.get_detector_coordinates()[i][-1]) for i in range(70)])
    if reverse:
        times = 5 - times
    earliest = np.where(h != 0, times[:, None], 6).min(axis=0)
    if (earliest == 6).any():
        raise ValueError("A mechanism has no detector support")
    means = []
    for t in range(5):
        selected = earliest >= t if t == 4 else earliest == t
        in_checks = np.flatnonzero(times == t)
        out_checks = np.flatnonzero(times == t + 1)
        if not reveal_middle and t in (1, 2, 3):
            continue
        if h[:, selected].sum() != h[np.r_[in_checks, out_checks]][:, selected].sum():
            raise ValueError("A block has detector support outside its parity masks")
        probabilities = np.ascontiguousarray(p[selected])
        incoming = projected_masks(h[:, selected], in_checks)
        outgoing = projected_masks(h[:, selected], out_checks)
        labels = projected_masks(logical[:, selected], list(range(4)))
        tags = np.ascontiguousarray((incoming << 18) | (outgoing << 4) | labels, dtype=np.uint32)
        key = (len(in_checks), len(out_checks), tags.tobytes(), probabilities.tobytes())
        if key not in cache:
            result = np.empty(4, dtype=np.float64)
            status = oracle(tags.ctypes.data, probabilities.ctypes.data, len(tags),
                            len(in_checks), len(out_checks), result.ctypes.data)
            if status:
                raise RuntimeError(f"Block oracle failed with status {status}")
            cache[key] = result
        means.append(cache[key])
        print(f"  block {t}, {len(tags)} mechanisms: {cache[key]}", flush=True)
    signs = np.asarray(means)
    return {"signs": signs.tolist(), "bit_ler": ((1 - signs.prod(axis=0)) / 2).tolist()}


def combine_bounds(boundary_path, subsets_path):
    boundary_bytes = boundary_path.read_bytes()
    subset_bytes = subsets_path.read_bytes()
    boundary, subset = map(json.loads, (boundary_bytes, subset_bytes))
    if not boundary["boundary_only"] or subset["boundary_only"] or subset["method"] != "three_fault_subset":
        raise ValueError("Expected boundary-only and full three-fault reports")

    bases = {}
    for basis in "XZ":
        ends = boundary["basis"][basis]
        blocks = subset["basis"][basis]
        if ends["circuit_sha256"] != blocks["circuit_sha256"]:
            raise ValueError(f"Circuit fingerprints differ for {basis}")
        directions = {}
        for direction in ("forward", "reverse"):
            exact = np.asarray(ends["directions"][direction]["signs"])
            partial = np.asarray(blocks["directions"][direction]["signs"])[1:4]
            if exact.shape != (2, 4) or partial.shape != (3, 4):
                raise ValueError(f"Unexpected block dimensions for {basis} {direction}")
            exact = np.minimum(1, exact + 2e-8)
            directions[direction] = ((1 - np.concatenate((exact, partial)).prod(axis=0)) / 2).tolist()
        bit_ler = np.maximum(directions["forward"], directions["reverse"])
        bases[basis] = {"directions": directions, "bit_ler_lower_bound": bit_ler.tolist(),
                        "mean_ler_lower_bound": float(bit_ler.mean())}
    return {
        "method": "exact_boundary_plus_three_fault_interior",
        "input_sha256": {path.name: sha256(content).hexdigest()
                         for path, content in ((boundary_path, boundary_bytes), (subsets_path, subset_bytes))},
        "basis": bases,
        "combined_ler_lower_bound": float(np.mean([v["mean_ler_lower_bound"] for v in bases.values()])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--subsets", action="store_true", help="Use only subsets of up to three faults for a conservative check")
    parser.add_argument("--combine", type=Path, nargs=2, metavar=("BOUNDARY", "SUBSETS"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.combine is not None:
        if args.boundary_only or args.subsets:
            parser.error("--combine reads existing reports and cannot be combined with calculation flags")
        result = combine_bounds(*args.combine)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    bounds = {}
    with tempfile.TemporaryDirectory(prefix="qec-oracle-") as temp:
        loader = load_subset_kernel if args.subsets else load_bound_kernel
        oracle = loader(Path(temp))
        cache = {}
        for basis, circuit in load_circuits().items():
            directions = {}
            for direction in ("forward", "reverse"):
                print(f"{basis} {direction}", flush=True)
                directions[direction] = block_oracle(circuit, oracle, cache, reverse=direction == "reverse",
                                                     reveal_middle=not args.boundary_only)
            bit_ler = np.maximum(directions["forward"]["bit_ler"], directions["reverse"]["bit_ler"])
            bounds[basis] = {"directions": directions, "bit_ler_lower_bound": bit_ler.tolist(),
                             "mean_ler_lower_bound": float(bit_ler.mean()),
                             "circuit_sha256": sha256(str(circuit).encode()).hexdigest()}
    result = {"basis": bounds, "boundary_only": args.boundary_only,
              "method": "three_fault_subset" if args.subsets else "walsh_hadamard",
              "combined_ler_lower_bound": float(np.mean([v["mean_ler_lower_bound"] for v in bounds.values()]))}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
