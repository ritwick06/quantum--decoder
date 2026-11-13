"""Build temporal likelihood tables for either recorded experiment."""

import argparse
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import stim

from stage01_support import load_circuits, problem_info, sha256, write_json
from temporal_decoder.decoder import EDGE, EXPANDED_FANOUT, FANOUT, ROUND_SIZES, circuit_digest, load_kernel


CANDIDATE = "temporal_v1"
BIDIRECTIONAL = "temporal_bidir_v1"
MIDPOINT = "temporal_midpoint_v1"
ADAPTIVE = "temporal_adaptive_v1"
SPECTRAL = "temporal_spectral_v1"
SUFFIX = "temporal_suffix_v1"
LOOKAHEAD = "temporal_lookahead_v1"
READOUT = "temporal_readout_v1"
EXPANDED = "temporal_expanded_v1"
CHECKPOINT = ROOT / "artifacts/checkpoints" / CANDIDATE
FALLBACK = ROOT / "artifacts/checkpoints/parity_v1"


def xor_moments(syndromes, logical, probabilities, size):
    """Distribution of the XOR and four logical sign moments under independent faults."""
    values = np.zeros((size, 5), dtype=np.float64)
    values[0] = 1
    indexes = np.arange(size)
    for syndrome, label, probability in zip(syndromes, logical, probabilities, strict=True):
        signs = np.array([1, *[1 - 2 * ((int(label) >> bit) & 1) for bit in range(4)]])
        values = (1 - probability) * values + probability * values[indexes ^ int(syndrome)] * signs
    return values


def projected_masks(checks, indexes):
    shifts = np.left_shift(np.uint32(1), np.arange(len(indexes), dtype=np.uint32))
    return shifts @ checks[indexes].astype(np.uint32)


def generate_basis(path, basis, circuit, kernel, *, reverse=False, spectral=None, reuse=None, expanded=False):
    information = problem_info(circuit)
    checks, logicals, probabilities = information["H"], information["L"], information["p"]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities >= 0.5)):
        raise ValueError(f"Unexpected independent fault probabilities in {basis}")
    times = np.array([circuit.get_detector_coordinates()[i][-1] for i in range(70)], dtype=int)
    if reverse:
        times = 5 - times
    groups = [np.flatnonzero(times == t).tolist() for t in range(6)]
    if tuple(map(len, groups)) != ROUND_SIZES:
        raise ValueError(f"Unexpected detector time slices in {basis}")
    earliest = np.min(np.where(checks != 0, times[:, None], 6), axis=0)
    latest = np.max(np.where(checks != 0, times[:, None], -1), axis=0)
    if np.any(earliest > latest) or np.any(latest - earliest > 1):
        raise ValueError(f"Fault mechanism is not supported on one or two neighboring slices in {basis}")
    logical = projected_masks(logicals, list(range(4)))
    masses = []
    for t, indexes in enumerate(groups):
        if expanded and t >= 4:
            break
        selected = earliest == t
        incoming = projected_masks(checks[:, selected], indexes)
        labels, p = logical[selected], probabilities[selected]
        if not expanded:
            marginal = xor_moments(incoming, labels, p, 1 << len(indexes))
        if t and not expanded:
            np.save(path / f"{basis}_marginal_{t}.npy", marginal[:, 0], allow_pickle=False)
        if t == 5:
            np.save(path / f"{basis}_terminal.npy", marginal, allow_pickle=False)
        else:
            outgoing = projected_masks(checks[:, selected], groups[t + 1])
            tags = np.ascontiguousarray((incoming << 18) | (outgoing << 4) | labels, dtype=np.uint32)
            width = EXPANDED_FANOUT if expanded else FANOUT
            table = np.zeros((1 << len(indexes), width), dtype=EDGE)
            key = (width, len(indexes), len(groups[t + 1]), tags.tobytes(), p.tobytes())
            edge_name = "expanded_edges" if expanded else "edges"
            output = path / f"{basis}_{edge_name}_{t}.npy"
            if reuse is not None and key in reuse:
                existing, retained = reuse[key]
                shutil.copyfile(existing, output)
                masses.append(retained)
                print(f"{basis} slice {t}: reused identical transition distribution", flush=True)
                continue
            if spectral is None:
                odds = np.ascontiguousarray(p / (1 - p))
                if expanded:
                    status = kernel.temporal_build_width(tags.ctypes.data, odds.ctypes.data, len(tags),
                                                         len(table), table.ctypes.data, width)
                else:
                    status = kernel.temporal_build(tags.ctypes.data, odds.ctypes.data, len(tags), len(table), table.ctypes.data)
                table["mass"] *= np.prod(1 - p)
            else:
                status = spectral(tags.ctypes.data, p.ctypes.data, len(tags), len(indexes),
                                  len(groups[t + 1]), table.ctypes.data)
            if status:
                raise RuntimeError(f"Temporal table generation failed for {basis} slice {t}")
            retained = float(table["mass"][:, :, 0].sum(dtype=np.float64))
            masses.append(retained)
            np.save(output, table, allow_pickle=False)
            if reuse is not None:
                reuse[key] = (output, retained)
            print(f"{basis} slice {t}: {len(tags)} mechanisms, retained unconditional mass {retained:.5f}", flush=True)
    return groups, masses


def generate_midpoint(path, basis, circuit, groups, incumbent):
    reverse = f"{basis}_reverse"
    names = [f"{basis}_edges_{t}" for t in range(2)]
    names += [f"{reverse}_edges_{t}" for t in range(3)]
    names += [f"{basis}_marginal_1", f"{reverse}_marginal_1", f"{reverse}_marginal_2"]
    for name in names:
        shutil.copyfile(incumbent / f"{name}.npy", path / f"{name}.npy")

    data = problem_info(circuit)
    times = np.empty(70, dtype=int)
    for t, indexes in enumerate(groups):
        times[indexes] = t
    checks = data["H"]
    earliest = np.min(np.where(checks, times[:, None], 6), axis=0)
    latest = np.max(np.where(checks, times[:, None], -1), axis=0)
    if np.any(earliest > latest) or np.any(latest - earliest > 1):
        raise ValueError(f"Unexpected fault span in {basis}")
    left = earliest < 2
    right = latest > 2
    if np.any(left & right):
        raise ValueError(f"A fault spans both sides of the midpoint in {basis}")
    middle = ~(left | right)
    parity = projected_masks(checks, groups[2])
    logical = projected_masks(data["L"], list(range(4)))
    p = data["p"]
    local = xor_moments(parity[middle], logical[middle], p[middle], 1 << 14)
    np.save(path / f"{basis}_middle.npy", local, allow_pickle=False)
    for name, selected in (("left", left | middle), ("right", right | middle)):
        likelihood = xor_moments(parity[selected], logical[selected], p[selected], 1 << 14)
        np.save(path / f"{basis}_middle_{name}.npy", likelihood[:, 0], allow_pickle=False)
    print(f"{basis}: {middle.sum()} exact middle-slice mechanisms, expected count {p[middle].sum():.6f}", flush=True)


def spectral_builder(path, *, suffix=False, lookahead=False):
    import ctypes

    source = ROOT / "temporal_decoder/spectral.cpp"
    library = path / "spectral.so"
    command = ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-march=x86-64-v2",
               "-fopenmp", str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    module = ctypes.CDLL(str(library))
    if lookahead:
        build = module.temporal_build_lookahead
    elif suffix:
        build = module.temporal_build_suffix
    else:
        build = module.temporal_build_spectral
    build.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    if lookahead:
        build.argtypes += [ctypes.c_void_p, ctypes.c_uint]
    elif not suffix:
        build.argtypes += [ctypes.c_uint, ctypes.c_uint]
    build.argtypes += [ctypes.c_void_p]
    build.restype = ctypes.c_int
    return build


def generate_suffix(path, prefix, circuit, groups, incumbent, build):
    for t in range(4):
        shutil.copyfile(incumbent / f"{prefix}_edges_{t}.npy", path / f"{prefix}_edges_{t}.npy")
    for t in range(1, 4):
        shutil.copyfile(incumbent / f"{prefix}_marginal_{t}.npy", path / f"{prefix}_marginal_{t}.npy")
    data = problem_info(circuit)
    early = np.concatenate(groups[:4])
    selected = ~data["H"][early].any(axis=0)
    incoming = projected_masks(data["H"][:, selected], groups[4])
    outgoing = projected_masks(data["H"][:, selected], groups[5])
    labels = projected_masks(data["L"][:, selected], list(range(4)))
    tags = np.ascontiguousarray((incoming << 18) | (outgoing << 4) | labels, dtype=np.uint32)
    p = np.ascontiguousarray(data["p"][selected])
    result = np.zeros((1 << 21, 5), dtype=np.float32)
    if build(tags.ctypes.data, p.ctypes.data, len(tags), result.ctypes.data):
        raise RuntimeError(f"Exact final-slice generation failed for {prefix}")
    np.save(path / f"{prefix}_suffix.npy", result, allow_pickle=False)
    np.save(path / f"{prefix}_suffix_mass.npy", result[:, 0].astype(np.float64), allow_pickle=False)
    print(f"{prefix}: {len(tags)} exact final-slice mechanisms, total mass {result[:, 0].sum(dtype=np.float64):.9f}", flush=True)


def generate_lookahead(path, prefix, circuit, groups, build, reuse):
    data = problem_info(circuit)
    checks = data["H"]
    for t in range(3):
        earlier = np.concatenate(groups[:t + 1])
        first, second = groups[t + 1:t + 3]
        selected = ~checks[earlier].any(axis=0) & checks[first + second].any(axis=0)
        a = projected_masks(checks[:, selected], first)
        b = projected_masks(checks[:, selected], second)
        p = np.ascontiguousarray(data["p"][selected])
        tags = np.ascontiguousarray((b << 18) | (a << 4), dtype=np.uint32)
        key = (tags.tobytes(), p.tobytes())
        matrix = path / f"{prefix}_lookahead_{t}.npy"
        index = path / f"{prefix}_lookahead_index_{t}.npy"
        if key in reuse:
            for original, target in zip(reuse[key], (matrix, index), strict=True):
                shutil.copyfile(original, target)
            print(f"{prefix} step {t}: reused joint parity table", flush=True)
            continue
        marginal = xor_moments(b, np.zeros(len(p), dtype=np.uint8), p, 1 << 14)[:, 0]
        parities = np.ascontiguousarray(np.argsort(-marginal, kind="stable")[:128], dtype=np.uint16)
        rank = np.full(1 << 14, -1, dtype=np.int16)
        rank[parities] = np.arange(len(parities), dtype=np.int16)
        result = np.empty((len(parities), 1 << 14), dtype=np.float32)
        if build(tags.ctypes.data, p.ctypes.data, len(tags), parities.ctypes.data,
                 len(parities), result.ctypes.data):
            raise RuntimeError(f"Joint parity generation failed for {prefix} step {t}")
        np.save(matrix, result, allow_pickle=False)
        np.save(index, rank, allow_pickle=False)
        reuse[key] = matrix, index
        print(f"{prefix} step {t}: stored second-parity mass {marginal[parities].sum():.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--bidirectional", action="store_true")
    modes.add_argument("--midpoint", action="store_true")
    modes.add_argument("--adaptive", action="store_true")
    modes.add_argument("--spectral", action="store_true")
    modes.add_argument("--suffix", action="store_true")
    modes.add_argument("--lookahead", action="store_true")
    modes.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    modes = {name: candidate for name, candidate in (("expanded", EXPANDED), ("lookahead", LOOKAHEAD), ("suffix", SUFFIX), ("spectral", SPECTRAL), ("adaptive", ADAPTIVE), ("midpoint", MIDPOINT),
                                                    ("bidirectional", BIDIRECTIONAL))}
    candidate = next((value for name, value in modes.items() if getattr(args, name)), CANDIDATE)
    destination = CHECKPOINT.parent / candidate
    design = {CANDIDATE: "STAGE04_TEMPORAL_EXPERIMENT.md", BIDIRECTIONAL: "BIDIRECTIONAL_TEMPORAL_EXPERIMENT.md",
              MIDPOINT: "MIDPOINT_TEMPORAL_EXPERIMENT.md", ADAPTIVE: "ADAPTIVE_BEAM_EXPERIMENT.md",
              SPECTRAL: "SPECTRAL_TEMPORAL_EXPERIMENT.md", SUFFIX: "EXACT_SUFFIX_EXPERIMENT.md",
              LOOKAHEAD: "TWO_SLICE_LOOKAHEAD_EXPERIMENT.md", EXPANDED: "EXPANDED_TRANSITIONS_EXPERIMENT.md"}[candidate]
    start = time.perf_counter()
    if destination.exists():
        raise FileExistsError(destination)
    if sys.version_info[:2] != (3, 13) or (np.__version__, stim.__version__) != ("2.2.6", "1.16.0"):
        raise RuntimeError("Use the pinned Python, NumPy and Stim environment")
    fallback = json.loads((FALLBACK / "provenance.json").read_text())
    for name, expected in fallback["files"].items():
        if sha256(FALLBACK / name) != expected:
            raise ValueError(f"Parity fallback changed: {name}")
    audit = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
    predecessor = CHECKPOINT.parent / BIDIRECTIONAL if candidate in (MIDPOINT, ADAPTIVE, SPECTRAL, SUFFIX) else CHECKPOINT
    if args.lookahead or args.expanded:
        predecessor = CHECKPOINT.parent / SUFFIX
    incumbent = json.loads((predecessor / "provenance.json").read_text()) if candidate != CANDIDATE else None
    if incumbent:
        for name, digest in incumbent["files"].items():
            if sha256(predecessor / name) != digest:
                raise ValueError(f"Predecessor checkpoint changed: {name}")
        previous_metadata = json.loads((predecessor / "model.json").read_text())
    compile_args = ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-march=x86-64-v2"]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-", dir=CHECKPOINT.parent) as temp:
        path = Path(temp) / candidate
        path.mkdir()
        source = ROOT / "temporal_decoder"
        for name in ("decoder.py", "kernel.cpp"):
            shutil.copyfile(source / name, path / name)
        shutil.copyfile(ROOT / "requirements.txt", path / "requirements.txt")
        (path / "parity_fallback").mkdir()
        for name in ("decoder.py", "model.json", "model.npz"):
            shutil.copyfile(FALLBACK / name, path / "parity_fallback" / name)
        subprocess.run([*compile_args, "kernel.cpp", "-o", "kernel.so"], cwd=path, check=True)
        kernel = load_kernel(path / "kernel.so")
        spectral = (spectral_builder(Path(temp), suffix=args.suffix, lookahead=args.lookahead)
                    if (args.spectral or args.suffix or args.lookahead) else None)
        if spectral is not None:
            shutil.copyfile(source / "spectral.cpp", path / "spectral.cpp")
        fingerprints, detector_groups, retained_mass, reuse = {}, {}, {}, {}
        for basis, circuit in load_circuits().items():
            fingerprints[basis] = circuit_digest(circuit)
            if fingerprints[basis] != audit["circuits"][basis]["canonical_circuit_sha256"]:
                raise ValueError(f"Audited circuit changed: {basis}")
            if args.lookahead or args.expanded:
                groups = previous_metadata["detector_groups"][basis]
                detector_groups[basis] = groups
                for name in incumbent["files"]:
                    if name.startswith(f"{basis}_") and name.endswith(".npy"):
                        shutil.copyfile(predecessor / name, path / name)
                for prefix, ordered in ((basis, groups), (f"{basis}_reverse", groups[::-1])):
                    retained_mass[prefix] = incumbent["retained_mass_by_slice"][prefix]
                    if args.expanded:
                        built_groups, retained_mass[f"{prefix}_expanded"] = generate_basis(
                            path, prefix, circuit, kernel, reverse=prefix.endswith("_reverse"), reuse=reuse, expanded=True)
                        if built_groups != ordered:
                            raise ValueError(f"Expanded detector groups changed for {prefix}")
                    else:
                        generate_lookahead(path, prefix, circuit, ordered, spectral, reuse)
            elif args.suffix:
                groups = previous_metadata["detector_groups"][basis]
                detector_groups[basis] = groups
                for prefix, ordered in ((basis, groups), (f"{basis}_reverse", groups[::-1])):
                    retained_mass[prefix] = incumbent["retained_mass_by_slice"][prefix][:4]
                    generate_suffix(path, prefix, circuit, ordered, predecessor, spectral)
            elif args.spectral:
                detector_groups[basis], retained_mass[basis] = generate_basis(
                    path, basis, circuit, kernel, spectral=spectral, reuse=reuse)
                prefix = f"{basis}_reverse"
                groups, retained_mass[prefix] = generate_basis(
                    path, prefix, circuit, kernel, reverse=True, spectral=spectral, reuse=reuse)
                if groups != detector_groups[basis][::-1]:
                    raise ValueError(f"Reversed detector groups changed the within-slice order for {basis}")
            elif args.midpoint:
                detector_groups[basis] = previous_metadata["detector_groups"][basis]
                generate_midpoint(path, basis, circuit, detector_groups[basis], predecessor)
            elif incumbent:
                detector_groups[basis] = previous_metadata["detector_groups"][basis]
                retained_mass[basis] = incumbent["retained_mass_by_slice"][basis]
                for name in incumbent["files"]:
                    if name.startswith(f"{basis}_") and name.endswith(".npy"):
                        shutil.copyfile(predecessor / name, path / name)
                prefix = f"{basis}_reverse"
                if args.bidirectional:
                    groups, retained_mass[prefix] = generate_basis(path, prefix, circuit, kernel, reverse=True)
                    if groups != detector_groups[basis][::-1]:
                        raise ValueError(f"Reversed detector groups changed the within-slice order for {basis}")
                else:
                    retained_mass[prefix] = incumbent["retained_mass_by_slice"][prefix]
            else:
                detector_groups[basis], retained_mass[basis] = generate_basis(path, basis, circuit, kernel)
        assets = {str(file.relative_to(path)): sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}
        version = {CANDIDATE: 1, BIDIRECTIONAL: 2, MIDPOINT: 3, ADAPTIVE: 4,
                   SPECTRAL: 2, SUFFIX: 5, LOOKAHEAD: 6, EXPANDED: 8}[candidate]
        write_json(path / "model.json", {"format_version": version, "canonical_circuit_sha256": fingerprints,
                                        "detector_groups": detector_groups, "assets": assets})
        elapsed = time.perf_counter() - start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if elapsed > 1200 or peak_rss > 5 * 1024**3:
            raise RuntimeError("Temporal table generation exceeded the approved budget")
        write_json(path / "provenance.json", {
            "candidate": candidate, "files": {**assets, "model.json": sha256(path / "model.json")},
            "build_seconds": elapsed, "build_peak_rss_bytes": peak_rss, "retained_mass_by_slice": retained_mass,
            "build_command": [*compile_args, "kernel.cpp", "-o", "kernel.so"],
            "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
            "python": platform.python_version(), "organizer_revision": audit["source_revision"],
            "fallback_provenance_sha256": sha256(FALLBACK / "provenance.json"),
            "design_sha256": sha256(ROOT / "docs" / design),
            "forward_provenance_sha256": sha256(predecessor / "provenance.json") if incumbent else None,
            "asset_generator_sha256": sha256(Path(__file__)),
        })
        if destination.exists():
            raise FileExistsError(destination)
        path.rename(destination)
    print(f"Built {candidate} in {elapsed:.1f}s with peak RSS {peak_rss / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
