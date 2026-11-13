"""Compare two fixed checkpoints on an independent Stim sample."""

import argparse
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import stim

from evaluate_temporal import EXPERIMENTS, checkpoint_decoder, locations
from stage01_support import load_circuits, sha256, write_json


def confirm(candidate, incumbent, shots, seeds):
    decoder, _ = checkpoint_decoder(candidate)
    previous, _ = checkpoint_decoder(incumbent)
    metrics = {}
    for basis, circuit in load_circuits().items():
        seed = seeds[basis]
        sampler = circuit.compile_detector_sampler(seed=seed)
        events, labels = sampler.sample(shots, separate_observables=True)
        before = previous(circuit, events)
        after = decoder(circuit, events)
        before_errors = np.count_nonzero(before != labels, axis=1)
        after_errors = np.count_nonzero(after != labels, axis=1)
        paired = (before_errors - after_errors) / labels.shape[1]
        metrics[basis] = {
            "seed": seed, "shots": shots,
            "detectors_sha256": hashlib.sha256(events.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "incumbent_errors": int(before_errors.sum()), "candidate_errors": int(after_errors.sum()),
            "paired_gain": float(paired.mean()), "paired_variance": float(paired.var(ddof=1)),
        }
    gain = sum(x["paired_gain"] for x in metrics.values()) / len(metrics)
    se = np.sqrt(sum(x["paired_variance"] / shots for x in metrics.values())) / len(metrics)
    ci = [float(gain - 1.95996398454 * se), float(gain + 1.95996398454 * se)]
    fingerprints = {
        name: sha256(locations(name)[0] / "provenance.json") for name in (candidate, incumbent)
    }
    return {
        "candidate": candidate, "incumbent": incumbent, "stim_version": stim.__version__,
        "checkpoint_provenance_sha256": fingerprints, "bases": metrics,
        "paired_gain": gain, "paired_gain_ci95": ci, "passed": ci[0] > 0,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=EXPERIMENTS, required=True)
    parser.add_argument("--incumbent", choices=EXPERIMENTS, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs=2, required=True, metavar=("X", "Z"))
    args = parser.parse_args()
    if args.shots < 2 or len(set(args.seeds)) != 2 or args.candidate == args.incumbent:
        parser.error("Use at least two shots, distinct seeds and distinct candidates")
    _, evidence = locations(args.candidate)
    output = evidence / "confirmation.json"
    if output.exists():
        raise FileExistsError(output)
    result = confirm(args.candidate, args.incumbent, args.shots, dict(zip("XZ", args.seeds, strict=True)))
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
