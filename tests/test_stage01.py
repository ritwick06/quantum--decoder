import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from stage01_support import (ROOT, SEEDS, check_binary, decode, decode_by_basis, inspect_csv,
    load_circuits, load_labelled, load_validation, logical_error_rate, reserved_mask, sha256)


class Stage01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuits = load_circuits()

    def test_zero_baseline_contract_both_bases(self):
        rng = np.random.default_rng(71)
        for basis in "XZ":
            for size in (0, 1, 17, 100000):
                with self.subTest(basis=basis, size=size):
                    events = rng.integers(0, 2, size=(size, 70), dtype=np.uint8).astype(bool)
                    before = events.copy()
                    output = decode(self.circuits[basis], events)
                    self.assertEqual(output.shape, (size, 4))
                    self.assertEqual(output.dtype, np.uint8)
                    self.assertFalse(output.any())
                    np.testing.assert_array_equal(events, before)
                    np.testing.assert_array_equal(output, decode(self.circuits[basis], events))

    def test_routing_and_order_with_distinguishable_outputs(self):
        events = np.zeros((5, 70), dtype=bool)
        events[:, :3] = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]]
        bases = np.array(["Z", "X", "Z", "X", "Z"])
        calls = []

        def sentinel(circuit, rows):
            basis = "X" if circuit is self.circuits["X"] else "Z"
            calls.append((basis, len(rows)))
            return np.column_stack((rows[:, :3], np.full(len(rows), basis == "Z")))

        expected = np.column_stack((events[:, :3], bases == "Z"))
        actual = decode_by_basis(self.circuits, events, bases, sentinel)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(calls, [("X", 2), ("Z", 3)])
        permutation = np.array([4, 2, 0, 3, 1])
        shuffled = decode_by_basis(self.circuits, events[permutation], bases[permutation], sentinel)
        np.testing.assert_array_equal(shuffled[np.argsort(permutation)], expected)

    def test_batch_independence_and_empty_wrapper(self):
        events = np.ones((19, 70), dtype=bool)
        bases = np.array(list("XZ") * 9 + ["X"])
        whole = decode_by_basis(self.circuits, events, bases)
        parts = np.concatenate([decode_by_basis(self.circuits, events[i:i+3], bases[i:i+3]) for i in range(0, 19, 3)])
        np.testing.assert_array_equal(whole, parts)
        self.assertEqual(decode_by_basis(self.circuits, np.zeros((0, 70), bool), np.array([], dtype=str)).shape, (0, 4))

    def test_invalid_inputs_and_outputs(self):
        for events in (np.zeros((2, 69), bool), np.full((2, 70), 2), np.full((2, 70), np.nan)):
            with self.assertRaises(ValueError):
                decode_by_basis(self.circuits, events, ["X", "Z"])
        with self.assertRaises(ValueError):
            decode_by_basis(self.circuits, np.zeros((2, 70), bool), ["X", "Y"])
        with self.assertRaises(ValueError):
            decode_by_basis(self.circuits, np.zeros((2, 70), bool), ["X"])
        for bad in (lambda c, d: np.zeros((len(d), 3)), lambda c, d: np.full((len(d), 4), 2),
                    lambda c, d: np.zeros((len(d)+1, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                decode_by_basis(self.circuits, np.zeros((2, 70), bool), ["X", "Z"], bad)

    def test_metric_counts_bits_not_failed_shots(self):
        truth = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        self.assertEqual(logical_error_rate(np.zeros_like(truth), truth), 0.25)
        with self.assertRaises(AssertionError):
            logical_error_rate(np.zeros((1, 4)), truth)

    def test_csv_schema_and_dataset_separation(self):
        with self.assertRaises(ValueError):
            load_labelled(ROOT / "quantum-error-correction-decoder/test.csv")
        header = "id,basis," + ",".join(f"D{i}" for i in range(70))
        row = "same_id,X," + ",".join(["0"] * 70)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            path.write_text(header + "\n" + row + "\n" + row + "\n")
            with self.assertRaises(ValueError):
                inspect_csv(path, labels=False)
        ids = np.array(["gen_X_00000", "gen_Z_99999", "gen_X_00042"])
        selection = reserved_mask(ids)
        np.testing.assert_array_equal(reserved_mask(ids[::-1]), selection[::-1])

    def test_official_source_is_unmodified(self):
        manifest = json.loads((ROOT / "organizer/SOURCE_MANIFEST.json").read_text())
        for path, info in manifest["files"].items():
            self.assertEqual(sha256(ROOT / "organizer" / path), info["sha256"], path)
        for path, digest in json.loads((ROOT / "artifacts/stage01/original_files.json").read_text()).items():
            self.assertEqual(sha256(ROOT / path), digest, path)

    def test_frozen_validation_contract(self):
        data = load_validation(ROOT / "data/validation")
        for basis, (events, labels) in data.items():
            self.assertEqual(events.shape, (100000, 70))
            self.assertEqual(labels.shape, (100000, 4))
            self.assertFalse(events.flags.writeable)
            self.assertFalse(labels.flags.writeable)
            check_binary(decode(self.circuits[basis], events), 4, "predictions")

    def test_fresh_process_predictions(self):
        code = '''
import hashlib, json
from stage01_support import ROOT, decode, load_circuits, load_validation
circuits = load_circuits()
data = load_validation(ROOT / "data/validation")
print(json.dumps({b: hashlib.sha256(decode(circuits[b], data[b][0]).tobytes()).hexdigest() for b in "XZ"}))
'''
        report = json.loads((ROOT / "artifacts/stage01/report.json").read_text())
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), report["validation"]["prediction_sha256"])

    def test_frozen_validation_replays_with_recorded_seeds(self):
        data = load_validation(ROOT / "data/validation")
        for basis in "XZ":
            events, labels = self.circuits[basis].compile_detector_sampler(seed=SEEDS[basis]).sample(
                100000, separate_observables=True)
            np.testing.assert_array_equal(events, data[basis][0])
            np.testing.assert_array_equal(labels, data[basis][1])


if __name__ == "__main__":
    unittest.main()
