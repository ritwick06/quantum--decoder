import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stim

from stage01_support import ROOT, load_circuits
from submission_pipeline import predict_checked
from team_decoder import decoder
from tools.train_parity import export_head, parity_moments


CHECKPOINT = ROOT / "artifacts/checkpoints/parity_v1"


class ParityMathTests(unittest.TestCase):
    def test_dem_moments_match_exhaustive_fault_enumeration(self):
        p = np.array([0.10, 0.25, 0.35])
        masks = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        logical = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        dx = np.zeros(2)
        ly = np.zeros(2)
        xy = np.zeros((2, 2))
        for faults in itertools.product((0, 1), repeat=3):
            mass = np.prod(np.where(faults, p, 1 - p))
            ds = (-1.0)**(masks @ faults)
            ls = (-1.0)**(logical @ faults)
            dx += mass * ds
            ly += mass * ls
            xy += mass * np.outer(ds, ls)
        for expected, actual in zip((dx, ly, xy), parity_moments(masks, logical, p), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_word_boundary_and_time_slice_numerators(self):
        groups = np.repeat(np.arange(6), decoder.ROUND_SIZES)
        rows = np.zeros((3, 70), dtype=bool)
        rows[0, [63, 64, 69]] = True
        rows[1, [60, 64]] = True
        rows[2, 63] = True
        words = decoder.pack_traces(rows)
        self.assertEqual(words[0].tolist(), [1 << 63, 33])
        masks = decoder.pack_traces(np.eye(70, dtype=bool)[[63, 64, 69]] | np.array([np.eye(70, dtype=bool)[60], np.zeros(70, dtype=bool), np.zeros(70, dtype=bool)]))
        times = np.array([[4, 5], [5, 5], [5, 5]])
        result = decoder.feature_numerators(words, decoder.round_numerators(rows, groups), masks, times)
        np.testing.assert_array_equal(result[:, 4:], [
            [0, 12, -28, -28, -28, -6, -12, -12],
            [2, 4, -28, -28, 28, -3, -4, 4],
            [0, 4, -28, 28, 28, -2, 4, 4],
        ])

    def test_export_folds_normalization_and_rejects_overflow(self):
        weights = np.array([0.5, 7.0, -3.5])
        mean = np.array([0.25, -0.5])
        scale = np.array([0.5, 0.25])
        bias, coefficients = export_head(weights, mean, scale)
        self.assertEqual(int(bias), -10 * 2**24)
        self.assertEqual(coefficients.tolist(), [2**23, -(2**23)])
        self.assertEqual(int(bias + np.array([16, -4]) @ coefficients), 0)
        with self.assertRaisesRegex(ValueError, "bound"):
            decoder.check_score_bound(0, [2**60])


class ParityCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuits = load_circuits()

    def test_batch_contract_and_mixed_basis_order(self):
        rows = np.random.default_rng(124).integers(0, 2, (4101, 140), dtype=np.uint8)[:, ::2]
        before = rows.copy()
        bases = np.where(np.arange(len(rows)) % 2 == 0, "X", "Z")
        with patch.object(decoder, "__file__", str(CHECKPOINT / "decoder.py")):
            for circuit in self.circuits.values():
                empty = decoder.decode(circuit, rows[:0])
                self.assertEqual(empty.shape, (0, 4))
                full = decoder.decode(circuit, rows)
                self.assertEqual(full.dtype, np.uint8)
                self.assertTrue(np.isin(full, [0, 1]).all())
                np.testing.assert_array_equal(full[:1], decoder.decode(circuit, rows[:1]))
                np.testing.assert_array_equal(full, np.concatenate([
                    decoder.decode(circuit, rows[:2010]), decoder.decode(circuit, rows[2010:])]))
            order = np.random.default_rng(52).permutation(len(rows))
            original = predict_checked(self.circuits, rows, bases, decoder.decode)
            shuffled = predict_checked(self.circuits, rows[order], bases[order], decoder.decode, batch_size=127)
        np.testing.assert_array_equal(rows, before)
        np.testing.assert_array_equal(original[order], shuffled)

    def test_bad_inputs_and_unknown_circuit(self):
        with patch.object(decoder, "__file__", str(CHECKPOINT / "decoder.py")):
            for bad in (np.zeros((1, 69)), np.zeros((1, 70)), np.full((1, 70), 2, dtype=int)):
                with self.assertRaises(ValueError):
                    decoder.decode(self.circuits["X"], bad)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                decoder.decode(stim.Circuit(), np.zeros((1, 70), dtype=bool))

    def test_missing_or_corrupt_assets_fail_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with self.assertRaisesRegex(ValueError, "assets"):
                decoder.load_model(directory)
            metadata = json.loads((CHECKPOINT / "model.json").read_text())
            (directory / "model.json").write_text(json.dumps(metadata))
            (directory / "model.npz").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                decoder.load_model(directory)

    def test_integer_score_tie_decodes_to_zero(self):
        masks = decoder.pack_traces(np.eye(70, dtype=bool)[[63]])
        head = (masks, np.array([[5, 5]]), np.zeros(8, dtype=np.int64), 0)
        groups = np.repeat(np.arange(6), decoder.ROUND_SIZES)
        with patch.object(decoder, "load_model", return_value={
            decoder.circuit_digest(self.circuits["X"]): (groups, [head] * 4),
        }):
            result = decoder.decode(self.circuits["X"], np.zeros((1, 70), dtype=bool))
        np.testing.assert_array_equal(result, np.zeros((1, 4), dtype=np.uint8))
