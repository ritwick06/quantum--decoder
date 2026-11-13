import ctypes
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import stim

from stage01_support import ROOT, load_circuits
from temporal_decoder import decoder
from tools.build_temporal import CHECKPOINT, xor_moments


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = decoder.load_kernel(CHECKPOINT / "kernel.so")

    def test_tables_match_small_exhaustive_fault_model(self):
        incoming = (1, 2, 3, 0)
        outgoing = (2, 1, 3, 1)
        logical = (1, 8, 5, 3)
        p = np.array([0.12, 0.08, 0.19, 0.04])
        tags = np.array([(i << 18) | (o << 4) | l for i, o, l in zip(incoming, outgoing, logical, strict=True)], dtype=np.uint32)
        odds = p / (1 - p)
        table = np.zeros((4, decoder.FANOUT), dtype=decoder.EDGE)
        self.assertEqual(self.kernel.temporal_build(tags.ctypes.data, odds.ctypes.data, len(tags), 4, table.ctypes.data), 0)
        expected = np.zeros((4, 4, 5))
        marginal = np.zeros((4, 5))
        for choices in itertools.product((False, True), repeat=len(tags)):
            a, b, word = 0, 0, 0
            for index, included in enumerate(choices):
                if included:
                    a ^= incoming[index]
                    b ^= outgoing[index]
                    word ^= logical[index]
            signed = np.array([1, *[(-1)**((word >> bit) & 1) for bit in range(4)]])
            marginal[a] += np.prod(np.where(choices, p, 1 - p)) * signed
            if sum(choices) <= 3:
                expected[a, b] += np.prod(odds[list(choices)]) * signed
        actual = np.zeros_like(expected)
        for i, row in enumerate(table):
            for edge in row[row["mass"][:, 0] > 0]:
                actual[i, edge["outgoing"]] = edge["mass"]
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(xor_moments(incoming, logical, p, 4), marginal, atol=1e-15)

    def test_cross_slice_parities_and_logical_cancellation(self):
        tables = [np.zeros((2, decoder.FANOUT), dtype=decoder.EDGE) for _ in range(5)]
        tables[0][0, 0] = (1, [0.7, -0.7, 0.7, 0.7, 0.7])
        tables[0][0, 1] = (0, [0.3, 0.3, 0.3, 0.3, 0.3])
        tables[1][0, 0] = (0, [1, -1, -1, 1, 1])
        for table in tables[2:]:
            table[0, 0] = (0, [1] * 5)
        marginals = [np.array([1.0, 0.0]) for _ in range(5)]
        terminal = np.array([[1.0] * 5, [0.0] * 5])
        table_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in marginals))
        observed = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.uint16)
        result = np.empty((2, 4), dtype=np.int8)
        status = self.kernel.temporal_decode(observed.ctypes.data, len(observed), table_ptrs, marginal_ptrs,
                                             terminal.ctypes.data, result.ctypes.data)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(result, [[0, 1, 0, 0], [-1, -1, -1, -1]])

    def test_checkpoint_batch_and_input_contract(self):
        rng = np.random.default_rng(914)
        rows = (rng.random((4101, 140)) < 0.1)[:, ::2]
        with patch.object(decoder, "HERE", CHECKPOINT):
            decoder.load_model.cache_clear()
            try:
                for circuit in load_circuits().values():
                    whole = decoder.decode(circuit, rows)
                    self.assertEqual(whole.shape, (len(rows), 4))
                    self.assertEqual(whole.dtype, np.uint8)
                    self.assertTrue(np.isin(whole, (0, 1)).all())
                    self.assertEqual(decoder.decode(circuit, rows[:0]).shape, (0, 4))
                    np.testing.assert_array_equal(whole[:1], decoder.decode(circuit, rows[:1]))
                    split = np.concatenate([decoder.decode(circuit, rows[:2000]), decoder.decode(circuit, rows[2000:])])
                    np.testing.assert_array_equal(whole, split)
                    with self.assertRaisesRegex(ValueError, "binary"):
                        decoder.decode(circuit, np.full((1, 70), 2, dtype=int))
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    decoder.decode(stim.Circuit(), rows[:1])
            finally:
                decoder.load_model.cache_clear()
