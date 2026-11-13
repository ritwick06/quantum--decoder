import ctypes
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stim

from stage01_support import ROOT, load_circuits
from temporal_decoder import decoder
from temporal_decoder import readout
from tools.build_temporal import ADAPTIVE, BACKWARD, CHECKPOINT, DEEP, EXPANDED, LOOKAHEAD, MIDPOINT, SUFFIX, assigned_slices, spectral_builder, xor_moments


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = decoder.load_kernel(CHECKPOINT / "kernel.so")

    def test_combination_uses_joint_mass_and_preserves_ambiguous_bits(self):
        forward = np.array([[1, 0.5, 0.1, 0.02, -0.6], [0, 0, 0, 0, 0]])
        reverse = np.array([[3, -1.5, -0.1, 0.06, 0.9], [0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(
            decoder.logical_decisions(forward + reverse),
            [[1, -1, -1, 0], [-1, -1, -1, -1]],
        )

    def test_readout_features_use_relative_masses_and_select_bits_independently(self):
        forward = np.array([[2., 1., 2., -2., 0.]])
        reverse = np.array([[1., -1., 1., -1., 0.]])
        prepared = readout.direction_features(forward, reverse)
        np.testing.assert_array_equal(prepared[-1], [[True, False, False, True]])
        design = readout.feature_columns(prepared, np.array([[1, 2, 3, 4, 5, 6]]),
                                         np.array([[1, 0, 0, 0]]), np.array([True]), 0)
        self.assertEqual(design.shape, (1, 26))
        np.testing.assert_allclose(design[0, :8], [0, 1, -1, 0, 0.5, -1, np.log(2), 1])
        np.testing.assert_array_equal(design[0, 14:20], [0.5, 1, 1.5, 2, 2.5, 3])
        np.testing.assert_array_equal(design[0, 20:], [-1, -2, -3, -4, -5, -6])

    def test_bidirectional_checkpoint_keeps_forward_predictions_and_row_order(self):
        path = CHECKPOINT.parent / "temporal_bidir_v1"
        rng = np.random.default_rng(740)
        events = rng.random((4101, 70)) < 0.13
        for basis, circuit in load_circuits().items():
            with patch.object(decoder, "HERE", CHECKPOINT):
                decoder.load_model.cache_clear()
                original = decoder.decode(circuit, events)
            with patch.object(decoder, "HERE", path):
                decoder.load_model.cache_clear()
                try:
                    kernel, fallback, models = decoder.load_model()
                    _, (forward, reverse) = models[decoder.circuit_digest(circuit)]
                    self.assertEqual(reverse[0], forward[0][::-1])
                    np.testing.assert_array_equal(
                        decoder.observed_slices(events, reverse[0]),
                        decoder.observed_slices(events, forward[0])[:, ::-1],
                    )
                    masses = decoder._decode_temporal(events, forward, kernel, return_moments=True)
                    one_direction = decoder.logical_decisions(masses)
                    fallback_rows = (one_direction < 0).any(axis=1)
                    one_direction[fallback_rows] = np.where(
                        one_direction[fallback_rows] < 0, fallback(circuit, events[fallback_rows]),
                        one_direction[fallback_rows],
                    )
                    np.testing.assert_array_equal(one_direction, original)
                    whole = decoder.decode(circuit, events)
                    split = np.concatenate([decoder.decode(circuit, events[:503]), decoder.decode(circuit, events[503:])])
                    np.testing.assert_array_equal(whole, split)
                    self.assertEqual(whole.dtype, np.uint8)
                    self.assertTrue(np.isin(whole, (0, 1)).all(), basis)
                finally:
                    decoder.load_model.cache_clear()

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

    def test_full_local_distribution_matches_all_fault_subsets(self):
        incoming = (1, 2, 3, 0, 3, 0)
        outgoing = (2, 1, 3, 1, 2, 0)
        logical = (1, 8, 5, 3, 10, 4)
        p = np.array([0.12, 0.08, 0.19, 0.04, 0.09, 0.1])
        tags = np.array([(i << 18) | (o << 4) | l for i, o, l in zip(incoming, outgoing, logical, strict=True)], dtype=np.uint32)
        table = np.zeros((4, decoder.FANOUT), dtype=decoder.EDGE)
        with tempfile.TemporaryDirectory() as temp:
            build = spectral_builder(Path(temp))
            self.assertEqual(build(tags.ctypes.data, p.ctypes.data, len(tags), 2, 2, table.ctypes.data), 0)
        expected = np.zeros((4, 4, 5))
        for selected in itertools.product((False, True), repeat=len(tags)):
            a, b, word = 0, 0, 0
            for i, included in enumerate(selected):
                if included:
                    a ^= incoming[i]
                    b ^= outgoing[i]
                    word ^= logical[i]
            signs = [1, *[(-1)**((word >> bit) & 1) for bit in range(4)]]
            expected[a, b] += np.prod(np.where(selected, p, 1 - p)) * np.array(signs)
        actual = np.zeros_like(expected)
        for a, row in enumerate(table):
            weights = row["mass"][:, 0]
            self.assertTrue(np.all(weights[:-1] >= weights[1:]))
            for edge in row[weights > 0]:
                actual[a, edge["outgoing"]] = edge["mass"]
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=2e-12)

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

    def test_midpoint_join_matches_exhaustive_distribution(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / MIDPOINT / "kernel.so")
        tables = [np.zeros((2, decoder.FANOUT), dtype=decoder.EDGE) for _ in range(5)]
        for t in (0, 2, 3):
            tables[t][0, 0] = (0, [1] * 5)
        p_left, p_right, p_middle = 0.25, 0.5, 0.125
        for t, p, signs in ((1, p_left, [-1, 1, 1, 1]), (4, p_right, [1, -1, 1, 1])):
            tables[t][0, 0] = (0, [1 - p] * 5)
            tables[t][0, 1] = (1, [p, *(p * s for s in signs)])
        middle = np.array([[1 - p_middle] * 5, [p_middle, p_middle, p_middle, -p_middle, p_middle]])
        marginals = [np.ones(2) for _ in tables]
        table_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((2, 6), dtype=np.uint16)
        observed[1, 2] = 1
        actual = np.empty((2, 5), dtype=np.float64)
        expected = np.zeros_like(actual)
        for a, b, c in itertools.product((0, 1), repeat=3):
            weight = np.prod(np.where([a, b, c], [p_left, p_right, p_middle],
                                      [1 - p_left, 1 - p_right, 1 - p_middle]))
            expected[a ^ b ^ c] += weight * np.array([1, 1 - 2 * a, 1 - 2 * b, 1 - 2 * c, 1])
        status = kernel.temporal_midpoint(observed.ctypes.data, len(observed), table_ptrs, marginal_ptrs,
                                          middle.ctypes.data, actual.ctypes.data)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_suffix_uses_last_observation_and_preserves_logical_signs(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / SUFFIX / "kernel.so")
        tables = [np.zeros((64, decoder.FANOUT), dtype=decoder.EDGE) for _ in range(4)]
        for table in tables[:3]:
            table[0, 0] = (0, [1.0] * 5)
        for mask in range(32):
            tables[3][0, mask] = (mask, [1.0] * 5)
        tables[3][0, 32] = (32, [0.5, -0.5, 0.5, 0.5, 0.5])
        terminal = np.zeros((64 * 128, 5), dtype=np.float32)
        terminal[0] = [1.0] * 5
        terminal[32 * 128 + 1] = [0.25, 0.25, -0.25, 0.25, 0.25]
        marginals = [np.ones(64) for _ in range(3)] + [terminal[:, 0].astype(np.float64)]
        table_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((2, 6), dtype=np.uint16)
        observed[1, 5] = 1
        actual = np.empty((2, 5), dtype=np.float64)
        status = kernel.temporal_suffix_moments(observed.ctypes.data, len(observed), table_ptrs, marginal_ptrs,
                                               terminal.ctypes.data, actual.ctypes.data)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, [[1.0] * 5, [0.125, -0.125, -0.125, 0.125, 0.125]])

    def test_joint_lookahead_retains_path_that_matches_two_slices(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / LOOKAHEAD / "kernel.so")
        tables = [np.zeros((64, decoder.FANOUT), dtype=decoder.EDGE) for _ in range(4)]
        for mask in range(32):
            tables[0][0, mask] = (mask, [1.0] * 5)
        tables[0][0, 32] = (32, [0.5, -0.5, 0.5, 0.5, 0.5])
        tables[1][32, 0] = (1, [1.0] * 5)
        for table in tables[2:]:
            table[0, 0] = (0, [1.0] * 5)
        terminal = np.ones((128, 5), dtype=np.float32)
        marginals = [np.ones(64) for _ in range(3)] + [np.ones(128)]
        indexes = [np.array([-1, 0], dtype=np.int16), np.full(2, -1, dtype=np.int16), np.full(2, -1, dtype=np.int16)]
        joint = np.zeros((1, 1 << 14), dtype=np.float32)
        joint[0, 32] = 1.0
        ptrs = lambda arrays: (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
        observed = np.zeros((2, 6), dtype=np.uint16)
        observed[1, 2] = 1
        actual = np.empty((2, 5), dtype=np.float64)
        status = kernel.temporal_lookahead_moments(
            observed.ctypes.data, len(observed), ptrs(tables), ptrs(marginals),
            terminal.ctypes.data, ptrs([joint] * 3), ptrs(indexes), actual.ctypes.data,
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, [[0] * 5, [0.5, -0.5, 0.5, 0.5, 0.5]])

    def test_expanded_transition_can_recover_path_after_the_original_fanout(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / EXPANDED / "kernel.so")
        width = decoder.EXPANDED_FANOUT
        tables = [np.zeros((64, width), dtype=decoder.EDGE) for _ in range(4)]
        for mask in range(decoder.FANOUT):
            tables[0][0, mask] = (mask, [1.0] * 5)
        tables[0][0, decoder.FANOUT] = (48, [0.5, -0.5, 0.5, 0.5, 0.5])
        tables[1][48, 0] = (0, [1.0] * 5)
        for table in tables[2:]:
            table[0, 0] = (0, [1.0] * 5)
        terminal = np.ones((128, 5), dtype=np.float32)
        marginals = [np.zeros(64), np.ones(64), np.ones(64), np.ones(128)]
        marginals[0][48] = 1.0
        table_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((1, 6), dtype=np.uint16)
        actual = np.empty((1, 5), dtype=np.float64)
        status = kernel.temporal_expanded_moments(
            observed.ctypes.data, 1, table_ptrs, marginal_ptrs, terminal.ctypes.data, actual.ctypes.data,
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, [[0.5, -0.5, 0.5, 0.5, 0.5]])

    def test_backward_likelihood_accounts_for_the_last_transition_before_pruning(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / BACKWARD / "kernel.so")
        width = decoder.EXPANDED_FANOUT
        tables = [np.zeros((64, width), dtype=decoder.EDGE) for _ in range(4)]
        for table in tables[:2]:
            table[0, 0] = (0, [1] * 5)
        for mask in range(32):
            tables[2][0, mask] = (mask, [1] * 5)
        tables[2][0, 32] = (32, [0.5, -0.5, 0.5, 0.5, 0.5])
        tables[3][32, 0] = (1, [0.25, 0.25, -0.25, 0.25, 0.25])
        suffix = np.zeros((65 * 128, 5), dtype=np.float32)
        suffix[128] = [0.5] * 5
        marginals = [np.ones(64) for _ in range(3)] + [suffix[:, 0].astype(np.float64)]
        offsets = np.array([0, 0, *([1] * 64)], dtype=np.uint32)
        incoming = np.array([(32, 0.25)], dtype=decoder.BACK_EDGE)
        residuals = np.array([1, 0, *range(2, 65)], dtype=np.uint16)
        table_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((1, 6), dtype=np.uint16)
        actual = np.empty((1, 5), dtype=np.float64)
        status = kernel.temporal_backward_moments(
            observed.ctypes.data, 1, table_ptrs, marginal_ptrs, suffix.ctypes.data,
            offsets.ctypes.data, incoming.ctypes.data, residuals.ctypes.data, actual.ctypes.data,
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, [[0.0625, -0.0625, -0.0625, 0.0625, 0.0625]])

    def test_deep_kernel_keeps_a_path_in_the_wider_beam_and_table(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / DEEP / "kernel.so")
        width = decoder.DEEP_FANOUT
        tables = [np.zeros((128, width), dtype=decoder.EDGE) for _ in range(4)]
        for mask in range(40):
            tables[0][0, mask] = (mask, [1] * 5)
        tables[1][39, :100] = (1, [1] * 5)
        tables[1][39, 100] = (0, [0.5, -0.5, 0.5, 0.5, 0.5])
        for table in tables[2:]:
            table[0, 0] = (0, [1] * 5)
        terminal = np.ones((128, 5), dtype=np.float32)
        marginals = [np.ones(128) for _ in range(4)]
        table_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 4)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((1, 6), dtype=np.uint16)
        actual = np.empty((1, 5), dtype=np.float64)
        status = kernel.temporal_deep_moments(observed.ctypes.data, 1, table_ptrs, marginal_ptrs,
                                             terminal.ctypes.data, actual.ctypes.data)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual, [[0.5, -0.5, 0.5, 0.5, 0.5]])

    def test_shifted_assignment_moves_only_interior_single_slice_faults(self):
        earliest = np.array([0, 0, 1, 1, 2, 3, 4, 4, 5])
        latest = np.array([0, 1, 1, 2, 2, 3, 4, 5, 5])
        np.testing.assert_array_equal(
            assigned_slices(earliest, latest, shifted=True), [0, 0, 0, 1, 1, 2, 3, 4, 5],
        )

    def test_wide_beam_recovers_a_path_pruned_by_narrow_beam(self):
        kernel = decoder.load_kernel(CHECKPOINT.parent / ADAPTIVE / "kernel.so")
        tables = [np.zeros((64, decoder.FANOUT), dtype=decoder.EDGE) for _ in range(5)]
        for mask in range(32):
            tables[0][0, mask] = (mask, [1.0] * 5)
        tables[0][0, 32] = (32, [0.5, -0.5, 0.5, 0.5, 0.5])
        tables[1][32, 0] = (0, [1.0] * 5)
        for table in tables[2:]:
            table[0, 0] = (0, [1.0] * 5)
        marginals = [np.ones(64) for _ in tables]
        terminal = np.ones((64, 5))
        table_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in tables))
        marginal_ptrs = (ctypes.c_void_p * 5)(*(a.ctypes.data for a in marginals))
        observed = np.zeros((1, 6), dtype=np.uint16)
        actual = np.empty((1, 5), dtype=np.float64)
        for infer, expected in ((kernel.temporal_moments, [[0] * 5]),
                                (kernel.temporal_wide_moments, [[0.5, -0.5, 0.5, 0.5, 0.5]])):
            status = infer(observed.ctypes.data, len(observed), table_ptrs, marginal_ptrs,
                           terminal.ctypes.data, actual.ctypes.data)
            self.assertEqual(status, 0)
            np.testing.assert_array_equal(actual, expected)

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
