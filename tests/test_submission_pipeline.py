import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from submission_pipeline import HEADER, load_test_data, predict_checked, validate_submission, write_submission


class SubmissionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.ids = ["shot_Z_1", "shot_X_2", "shot_Z_3"]
        self.predictions = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 1, 0]], dtype=np.uint8)

    def csv_file(self, header, rows, name="fixture.csv"):
        path = self.folder / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def valid_rows(self):
        return [[shot, *[str(int(value)) for value in row]] for shot, row in zip(self.ids, self.predictions)]

    def test_round_trip_preserves_distinct_values_and_ids(self):
        output = self.folder / "result.csv"
        result = write_submission(output, self.ids, self.predictions)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(validate_submission(output, self.ids, self.predictions), result)
        with output.open(newline="") as stream:
            reader = csv.reader(stream)
            self.assertEqual(next(reader), HEADER)
            self.assertEqual(list(reader), self.valid_rows())

    def test_rejects_wrong_or_extra_headers(self):
        for header in (HEADER[::-1], HEADER + ["index"], ["id", "L0", "L1", "L3", "L2"], ["\ufeffid", *HEADER[1:]]):
            with self.subTest(header=header):
                path = self.csv_file(header, self.valid_rows())
                with self.assertRaises(ValueError):
                    validate_submission(path, self.ids)

    def test_rejects_missing_extra_duplicate_and_reordered_rows(self):
        rows = self.valid_rows()
        cases = [rows[:-1], rows + [["extra", "0", "0", "0", "0"]],
                 [rows[0], rows[0], rows[2]], rows[::-1],
                 [["", *rows[0][1:]], *rows[1:]], [["unknown", *rows[0][1:]], *rows[1:]]]
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                path = self.csv_file(HEADER, case)
                with self.assertRaises(ValueError):
                    validate_submission(path, self.ids)

    def test_rejects_nonbinary_blank_nan_and_wrong_width(self):
        for value in ("", "NaN", "0.0", "2", "-1", "True", " 0", "1 "):
            rows = self.valid_rows()
            rows[0][1] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_submission(self.csv_file(HEADER, rows), self.ids)
        for row in ([self.ids[0], "0"], [self.ids[0], "0", "0", "0", "0", "0"]):
            with self.assertRaises(ValueError):
                validate_submission(self.csv_file(HEADER, [row, *self.valid_rows()[1:]]), self.ids)

    def test_rejects_prediction_changes_that_remain_binary(self):
        rows = self.valid_rows()
        rows[0][1] = "0"
        with self.assertRaisesRegex(ValueError, "differ from decoder"):
            validate_submission(self.csv_file(HEADER, rows), self.ids, self.predictions)

    def test_rejects_invalid_utf8(self):
        path = self.folder / "bad.csv"
        path.write_bytes(b"id,L0,L1,L2,L3\n\xff,0,0,0,0\n")
        with self.assertRaises(UnicodeDecodeError):
            validate_submission(path, self.ids)

    def test_writer_preserves_existing_file_and_removes_failed_temporary(self):
        path = self.folder / "existing.csv"
        path.write_bytes(b"preserve me")
        with self.assertRaises(FileExistsError):
            write_submission(path, self.ids, self.predictions)
        self.assertEqual(path.read_bytes(), b"preserve me")
        new_path = self.folder / "bad.csv"
        before = set(self.folder.iterdir())
        with self.assertRaises(ValueError):
            write_submission(new_path, [self.ids[0]] * 3, self.predictions)
        self.assertEqual(set(self.folder.iterdir()), before)
        with self.assertRaises(ValueError):
            write_submission(new_path, self.ids, np.full((3, 4), np.nan))
        self.assertFalse(new_path.exists())

    def test_test_loader_requires_unlabelled_schema_and_template_order(self):
        detectors = [f"D{i}" for i in range(70)]
        header = ["id", "basis", *detectors]
        rows = [[shot, basis, *(["0"] * 70)] for shot, basis in zip(self.ids, ("Z", "X", "Z"))]
        test = self.csv_file(header, rows, "test.csv")
        sample = self.csv_file(HEADER, self.valid_rows(), "sample.csv")
        data, summary = load_test_data(test, sample)
        self.assertEqual(data["id"].tolist(), self.ids)
        self.assertEqual(summary["basis_counts"], {"X": 1, "Z": 2})
        self.assertNotIn("observable_flips", data)
        self.csv_file(HEADER, self.valid_rows()[::-1], "sample.csv")
        with self.assertRaises(ValueError):
            load_test_data(test, sample)
        self.csv_file([*header, "L0", "L1", "L2", "L3"], [r + ["0"] * 4 for r in rows], "test.csv")
        with self.assertRaises(ValueError):
            load_test_data(test, sample)

    def test_mixed_basis_routing_and_chunked_inference(self):
        circuits = {"X": object(), "Z": object()}
        events = np.zeros((3, 70), dtype=bool)
        events[:, :3] = self.predictions[:, :3]
        bases = np.array(["Z", "X", "Z"])

        def sentinel(circuit, rows):
            return np.column_stack((rows[:, :3], np.full(len(rows), circuit is circuits["Z"])))

        expected = np.column_stack((events[:, :3], bases == "Z"))
        for size in (None, 1, 2, 100):
            np.testing.assert_array_equal(predict_checked(circuits, events, bases, sentinel, size), expected)
        self.assertEqual(predict_checked(circuits, np.zeros((0, 70), bool), np.array([], dtype=str), sentinel).shape, (0, 4))
        with self.assertRaises(ValueError):
            predict_checked(circuits, events, bases, sentinel, 0)
        with self.assertRaises(ValueError):
            predict_checked(circuits, events, ["X"], sentinel)


if __name__ == "__main__":
    unittest.main()
