"""Strict CSV preparation around an explicitly supplied decoder."""

import csv
from itertools import zip_longest
import os
from pathlib import Path
import tempfile

import numpy as np

from stage01_support import check_binary, decode_by_basis, inspect_csv, sha256
from hackathon_2026 import load_kaggle_csv, save_submission

HEADER = ["id", "L0", "L1", "L2", "L3"]


def validate_submission(path, expected_ids, expected_predictions=None):
    ids = [str(value) for value in expected_ids]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected IDs must be nonempty and unique")
    if expected_predictions is not None:
        expected_predictions = check_binary(expected_predictions, 4, "expected predictions")
        if len(expected_predictions) != len(ids):
            raise ValueError("Expected prediction count differs from ID count")
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError("Header must be exactly id,L0,L1,L2,L3")
        for index, (row, shot_id) in enumerate(zip_longest(reader, ids)):
            if row is None or shot_id is None:
                raise ValueError("Submission row count differs from test data")
            if len(row) != 5:
                raise ValueError(f"Row {index + 2}: expected exactly five columns")
            if not row[0] or row[0] in seen:
                raise ValueError(f"Row {index + 2}: missing or duplicate ID")
            if row[0] != shot_id:
                raise ValueError(f"Row {index + 2}: ID differs from template order")
            if any(value not in ("0", "1") for value in row[1:]):
                raise ValueError(f"Row {index + 2}: predictions must be literal 0 or 1")
            if expected_predictions is not None and row[1:] != [str(int(v)) for v in expected_predictions[index]]:
                raise ValueError(f"Row {index + 2}: predictions differ from decoder output")
            seen.add(row[0])
    return {"rows": len(seen), "columns": HEADER, "sha256": sha256(path),
            "id_order_matches": True, "binary": True, "duplicate_ids": 0, "encoding": "UTF-8"}


def load_test_data(test_path, sample_path):
    # Schema validation happens before the organizer loader can coerce values to bool.
    summary = inspect_csv(test_path, labels=False)
    if summary["rows"] == 0:
        raise ValueError("Test data must contain at least one shot")
    data = load_kaggle_csv(test_path)
    validate_submission(sample_path, data["id"])
    return data, summary


def predict_checked(circuits, events, bases, decoder, batch_size=None):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    events = check_binary(events, 70, "detector_events")
    bases = np.asarray(bases)
    if bases.shape != (len(events),) or not np.isin(bases, ("X", "Z")).all():
        raise ValueError("One valid X/Z basis is required per shot")
    output = np.empty((len(events), 4), dtype=np.uint8)
    step = batch_size or max(1, len(events))
    for start in range(0, len(events), step):
        stop = min(start + step, len(events))
        output[start:stop] = decode_by_basis(circuits, events[start:stop], bases[start:stop], decoder)
    return output


def write_submission(path, ids, predictions):
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace {destination}")
    predictions = check_binary(predictions, 4, "predictions")
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != len(predictions):
        raise ValueError("One ID is required per prediction")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".csv", delete=False) as stream:
            temporary = Path(stream.name)
        save_submission(temporary, ids, predictions)
        result = validate_submission(temporary, ids, predictions)
        # A same-filesystem link publishes the validated bytes without replacing an existing path.
        os.link(temporary, destination)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
