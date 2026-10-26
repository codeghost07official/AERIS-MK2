import csv
import struct

import pytest

from app.services.nasa_data import (
    DatasetValidationError,
    audit_channel_alignment,
    inspect_npy_matrix,
    iter_npy_rows,
    load_unambiguous_labels,
)


def write_npy(path, rows, *, dtype="<f8"):
    columns = len(rows[0])
    header = {
        "descr": dtype,
        "fortran_order": False,
        "shape": (len(rows), columns),
    }
    header_text = repr(header).encode("latin1")
    padding = (16 - ((10 + len(header_text) + 1) % 16)) % 16
    header_text += b" " * padding + b"\n"
    with path.open("wb") as stream:
        stream.write(b"\x93NUMPY")
        stream.write(bytes((1, 0)))
        stream.write(struct.pack("<H", len(header_text)))
        stream.write(header_text)
        for row in rows:
            stream.write(struct.pack("<" + "d" * columns, *row))


def test_streams_numpy_matrix_rows(tmp_path):
    path = tmp_path / "channel.npy"
    write_npy(path, [(1.0, 2.0), (3.0, 4.0)])

    metadata = inspect_npy_matrix(path)
    assert (metadata.rows, metadata.columns, metadata.dtype) == (2, 2, "<f8")
    assert list(iter_npy_rows(metadata)) == [(1.0, 2.0), (3.0, 4.0)]


def test_rejects_bad_shapes_and_non_finite_values(tmp_path):
    path = tmp_path / "invalid.npy"
    write_npy(path, [(1.0, float("nan"))])
    metadata = inspect_npy_matrix(path)

    with pytest.raises(DatasetValidationError, match="Non-finite"):
        list(iter_npy_rows(metadata))

    path.write_bytes(b"not a numpy file")
    with pytest.raises(DatasetValidationError, match="signature"):
        inspect_npy_matrix(path)


def write_labels(path, rows):
    fields = ["chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label(channel, intervals, length):
    return {
        "chan_id": channel,
        "spacecraft": "SMAP",
        "anomaly_sequences": repr(intervals),
        "class": "[point]",
        "num_values": str(length),
    }


def test_conflicting_and_missing_labels_are_excluded(tmp_path):
    path = tmp_path / "labeled_anomalies.csv"
    write_labels(
        path,
        [
            label("P-2", [[2, 5]], 10),
            label("P-2", [[3, 6]], 10),
            label("A-1", [[1, 2]], 8),
        ],
    )

    audit = load_unambiguous_labels(path)
    alignment = audit_channel_alignment(
        {"P-2": (5, 25), "A-1": (4, 25), "T-10": (3, 55)},
        {"P-2": (10, 25), "A-1": (8, 25), "T-10": (3, 55)},
        audit,
    )

    assert audit.duplicate_ids == ("P-2",)
    assert len(audit.duplicate_rows["P-2"]) == 2
    assert set(audit.valid_labels) == {"A-1"}
    assert alignment.duplicate_label_ids == ("P-2",)
    assert alignment.test_without_labels == ("T-10",)
    assert alignment.label_length_mismatches == ()


def test_rejects_out_of_range_label_intervals(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels(path, [label("A-1", [[0, 8]], 8)])

    audit = load_unambiguous_labels(path)

    assert not audit.valid_labels
    assert len(audit.invalid_rows) == 1
