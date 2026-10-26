import ast
import csv
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


class DatasetValidationError(ValueError):
    """Raised when NASA telemetry or its label metadata is invalid."""


@dataclass(frozen=True)
class NpyMatrixMetadata:
    path: Path
    rows: int
    columns: int
    dtype: str
    data_offset: int
    byte_order: str
    struct_code: str
    item_size: int


def inspect_npy_matrix(path: Path, *, max_bytes: int = 256 * 1024 * 1024) -> NpyMatrixMetadata:
    path = Path(path)
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise DatasetValidationError(f"Unable to read NumPy file: {path}") from exc

    with stream:
        if stream.read(6) != b"\x93NUMPY":
            raise DatasetValidationError(f"Invalid NumPy file signature: {path.name}")
        version = stream.read(2)
        if len(version) != 2:
            raise DatasetValidationError(f"Truncated NumPy version: {path.name}")
        major, minor = version
        if major == 1:
            length_bytes, length_format, encoding = 2, "<H", "latin1"
        elif major in (2, 3):
            length_bytes, length_format = 4, "<I"
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise DatasetValidationError(
                f"Unsupported NumPy format version {major}.{minor}: {path.name}"
            )

        raw_length = stream.read(length_bytes)
        if len(raw_length) != length_bytes:
            raise DatasetValidationError(f"Truncated NumPy header: {path.name}")
        header_length = struct.unpack(length_format, raw_length)[0]
        if header_length > 1_000_000:
            raise DatasetValidationError(f"NumPy header is too large: {path.name}")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise DatasetValidationError(f"Truncated NumPy header: {path.name}")
        data_offset = stream.tell()

    try:
        header = ast.literal_eval(header_bytes.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise DatasetValidationError(f"Malformed NumPy header: {path.name}") from exc
    if not isinstance(header, dict):
        raise DatasetValidationError(f"Malformed NumPy header: {path.name}")

    shape = header.get("shape")
    dtype = header.get("descr")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise DatasetValidationError(
            f"Expected a non-empty 2-D telemetry matrix: {path.name}"
        )
    if header.get("fortran_order") is not False:
        raise DatasetValidationError(
            f"Fortran-ordered arrays are not supported: {path.name}"
        )
    if not isinstance(dtype, str) or not re.fullmatch(r"[<>=|][f](4|8)", dtype):
        raise DatasetValidationError(
            f"Expected a float32 or float64 NumPy array: {path.name}"
        )

    item_size = int(dtype[-1])
    byte_order = dtype[0]
    if byte_order in ("<", "|"):
        endian = "<"
    elif byte_order == ">":
        endian = ">"
    else:
        endian = "<" if struct.pack("=I", 1)[0] == 1 else ">"
    payload_size = shape[0] * shape[1] * item_size
    file_size = path.stat().st_size
    if payload_size > max_bytes:
        raise DatasetValidationError(
            f"NumPy matrix exceeds the {max_bytes}-byte read limit: {path.name}"
        )
    if file_size != data_offset + payload_size:
        raise DatasetValidationError(
            f"NumPy payload size does not match its shape: {path.name}"
        )

    return NpyMatrixMetadata(
        path=path,
        rows=shape[0],
        columns=shape[1],
        dtype=dtype,
        data_offset=data_offset,
        byte_order=endian,
        struct_code="f" if item_size == 4 else "d",
        item_size=item_size,
    )


def iter_npy_rows(
    metadata: NpyMatrixMetadata, *, validate_finite: bool = True
) -> Iterator[tuple[float, ...]]:
    row_bytes = metadata.columns * metadata.item_size
    row_format = metadata.byte_order + metadata.struct_code * metadata.columns
    try:
        with metadata.path.open("rb") as stream:
            stream.seek(metadata.data_offset)
            for row_index in range(metadata.rows):
                raw = stream.read(row_bytes)
                if len(raw) != row_bytes:
                    raise DatasetValidationError(
                        f"Truncated NumPy data at row {row_index}: "
                        f"{metadata.path.name}"
                    )
                row = struct.unpack(row_format, raw)
                if validate_finite and not all(math.isfinite(value) for value in row):
                    raise DatasetValidationError(
                        f"Non-finite value at row {row_index}: {metadata.path.name}"
                    )
                yield row
    except OSError as exc:
        raise DatasetValidationError(
            f"Unable to stream NumPy data: {metadata.path.name}"
        ) from exc


@dataclass(frozen=True)
class ChannelLabels:
    channel_id: str
    spacecraft: str
    num_values: int
    intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LabelAudit:
    valid_labels: dict[str, ChannelLabels]
    duplicate_ids: tuple[str, ...]
    duplicate_rows: dict[str, tuple[dict[str, str], ...]]
    invalid_rows: tuple[str, ...]


def load_unambiguous_labels(path: Path) -> LabelAudit:
    try:
        stream: TextIO = Path(path).open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise DatasetValidationError(f"Unable to read label CSV: {path}") from exc

    try:
        with stream:
            reader = csv.DictReader(stream)
            required = {
                "chan_id",
                "spacecraft",
                "anomaly_sequences",
                "num_values",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise DatasetValidationError(
                    "Label CSV is missing required NASA anomaly metadata columns."
                )
            rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise DatasetValidationError("Unable to parse label CSV.") from exc

    counts = Counter(row.get("chan_id", "") for row in rows)
    duplicates = tuple(sorted(channel for channel, count in counts.items() if count > 1))
    duplicate_rows = {
        channel: tuple(row for row in rows if row.get("chan_id") == channel)
        for channel in duplicates
    }
    labels: dict[str, ChannelLabels] = {}
    invalid: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        channel_id = row.get("chan_id", "")
        if channel_id in duplicates:
            continue
        try:
            num_values = int(row["num_values"])
            if num_values <= 0:
                raise ValueError("num_values must be positive")
            raw_intervals = ast.literal_eval(row["anomaly_sequences"])
            if not isinstance(raw_intervals, list):
                raise ValueError("anomaly_sequences must be a list")
            intervals: list[tuple[int, int]] = []
            for interval in raw_intervals:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or any(not isinstance(index, int) for index in interval)
                ):
                    raise ValueError("anomaly interval must contain two integers")
                start, end = interval
                if start < 0 or start > end or end >= num_values:
                    raise ValueError("anomaly interval is outside the labeled sequence")
                intervals.append((start, end))
            labels[channel_id] = ChannelLabels(
                channel_id=channel_id,
                spacecraft=row["spacecraft"],
                num_values=num_values,
                intervals=tuple(intervals),
            )
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            invalid.append(f"row {row_number} ({channel_id}): {exc}")

    return LabelAudit(
        valid_labels=labels,
        duplicate_ids=duplicates,
        duplicate_rows=duplicate_rows,
        invalid_rows=tuple(invalid),
    )


@dataclass(frozen=True)
class ChannelAlignment:
    matched_ids: tuple[str, ...]
    train_without_test: tuple[str, ...]
    test_without_train: tuple[str, ...]
    train_without_labels: tuple[str, ...]
    test_without_labels: tuple[str, ...]
    label_length_mismatches: tuple[str, ...]
    duplicate_label_ids: tuple[str, ...]
    invalid_label_rows: tuple[str, ...]


def audit_channel_alignment(
    train_shapes: dict[str, tuple[int, ...]],
    test_shapes: dict[str, tuple[int, ...]],
    labels: LabelAudit,
) -> ChannelAlignment:
    train_ids, test_ids = set(train_shapes), set(test_shapes)
    test_missing_label = test_ids - set(labels.valid_labels) - set(labels.duplicate_ids)
    mismatches = []
    for channel_id in sorted(test_ids & set(labels.valid_labels)):
        shape = test_shapes[channel_id]
        if not shape or shape[0] != labels.valid_labels[channel_id].num_values:
            mismatches.append(channel_id)
    return ChannelAlignment(
        matched_ids=tuple(sorted(train_ids & test_ids)),
        train_without_test=tuple(sorted(train_ids - test_ids)),
        test_without_train=tuple(sorted(test_ids - train_ids)),
        train_without_labels=tuple(
            sorted(train_ids - set(labels.valid_labels) - set(labels.duplicate_ids))
        ),
        test_without_labels=tuple(sorted(test_missing_label)),
        label_length_mismatches=tuple(mismatches),
        duplicate_label_ids=labels.duplicate_ids,
        invalid_label_rows=labels.invalid_rows,
    )
