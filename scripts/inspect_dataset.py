import argparse
import ast
import csv
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("invalid NumPy magic")
        major, minor = stream.read(2)
        if major == 1:
            length_bytes = 2
            length_format = "<H"
            encoding = "latin1"
        elif major in (2, 3):
            length_bytes = 4
            length_format = "<I"
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"unsupported NumPy format version {major}.{minor}")
        raw_length = stream.read(length_bytes)
        if len(raw_length) != length_bytes:
            raise ValueError("truncated NumPy header length")
        header_length = struct.unpack(length_format, raw_length)[0]
        header_data = stream.read(header_length)
        if len(header_data) != header_length:
            raise ValueError("truncated NumPy header")
    header = ast.literal_eval(header_data.decode(encoding).strip())
    if not isinstance(header, dict) or "shape" not in header or "descr" not in header:
        raise ValueError("NumPy header is missing shape or dtype")
    return header


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "chan_id" not in reader.fieldnames:
            raise ValueError("CSV has no chan_id column")
        return list(reader)


def has_hdf5_signature(path: Path) -> bool:
    signature = b"\x89HDF\r\n\x1a\n"
    file_size = path.stat().st_size
    offset = 0
    with path.open("rb") as stream:
        while offset + len(signature) <= file_size:
            stream.seek(offset)
            if stream.read(len(signature)) == signature:
                return True
            offset = 512 if offset == 0 else offset * 2
    return False


def describe_dataset(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset directory does not exist: {root}", file=sys.stderr)
        return 2

    files = sorted(path for path in root.rglob("*") if path.is_file())
    print(f"Dataset root: {root.resolve()}")
    print(f"Files discovered: {len(files)}")
    print("\nFormat counts:")
    for extension, count in sorted(
        Counter(path.suffix.lower() or "<no extension>" for path in files).items()
    ):
        print(f"  {extension}: {count}")

    errors = 0
    split_shapes: dict[str, dict[str, tuple[int, ...]]] = {
        "train": {},
        "test": {},
    }
    for path in files:
        relative = path.relative_to(root)
        detail = ""
        if path.suffix.lower() == ".npy":
            try:
                header = read_npy_header(path)
                detail = f"shape={header['shape']} dtype={header['descr']}"
                if (
                    len(relative.parts) >= 3
                    and relative.parts[0] == "data"
                    and relative.parts[1] in split_shapes
                ):
                    split_shapes[relative.parts[1]][path.stem] = tuple(
                        header["shape"]
                    )
            except (OSError, ValueError, SyntaxError, UnicodeError, struct.error) as exc:
                errors += 1
                detail = f"ERROR={type(exc).__name__}: {exc}"
        elif path.suffix.lower() == ".csv":
            try:
                rows = read_label_rows(path)
                detail = f"rows={len(rows)} columns={list(rows[0]) if rows else []}"
            except (OSError, csv.Error, UnicodeError, ValueError) as exc:
                errors += 1
                detail = f"ERROR={type(exc).__name__}: {exc}"
        elif path.suffix.lower() in (".h5", ".hdf5"):
            try:
                detail = (
                    "HDF5 signature confirmed"
                    if has_hdf5_signature(path)
                    else "HDF5 signature not found"
                )
                if detail.endswith("not found"):
                    errors += 1
            except OSError as exc:
                errors += 1
                detail = f"ERROR={type(exc).__name__}: {exc}"
        elif path.suffix.lower() == ".log":
            try:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    line_count = sum(1 for _ in stream)
                detail = f"lines={line_count}"
            except OSError as exc:
                errors += 1
                detail = f"ERROR={type(exc).__name__}: {exc}"
        print(f"  {relative} | {path.stat().st_size} bytes | {detail}".rstrip())

    label_path = root / "labeled_anomalies.csv"
    if label_path.exists():
        try:
            labels = read_label_rows(label_path)
        except (OSError, csv.Error, UnicodeError, ValueError) as exc:
            print(f"\nERROR: unable to inspect labels: {type(exc).__name__}: {exc}")
            return 1

        channel_counts = Counter(row["chan_id"] for row in labels)
        duplicates = sorted(channel for channel, count in channel_counts.items() if count > 1)
        print("\nLabel summary:")
        print(f"  Rows: {len(labels)}")
        print(f"  Distinct channel IDs: {len(channel_counts)}")
        print(f"  Duplicate channel IDs: {duplicates or 'none'}")
        for channel in duplicates:
            entries = [
                f"spacecraft={row.get('spacecraft', '')}, "
                f"anomaly_sequences={row.get('anomaly_sequences', '')}"
                for row in labels
                if row["chan_id"] == channel
            ]
            print(f"    {channel}: {'; '.join(entries)}")

        for split in ("train", "test"):
            split_dir = root / "data" / split
            if not split_dir.is_dir():
                print(f"  {split}: directory not found at {split_dir.relative_to(root)}")
                continue
            array_ids = {path.stem for path in split_dir.glob("*.npy")}
            label_ids = set(channel_counts)
            print(f"  {split} arrays: {len(array_ids)}")
            print(f"  {split} arrays without a label row: {sorted(array_ids - label_ids) or 'none'}")
            print(f"  Label IDs without a {split} array: {sorted(label_ids - array_ids) or 'none'}")
            if split == "test":
                length_mismatches = []
                for row in labels:
                    shape = split_shapes[split].get(row["chan_id"])
                    if shape and row.get("num_values", "").isdigit():
                        if shape[0] != int(row["num_values"]):
                            length_mismatches.append(
                                (row["chan_id"], shape[0], row["num_values"])
                            )
                print(
                    "  Test/label length mismatches: "
                    f"{length_mismatches or 'none'}"
                )

        if duplicates:
            print(
                "  Review duplicate label rows before computing label-based "
                "evaluation metrics."
            )

    print(f"\nInspection errors: {errors}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect dataset file formats, NumPy headers, and label mappings."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Dataset",
        help="dataset directory (defaults to the discovered project Dataset directory)",
    )
    args = parser.parse_args()
    return describe_dataset(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
