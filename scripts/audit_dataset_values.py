import argparse
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.nasa_data import (
    DatasetValidationError,
    audit_channel_alignment,
    inspect_npy_matrix,
    iter_npy_rows,
    load_unambiguous_labels,
)


def inspect_split(root: Path, split: str) -> tuple[dict[str, tuple[int, ...]], int]:
    directory = root / "data" / split
    if not directory.is_dir():
        raise DatasetValidationError(f"Missing {split} directory: {directory}")

    shapes: dict[str, tuple[int, ...]] = {}
    widths: Counter[int] = Counter()
    total_values = 0
    print(f"\n{split.upper()} arrays:")
    for path in sorted(directory.glob("*.npy")):
        metadata = inspect_npy_matrix(path)
        total = 0
        zeros = 0
        minimum = math.inf
        maximum = -math.inf
        first_row = None
        for row in iter_npy_rows(metadata):
            if first_row is None:
                first_row = row
            total += math.fsum(row)
            zeros += sum(value == 0 for value in row)
            minimum = min(minimum, min(row))
            maximum = max(maximum, max(row))
        count = metadata.rows * metadata.columns
        mean = total / count
        shapes[path.stem] = (metadata.rows, metadata.columns)
        widths[metadata.columns] += 1
        total_values += count
        print(
            f"  {path.stem}: shape=({metadata.rows}, {metadata.columns}) "
            f"dtype={metadata.dtype} finite=yes min={minimum:.6g} "
            f"max={maximum:.6g} mean={mean:.6g} zeros={zeros} "
            f"first_row_sample={first_row[:3] if first_row else ()}"
        )
    print(
        f"{split.upper()} summary: channels={len(shapes)} widths={dict(sorted(widths.items()))} "
        f"values={total_values:,}"
    )
    return shapes, total_values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream NASA SMAP/MSL NumPy arrays and audit channel/label alignment "
            "without loading full files into memory."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
    )
    args = parser.parse_args()
    root = args.dataset
    try:
        train_shapes, train_values = inspect_split(root, "train")
        test_shapes, test_values = inspect_split(root, "test")
        labels = load_unambiguous_labels(root / "labeled_anomalies.csv")
    except (DatasetValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    alignment = audit_channel_alignment(train_shapes, test_shapes, labels)
    print("\nLABEL AND CHANNEL AUDIT:")
    print(f"  unambiguous labels: {len(labels.valid_labels)}")
    print(f"  conflicting duplicate IDs (excluded): {list(labels.duplicate_ids)}")
    for channel_id, rows in labels.duplicate_rows.items():
        for row in rows:
            print(
                f"    conflicting {channel_id}: spacecraft={row['spacecraft']} "
                f"num_values={row['num_values']} class={row['class']} "
                f"intervals={row['anomaly_sequences']}"
            )
    print(f"  malformed label rows: {list(labels.invalid_rows)}")
    print("  unambiguous label metadata:")
    for channel_id, label in sorted(labels.valid_labels.items()):
        print(
            f"    {channel_id}: spacecraft={label.spacecraft} "
            f"num_values={label.num_values} intervals={label.intervals}"
        )
    print(f"  train/test channel IDs matched: {len(alignment.matched_ids)}")
    print(f"  train channels without test: {list(alignment.train_without_test)}")
    print(f"  test channels without train: {list(alignment.test_without_train)}")
    print(f"  train channels without unique labels: {list(alignment.train_without_labels)}")
    print(f"  test channels without unique labels: {list(alignment.test_without_labels)}")
    print(f"  test/label length mismatches: {list(alignment.label_length_mismatches)}")
    excluded_ids = sorted(
        set(alignment.duplicate_label_ids)
        | set(alignment.test_without_labels)
        | set(alignment.label_length_mismatches)
    )
    print(f"  excluded from label-based evaluation: {excluded_ids}")
    print(
        "  note: train and test arrays are separate recorded splits; this audit "
        "does not imply a trained model or validate temporal ordering."
    )
    print(
        f"\nTOTAL: train_values={train_values:,} test_values={test_values:,} "
        f"combined_values={train_values + test_values:,}"
    )
    return 0 if not labels.invalid_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
