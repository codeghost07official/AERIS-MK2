import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Training dependencies are missing. Install requirements.txt into the "
        "existing Python 3.12 environment, then rerun this script."
    ) from exc

from app.services.classification_evaluation import calculate_binary_metrics
from app.services.nasa_data import (
    DatasetValidationError,
    audit_channel_alignment,
    inspect_npy_matrix,
    load_unambiguous_labels,
)
from app.services.regression_evaluation import EvaluationError
from app.services.sequence_split import SplitError, chronological_split


MODEL_VERSION = "aeris-linear-forecaster-v1"
TRAINING_FRACTION = 0.6
VALIDATION_FRACTION = 0.2
TESTING_FRACTION = 0.2
THRESHOLD_QUANTILE = 0.99


class TrainingError(RuntimeError):
    """Raised when the real NASA dataset cannot be used for a valid fit."""


def load_channel_array(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    metadata = inspect_npy_matrix(path)
    if (metadata.rows, metadata.columns) != expected_shape:
        raise TrainingError(f"Array shape changed while reading {path.name}.")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Unable to load NumPy array {path.name}.") from exc
    if values.shape != expected_shape or values.dtype.kind != "f":
        raise TrainingError(
            f"{path.name} must be a float matrix of shape {expected_shape}."
        )
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise TrainingError(f"{path.name} contains non-finite telemetry values.")
    return values


def sequence_pairs(
    values: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    if end - start < 2:
        raise TrainingError("A chronological partition needs at least two records.")
    section = values[start:end]
    return section[:-1], section[1:]


def residual_scores(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(actual - predicted), axis=1))


def anomaly_targets(length: int, intervals: tuple[tuple[int, int], ...]) -> np.ndarray:
    targets = np.zeros(length, dtype=bool)
    for start, end in intervals:
        targets[start : end + 1] = True
    return targets


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise TrainingError(f"Unable to write artifact {path}.") from exc


def write_atomic_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        joblib.dump(payload, temporary_path, compress=3)
        with temporary_path.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except (OSError, ValueError, TypeError) as exc:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise TrainingError(f"Unable to write model artifact {path}.") from exc


def train_dataset(dataset_root: Path, output_directory: Path) -> dict[str, Any]:
    train_directory = dataset_root / "data" / "train"
    test_directory = dataset_root / "data" / "test"
    if not train_directory.is_dir() or not test_directory.is_dir():
        raise TrainingError(
            "Expected Dataset/data/train and Dataset/data/test directories."
        )

    train_paths = {path.stem: path for path in train_directory.glob("*.npy")}
    test_paths = {path.stem: path for path in test_directory.glob("*.npy")}
    if not train_paths or not test_paths:
        raise TrainingError("No training or test NumPy arrays were found.")
    labels = load_unambiguous_labels(dataset_root / "labeled_anomalies.csv")

    train_shapes: dict[str, tuple[int, int]] = {}
    test_shapes: dict[str, tuple[int, int]] = {}
    for channel_id, path in train_paths.items():
        metadata = inspect_npy_matrix(path)
        train_shapes[channel_id] = (metadata.rows, metadata.columns)
    for channel_id, path in test_paths.items():
        metadata = inspect_npy_matrix(path)
        test_shapes[channel_id] = (metadata.rows, metadata.columns)

    alignment = audit_channel_alignment(train_shapes, test_shapes, labels)
    if alignment.train_without_test or alignment.test_without_train:
        raise TrainingError(
            "Training and test channel IDs differ. "
            f"train-only={list(alignment.train_without_test)}, "
            f"test-only={list(alignment.test_without_train)}"
        )
    if set(train_paths) != set(test_paths):
        raise TrainingError("Training and test channel inventories do not match.")
    for channel_id in sorted(train_paths):
        if train_shapes[channel_id][1] != test_shapes[channel_id][1]:
            raise TrainingError(
                f"Feature count differs between train/test for {channel_id}."
            )

    model_channels: dict[str, dict[str, Any]] = {}
    channel_reports = []
    squared_error_sum = 0.0
    absolute_percentage_error_sum = 0.0
    total_scalar_count = 0
    nonzero_target_count = 0
    zero_target_count = 0
    confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    classification_sample_count = 0
    regression_channels = []
    classification_channels = []

    for channel_id in sorted(train_paths):
        train_values = load_channel_array(train_paths[channel_id], train_shapes[channel_id])
        test_values = load_channel_array(test_paths[channel_id], test_shapes[channel_id])
        if train_values.shape[1] < 1 or train_values.shape[0] < 10:
            raise TrainingError(f"Not enough training records for {channel_id}.")
        try:
            split = chronological_split(
                train_values,
                training_fraction=TRAINING_FRACTION,
                validation_fraction=VALIDATION_FRACTION,
                testing_fraction=TESTING_FRACTION,
            )
        except SplitError as exc:
            raise TrainingError(f"Unable to split {channel_id}: {exc}") from exc

        training_end = split.training_end
        validation_end = split.validation_end
        train_x, train_y = sequence_pairs(train_values, 0, training_end)
        validation_x, validation_y = sequence_pairs(
            train_values, training_end, validation_end
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_train_x = x_scaler.fit_transform(train_x)
        scaled_train_y = y_scaler.fit_transform(train_y)
        regressor = LinearRegression()
        regressor.fit(scaled_train_x, scaled_train_y)

        validation_predictions = y_scaler.inverse_transform(
            regressor.predict(x_scaler.transform(validation_x))
        )
        validation_residuals = residual_scores(validation_y, validation_predictions)
        if not np.isfinite(validation_residuals).all():
            raise TrainingError(
                f"Validation residuals are non-finite for {channel_id}."
            )
        threshold = float(np.quantile(validation_residuals, THRESHOLD_QUANTILE))

        test_x, test_y = sequence_pairs(test_values, 0, test_values.shape[0])
        test_predictions = y_scaler.inverse_transform(
            regressor.predict(x_scaler.transform(test_x))
        )
        if not np.isfinite(test_predictions).all():
            raise TrainingError(f"Test predictions are non-finite for {channel_id}.")
        errors = test_y - test_predictions
        squared_error_sum += float(np.square(errors).sum(dtype=np.float64))
        total_scalar_count += int(errors.size)
        nonzero_targets = test_y != 0
        nonzero_target_count += int(nonzero_targets.sum())
        zero_target_count += int(test_y.size - int(nonzero_targets.sum()))
        if nonzero_targets.any():
            absolute_percentage_error_sum += float(
                (
                    np.abs(errors[nonzero_targets])
                    / np.abs(test_y[nonzero_targets])
                    * 100
                ).sum(dtype=np.float64)
            )

        test_residuals = residual_scores(test_y, test_predictions)
        channel_labels = labels.valid_labels.get(channel_id)
        channel_report: dict[str, Any] = {
            "channel_id": channel_id,
            "feature_count": int(train_values.shape[1]),
            "training_record_count": int(training_end),
            "validation_record_count": int(validation_end - training_end),
            "test_record_count": int(test_values.shape[0]),
            "regression_sample_count": int(test_y.size),
            "residual_threshold": threshold,
            "threshold_calibration_quantile": THRESHOLD_QUANTILE,
        }
        if channel_labels and channel_labels.num_values == test_values.shape[0]:
            actual_flags = anomaly_targets(
                test_values.shape[0], channel_labels.intervals
            )[1:]
            predicted_flags = test_residuals > threshold
            channel_metrics = calculate_binary_metrics(
                actual_flags.tolist(), predicted_flags.tolist()
            )
            for name, value in channel_metrics["confusion_matrix"].items():
                confusion[name] += int(value)
            classification_sample_count += int(channel_metrics["sample_count"])
            channel_report["classification_evaluation"] = channel_metrics
            classification_channels.append(channel_id)
        else:
            channel_report["classification_evaluation"] = None
            if channel_id not in labels.duplicate_ids:
                if channel_labels:
                    channel_report["classification_exclusion"] = (
                        "test length does not match labeled sequence length"
                    )
                else:
                    channel_report["classification_exclusion"] = (
                        "no unique ground-truth label row"
                    )

        model_channels[channel_id] = {
            "feature_count": int(train_values.shape[1]),
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "regressor": regressor,
            "residual_threshold": threshold,
        }
        channel_reports.append(channel_report)
        regression_channels.append(channel_id)

    if not total_scalar_count:
        raise TrainingError("No held-out predictions were generated.")
    mse = squared_error_sum / total_scalar_count
    regression_metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape_percent": (
            absolute_percentage_error_sum / nonzero_target_count
            if nonzero_target_count
            else None
        ),
        "sample_count": total_scalar_count,
        "zero_target_count": zero_target_count,
        "mape_excluded_zero_count": zero_target_count,
    }

    if classification_sample_count:
        precision_denominator = confusion["true_positive"] + confusion["false_positive"]
        recall_denominator = confusion["true_positive"] + confusion["false_negative"]
        precision = (
            confusion["true_positive"] / precision_denominator
            if precision_denominator
            else None
        )
        recall = (
            confusion["true_positive"] / recall_denominator
            if recall_denominator
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        classification_metrics = {
            "confusion_matrix": confusion,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": (
                confusion["true_positive"] + confusion["true_negative"]
            )
            / classification_sample_count,
            "sample_count": classification_sample_count,
        }
    else:
        classification_metrics = None

    report: dict[str, Any] = {
        "artifact_id": uuid.uuid4().hex,
        "model_version": MODEL_VERSION,
        "model_name": "Per-channel one-step Linear Regression forecaster",
        "training_dataset": (
            "NASA SMAP/MSL supplied Dataset/data/train arrays."
        ),
        "prediction_method": (
            "Predict each feature at record t+1 from the complete feature vector "
            "at record t. Scale inputs and targets using training-partition "
            "statistics only. Flag a row when per-row prediction RMSE exceeds "
            "that channel's 99th-percentile validation residual."
        ),
        "split_method": (
            "For each supplied training sequence, use ordered contiguous "
            "60% fit, 20% validation, and 20% unused partitions. Construct "
            "one-step pairs within each partition so no pair crosses a "
            "partition boundary. Fit scalers and regressors on the fit partition "
            "only; calibrate thresholds on validation only. The separate "
            "supplied NASA test arrays are the final held-out evaluation set."
        ),
        "evaluation_dataset": (
            "NASA SMAP/MSL supplied Dataset/data/test arrays; one-step "
            "predictions are evaluated within each channel's test sequence."
        ),
        "threshold_calibration": (
            "Per-channel 99th percentile of per-row multivariate RMSE on the "
            "chronological validation partition."
        ),
        "preprocessing": {
            "input_scaler": "StandardScaler, fitted on the fit partition only",
            "target_scaler": "StandardScaler, fitted on the fit partition only",
            "target_shape": "Complete next-record feature vector (multi-output)",
            "threshold_quantile": THRESHOLD_QUANTILE,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "label_interval_interpretation": (
            "CSV anomaly interval endpoints are treated as inclusive. "
            "P-2 is excluded because its label rows conflict; T-10 is excluded "
            "because it has no label row."
        ),
        "regression_metrics_scope": (
            "Aggregated scalar feature predictions over all channels in the "
            "provided test arrays, including channels without labels."
        ),
        "regression_metrics": regression_metrics,
        "classification_metrics_scope": (
            "Per-record anomaly flags compared only with unique, length-matched "
            "test labels. This is a separate measure from regression error."
        ),
        "classification_metrics": classification_metrics,
        "regression_channels": regression_channels,
        "classification_channels": classification_channels,
        "excluded_from_classification": {
            "duplicate_labels": list(labels.duplicate_ids),
            "missing_unique_labels": list(alignment.test_without_labels),
            "length_mismatches": list(alignment.label_length_mismatches),
            "malformed_rows": list(labels.invalid_rows),
        },
        "training_channel_count": len(train_paths),
        "evaluation_channels": regression_channels,
        "training_feature_count_distribution": {
            str(width): sum(shape[1] == width for shape in train_shapes.values())
            for width in sorted({shape[1] for shape in train_shapes.values()})
        },
        "threshold_quantile": THRESHOLD_QUANTILE,
        "channels": channel_reports,
        "limitations": [
            "The available curriculum summary names Linear Regression but does not explicitly establish residual-threshold anomaly detection as a taught method; obtain teacher approval before presenting anomaly flags as syllabus-compliant.",
            "The chronological partition assumes each file's row order is temporal; this should be confirmed from authoritative dataset documentation.",
            "The train and test arrays are treated as the supplied split; metadata alone does not establish their collection chronology or independence.",
            "MAPE excludes zero targets and can be unstable for nonzero targets close to zero.",
            "Regression metrics summarize numerical forecasting and are not classification accuracy.",
        ],
    }
    if not labels.valid_labels:
        report["classification_metrics"] = None
        report["classification_metrics_scope"] = (
            "Unavailable because no unambiguous ground-truth labels are present."
        )

    bundle = {
        "artifact_id": report["artifact_id"],
        "model_version": MODEL_VERSION,
        "model_name": report["model_name"],
        "prediction_method": report["prediction_method"],
        "channels": model_channels,
        "evaluation": report,
    }
    model_path = output_directory / "aeris_linear_forecaster.joblib"
    report_path = output_directory / "evaluation_report.json"
    write_atomic_joblib(model_path, bundle)
    write_atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a per-channel one-step Linear Regression forecaster on the "
            "local NASA SMAP/MSL training arrays."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="NASA dataset directory (default: project Dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="artifact output directory (default: project models)",
    )
    args = parser.parse_args()
    try:
        report = train_dataset(args.dataset, args.output)
    except (
        DatasetValidationError,
        EvaluationError,
        TrainingError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("Training completed on the supplied dataset.")
    print(f"Model artifact: {args.output / 'aeris_linear_forecaster.joblib'}")
    print(f"Evaluation report: {args.output / 'evaluation_report.json'}")
    print(f"Trained channels: {report['training_channel_count']}")
    print(f"Regression metrics: {json.dumps(report['regression_metrics'])}")
    print(
        "Classification metrics: "
        f"{json.dumps(report['classification_metrics'])}"
    )
    print(
        "Classification exclusions: "
        f"{json.dumps(report['excluded_from_classification'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
