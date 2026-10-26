import json
import math
import pickle
import zlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_VERSION = "aeris-linear-forecaster-v1"
DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
MODEL_FILENAME = "aeris_linear_forecaster.joblib"
REPORT_FILENAME = "evaluation_report.json"
INFERENCE_BATCH_ROWS = 8_192


class ModelArtifactError(RuntimeError):
    """Raised when a trained-model artifact is missing or incompatible."""


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def artifact_paths(directory: Path = DEFAULT_MODEL_DIRECTORY) -> tuple[Path, Path]:
    directory = Path(directory)
    return directory / MODEL_FILENAME, directory / REPORT_FILENAME


def load_evaluation_report(directory: Path = DEFAULT_MODEL_DIRECTORY) -> dict[str, Any]:
    model_path, report_path = artifact_paths(directory)
    if not model_path.is_file() or not report_path.is_file():
        raise ModelArtifactError(
            "No trained model artifacts are available. Run "
            "scripts\\train_model.py locally before using model inference."
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError("The model evaluation report is unreadable.") from exc
    metrics = report.get("regression_metrics") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("model_version") != MODEL_VERSION
        or not isinstance(report.get("artifact_id"), str)
        or not report["artifact_id"]
        or not isinstance(report.get("model_name"), str)
        or not isinstance(report.get("prediction_method"), str)
        or not isinstance(report.get("split_method"), str)
        or not isinstance(report.get("evaluation_dataset"), str)
        or not isinstance(report.get("channels"), list)
        or not isinstance(metrics, dict)
        or any(
            not _is_finite_number(metrics.get(key))
            for key in ("mse", "rmse")
        )
        or (
            metrics.get("mape_percent") is not None
            and not _is_finite_number(metrics.get("mape_percent"))
        )
        or type(metrics.get("sample_count")) is not int
        or type(metrics.get("mape_excluded_zero_count")) is not int
        or not isinstance(report.get("regression_channels"), list)
        or not isinstance(report.get("classification_channels"), list)
        or not isinstance(report.get("excluded_from_classification"), dict)
    ):
        raise ModelArtifactError("The model evaluation report is incompatible.")
    classification = report.get("classification_metrics")
    if classification is not None:
        if (
            not isinstance(classification, dict)
            or type(classification.get("sample_count")) is not int
            or any(
                classification.get(key) is not None
                and not _is_finite_number(classification.get(key))
                for key in ("precision", "recall", "f1_score")
            )
            or not _is_finite_number(classification.get("accuracy"))
        ):
            raise ModelArtifactError("The model evaluation report is incompatible.")
    for channel in report["channels"]:
        if (
            not isinstance(channel, dict)
            or not isinstance(channel.get("channel_id"), str)
            or type(channel.get("feature_count")) is not int
            or channel["feature_count"] <= 0
        ):
            raise ModelArtifactError("The model evaluation report is incompatible.")
    return report


@lru_cache(maxsize=2)
def _load_bundle_cached(model_path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    try:
        import joblib

        bundle = joblib.load(model_path)
    except (
        ImportError,
        OSError,
        ValueError,
        EOFError,
        TypeError,
        AttributeError,
        KeyError,
        pickle.UnpicklingError,
        zlib.error,
    ) as exc:
        raise ModelArtifactError(
            "The trained model artifact is unreadable or incompatible."
        ) from exc
    if (
        not isinstance(bundle, dict)
        or bundle.get("model_version") != MODEL_VERSION
        or not isinstance(bundle.get("channels"), dict)
    ):
        raise ModelArtifactError("The trained model artifact has an invalid format.")
    return bundle


def load_model_bundle(directory: Path = DEFAULT_MODEL_DIRECTORY) -> dict[str, Any]:
    model_path, _ = artifact_paths(directory)
    if not model_path.is_file():
        raise ModelArtifactError(
            "No trained model is available. Run scripts\\train_model.py locally."
        )
    try:
        modified_ns = model_path.stat().st_mtime_ns
        bundle = _load_bundle_cached(str(model_path.resolve()), modified_ns)
    except OSError as exc:
        raise ModelArtifactError("Unable to access the trained model artifact.") from exc
    return bundle


def available_channels(directory: Path = DEFAULT_MODEL_DIRECTORY) -> list[dict[str, Any]]:
    report = load_evaluation_report(directory)
    return [
        {
            "channel_id": channel["channel_id"],
            "feature_count": channel["feature_count"],
        }
        for channel in report["channels"]
        if isinstance(channel, dict)
        and isinstance(channel.get("channel_id"), str)
        and isinstance(channel.get("feature_count"), int)
    ]


def predict_channel(
    values: list[list[float]] | Mapping[str, list[float]],
    channel_id: str,
    *,
    directory: Path = DEFAULT_MODEL_DIRECTORY,
    record_indices: list[int] | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ModelArtifactError(
            "NumPy is required for model inference. Install requirements.txt."
        ) from exc

    bundle = load_model_bundle(directory)
    report = load_evaluation_report(directory)
    if bundle.get("artifact_id") != report.get("artifact_id"):
        raise ModelArtifactError(
            "The trained model and evaluation report are from different runs."
        )
    channel = bundle["channels"].get(channel_id)
    if not isinstance(channel, dict):
        raise ModelArtifactError(
            f"No trained model exists for channel {channel_id!r}."
        )
    feature_count = channel.get("feature_count")
    if type(feature_count) is not int or feature_count <= 0:
        raise ModelArtifactError(f"Model feature metadata is invalid for {channel_id}.")
    threshold = channel.get("residual_threshold")
    if not _is_finite_number(threshold):
        raise ModelArtifactError(f"Residual threshold is invalid for {channel_id}.")
    if isinstance(values, Mapping):
        expected_names = [f"sensor_{index}" for index in range(1, feature_count + 1)]
        if len(values) != feature_count or any(
            name not in values for name in expected_names
        ):
            raise ValueError(
                f"Channel {channel_id} requires exactly {feature_count} sensor columns."
            )
        record_count = len(values[expected_names[0]])
        if any(len(values[name]) != record_count for name in expected_names):
            raise ValueError("Model sensor columns must have equal record counts.")
    else:
        if any(len(row) != feature_count for row in values):
            raise ValueError(
                f"Channel {channel_id} requires exactly {feature_count} sensor columns."
            )
        record_count = len(values)
    if record_count < 2:
        raise ValueError(
            "At least two rows are needed for one-step-ahead forecasting."
        )
    if record_indices is not None:
        if len(record_indices) != record_count or any(
            current != previous + 1
            for previous, current in zip(record_indices, record_indices[1:])
        ):
            raise ValueError(
                "Model forecasting requires consecutive record_index values. "
                "Descriptive analysis is still available."
            )

    actual_feature_values: list[float] = []
    predicted_feature_values: list[float] = []
    residual_score_values: list[float] = []
    anomaly_flags: list[bool] = []
    try:
        for start in range(0, record_count - 1, INFERENCE_BATCH_ROWS):
            stop = min(start + INFERENCE_BATCH_ROWS, record_count - 1)
            if isinstance(values, Mapping):
                batch = np.column_stack(
                    [values[name][start : stop + 1] for name in expected_names]
                ).astype(np.float64, copy=False)
            else:
                batch = np.asarray(values[start : stop + 1], dtype=np.float64)
            if (
                batch.ndim != 2
                or batch.shape != (stop - start + 1, feature_count)
                or not np.isfinite(batch).all()
            ):
                raise ValueError("Model input must be a finite two-dimensional matrix.")
            batch_actual = batch[1:]
            scaled_inputs = channel["x_scaler"].transform(batch[:-1])
            scaled_predictions = channel["regressor"].predict(scaled_inputs)
            predictions = channel["y_scaler"].inverse_transform(scaled_predictions)
            if (
                predictions.shape != batch_actual.shape
                or not np.isfinite(predictions).all()
            ):
                raise ModelArtifactError("The model produced invalid prediction values.")
            scores = np.sqrt(
                np.mean(np.square(batch_actual - predictions), axis=1)
            )
            if not np.isfinite(scores).all():
                raise ModelArtifactError("The model produced invalid residual scores.")
            actual_feature_values.extend(batch_actual[:, 0].tolist())
            predicted_feature_values.extend(predictions[:, 0].tolist())
            residual_score_values.extend(scores.tolist())
            anomaly_flags.extend((scores > channel["residual_threshold"]).tolist())
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ModelArtifactError(
            f"Saved preprocessing or model is incompatible for {channel_id}."
        ) from exc

    return {
        "channel_id": channel_id,
        "feature_count": feature_count,
        "model_name": bundle.get(
            "model_name", "Per-channel one-step Linear Regression forecaster"
        ),
        "prediction_method": bundle.get("prediction_method", ""),
        "record_indices": (
            record_indices[1:]
            if record_indices is not None
            else list(range(1, record_count))
        ),
        "actual_values": actual_feature_values,
        "predicted_values": predicted_feature_values,
        "residual_scores": residual_score_values,
        "anomaly_flags": anomaly_flags,
        "threshold": float(threshold),
        "model_version": bundle["model_version"],
        "evaluation": report,
    }


def summarize_prediction(result: dict[str, Any]) -> dict[str, Any]:
    flags = result["anomaly_flags"]
    indices = result["record_indices"]
    flagged_index_sample = [
        index for index, flagged in zip(indices, flags) if bool(flagged)
    ][:10]
    return {
        "channel_id": result["channel_id"],
        "feature_count": result["feature_count"],
        "record_count": len(indices),
        "model_name": result.get(
            "model_name", "Per-channel one-step Linear Regression forecaster"
        ),
        "prediction_method": result.get(
            "prediction_method",
            "Predict the next feature vector from the previous feature vector; "
            "flag residual RMSE above the validation-calibrated threshold.",
        ),
        "threshold": result["threshold"],
        "anomaly_count": sum(bool(flagged) for flagged in flags),
        "flagged_record_indices": flagged_index_sample,
        "evaluation": result["evaluation"],
    }
