import csv
import io
import math
import re
import statistics
from typing import Any


SENSOR_NAME = re.compile(r"sensor_[1-9][0-9]*\Z")


class TelemetryValidationError(ValueError):
    """Raised when an uploaded CSV does not match the documented telemetry format."""


def _downsample(
    values: list[float],
    record_indices: list[int],
    max_points: int,
) -> list[dict[str, float | int]]:
    step = max(1, math.ceil(len(values) / max_points))
    indexes = list(range(0, len(values), step))
    if indexes[-1] != len(values) - 1:
        indexes.append(len(values) - 1)
    return [
        {"x": record_indices[index], "y": values[index]}
        for index in indexes
    ]


def _histogram(values: list[float], bins: int = 10) -> dict[str, list[Any]]:
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return {"labels": [f"{minimum:.6g}"], "counts": [len(values)]}

    width = (maximum - minimum) / bins
    if not math.isfinite(width) or width <= 0:
        raise TelemetryValidationError(
            "Sensor values span a range too large for a reliable histogram."
        )
    counts = [0] * bins
    for value in values:
        index = min(int((value - minimum) / width), bins - 1)
        counts[index] += 1
    labels = [
        f"{minimum + index * width:.4g}–{minimum + (index + 1) * width:.4g}"
        for index in range(bins)
    ]
    return {"labels": labels, "counts": counts}


def analyze_csv(
    file_bytes: bytes,
    *,
    max_records: int = 100_000,
    max_sensors: int = 55,
    max_chart_points: int = 400,
) -> dict[str, Any]:
    if not file_bytes:
        raise TelemetryValidationError("The uploaded CSV is empty.")
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TelemetryValidationError(
            "The CSV must be encoded as UTF-8."
        ) from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None)
    except csv.Error as exc:
        raise TelemetryValidationError("The CSV header is malformed.") from exc

    if not header or any(not column.strip() for column in header):
        raise TelemetryValidationError("The CSV must have a non-empty header row.")
    header = [column.strip() for column in header]
    if len(set(header)) != len(header):
        raise TelemetryValidationError("The CSV contains duplicate column names.")
    if header[0] != "record_index":
        raise TelemetryValidationError(
            "The first column must be named record_index."
        )

    sensor_names = header[1:]
    if not sensor_names or any(not SENSOR_NAME.fullmatch(name) for name in sensor_names):
        raise TelemetryValidationError(
            "Sensor columns must be named sensor_1, sensor_2, and so on."
        )
    expected_names = [f"sensor_{number}" for number in range(1, len(sensor_names) + 1)]
    if sensor_names != expected_names:
        raise TelemetryValidationError(
            "Sensor columns must be consecutive and ordered from sensor_1."
        )
    if len(sensor_names) > max_sensors:
        raise TelemetryValidationError(
            f"At most {max_sensors} sensor columns are supported."
        )

    values_by_sensor: dict[str, list[float]] = {
        name: [] for name in sensor_names
    }
    record_indices: list[int] = []
    previous_record_index = -1
    try:
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise TelemetryValidationError(
                    f"Row {row_number} has {len(row)} columns; expected {len(header)}."
                )
            if len(values_by_sensor[sensor_names[0]]) >= max_records:
                raise TelemetryValidationError(
                    f"The CSV exceeds the {max_records:,}-record limit."
                )
            try:
                record_index_value = int(row[0].strip())
            except ValueError as exc:
                raise TelemetryValidationError(
                    f"record_index on row {row_number} must be an integer."
                ) from exc
            if record_index_value < 0 or record_index_value <= previous_record_index:
                raise TelemetryValidationError(
                    f"record_index on row {row_number} must be a strictly increasing "
                    "non-negative integer."
                )
            previous_record_index = record_index_value
            record_indices.append(record_index_value)

            parsed_values: list[float] = []
            for name, raw_value in zip(sensor_names, row[1:]):
                try:
                    value = float(raw_value.strip())
                except ValueError as exc:
                    raise TelemetryValidationError(
                        f"{name} on row {row_number} must be numeric; missing values "
                        "are not accepted."
                    ) from exc
                if not math.isfinite(value):
                    raise TelemetryValidationError(
                        f"{name} on row {row_number} must be finite; NaN and infinity "
                        "are not accepted."
                    )
                parsed_values.append(value)
            for name, value in zip(sensor_names, parsed_values):
                values_by_sensor[name].append(value)
    except csv.Error as exc:
        raise TelemetryValidationError("The CSV contains malformed rows.") from exc

    record_count = len(values_by_sensor[sensor_names[0]])
    if not record_count:
        raise TelemetryValidationError("The CSV must contain at least one data row.")

    series: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, values in values_by_sensor.items():
        try:
            mean_value = statistics.fmean(values)
            median_value = statistics.median(values)
            standard_deviation = statistics.pstdev(values)
        except (OverflowError, ValueError) as exc:
            raise TelemetryValidationError(
                f"{name} values exceed the range supported by descriptive statistics."
            ) from exc
        if not all(
            math.isfinite(number)
            for number in (mean_value, median_value, standard_deviation)
        ):
            raise TelemetryValidationError(
                f"{name} values exceed the range supported by descriptive statistics."
            )
        histogram = _histogram(values)
        summaries.append(
            {
                "name": name,
                "minimum": min(values),
                "maximum": max(values),
                "mean": mean_value,
                "median": median_value,
                "standard_deviation": standard_deviation,
                "first_value": values[0],
                "last_value": values[-1],
                "net_change": values[-1] - values[0],
                "histogram": histogram,
            }
        )
        if len(series) < 8:
            series.append(
                {
                    "name": name,
                    "points": _downsample(
                        values, record_indices, max_chart_points
                    ),
                }
            )

    return {
        "record_count": record_count,
        "sensor_count": len(sensor_names),
        "record_indices": record_indices,
        "sensors": summaries,
        "sensor_values": values_by_sensor,
        "chart_series": series,
        "chart_note": (
            f"Showing {len(series)} of {len(sensor_names)} sensors; plotted points "
            f"are sampled from the {record_count:,} uploaded records."
        ),
        "model_status": "No model predictions or evaluation metrics are available.",
    }


def summarize_for_gemini(
    analysis: dict[str, Any],
    model_result: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_count": analysis["record_count"],
        "sensor_count": analysis["sensor_count"],
        "summarized_sensor_count": min(8, analysis["sensor_count"]),
        "sensors": [
            {
                key: sensor[key]
                for key in (
                    "name",
                    "minimum",
                    "maximum",
                    "mean",
                    "median",
                    "standard_deviation",
                    "first_value",
                    "last_value",
                    "net_change",
                )
            }
            for sensor in analysis["sensors"][:8]
        ],
        "model_status": analysis["model_status"],
    }
    if model_result:
        summary["model_output"] = {
            "model_name": model_result["model_name"],
            "channel_id": model_result["channel_id"],
            "prediction_method": model_result["prediction_method"],
            "residual_threshold": model_result["threshold"],
            "flagged_record_count": model_result["anomaly_count"],
            "flagged_record_indices_sample": model_result[
                "flagged_record_indices"
            ][:10],
            "prediction_sample": model_result.get("prediction_preview", [])[:5],
        }
    if evaluation:
        summary["held_out_evaluation"] = {
            "regression_metrics": evaluation.get("regression_metrics"),
            "classification_metrics": evaluation.get("classification_metrics"),
            "classification_metrics_scope": evaluation.get(
                "classification_metrics_scope"
            ),
        }
    return summary
