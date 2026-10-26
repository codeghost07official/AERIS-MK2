import json

import pytest

from app.services.telemetry import (
    TelemetryValidationError,
    analyze_csv,
    summarize_for_gemini,
)


def test_analyzes_numeric_csv_and_bounds_chart_points():
    result = analyze_csv(
        b"record_index,sensor_1,sensor_2\n0,1,4\n1,3,4\n2,5,4\n",
        max_chart_points=2,
    )

    assert result["record_count"] == 3
    assert result["sensor_count"] == 2
    assert result["sensors"][0]["mean"] == 3
    assert result["sensors"][0]["median"] == 3
    assert result["sensors"][0]["first_value"] == 1
    assert result["sensors"][0]["last_value"] == 5
    assert result["sensors"][0]["net_change"] == 4
    assert result["sensors"][0]["standard_deviation"] == pytest.approx(1.633, abs=1e-5)
    assert len(result["chart_series"][0]["points"]) == 2
    assert result["chart_series"][0]["points"][-1] == {"x": 2, "y": 5}
    assert result["model_status"].startswith("No model predictions")
    json.dumps(result, allow_nan=False)

    summary = summarize_for_gemini(result)
    assert summary["summarized_sensor_count"] == 2
    assert "chart_series" not in summary
    assert "filename" not in summary


def test_gemini_summary_receives_bounded_computed_model_predictions():
    analysis = analyze_csv(b"record_index,sensor_1\n0,1\n1,2\n")
    summary = summarize_for_gemini(
        analysis,
        model_result={
            "model_name": "Linear Regression",
            "channel_id": "A-1",
            "prediction_method": "one step",
            "threshold": 0.5,
            "anomaly_count": 1,
            "flagged_record_indices": [1],
            "prediction_preview": [
                {
                    "record_index": index,
                    "actual": float(index),
                    "predicted": float(index) - 0.1,
                    "residual_score": 0.1,
                    "flagged": False,
                }
                for index in range(8)
            ],
        },
    )

    output = summary["model_output"]
    assert output["prediction_sample"][0]["predicted"] == -0.1
    assert len(output["prediction_sample"]) == 5
    assert "sensor_values" not in summary


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "empty"),
        (b"time,sensor_1\n0,1\n", "record_index"),
        (b"record_index,value\n0,1\n", "Sensor columns"),
        (b"record_index,sensor_2\n0,1\n", "consecutive"),
        (b"record_index,sensor_1\n0,\n", "missing values"),
        (b"record_index,sensor_1\n0,NaN\n", "finite"),
        (b"record_index,sensor_1\n1,1\n1,2\n", "strictly increasing"),
        (b"record_index,sensor_1,sensor_1\n0,1,2\n", "duplicate"),
    ],
)
def test_rejects_invalid_csv(payload, message):
    with pytest.raises(TelemetryValidationError, match=message):
        analyze_csv(payload)


def test_rejects_record_and_sensor_limits():
    with pytest.raises(TelemetryValidationError, match="record limit"):
        analyze_csv(b"record_index,sensor_1\n0,1\n1,2\n", max_records=1)

    with pytest.raises(TelemetryValidationError, match="At most 1"):
        analyze_csv(
            b"record_index,sensor_1,sensor_2\n0,1,2\n",
            max_sensors=1,
        )


def test_accepts_utf8_bom_and_blank_lines():
    result = analyze_csv(
        b"\xef\xbb\xbfrecord_index,sensor_1\r\n0,2\r\n\r\n1,4\r\n"
    )

    assert result["record_count"] == 2
    assert result["sensors"][0]["mean"] == 3


def test_chart_points_use_record_indices_for_prediction_alignment():
    result = analyze_csv(
        b"record_index,sensor_1\n10,1\n11,2\n12,3\n",
        max_chart_points=3,
    )

    assert result["chart_series"][0]["points"] == [
        {"x": 10, "y": 1.0},
        {"x": 11, "y": 2.0},
        {"x": 12, "y": 3.0},
    ]
