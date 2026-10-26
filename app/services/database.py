import json
from typing import Any

import psycopg
from psycopg.rows import dict_row


def get_recent_analyses(database_url: str, limit: int = 50) -> list[dict[str, Any]]:
    with psycopg.connect(
        database_url, row_factory=dict_row, connect_timeout=10
    ) as connection:
        return connection.execute(
            """
            SELECT id, uploaded_filename, channel_id, record_count,
                   anomaly_count, prediction_summary, evaluation_metrics,
                   descriptive_statistics, gemini_insights, created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()


def save_analysis(
    database_url: str,
    *,
    filename: str,
    record_count: int,
    sensor_count: int,
    sensor_summaries: list[dict[str, Any]],
    gemini_insights: dict[str, str] | None,
    model_result: dict[str, Any] | None = None,
) -> None:
    summary = f"Descriptive analysis of {sensor_count} sensor channel(s)."
    channel_id = None
    anomaly_count = None
    metrics: dict[str, Any] = {}
    if model_result:
        channel_id = model_result["channel_id"]
        anomaly_count = model_result["anomaly_count"]
        report = model_result["evaluation_report"]
        metrics = {
            "model_name": report["model_name"],
            "regression_metrics": report["regression_metrics"],
            "classification_metrics": report["classification_metrics"],
        }
        summary += (
            f" {anomaly_count} record(s) flagged by the residual threshold "
            f"for channel {channel_id}; flags are not confirmed anomalies."
        )
    else:
        summary += " Model predictions and evaluation were unavailable."
    with psycopg.connect(database_url, connect_timeout=10) as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                uploaded_filename, channel_id, record_count, anomaly_count,
                prediction_summary, evaluation_metrics, descriptive_statistics,
                gemini_insights
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
            """,
            (
                filename,
                channel_id,
                record_count,
                anomaly_count,
                summary,
                json.dumps(metrics, allow_nan=False),
                json.dumps(
                    [
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
                        for sensor in sensor_summaries
                    ]
                ),
                json.dumps(gemini_insights) if gemini_insights else None,
            ),
        )
