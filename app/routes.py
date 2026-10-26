from pathlib import Path

import psycopg
from flask import Blueprint, abort, current_app, render_template, request, session
from werkzeug.utils import secure_filename

from app.auth import login_required, valid_csrf_token
from app.services.database import get_recent_analyses, save_analysis
from app.services.gemini import GeminiUnavailableError, generate_insights
from app.services.anomaly_model import (
    ModelArtifactError,
    load_evaluation_report,
    predict_channel,
    summarize_prediction,
)
from app.services.telemetry import (
    TelemetryValidationError,
    analyze_csv,
    summarize_for_gemini,
)


main_blueprint = Blueprint("main", __name__)


def _model_directory() -> Path:
    return Path(current_app.config["MODEL_ARTIFACT_DIRECTORY"])


def _model_choices() -> tuple[list[dict], dict | None, str | None]:
    try:
        report = load_evaluation_report(_model_directory())
        channels = [
            {
                "channel_id": channel["channel_id"],
                "feature_count": channel["feature_count"],
            }
            for channel in report["channels"]
            if isinstance(channel, dict)
            and isinstance(channel.get("channel_id"), str)
            and isinstance(channel.get("feature_count"), int)
        ]
        if not channels:
            return [], report, "The evaluation report contains no usable model channels."
        return channels, report, None
    except ModelArtifactError as exc:
        return [], None, str(exc)


@main_blueprint.get("/")
@login_required
def dashboard() -> str:
    channels, evaluation_report, model_message = _model_choices()
    return render_template(
        "dashboard.html",
        username=session["username"],
        model_channels=channels,
        model_message=model_message,
        evaluation_report=evaluation_report,
    )


@main_blueprint.post("/analyze")
@login_required
def analyze() -> str | tuple[str, int]:
    if not valid_csrf_token():
        abort(400, description="The form expired. Reload the page and try again.")

    channels, evaluation_report, artifact_message = _model_choices()
    dashboard_context = {
        "username": session["username"],
        "model_channels": channels,
        "model_message": artifact_message,
        "evaluation_report": evaluation_report,
    }
    upload = request.files.get("telemetry_file")
    if upload is None or not upload.filename:
        return render_template(
            "dashboard.html", **dashboard_context,
            upload_error="Choose a CSV file to analyze.",
        ), 400

    filename = secure_filename(upload.filename)
    if not filename or not filename.lower().endswith(".csv"):
        return render_template(
            "dashboard.html", **dashboard_context,
            upload_error="Upload a file with a .csv extension.",
        ), 400

    file_bytes = upload.read(current_app.config["MAX_UPLOAD_BYTES"] + 1)
    if len(file_bytes) > current_app.config["MAX_UPLOAD_BYTES"]:
        return render_template(
            "dashboard.html", **dashboard_context,
            upload_error="The CSV exceeds the 5 MB upload limit.",
        ), 413

    try:
        analysis = analyze_csv(
            file_bytes,
            max_records=current_app.config["MAX_TELEMETRY_RECORDS"],
            max_sensors=current_app.config["MAX_TELEMETRY_SENSORS"],
            max_chart_points=current_app.config["MAX_CHART_POINTS"],
        )
    except TelemetryValidationError as exc:
        return render_template(
            "dashboard.html", **dashboard_context,
            upload_error=str(exc),
        ), 400

    model_result = None
    model_warning = None
    selected_channel = request.form.get("channel_id", "").strip()
    if selected_channel:
        if request.form.get("confirm_channel_mapping") != "yes":
            model_warning = (
                "Confirm that the ordered CSV sensor columns match the selected "
                "NASA channel. Descriptive analysis is still available."
            )
        else:
            try:
                values = analysis["sensor_values"]
                prediction = predict_channel(
                    values,
                    selected_channel,
                    directory=_model_directory(),
                    record_indices=analysis["record_indices"],
                )
                model_result = summarize_prediction(prediction)
                model_result["evaluation_report"] = prediction["evaluation"]
                model_result["prediction_preview"] = [
                    {
                        "record_index": record_index,
                        "actual": prediction["actual_values"][position],
                        "predicted": prediction["predicted_values"][position],
                        "residual_score": prediction["residual_scores"][position],
                        "flagged": bool(prediction["anomaly_flags"][position]),
                    }
                    for position, record_index in enumerate(
                        prediction["record_indices"][-10:],
                        start=max(0, len(prediction["record_indices"]) - 10),
                    )
                ]
                model_result["prediction_chart"] = _prediction_chart_data(
                    prediction,
                    current_app.config["MAX_CHART_POINTS"],
                )
                analysis["model_status"] = (
                    "Predictions and residual-threshold flags are shown for the "
                    f"selected model channel {selected_channel}. Flags are not "
                    "confirmed anomalies."
                )
            except (ValueError, ModelArtifactError) as exc:
                model_warning = str(exc)
    else:
        model_warning = (
            artifact_message
            if not channels
            else "Select a trained channel to run model inference."
        )

    if model_result is None and evaluation_report:
        analysis["model_status"] = (
            "No prediction was run for this upload. A saved held-out model "
            "evaluation is available, but it does not describe this file."
        )
    summary = summarize_for_gemini(
        analysis,
        model_result=model_result,
        evaluation=(
            model_result["evaluation_report"]
            if model_result
            else evaluation_report
        ),
    )
    insights = None
    try:
        insights = generate_insights(
            summary,
            api_key=current_app.config["GEMINI_API_KEY"],
            model=current_app.config["GEMINI_MODEL"],
        )
        gemini_message = None
    except GeminiUnavailableError as exc:
        gemini_message = str(exc)
        current_app.logger.info("Gemini insights unavailable: %s", exc)

    database_message = None
    database_url = current_app.config["NEON_DATABASE_URL"]
    if database_url:
        try:
            save_analysis(
                database_url,
                filename=filename,
                record_count=analysis["record_count"],
                sensor_count=analysis["sensor_count"],
                sensor_summaries=analysis["sensors"],
                gemini_insights=insights,
                model_result=model_result,
            )
        except psycopg.Error as exc:
            current_app.logger.error(
                "Unable to save analysis summary (%s).", type(exc).__name__
            )
            database_message = (
                "The analysis succeeded, but its summary could not be saved "
                "to history."
            )
    else:
        database_message = (
            "History is not saved because NEON_DATABASE_URL is not configured."
        )

    return render_template(
        "results.html",
        filename=filename,
        analysis=analysis,
        model_result=model_result,
        model_warning=model_warning,
        model_channels=channels,
        evaluation_report=evaluation_report,
        insights=insights,
        gemini_message=gemini_message,
        database_message=database_message,
    )


def _prediction_chart_data(
    prediction: dict,
    max_points: int,
) -> dict:
    indices = prediction["record_indices"]
    actual = prediction["actual_values"]
    predicted = prediction["predicted_values"]
    flags = prediction["anomaly_flags"]
    step = max(1, (len(indices) + max_points - 1) // max_points)
    sample_indices = list(range(0, len(indices), step))
    if sample_indices[-1] != len(indices) - 1:
        sample_indices.append(len(indices) - 1)
    actual_points = [
        {"x": indices[position], "y": actual[position]}
        for position in sample_indices
    ]
    predicted_points = [
        {"x": indices[position], "y": predicted[position]}
        for position in sample_indices
    ]
    anomaly_points = []
    anomaly_step = max(1, (sum(flags) + 499) // 500)
    anomalies_seen = 0
    for position, (record_index, flagged) in enumerate(zip(indices, flags)):
        if flagged:
            if anomalies_seen % anomaly_step == 0:
                anomaly_points.append(
                    {"x": record_index, "y": actual[position]}
                )
            anomalies_seen += 1
    return {
        "feature_name": "sensor_1",
        "actual": actual_points,
        "predicted": predicted_points,
        "anomalies": anomaly_points,
        "anomaly_point_limit": 500,
    }


@main_blueprint.get("/history")
@login_required
def history() -> str:
    database_url = current_app.config["NEON_DATABASE_URL"]
    if not database_url:
        return render_template(
            "history.html",
            analyses=[],
            history_message="History is unavailable until NEON_DATABASE_URL is configured.",
        )

    try:
        analyses = get_recent_analyses(database_url)
    except psycopg.Error as exc:
        current_app.logger.error(
            "Unable to retrieve analysis history (%s).", type(exc).__name__
        )
        return render_template(
            "history.html",
            analyses=[],
            history_message=(
                "The history database could not be reached. Check the server logs "
                "or try again later."
            ),
        )

    return render_template(
        "history.html",
        analyses=analyses,
        history_message=(
            "No saved analyses yet. Results will appear here after analysis is "
            "completed."
            if not analyses
            else None
        ),
    )
