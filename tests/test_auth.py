import io
from pathlib import Path

from werkzeug.security import generate_password_hash

from app import create_app


def make_client():
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-only-secret-for-unit-tests-32",
            "AERIS_USERS": {"operator": generate_password_hash("correct-password")},
            "NEON_DATABASE_URL": "",
            "MODEL_ARTIFACT_DIRECTORY": str(
                Path(__file__).parent / "missing-model-artifacts"
            ),
        }
    )
    return app.test_client()


def test_dashboard_requires_login():
    response = make_client().get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")


def test_login_and_dashboard():
    client = make_client()
    login_page = client.get("/login")
    with client.session_transaction() as session:
        csrf_token = session["csrf_token"]

    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token,
            "username": "operator",
            "password": "correct-password",
        },
    )

    assert response.status_code == 302
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert b"Model analysis is not enabled yet" in dashboard.data
    assert b"Welcome, operator" in dashboard.data
    assert login_page.status_code == 200


def test_login_rejects_invalid_password():
    client = make_client()
    client.get("/login")
    with client.session_transaction() as session:
        csrf_token = session["csrf_token"]

    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token,
            "username": "operator",
            "password": "incorrect",
        },
    )

    assert response.status_code == 200
    assert b"username or password is incorrect" in response.data
    assert client.get("/").status_code == 302


def test_login_rejects_missing_csrf_token():
    response = make_client().post(
        "/login", data={"username": "operator", "password": "correct-password"}
    )

    assert response.status_code == 400


def test_logout_clears_session():
    client = make_client()
    csrf_token = authenticate(client)

    response = client.post("/logout", data={"csrf_token": csrf_token})

    assert response.status_code == 302
    assert client.get("/").status_code == 302


def authenticate(client):
    client.get("/login")
    with client.session_transaction() as session:
        csrf_token = session["csrf_token"]
    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token,
            "username": "operator",
            "password": "correct-password",
        },
    )
    assert response.status_code == 302
    client.get("/")
    with client.session_transaction() as session:
        return session["csrf_token"]


def test_upload_runs_local_analysis_without_model(monkeypatch):
    from app import routes

    def no_gemini(*args, **kwargs):
        raise routes.GeminiUnavailableError("Gemini is not configured.")

    monkeypatch.setattr(routes, "generate_insights", no_gemini)
    client = make_client()
    csrf_token = authenticate(client)

    response = client.post(
        "/analyze",
        data={
            "csrf_token": csrf_token,
            "telemetry_file": (
                io.BytesIO(
                    b"record_index,sensor_1\n0,1\n1,3\n2,5\n"
                ),
                "telemetry.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Descriptive results only" in response.data
    assert b"3 records" in response.data
    assert b"No trained model artifacts are available" in response.data
    assert b"Gemini is not configured" in response.data


def test_upload_runs_channel_model_only_after_explicit_mapping_confirmation(
    monkeypatch,
):
    from app import routes

    report = {
        "model_name": "Synthetic test forecaster",
        "regression_metrics": {
            "mse": 1.0,
            "rmse": 1.0,
            "mape_percent": 10.0,
            "sample_count": 10,
            "mape_excluded_zero_count": 0,
        },
        "regression_channels": ["A-1"],
        "classification_metrics": None,
        "classification_metrics_scope": "No valid labels.",
        "classification_channels": [],
        "excluded_from_classification": {
            "duplicate_labels": [],
            "missing_unique_labels": [],
            "length_mismatches": [],
        },
        "split_method": "Chronological held-out evaluation.",
        "evaluation_dataset": "Synthetic unit-test sequences.",
    }
    monkeypatch.setattr(
        routes,
        "_model_choices",
        lambda: ([{"channel_id": "A-1", "feature_count": 1}], report, None),
    )

    def should_not_predict(*args, **kwargs):
        raise AssertionError("inference must require mapping confirmation")

    def no_gemini(*args, **kwargs):
        raise routes.GeminiUnavailableError("Gemini is not configured.")

    monkeypatch.setattr(routes, "predict_channel", should_not_predict)
    monkeypatch.setattr(routes, "generate_insights", no_gemini)
    client = make_client()
    csrf_token = authenticate(client)

    response = client.post(
        "/analyze",
        data={
            "csrf_token": csrf_token,
            "channel_id": "A-1",
            "telemetry_file": (
                io.BytesIO(b"record_index,sensor_1\n0,1\n1,2\n"),
                "telemetry.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Confirm that the ordered CSV sensor columns match" in response.data
    assert b"Descriptive results only" in response.data


def test_upload_displays_verified_model_output_and_held_out_metrics(monkeypatch):
    from app import routes

    report = {
        "model_version": "aeris-linear-forecaster-v1",
        "model_name": "A-1 Linear Regression forecaster",
        "prediction_method": "One-step Linear Regression.",
        "split_method": "Chronological train and validation, separate held-out test.",
        "evaluation_dataset": "NASA test array A-1.",
        "regression_metrics": {
            "mse": 1.0,
            "rmse": 1.0,
            "mape_percent": 10.0,
            "sample_count": 10,
            "mape_excluded_zero_count": 0,
        },
        "regression_channels": ["A-1"],
        "classification_metrics": None,
        "classification_metrics_scope": "Unavailable without valid labels.",
        "classification_channels": [],
        "excluded_from_classification": {
            "duplicate_labels": [],
            "missing_unique_labels": [],
            "length_mismatches": [],
        },
    }
    monkeypatch.setattr(
        routes,
        "_model_choices",
        lambda: ([{"channel_id": "A-1", "feature_count": 1}], report, None),
    )

    def predict(values, channel_id, **kwargs):
        assert values == {"sensor_1": [1.0, 2.0, 3.0]}
        assert channel_id == "A-1"
        return {
            "channel_id": "A-1",
            "feature_count": 1,
            "model_name": report["model_name"],
            "prediction_method": report["prediction_method"],
            "threshold": 0.5,
            "record_indices": [1, 2],
            "actual_values": [2.0, 3.0],
            "predicted_values": [1.8, 2.5],
            "residual_scores": [0.2, 0.6],
            "anomaly_flags": [False, True],
            "evaluation": report,
        }

    def no_gemini(*args, **kwargs):
        raise routes.GeminiUnavailableError("Gemini is not configured.")

    monkeypatch.setattr(routes, "predict_channel", predict)
    monkeypatch.setattr(routes, "generate_insights", no_gemini)
    client = make_client()
    csrf_token = authenticate(client)

    response = client.post(
        "/analyze",
        data={
            "csrf_token": csrf_token,
            "channel_id": "A-1",
            "confirm_channel_mapping": "yes",
            "telemetry_file": (
                io.BytesIO(
                    b"record_index,sensor_1\n0,1\n1,2\n2,3\n"
                ),
                "telemetry.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Model output for A-1" in response.data
    assert b"Held-out RMSE" in response.data
    assert b"2.5" in response.data
    assert b"Residual threshold exceeded" in response.data
    assert b"Unavailable without valid labels" in response.data


def test_upload_rejects_arbitrary_csv_columns():
    client = make_client()
    csrf_token = authenticate(client)
    response = client.post(
        "/analyze",
        data={
            "csrf_token": csrf_token,
            "telemetry_file": (
                io.BytesIO(b"time,temperature\n0,12\n"),
                "telemetry.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"record_index" in response.data
