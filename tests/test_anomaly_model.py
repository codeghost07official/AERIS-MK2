import json

import pytest

from app.services.anomaly_model import (
    ModelArtifactError,
    artifact_paths,
    load_evaluation_report,
    load_model_bundle,
)


def test_missing_model_artifacts_explain_how_to_train(tmp_path):
    with pytest.raises(ModelArtifactError, match=r"scripts\\train_model\.py"):
        load_evaluation_report(tmp_path)

    with pytest.raises(ModelArtifactError, match="No trained model"):
        load_model_bundle(tmp_path)


def test_corrupt_evaluation_report_is_rejected(tmp_path):
    model_path, report_path = artifact_paths(tmp_path)
    model_path.write_bytes(b"not a joblib artifact")
    report_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="unreadable"):
        load_evaluation_report(tmp_path)

    report_path.write_text(
        json.dumps({"model_version": "aeris-linear-forecaster-v1", "channels": []}),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="unreadable or incompatible"):
        load_model_bundle(tmp_path)


def test_incompatible_evaluation_report_is_rejected(tmp_path):
    model_path = tmp_path / "aeris_linear_forecaster.joblib"
    model_path.write_bytes(b"placeholder")
    report_path = tmp_path / "evaluation_report.json"
    report_path.write_text(
        json.dumps({"model_version": "unknown", "channels": []}),
        encoding="utf-8",
    )

    with pytest.raises(ModelArtifactError, match="incompatible"):
        load_evaluation_report(tmp_path)
