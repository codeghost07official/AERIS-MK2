import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from app.services.anomaly_model import predict_channel
from scripts.train_model import train_dataset


def test_synthetic_training_creates_loadable_artifacts_and_inference(tmp_path):
    dataset = tmp_path / "Dataset"
    train_directory = dataset / "data" / "train"
    test_directory = dataset / "data" / "test"
    train_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)

    train_values = np.column_stack(
        (np.linspace(1, 20, 30), np.linspace(2, 12, 30))
    ).astype(np.float64)
    test_values = np.column_stack(
        (np.linspace(21, 26, 10), np.linspace(13, 18, 10))
    ).astype(np.float64)
    np.save(train_directory / "A-1.npy", train_values)
    np.save(test_directory / "A-1.npy", test_values)
    (dataset / "labeled_anomalies.csv").write_text(
        "chan_id,spacecraft,anomaly_sequences,num_values\n"
        'A-1,SMAP,[],10\n',
        encoding="utf-8",
    )

    output = tmp_path / "models"
    report = train_dataset(dataset, output)

    assert (output / "aeris_linear_forecaster.joblib").is_file()
    saved_report = json.loads(
        (output / "evaluation_report.json").read_text(encoding="utf-8")
    )
    assert saved_report["artifact_id"] == report["artifact_id"]
    assert report["evaluation_dataset"].startswith("NASA SMAP/MSL")
    assert report["regression_metrics"]["sample_count"] == 18
    assert report["preprocessing"]["input_scaler"].startswith("StandardScaler")
    assert report["software_versions"]["scikit_learn"]

    prediction = predict_channel(
        test_values.tolist(),
        "A-1",
        directory=output,
        record_indices=list(range(10, 20)),
    )
    assert len(prediction["predicted_values"]) == 9
    assert len(prediction["anomaly_flags"]) == 9
    columnar_prediction = predict_channel(
        {
            "sensor_1": test_values[:, 0].tolist(),
            "sensor_2": test_values[:, 1].tolist(),
        },
        "A-1",
        directory=output,
    )
    assert columnar_prediction["predicted_values"] == prediction["predicted_values"]


def test_inference_rejects_wrong_feature_count_after_training(tmp_path):
    dataset = tmp_path / "Dataset"
    train_directory = dataset / "data" / "train"
    test_directory = dataset / "data" / "test"
    train_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)
    np.save(train_directory / "A-1.npy", np.arange(60).reshape(30, 2).astype(float))
    np.save(test_directory / "A-1.npy", np.arange(20).reshape(10, 2).astype(float))
    (dataset / "labeled_anomalies.csv").write_text(
        "chan_id,spacecraft,anomaly_sequences,num_values\n",
        encoding="utf-8",
    )
    output = tmp_path / "models"
    train_dataset(dataset, output)

    with pytest.raises(ValueError, match="exactly 2 sensor columns"):
        predict_channel([[1.0], [2.0]], "A-1", directory=output)
