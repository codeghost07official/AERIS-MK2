import pytest

from app.services.regression_evaluation import (
    EvaluationError,
    calculate_regression_metrics,
)


def test_calculates_regression_metrics_and_excludes_zero_targets_from_mape():
    metrics = calculate_regression_metrics([0, 2, 4], [1, 3, 2])

    assert metrics["mse"] == pytest.approx(2)
    assert metrics["rmse"] == pytest.approx(2**0.5)
    assert metrics["mape_percent"] == pytest.approx(50)
    assert metrics["sample_count"] == 3
    assert metrics["zero_target_count"] == 1
    assert metrics["mape_excluded_zero_count"] == 1


def test_mape_is_unavailable_when_all_targets_are_zero():
    metrics = calculate_regression_metrics([0, 0], [1, -1])

    assert metrics["mape_percent"] is None
    assert metrics["mape_excluded_zero_count"] == 2
    assert metrics["mse"] == 1


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([], []),
        ([1], []),
        ([float("nan")], [1]),
        ([1], [float("inf")]),
    ],
)
def test_rejects_invalid_metric_inputs(actual, predicted):
    with pytest.raises(EvaluationError):
        calculate_regression_metrics(actual, predicted)
