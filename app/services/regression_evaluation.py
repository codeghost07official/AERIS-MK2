import math
from typing import Iterable


class EvaluationError(ValueError):
    """Raised when regression metrics cannot be computed reliably."""


def calculate_regression_metrics(
    actual_values: Iterable[float], predicted_values: Iterable[float]
) -> dict[str, float | int | None]:
    actual = list(actual_values)
    predicted = list(predicted_values)
    if not actual or len(actual) != len(predicted):
        raise EvaluationError(
            "Actual and predicted values must be non-empty and have equal lengths."
        )
    if any(
        not math.isfinite(value)
        for value in (*actual, *predicted)
    ):
        raise EvaluationError("Regression metrics require finite values.")

    try:
        squared_errors = [
            (observed - predicted_value) ** 2
            for observed, predicted_value in zip(actual, predicted)
        ]
        mse = math.fsum(squared_errors) / len(actual)
        rmse = math.sqrt(mse)
    except (OverflowError, ValueError) as exc:
        raise EvaluationError("Regression errors exceed the supported numeric range.") from exc
    if not math.isfinite(mse) or not math.isfinite(rmse):
        raise EvaluationError("Regression errors exceed the supported numeric range.")

    nonzero_pairs = [
        (observed, predicted_value)
        for observed, predicted_value in zip(actual, predicted)
        if observed != 0
    ]
    if nonzero_pairs:
        try:
            percentage_errors = [
                abs(observed - predicted_value) / abs(observed) * 100
                for observed, predicted_value in nonzero_pairs
            ]
            mape: float | None = math.fsum(percentage_errors) / len(percentage_errors)
        except (OverflowError, ValueError) as exc:
            raise EvaluationError("MAPE exceeds the supported numeric range.") from exc
        if not math.isfinite(mape):
            raise EvaluationError("MAPE exceeds the supported numeric range.")
    else:
        mape = None

    return {
        "mse": mse,
        "rmse": rmse,
        "mape_percent": mape,
        "sample_count": len(actual),
        "zero_target_count": len(actual) - len(nonzero_pairs),
        "mape_excluded_zero_count": len(actual) - len(nonzero_pairs),
    }


def calculate_mape(
    actual_values: Iterable[float], predicted_values: Iterable[float]
) -> tuple[float | None, int]:
    metrics = calculate_regression_metrics(actual_values, predicted_values)
    return metrics["mape_percent"], int(metrics["mape_excluded_zero_count"])
