from collections.abc import Iterable
from typing import Any


class ClassificationEvaluationError(ValueError):
    """Raised when binary predictions and labels cannot be compared."""


def calculate_binary_metrics(
    actual_labels: Iterable[bool], predicted_labels: Iterable[bool]
) -> dict[str, int | float | None]:
    actual = list(actual_labels)
    predicted = list(predicted_labels)
    if not actual or len(actual) != len(predicted):
        raise ClassificationEvaluationError(
            "Binary labels and predictions must be non-empty and equal in length."
        )
    if any(type(value) is not bool for value in (*actual, *predicted)):
        raise ClassificationEvaluationError("Binary labels must contain booleans.")

    true_positive = sum(expected and predicted_value for expected, predicted_value in zip(actual, predicted))
    true_negative = sum(not expected and not predicted_value for expected, predicted_value in zip(actual, predicted))
    false_positive = sum(not expected and predicted_value for expected, predicted_value in zip(actual, predicted))
    false_negative = sum(expected and not predicted_value for expected, predicted_value in zip(actual, predicted))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else None
    )
    recall = true_positive / recall_denominator if recall_denominator else None
    if precision is None or recall is None or precision + recall == 0:
        f1_score = None
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (true_positive + true_negative) / len(actual)

    return {
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "sample_count": len(actual),
    }
