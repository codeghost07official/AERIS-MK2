import math
from dataclasses import dataclass
from typing import Sequence, TypeVar


Value = TypeVar("Value")


class SplitError(ValueError):
    """Raised when a chronological sequence cannot be split safely."""


@dataclass(frozen=True)
class SequencePartitions:
    training: tuple[Value, ...]
    validation: tuple[Value, ...]
    testing: tuple[Value, ...]
    training_end: int
    validation_end: int


def chronological_split(
    sequence: Sequence[Value],
    *,
    training_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    testing_fraction: float = 0.2,
) -> SequencePartitions:
    fractions = (training_fraction, validation_fraction, testing_fraction)
    if any(
        not math.isfinite(fraction) or fraction <= 0 or fraction >= 1
        for fraction in fractions
    ):
        raise SplitError("Each split fraction must be greater than zero and less than one.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise SplitError("Training, validation, and testing fractions must sum to one.")

    training_end = int(len(sequence) * training_fraction)
    validation_end = int(len(sequence) * (training_fraction + validation_fraction))
    if training_end < 1 or validation_end <= training_end or validation_end >= len(sequence):
        raise SplitError(
            "The sequence is too short for non-empty chronological partitions."
        )

    return SequencePartitions(
        training=tuple(sequence[:training_end]),
        validation=tuple(sequence[training_end:validation_end]),
        testing=tuple(sequence[validation_end:]),
        training_end=training_end,
        validation_end=validation_end,
    )
