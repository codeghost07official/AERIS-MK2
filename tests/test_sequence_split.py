import pytest

from app.services.sequence_split import SplitError, chronological_split


def test_chronological_split_is_ordered_non_overlapping_and_repeatable():
    sequence = list(range(100))

    first = chronological_split(sequence)
    second = chronological_split(sequence)

    assert first == second
    assert first.training == tuple(range(60))
    assert first.validation == tuple(range(60, 80))
    assert first.testing == tuple(range(80, 100))
    assert set(first.training).isdisjoint(first.validation)
    assert set(first.validation).isdisjoint(first.testing)


def test_chronological_split_rejects_invalid_fractions_and_short_data():
    with pytest.raises(SplitError):
        chronological_split(list(range(10)), training_fraction=0.5)
    with pytest.raises(SplitError):
        chronological_split(list(range(10)), training_fraction=float("nan"))
    with pytest.raises(SplitError):
        chronological_split([1, 2])
