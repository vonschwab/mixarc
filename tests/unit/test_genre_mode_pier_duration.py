"""Genre-mode piers are algorithm-selected, so they must obey the configured
duration window. User-supplied seeds (artist/seeds mode) remain exempt."""
import numpy as np
import pytest

from src.playlist.genre_mode import filter_member_indices_by_duration


def test_overlong_candidate_is_excluded():
    # durations_ms indexed by bundle position; index 2 is 77:42 = 4662s
    durations = np.array([180_000, 240_000, 4_662_000, 300_000], dtype=float)
    kept, removed = filter_member_indices_by_duration(
        [0, 1, 2, 3], durations, min_seconds=46, max_seconds=720
    )
    assert kept == [0, 1, 3]
    assert removed == 1


def test_too_short_candidate_is_excluded():
    durations = np.array([30_000, 240_000], dtype=float)
    kept, removed = filter_member_indices_by_duration(
        [0, 1], durations, min_seconds=46, max_seconds=720
    )
    assert kept == [1]
    assert removed == 1


def test_boundaries_are_inclusive():
    durations = np.array([46_000, 720_000], dtype=float)
    kept, _ = filter_member_indices_by_duration(
        [0, 1], durations, min_seconds=46, max_seconds=720
    )
    assert kept == [0, 1]


def test_missing_duration_is_kept_not_dropped():
    """A null/zero duration means unknown, not invalid — dropping it would
    silently shrink the member set on incomplete metadata."""
    durations = np.array([0, 240_000], dtype=float)
    kept, removed = filter_member_indices_by_duration(
        [0, 1], durations, min_seconds=46, max_seconds=720
    )
    assert kept == [0, 1]
    assert removed == 0


def test_empty_input_returns_empty():
    assert filter_member_indices_by_duration([], np.array([]), min_seconds=46, max_seconds=720) == ([], 0)
