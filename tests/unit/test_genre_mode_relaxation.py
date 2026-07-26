# tests/unit/test_genre_mode_relaxation.py
from src.playlist import genre_mode
from tests.unit.test_genre_mode_membership import FakeSteering, conn, steering  # noqa: F401


def test_no_relaxation_when_pool_is_big_enough(conn, steering):  # noqa: F811
    ids, _sims, thr = genre_mode.resolve_pool_with_relaxation(
        conn, steering, {"shoegaze"}, "shoegaze",
        start_threshold=0.5, steps=[0.35, 0.2], min_tracks=2,
    )
    assert thr == 0.5
    assert len(ids) >= 2


def test_relaxes_until_min_tracks_met(conn, steering):  # noqa: F811
    # At 0.99 only the 2 exact shoegaze tracks exist; need 3 -> must step down to
    # 0.5, where dream_pop (t3) joins.
    ids, _sims, thr = genre_mode.resolve_pool_with_relaxation(
        conn, steering, {"shoegaze"}, "shoegaze",
        start_threshold=0.99, steps=[0.5], min_tracks=3,
    )
    assert thr == 0.5
    assert ids == {"t1", "t2", "t3"}


def test_returns_widest_pool_when_never_satisfied(conn, steering):  # noqa: F811
    # Nothing can reach 99 tracks; must return the widest attempt, not raise.
    ids, _sims, thr = genre_mode.resolve_pool_with_relaxation(
        conn, steering, {"shoegaze"}, "shoegaze",
        start_threshold=0.99, steps=[0.5, 0.005], min_tracks=99,
    )
    assert thr == 0.005
    assert len(ids) >= 3


def test_steps_above_start_threshold_are_ignored(conn, steering):  # noqa: F811
    ids, _sims, thr = genre_mode.resolve_pool_with_relaxation(
        conn, steering, {"shoegaze"}, "shoegaze",
        start_threshold=0.5, steps=[0.9, 0.2], min_tracks=99,
    )
    assert thr == 0.2
