"""The live builder must order ARTIST piers alone and insert tag-steering anchors
into the gaps -- never re-order anchors as co-equal piers.

Fixture shape (sonic space is 2-D, all unit vectors so cosines are exact):
  a0,a1,a2 = three "seed artist" piers spread along the arc;
  k0,k1    = on-tag ANCHORS, each closest to one interior region;
  f0..f3   = bridge filler.
Without the fix, _order_seeds_by_bridgeability is free to seat an anchor at
either terminal. With it, terminals are artist piers by construction.
"""
import logging
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist

_IDS = ["a0", "a1", "a2", "k0", "k1", "f0", "f1", "f2", "f3"]


def _unit(theta: float) -> list[float]:
    return [float(np.cos(theta)), float(np.sin(theta))]


# Angles in radians: artist piers at 0.0, 0.6, 1.2; anchors at 0.30 and 0.90
# (i.e. squarely inside gap 0 and gap 1); filler interleaved.
_ANGLES = [0.00, 0.60, 1.20, 0.30, 0.90, 0.15, 0.45, 0.75, 1.05]
_X = np.array([_unit(t) for t in _ANGLES], dtype=float)


def _bundle() -> ArtifactBundle:
    track_ids = np.array(_IDS, dtype=object)
    artists = np.array(
        ["Seed", "Seed", "Seed", "Foreign A", "Foreign B", "F0", "F1", "F2", "F3"],
        dtype=object,
    )
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([str(a).lower() for a in artists], dtype=object),
        track_artists=artists,
        track_titles=np.array([f"Title {i}" for i in range(len(_IDS))], dtype=object),
        X_sonic=_X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=None,
        X_genre_smoothed=None,
        genre_vocab=None,
        track_id_to_index={str(t): i for i, t in enumerate(track_ids)},
    )


def _cfg() -> PierBridgeConfig:
    return PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
        mini_pier_enabled=False,
        corridor_width_percentile=0.0,
    )


def _run(**kwargs):
    bundle = _bundle()
    return build_pier_bridge_playlist(
        # Anchors LAST, exactly as playlist_generator appends them.
        seed_track_ids=["a0", "a1", "a2", "k0", "k1"],
        total_tracks=9,
        bundle=bundle,
        candidate_pool_indices=[5, 6, 7, 8],
        cfg=_cfg(),
        **kwargs,
    )


def _run_total(total_tracks, **kwargs):
    """Like _run, but with an overridable total_tracks -- needed when a test
    drops anchors down to 3 piers, since the filler pool (f0..f3, 4 tracks)
    can't fill 6 interior slots (2 segments x 3) at the default total_tracks=9."""
    bundle = _bundle()
    return build_pier_bridge_playlist(
        seed_track_ids=["a0", "a1", "a2", "k0", "k1"],
        total_tracks=total_tracks,
        bundle=bundle,
        candidate_pool_indices=[5, 6, 7, 8],
        cfg=_cfg(),
        **kwargs,
    )


def _pier_order(result, bundle_ids=_IDS):
    """Piers, in realized playlist order."""
    piers = {"a0", "a1", "a2", "k0", "k1"}
    return [t for t in result.track_ids if t in piers]


def test_anchors_land_in_gaps_never_terminal_never_adjacent():
    res = _run(tag_anchor_track_ids={"k0", "k1"})
    assert res.success, res.failure_reason
    order = _pier_order(res)
    assert order[0].startswith("a") and order[-1].startswith("a"), order
    assert not any(
        x.startswith("k") and y.startswith("k") for x, y in zip(order, order[1:])
    ), order
    assert set(order) == {"a0", "a1", "a2", "k0", "k1"}, order


def test_rollback_flag_restores_legacy_co_equal_ordering():
    """gap_insertion=False must take the untouched legacy path: all five piers go
    through _order_seeds_by_bridgeability together."""
    legacy = _run(tag_anchor_track_ids={"k0", "k1"}, tag_anchor_gap_insertion=False)
    none_passed = _run(tag_anchor_track_ids=None)
    assert legacy.success and none_passed.success
    assert list(legacy.track_ids) == list(none_passed.track_ids)


def test_no_anchor_ids_is_byte_identical_to_today():
    """The safety property: absent anchor ids, ordering is the legacy result."""
    a = _run(tag_anchor_track_ids=None)
    b = _run(tag_anchor_track_ids=set())
    assert list(a.track_ids) == list(b.track_ids)


# ---------------------------------------------------------------------------
# Review follow-up (Important #1 + #2): drop/fallback path coverage.
# ---------------------------------------------------------------------------

# A second, smaller fixture: two artist piers (a0, a1) => exactly ONE interior
# gap, so K > P-1 clamping is exercisable with three ranked anchors k0 > k1 > k2.
_CLAMP_IDS = ["a0", "a1", "k0", "k1", "k2", "f0", "f1", "f2"]
_CLAMP_ANGLES = [0.00, 0.60, 0.20, 0.30, 0.40, 0.10, 0.35, 0.50]
_CLAMP_X = np.array([_unit(t) for t in _CLAMP_ANGLES], dtype=float)


def _clamp_bundle() -> ArtifactBundle:
    track_ids = np.array(_CLAMP_IDS, dtype=object)
    artists = np.array(
        ["Seed", "Seed", "Foreign A", "Foreign B", "Foreign C", "F0", "F1", "F2"],
        dtype=object,
    )
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([str(a).lower() for a in artists], dtype=object),
        track_artists=artists,
        track_titles=np.array([f"Title {i}" for i in range(len(_CLAMP_IDS))], dtype=object),
        X_sonic=_CLAMP_X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=None,
        X_genre_smoothed=None,
        genre_vocab=None,
        track_id_to_index={str(t): i for i, t in enumerate(track_ids)},
    )


def test_clamp_drops_lowest_ranked_anchors_when_k_exceeds_gaps():
    """2 artist piers => 1 gap; 3 ranked anchors (k0 > k1 > k2). K > P-1 clamps
    the two lowest-ranked anchors regardless of score -- only k0 survives."""
    res = build_pier_bridge_playlist(
        # Rank order = order of appearance: k0 then k1 then k2.
        seed_track_ids=["a0", "a1", "k0", "k1", "k2"],
        total_tracks=5,
        bundle=_clamp_bundle(),
        candidate_pool_indices=[5, 6, 7],
        cfg=_cfg(),
        tag_anchor_track_ids={"k0", "k1", "k2"},
        tag_anchor_min_bridge=-1.0,  # isolate the clamp from score gating
    )
    assert res.success, res.failure_reason
    piers = {"a0", "a1", "k0", "k1", "k2"}
    order = [t for t in res.track_ids if t in piers]
    assert set(order) == {"a0", "a1", "k0"}, order
    assert "k1" not in res.track_ids and "k2" not in res.track_ids


def test_all_anchors_dropped_unbridgeable_still_builds():
    """min_bridge set unreachably high -- both anchors are islands, both are
    dropped, and the run still succeeds with only the artist piers seated.
    total_tracks=7 (not the module default 9): with both anchors dropped only
    3 piers remain (2 segments), and 9 tracks would need 6 interior slots from
    a 4-track filler pool -- infeasible for reasons unrelated to this test."""
    res = _run_total(7, tag_anchor_track_ids={"k0", "k1"}, tag_anchor_min_bridge=0.999)
    assert res.success, res.failure_reason
    order = _pier_order(res)
    assert set(order) == {"a0", "a1", "a2"}, order
    assert order[0].startswith("a") and order[-1].startswith("a"), order
    assert "k0" not in res.track_ids and "k1" not in res.track_ids


def test_unmatched_anchor_ids_warn_and_fall_back_to_legacy(caplog):
    """Anchor ids that resolve in the bundle but aren't among the piers can't
    be placed -- WARN, don't silently no-op, and degrade to legacy co-equal
    ordering (same result as no anchor ids at all)."""
    with caplog.at_level(logging.WARNING):
        res = _run(tag_anchor_track_ids={"f0", "f1"})
    assert res.success, res.failure_reason
    assert any(
        "NONE matched a pier" in r.message
        for r in caplog.records if r.name.startswith("src.playlist.pier_bridge_builder")
    ), [r.message for r in caplog.records]
    baseline = _run(tag_anchor_track_ids=None)
    assert list(res.track_ids) == list(baseline.track_ids)


def test_single_artist_pier_warns_and_falls_back_to_legacy(caplog):
    """With fewer than 2 artist piers there is no interior gap -- WARN, don't
    silently no-op, and degrade to legacy co-equal ordering of the same seeds."""
    bundle = _bundle()
    common = dict(
        seed_track_ids=["a0", "k0"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=[5, 6, 7, 8],
        cfg=_cfg(),
    )
    with caplog.at_level(logging.WARNING):
        res = build_pier_bridge_playlist(tag_anchor_track_ids={"k0"}, **common)
    assert res.success, res.failure_reason
    assert any(
        "only 1 artist pier" in r.message
        for r in caplog.records if r.name.startswith("src.playlist.pier_bridge_builder")
    ), [r.message for r in caplog.records]
    baseline = build_pier_bridge_playlist(tag_anchor_track_ids=None, **common)
    assert baseline.success, baseline.failure_reason
    assert list(res.track_ids) == list(baseline.track_ids)


def test_dropped_anchor_does_not_trigger_false_seed_count_mismatch(caplog):
    """Regression for review Important #1: a legitimately dropped anchor must
    NOT fire the 'seed count mismatch' warning, and stats['num_seeds'] must
    reflect the pier count that actually reached the playlist (3), not the
    pre-drop request count (5)."""
    with caplog.at_level(logging.WARNING):
        res = _run_total(7, tag_anchor_track_ids={"k0", "k1"}, tag_anchor_min_bridge=0.999)
    assert res.success, res.failure_reason
    assert not any(
        "seed count mismatch" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
    assert res.stats["num_seeds"] == 3, res.stats["num_seeds"]


def test_placed_anchors_count_toward_num_seeds_when_none_dropped():
    """Companion positive control: when no anchor is dropped, stats['num_seeds']
    still counts all 5 piers (artist piers + both placed anchors)."""
    res = _run(tag_anchor_track_ids={"k0", "k1"})
    assert res.success, res.failure_reason
    assert res.stats["num_seeds"] == 5, res.stats["num_seeds"]
