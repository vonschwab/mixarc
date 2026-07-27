"""The live builder must order ARTIST piers alone and insert tag-steering anchors
into the gaps -- never re-order anchors as co-equal piers.

Fixture shape (sonic space is 2-D, all unit vectors so cosines are exact):
  a0,a1,a2 = three "seed artist" piers spread along the arc;
  k0,k1    = on-tag ANCHORS, each closest to one interior region;
  f0..f3   = bridge filler.
Without the fix, _order_seeds_by_bridgeability is free to seat an anchor at
either terminal. With it, terminals are artist piers by construction.
"""
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
