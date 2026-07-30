import types
from pathlib import Path

import numpy as np
import pytest

from src.playlist.artist_style import (
    ArtistStyleConfig,
    build_genre_neighbor_candidate_pool,
    cluster_artist_tracks,
    order_clusters,
    reorder_avoiding_low_support_terminal,
    compute_within_artist_support,
    _finite_median,
    _robust_energy_span,
    _slot_targets_by_quantile,
    _slot_proximity,
    _medoids_for_cluster,
    _dedupe_artist_indices,
    _support_penalty,
    load_artist_energy_values,
    compute_pier_bridgeability,
    seed_genre_relevance_mask,
)


class DummyBundle:
    def __init__(self, X_sonic, artist_keys, track_ids, track_artists=None, track_titles=None):
        self.X_sonic = X_sonic
        self.X_sonic_start = None
        self.X_sonic_mid = None
        self.X_sonic_end = None
        self.artist_keys = artist_keys
        self.track_ids = track_ids
        self.track_artists = track_artists
        self.track_titles = track_titles
        self.track_id_to_index = {str(t): i for i, t in enumerate(track_ids)}
        # ArtifactBundle has Optional[np.ndarray] = None for these; the prod
        # cluster_artist_tracks code path checks them, so the mock must mirror.
        self.durations_ms = None
        self.X_genre_raw = None
        self.X_genre_smoothed = None
        self.genre_vocab = None


def test_artist_style_config_has_energy_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.medoid_energy_weight == 0.0          # opt-in: off by default
    assert cfg.energy_feature == "arousal_p50"
    assert cfg.energy_slot_lo_pct == 10.0
    assert cfg.energy_slot_hi_pct == 90.0


def test_selects_multiple_clusters_and_medoids():
    # Two clear clusters on unit vectors
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],  # cluster 1
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],  # cluster 2
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    cfg = ArtistStyleConfig(cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True)
    clusters, medoids, medoids_by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle,
        artist_name="A",
        cfg=cfg,
        random_seed=0,
    )
    assert len(clusters) == 2
    assert len(medoids_by_cluster) == 2
    ordered = order_clusters(medoids, X_norm)
    assert set(ordered) == set(medoids)


# test_balanced_candidate_pool_respects_per_cluster_limits deleted (Phase 1
# Task 8): build_balanced_candidate_pool itself was deleted -- corridor
# pooling's library-wide eligible universe replaces the per-cluster external
# pool it used to build. See src/playlist/artist_style.py's retirement
# comment where the function used to live.


def test_genre_neighbor_pool_recovers_low_sonic_genre_match():
    artist_keys = np.array(["seed", "seed", "genre-match", "conflict"])
    track_ids = np.array(["s0", "s1", "genre_match", "conflict"])
    X = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],  # low sonic similarity to seed piers
        [0.0, 1.0],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.genre_vocab = np.array(["indie pop", "punk", "rnb", "house", "soul"])
    bundle.X_genre_raw = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
    ], dtype=float)
    bundle.X_genre_smoothed = bundle.X_genre_raw.copy()

    pool = build_genre_neighbor_candidate_pool(
        bundle=bundle,
        pier_indices=[0, 1],
        artist_key="seed",
        pool_size=10,
        min_similarity=0.3,
        min_confidence=0.5,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
        genre_method="weighted_jaccard",
    )

    assert "genre_match" in pool
    assert "conflict" not in pool


def test_allowed_invariant_helper():
    from src.playlist.pipeline import enforce_allowed_invariant
    allowed = {"1", "2"}
    enforce_allowed_invariant(["1", "2"], allowed, context="test")
    with pytest.raises(ValueError):
        enforce_allowed_invariant(["1", "3"], allowed, context="test")


# test_bridge_endpoint_gate_blocks_low_sim and
# test_narrow_bridge_floor_default_blocks_low_sim deleted (Phase 1 Task 8):
# both drove _build_segment_candidate_pool_scored directly (the legacy
# segment-scored pool builder), which was deleted along with the
# legacy/corridor pooling selector. Corridor's equivalent bridge-floor-style
# gating (percentile min-sim membership) is covered by
# tests/unit/test_corridor_builder.py.


def test_seed_artist_disallowed_in_interiors_when_enabled():
    # For artist playlist runs, we want "seed artist = piers only" when the policy is enabled,
    # even if seed-artist candidates would otherwise win.
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # pier A (seed artist)
        [0.0, 1.0],                   # pier B (seed artist)
        [np.sqrt(0.5), np.sqrt(0.5)], # seed-artist candidate (would win on bridge score)
        [0.6, 0.8],                   # other-artist candidate
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["seed", "seed", "seed", "other"]),
        track_ids=np.array(["a0", "b0", "seed_cand", "other_cand"]),
        track_artists=np.array(["Seed Artist", "Seed Artist", "Seed Artist feat. X", "Other Artist"]),
        track_titles=np.array(["A", "B", "Bridge Seed", "Bridge Other"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
            disallow_seed_artist_in_interiors=True,
        ),
        allowed_track_ids_set={"a0", "b0", "seed_cand", "other_cand"},
    )
    assert result.success
    assert result.track_ids[1] == "other_cand"


def test_builder_wires_roam_detour_when_enabled(monkeypatch):
    # Roam corridors (Phase 1): when the flag is on, the builder must compute the
    # per-segment on-manifold sonic detour and pass it to the beam; when off, the
    # beam gets None. Both must still produce a valid playlist (never-fail).
    import src.playlist.pier_bridge_builder as pbb

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5)],
        [0.6, 0.8],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c", "d"]),
        track_ids=np.array(["a0", "b0", "c0", "d0"]),
        track_artists=np.array(["A", "B", "C", "D"]),
        track_titles=np.array(["A", "B", "C", "D"]),
    )
    captured = {}
    real_beam = pbb._beam_search_segment

    def _spy(*args, **kwargs):
        captured["roam"] = kwargs.get("roam_detour_sonic")
        return real_beam(*args, **kwargs)

    monkeypatch.setattr(pbb, "_beam_search_segment", _spy)

    def _run(enabled):
        captured.clear()
        return pbb.build_pier_bridge_playlist(
            seed_track_ids=["a0", "b0"],
            total_tracks=3,
            bundle=bundle,
            candidate_pool_indices=[2, 3],
            cfg=pbb.PierBridgeConfig(
                transition_floor=0.0,
                bridge_floor=0.0,
                weight_bridge=1.0,
                weight_transition=0.0,
                eta_destination_pull=0.0,
                roam_corridors_enabled=enabled,
                roam_width_sonic=0.0,
                roam_penalty_slope=5.0,
            ),
            allowed_track_ids_set={"a0", "b0", "c0", "d0"},
        )

    res_on = _run(enabled=True)
    assert res_on.success
    assert captured["roam"] is not None          # detour wired through when enabled
    res_off = _run(enabled=False)
    assert res_off.success
    assert captured["roam"] is None              # not computed when disabled


# test_one_artist_per_segment_collapses_feat_with_variants deleted (Phase 1
# Task 8): drove _build_segment_candidate_pool_scored's
# collapse_segment_pool_by_artist path directly (deleted -- corridor doesn't
# collapse the pool by artist at build time; the beam enforces per-segment
# artist diversity on its own, per CLAUDE.md's project-specific gotcha on
# collapse_segment_pool_by_artist).


def test_seed_track_key_collision_excludes_duplicate_song():
    # Prevent "same song twice" when a pier seed and a candidate share (artist_key,title_key)
    # but have different track_ids (Yo La Tengo "Autumn Sweater" style).
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # seed 1
        [0.0, 1.0],                   # seed 2
        [np.sqrt(0.5), np.sqrt(0.5)], # duplicate of seed 1 by track_key
        [0.6, 0.8],                   # valid bridge candidate
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["yo la tengo", "other", "yo la tengo", "x"]),
        track_ids=np.array(["seed_autumn", "seed_other", "dup_autumn", "x_song"]),
        track_artists=np.array(["Yo La Tengo", "Other", "Yo La Tengo", "X"]),
        track_titles=np.array(["Autumn Sweater", "Other Seed", "Autumn Sweater", "Bridge"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["seed_autumn", "seed_other"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
        ),
        allowed_track_ids_set={"seed_autumn", "seed_other", "dup_autumn", "x_song"},
    )
    assert result.success
    assert "dup_autumn" not in result.track_ids
    assert result.track_ids[1] == "x_song"


def test_one_per_artist_cap_applies_across_segments_but_exempts_seeds():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # seed A, artist seed-a
        [0.0, 1.0],                   # seed B, artist seed-b
        [-1.0, 0.0],                  # seed C, artist repeat
        [np.sqrt(0.5), np.sqrt(0.5)], # repeat artist, segment A->B winner
        [-0.6, 0.8],                  # repeat artist, segment B->C would win without global cap
        [-0.4, 0.9],                  # other artist, valid fallback for B->C
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["seed-a", "seed-b", "repeat", "repeat", "repeat", "other"]),
        track_ids=np.array(["seed_a", "seed_b", "seed_c", "repeat_1", "repeat_2", "other_1"]),
        track_artists=np.array(["Seed A", "Seed B", "Repeat", "Repeat", "Repeat", "Other"]),
        track_titles=np.array(["A", "B", "C", "Repeat 1", "Repeat 2", "Other 1"]),
    )

    result = build_pier_bridge_playlist(
        seed_track_ids=["seed_a", "seed_b", "seed_c"],
        total_tracks=5,
        bundle=bundle,
        candidate_pool_indices=[3, 4, 5],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
            max_non_seed_tracks_per_artist=1,
            # Phase 1 Task 8: corridor is the sole pooling path. bridge_floor=0.0
            # was this fixture's "admit everyone" sentinel under legacy's
            # fixed-floor gate; corridor's percentile-based membership needs
            # the equivalent -- width_percentile=0.0 admits the full
            # 3-candidate universe (repeat_1, repeat_2, other_1) into every
            # segment's corridor so the cross-segment artist cap this test
            # exists to demonstrate can actually be exercised.
            corridor_width_percentile=0.0,
        ),
        allowed_track_ids_set={"seed_a", "seed_b", "seed_c", "repeat_1", "repeat_2", "other_1"},
    )

    assert result.success
    assert result.track_ids == ["seed_a", "repeat_1", "seed_b", "other_1", "seed_c"]
    assert result.stats["artist_counts"]["repeat"] == 2
    assert result.stats["non_seed_artist_counts"]["repeat"] == 1


# test_segment_pool_is_endpoint_local_not_global_seed_max deleted (Phase 1
# Task 8): drove _build_segment_candidate_pool_scored directly (deleted).
# Corridor's per-segment membership is inherently endpoint-local (min-sim
# to THIS segment's two anchors), covered by test_corridor_builder.py.


def test_progress_monotonicity_in_beam_search_segment():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X = np.array([
        [1.0, 0.0],  # pier A
        [0.0, 1.0],  # pier B
        [0.9, 0.4],  # early-ish
        [0.7, 0.7],  # mid
        [0.4, 0.9],  # late-ish
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    candidates = [2, 3, 4]
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_monotonic_epsilon=0.0,
        progress_penalty_weight=1.0,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        2,
        candidates,
        X_norm,  # X_full (transition space)
        X_norm,  # X_full_norm (sonic space)
        None,
        None,
        None,
        None,
        cfg,
        20,
    )
    assert err is None
    assert path is not None and len(path) == 2

    vec_a = X_norm[0]
    vec_b = X_norm[1]
    d = vec_b - vec_a
    denom = float(np.dot(d, d))
    assert denom > 1e-9

    ts = []
    for idx in path:
        t_raw = float(np.dot((X_norm[int(idx)] - vec_a), d) / denom)
        ts.append(max(0.0, min(1.0, t_raw)))
    assert ts[0] <= ts[1] + 1e-9


def test_transition_floor_no_longer_hard_gates_low_t():
    # Roam-only design (2026-06-25 centered-transition fix, design §5): the
    # ``transition_floor`` HARD GATE is removed. A low-T edge that the legacy floor
    # would have rejected (cosine ~0.20 < the old 0.35 floor) is now ACCEPTED in the
    # legacy beam path — roam's worst-edge minimax, not a hard floor, is the quality
    # mechanism (and removing the gate is strictly fewer cascade triggers → budget-safe).
    # ``transition_floor`` is retained only as an inert param for cascade callers.
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist
    from src.playlist.run_audit import InfeasibleHandlingConfig

    X = np.array([
        [1.0, 0.0],         # pier A
        [0.0, 1.0],         # pier B
        [0.2, 0.98],        # candidate: A->cand cosine ~0.20, below the old 0.35 floor
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(transition_floor=0.35, bridge_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(guarantee_feasible=False),
    )
    # The old hard gate would have failed this build (no edge ≥ 0.35); it no longer gates.
    assert result.success
    assert "c0" in result.track_ids


def test_soft_genre_penalty_changes_ranking_without_gating():
    # Candidate "low" is slightly better on transitions, but has very low genre sim (< threshold),
    # so with a soft penalty enabled it should be outranked by "high".
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],   # pier A
        [0.0, 1.0],   # pier B
        [0.71, 0.71],   # cand low-genre: slightly better transitions overall
        [0.65, 0.76],   # cand high-genre: slightly worse transitions, better genre
    ])
    X_genre = np.array([
        [1.0, 0.0],                              # A
        [1.0, 0.0],                              # B
        [0.10, float(np.sqrt(1 - 0.10**2))],      # low genre sim to A
        [0.30, float(np.sqrt(1 - 0.30**2))],      # high genre sim to A
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "low", "high"]),
        track_ids=np.array(["a0", "b0", "c_low", "c_high"]),
        track_titles=np.array(["A", "B", "Low", "High"]),
    )
    base_cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        eta_destination_pull=0.0,
        weight_bridge=0.0,
        weight_transition=1.0,
        genre_tiebreak_weight=0.0,
        genre_penalty_threshold=0.20,
        genre_penalty_strength=0.0,
        # Phase 1 Task 8: corridor is the sole pooling path. bridge_floor=0.0
        # was this fixture's "admit everyone" sentinel under legacy's
        # fixed-floor gate; corridor's percentile-based membership needs the
        # equivalent -- width_percentile=0.0 admits the full 2-candidate
        # universe (c_low, c_high) so the beam-level soft-genre-penalty
        # differentiation this test exists to demonstrate can actually
        # compete between both (inherited by `penalized` below via
        # **base_cfg.__dict__).
        corridor_width_percentile=0.0,
    )
    base = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=base_cfg,
        X_genre_smoothed=X_genre,
        allowed_track_ids_set={"a0", "b0", "c_low", "c_high"},
    )
    assert base.success
    assert base.track_ids[1] == "c_low"

    penalized = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(**{**base_cfg.__dict__, "genre_penalty_strength": 0.10}),
        X_genre_smoothed=X_genre,
        allowed_track_ids_set={"a0", "b0", "c_low", "c_high"},
    )
    assert penalized.success
    assert penalized.track_ids[1] == "c_high"


def test_soft_genre_penalty_per_mode_resolution():
    """Per-mode soft_genre_penalty keys override the legacy flat defaults."""
    from src.playlist.config import resolve_pier_bridge_tuning

    overrides = {
        "pier_bridge": {
            "soft_genre_penalty_threshold_strict": 0.82,
            "soft_genre_penalty_threshold_narrow": 0.78,
            "soft_genre_penalty_threshold_dynamic": 0.55,
            "soft_genre_penalty_strength_strict": 0.40,
            "soft_genre_penalty_strength_narrow": 0.30,
            "soft_genre_penalty_strength_dynamic": 0.15,
        }
    }

    for mode, expected_threshold, expected_strength in [
        ("strict", 0.82, 0.40),
        ("narrow", 0.78, 0.30),
        ("dynamic", 0.55, 0.15),
    ]:
        tuning, sources = resolve_pier_bridge_tuning(
            mode=mode, similarity_floor=0.20, overrides=overrides
        )
        assert tuning.genre_penalty_threshold == expected_threshold, (
            f"mode={mode}: expected threshold {expected_threshold}, "
            f"got {tuning.genre_penalty_threshold}"
        )
        assert tuning.genre_penalty_strength == expected_strength, (
            f"mode={mode}: expected strength {expected_strength}, "
            f"got {tuning.genre_penalty_strength}"
        )
        assert sources["genre_penalty_threshold"] == (
            f"pier_bridge.soft_genre_penalty_threshold_{mode}"
        )
        assert sources["genre_penalty_strength"] == (
            f"pier_bridge.soft_genre_penalty_strength_{mode}"
        )

    # Modes without per-mode keys fall back to the flat default (0.20 / 0.10)
    tuning_off, sources_off = resolve_pier_bridge_tuning(
        mode="off", similarity_floor=0.20, overrides=overrides
    )
    assert tuning_off.genre_penalty_threshold == 0.20
    assert tuning_off.genre_penalty_strength == 0.10
    assert sources_off["genre_penalty_threshold"] == "default"
    assert sources_off["genre_penalty_strength"] == "default"


def test_local_sonic_edge_penalty_changes_ranking_without_gating():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X_transition = np.array([
        [1.0, 0.0],   # pier A
        [1.0, 0.0],   # pier B
        [1.0, 0.0],   # low-sonic candidate: best transition score
        [0.9, 0.4],   # coherent candidate: slightly weaker transition score
    ])
    X_transition = X_transition / (np.linalg.norm(X_transition, axis=1, keepdims=True) + 1e-12)
    X_sonic = np.array([
        [1.0, 0.0],   # pier A
        [1.0, 0.0],   # pier B
        [-1.0, 0.0],  # low-sonic candidate: locally opposed to both piers
        [1.0, 0.0],   # coherent candidate
    ])
    X_sonic = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
    base_cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        weight_bridge=0.0,
        weight_transition=1.0,
        eta_destination_pull=0.0,
        progress_enabled=False,
    )

    base_path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        base_cfg,
        10,
    )
    assert err is None
    assert base_path == [2]

    local_stats = {}
    penalized_path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            **{
                **base_cfg.__dict__,
                "local_sonic_edge_penalty_enabled": True,
                "local_sonic_edge_penalty_threshold": 0.5,
                "local_sonic_edge_penalty_strength": 1.0,
            }
        ),
        10,
        local_sonic_stats=local_stats,
    )
    assert err is None
    assert penalized_path == [3]
    assert local_stats["local_sonic_penalty_hits"] >= 2
    assert local_stats["local_sonic_gate_rejected"] == 0


def test_local_sonic_edge_floor_applies_to_final_destination_edge():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X_transition = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    X_sonic = np.array([
        [1.0, 0.0],   # pier A
        [-1.0, 0.0],  # pier B
        [1.0, 0.0],   # candidate: good after A, bad into B
    ])
    local_stats = {}

    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            transition_floor=-1.0,
            bridge_floor=-1.0,
            weight_bridge=0.0,
            weight_transition=1.0,
            eta_destination_pull=0.0,
            progress_enabled=False,
            local_sonic_edge_floor=0.0,
        ),
        10,
        local_sonic_stats=local_stats,
    )
    assert path is None
    assert err == "no valid final connection to destination"
    assert local_stats["local_sonic_gate_rejected"] == 1


# ---------------------------------------------------------------------------
# Task 5: energy-slot wiring in cluster_artist_tracks
# ---------------------------------------------------------------------------

def _two_cluster_bundle():
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],   # cluster 1 (indices 0,1,2)
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],   # cluster 2 (indices 3,4,5)
    ])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def test_cluster_artist_tracks_energy_weight_zero_matches_none():
    bundle = _two_cluster_bundle()
    cfg_off = ArtistStyleConfig(cluster_k_min=2, cluster_k_max=2, enabled=True)
    cfg_zero = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=0.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg_off, random_seed=0)
    zeroed = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg_zero, random_seed=0, energy_values=energy
    )
    assert sorted(base[1]) == sorted(zeroed[1])   # identical medoids


def test_cluster_artist_tracks_energy_runs_and_returns_medoids():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    clusters, medoids, by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=energy
    )
    assert len(clusters) == 2
    assert len(medoids) == 2


def test_cluster_artist_tracks_inert_on_flat_energy():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    flat = np.zeros(6)   # zero span => energy term inert, must not crash
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg, random_seed=0)
    flatted = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=flat
    )
    assert sorted(base[1]) == sorted(flatted[1])


def test_transition_floor_config_override_per_mode():
    from src.playlist.config import default_ds_config

    cfg = default_ds_config(
        "dynamic",
        playlist_len=30,
        overrides={"constraints": {"transition_floor_dynamic": 0.55}},
    )
    assert cfg.construct.transition_floor == 0.55

    cfg2 = default_ds_config(
        "narrow",
        playlist_len=30,
        overrides={"constraints": {"transition_floor_narrow": 0.66}},
    )
    assert cfg2.construct.transition_floor == 0.66


def test_pier_bridge_tuning_defaults_per_mode():
    from src.playlist.config import resolve_pier_bridge_tuning

    # Phase 1-3A relaxed bridge_floor: dynamic 0.03 -> 0.02, narrow 0.08 -> 0.05
    # (commit 34f2948).
    dyn, _ = resolve_pier_bridge_tuning(mode="dynamic", similarity_floor=0.2, overrides={})
    assert dyn.transition_floor == 0.35
    assert dyn.bridge_floor == 0.02
    assert dyn.weight_bridge == 0.6
    assert dyn.weight_transition == 0.4
    assert dyn.genre_tiebreak_weight == 0.05
    assert dyn.genre_penalty_threshold == 0.20
    assert dyn.genre_penalty_strength == 0.10

    nar, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.2, overrides={})
    assert nar.transition_floor == 0.45
    assert nar.bridge_floor == 0.05
    assert nar.weight_bridge == 0.7
    assert nar.weight_transition == 0.3
    assert nar.genre_tiebreak_weight == 0.05
    assert nar.genre_penalty_threshold == 0.20
    assert nar.genre_penalty_strength == 0.10


def test_pier_bridge_tuning_overrides_and_penalty_clamp():
    from src.playlist.config import resolve_pier_bridge_tuning

    tuning, sources = resolve_pier_bridge_tuning(
        mode="narrow",
        similarity_floor=0.2,
        overrides={
            "pier_bridge": {
                "bridge_floor_narrow": 0.12,
                "soft_genre_penalty_strength": 1.5,  # should clamp to 1.0
            }
        },
    )
    assert tuning.bridge_floor == 0.12
    assert sources["bridge_floor"] == "pier_bridge.bridge_floor_narrow"
    assert tuning.genre_penalty_strength == 1.0
    assert sources["genre_penalty_strength"] == "pier_bridge.soft_genre_penalty_strength"


def test_build_ds_overrides_includes_pier_bridge():
    from src.playlist_generator import build_ds_overrides

    ds_cfg = {
        "constraints": {"transition_floor_dynamic": 0.55},
        "pier_bridge": {"bridge_floor_dynamic": 0.04},
    }
    overrides = build_ds_overrides(ds_cfg)
    assert overrides["constraints"]["transition_floor_dynamic"] == 0.55
    assert overrides["pier_bridge"]["bridge_floor_dynamic"] == 0.04


# test_high_bridge_floor_fails_segment deleted (Phase 1 Task 8, investigated
# not just reflexively removed): its premise -- a high bridge_floor/
# transition_floor makes an infeasible segment hard-FAIL the whole build --
# was already stale independent of corridor. Traced via --log-cli-level=DEBUG:
# with seed_track_ids=["a0","b0"] and total_tracks=2, interior_len=0, so the
# segment takes beam.py's interior_length==0 fast path (direct pier_a->pier_b
# edge scoring) which does not hard-gate on bridge_floor/transition_floor --
# it scored the direct edge at transition=0.707 (well below the test's
# transition_floor=0.8) and still returned success=True, matching the SAME
# "soft floor, not a hard gate" behavior the adjacent (kept)
# test_transition_floor_no_longer_hard_gates_low_t already documents and
# asserts for. Not a corridor-vs-legacy difference -- corridor's own pooling
# is provably irrelevant here (interior_len=0 never consults the segment
# pool at all).


# test_bridge_floor_backoff_disabled_infeasible_segment_fails and
# test_bridge_floor_backoff_enabled_succeeds_and_records_attempts deleted
# (Phase 1 Task 8): both exercised infeasible_handling's bridge_floor backoff
# cascade (_bridge_floor_attempts), which was the legacy per-segment
# relaxation ladder Task 4 already gated off for corridor and Task 8 deleted
# outright -- the corridor widening ladder
# (_run_corridor_widening_ladder, tested by
# tests/integration/test_corridor_pooling.py's
# test_corridor_widening_ladder_recovers_stressed_segment) is the sole
# segment-level recovery mechanism now. `infeasible_handling.enabled`/
# `.backoff_steps`/etc. are retired config fields (warned via
# src.playlist_gui.worker._warn_retired_keys); `.guarantee_feasible` remains
# live and is exercised elsewhere.


def test_run_audit_writer_creates_markdown_report(tmp_path):
    from src.playlist.pier_bridge_builder import build_pier_bridge_playlist, PierBridgeConfig
    from src.playlist.run_audit import (
        InfeasibleHandlingConfig,
        RunAuditConfig,
        RunAuditContext,
        RunAuditEvent,
        now_utc_iso,
        write_markdown_report,
    )

    x = 0.04
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [x, float(np.sqrt(1 - x**2))],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_artists=np.array(["A", "B", "C"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    events = [
        RunAuditEvent(
            kind="preflight",
            ts_utc=now_utc_iso(),
            payload={
                "tuning": {"mode": "narrow", "bridge_floor": 0.08, "transition_floor": 0.0},
                "tuning_sources": {},
                "pool_summary": {"allowed_ids_count": 3},
            },
        )
    ]
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(bridge_floor=0.08, transition_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(enabled=True, backoff_steps=(0.03,)),
        audit_config=RunAuditConfig(enabled=True, include_top_k=5),
        audit_events=events,
    )
    assert result.success
    events.append(
        RunAuditEvent(
            kind="final_success",
            ts_utc=now_utc_iso(),
            payload={
                "playlist_tracks": [
                    {"track_id": tid, "artist": "", "title": ""} for tid in result.track_ids
                ],
                "weakest_edges": [],
                "summary_stats": {"final_playlist_size": len(result.track_ids)},
            },
        )
    )
    ctx = RunAuditContext(
        timestamp_utc=now_utc_iso(),
        run_id="test_run",
        cohesion_mode="narrow",
        seed_track_id="a0",
        seed_artist="A",
        dry_run=True,
        artifact_path="dummy",
        sonic_variant="raw",
        allowed_ids_count=3,
        pool_source="test",
        artist_style_enabled=False,
        extra={},
    )
    out_path = write_markdown_report(
        context=ctx,
        events=events,
        path=tmp_path / "audit.md",
        max_bytes=200000,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "## 2) Effective Tuning" in text
    assert "## 4) Segment Diagnostics" in text
    assert "### Segment 0" in text
    assert "#### Attempt 1" in text


def test_finite_median_ignores_nan():
    assert _finite_median(np.array([1.0, np.nan, 3.0])) == 2.0
    assert np.isnan(_finite_median(np.array([np.nan, np.nan])))


def test_robust_energy_span_uses_percentiles():
    vals = np.arange(10, dtype=float)  # 0..9
    span = _robust_energy_span(vals, 10.0, 90.0)
    assert span is not None
    lo, hi = span
    assert lo == pytest.approx(0.9)
    assert hi == pytest.approx(8.1)


def test_robust_energy_span_none_when_flat_or_sparse():
    assert _robust_energy_span(np.array([5.0, 5.0, 5.0]), 10.0, 90.0) is None  # zero span
    assert _robust_energy_span(np.array([np.nan, 1.0]), 10.0, 90.0) is None     # <2 finite


def test_slot_targets_quantile_uniform_is_evenly_spaced():
    # Uniform distribution: quantiles map ~linearly to values, so quantile-spacing
    # reduces to even value-spacing. medians order: cluster0 high, cluster1 low, cluster2 mid.
    energy = np.linspace(0.0, 10.0, 101)
    targets = _slot_targets_by_quantile([2.0, 0.0, 1.0], energy, 10.0, 90.0)
    assert targets[1] == pytest.approx(1.0, abs=0.2)    # lowest cluster -> p10 -> ~1.0
    assert targets[2] == pytest.approx(5.0, abs=0.2)    # mid -> p50 -> ~5.0
    assert targets[0] == pytest.approx(9.0, abs=0.2)    # highest cluster -> p90 -> ~9.0


def test_slot_targets_quantile_follows_density_not_range():
    # 8 high-energy tracks, 2 low: evenly-spaced quantiles land mostly in the dense
    # high mass, NOT evenly across the 0..5 value range (the representativeness fix).
    energy = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # 4 clusters ranked low->high by median energy
    targets = _slot_targets_by_quantile([0.0, 2.5, 5.0, 5.0], energy, 10.0, 90.0)
    assert targets[0] < 1.0                              # only the lowest reaches the sparse low end
    assert all(t == pytest.approx(5.0) for t in targets[1:])  # the rest follow the dense mass
    # (even value-spacing would have put a target at ~1.67 in the sparse middle)


def test_slot_targets_quantile_single_cluster_is_mid_quantile():
    energy = np.linspace(0.0, 10.0, 101)
    assert _slot_targets_by_quantile([3.0], energy, 10.0, 90.0) == [pytest.approx(5.0, abs=0.2)]


def test_slot_targets_quantile_nan_median_stays_nan():
    energy = np.linspace(0.0, 10.0, 101)
    targets = _slot_targets_by_quantile([np.nan, 1.0], energy, 10.0, 90.0)
    assert np.isnan(targets[0])


def test_slot_targets_quantile_sparse_energy_is_inert():
    targets = _slot_targets_by_quantile([0.0, 5.0], np.array([1.0]), 10.0, 90.0)
    assert all(np.isnan(t) for t in targets)   # <2 finite values -> all NaN (inert)


def test_slot_proximity_peaks_at_target_and_zeros_for_nan():
    z = np.array([5.0, 0.0, 10.0, np.nan])
    prox = _slot_proximity(z, target=5.0, span_width=10.0)
    assert prox[0] == pytest.approx(1.0)     # at target
    assert prox[1] == pytest.approx(0.5)     # half a span away
    assert prox[2] == pytest.approx(0.5)
    assert prox[3] == 0.0                     # NaN energy -> neutral (no bonus)


def test_slot_proximity_inert_when_target_nan():
    z = np.array([1.0, 2.0])
    assert np.all(_slot_proximity(z, target=np.nan, span_width=10.0) == 0.0)
    # span_width <= 0 is also inert (all zeros)
    assert np.all(_slot_proximity(z, target=5.0, span_width=0.0) == 0.0)


def test_dedupe_artist_indices_prefers_studio_and_collapses_dupes():
    # 0 studio "On a Plain" | 1 live version | 2 duplicate studio (lowercase)
    # | 3 unique song | 4 a song that exists ONLY as a live cut
    titles = np.array([
        "On a Plain",
        "On A Plain (Live In Tokyo)",
        "On a plain",
        "About a Girl",
        "Scoff (Live at Pine Street)",
    ])
    durations = np.array([200000, 210000, 200000, 150000, 180000], dtype=float)
    kept = _dedupe_artist_indices([0, 1, 2, 3, 4], titles, durations)
    assert 1 not in kept            # live "On a Plain" demoted (studio present)
    assert 2 not in kept            # duplicate studio collapsed
    assert 0 in kept                # one canonical studio "On a Plain" survives
    assert 3 in kept                # unique song kept
    assert 4 in kept                # sole-version live cut kept
    assert kept == sorted(kept)


def test_dedupe_artist_indices_no_titles_passthrough():
    # No titles -> can't group -> every index passes through unchanged.
    assert _dedupe_artist_indices([0, 1, 2], None, None) == [0, 1, 2]


def _centroid_for(X, indices):
    c = X[indices].mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


def test_medoid_energy_term_pulls_to_slot():
    # 3 candidates, near-identical sonic centrality; energy proximity favors index 1.
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    rng = np.random.default_rng(0)

    # Baseline (no energy): pick by sonic alone, top_k=1 => deterministic argmax.
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng, 1,
        None, None, 0.7, 0.3,
    )
    # Energy strongly favors index 1.
    rng2 = np.random.default_rng(0)
    energized = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng2, 1,
        None, None, 0.7, 0.3,
        10.0, np.array([0.0, 1.0, 0.0]),   # energy_weight, energy_proximity
    )
    assert energized == [1]


def test_medoid_energy_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
    )
    with_zero = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
        0.0, np.array([1.0, 0.0, 0.0]),   # weight 0 => proximity ignored
    )
    assert with_zero == base


def _write_energy_sidecar(tmp_path, track_ids, arousal):
    energy_dir = Path(tmp_path) / "energy"
    energy_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        energy_dir / "energy_sidecar.npz",
        track_ids=np.array(track_ids, dtype=object),
        arousal_p50=np.array(arousal, dtype=np.float32),
    )


def test_load_artist_energy_values_returns_zscored():
    import tempfile
    import shutil
    tmp_path = tempfile.mkdtemp()
    try:
        track_ids = ["a", "b", "c"]
        _write_energy_sidecar(tmp_path, track_ids, [1.0, 3.0, 5.0])
        bundle = types.SimpleNamespace(
            track_ids=np.array(track_ids), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=1.0, energy_feature="arousal_p50")
        vals = load_artist_energy_values(bundle, cfg)
        assert vals is not None and vals.shape == (3,)
        assert vals[0] < vals[1] < vals[2]            # preserves ordering
        assert abs(float(np.mean(vals))) < 1e-6        # z-scored => ~zero mean
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_inert_when_weight_zero():
    import tempfile
    import shutil
    tmp_path = tempfile.mkdtemp()
    try:
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a"]), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        assert load_artist_energy_values(bundle, ArtistStyleConfig()) is None
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_warns_when_sidecar_missing(caplog):
    import tempfile
    import shutil
    import logging
    tmp_path = tempfile.mkdtemp()
    try:
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a"]), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=0.5)
        with caplog.at_level(logging.WARNING):
            assert load_artist_energy_values(bundle, cfg) is None
        assert any("energy sidecar missing" in r.message for r in caplog.records)
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_warns_when_no_finite(caplog):
    # Sidecar exists but its track_ids do NOT overlap the bundle's track_ids,
    # so load_energy_matrix returns an all-NaN column. This is the production
    # "configured-knob-must-act" path: an artist whose tracks aren't in the
    # sidecar. Must return None AND log a "no finite" WARNING.
    import tempfile
    import shutil
    import logging
    tmp_path = tempfile.mkdtemp()
    try:
        _write_energy_sidecar(tmp_path, ["x", "y", "z"], [1.0, 3.0, 5.0])
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a", "b", "c"]),  # disjoint from sidecar ids
            artifact_path=Path(tmp_path) / "artifact.npz",
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=0.5, energy_feature="arousal_p50")
        with caplog.at_level(logging.WARNING):
            assert load_artist_energy_values(bundle, cfg) is None
        assert any("no finite" in r.message for r in caplog.records)
    finally:
        shutil.rmtree(tmp_path)


def test_medoid_popularity_term_breaks_tie_toward_popular():
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3)
    pop = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3, 0.0, None,
        5.0, np.array([0.0, 1.0, 0.0]))   # popularity_weight, popularity_values
    assert pop == [1]                       # strong popularity on index 1 wins the pick
    del base                                # baseline computed only to mirror the call shape


def test_medoid_popularity_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3)
    z = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3, 0.0, None,
        0.0, np.array([1.0, 0.0, 0.0]))   # popularity weight 0 -> ignored
    assert z == base


def _unit_rows(rows):
    X = np.array(rows, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def test_bridgeability_kth_rank_same_artist_exclusion():
    # 5-dim basis geometry. Artist 'a' rows: 0 (e1), 1 (e2), 2 (e3), 9 (e2 twin of 1).
    # Library rows: 3,4 (e1 — two neighbors for member 0), 5 (e3 — ONE neighbor for
    # member 2), 6,7 (e4), 8 (e5) filler.
    e = np.eye(5)
    X = _unit_rows([e[0], e[1], e[2],          # members 0,1,2 (artist 'a')
                    e[0], e[0], e[2],          # lib 3,4,5
                    e[3], e[3], e[4],          # lib 6,7,8
                    e[1]])                     # 9 = same-artist twin of member 1
    # muq calibration band (what resolve_transition_calib(None) returns)
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)

    t = compute_pier_bridgeability(X, [0, 1, 2], [0, 1, 2, 9], k=2,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    assert t.shape == (3,)
    assert t[0] > 0.9        # member 0: two e1 library neighbors -> kth cos = 1.0
    assert t[1] < 0.05       # member 1: only close row is same-artist twin 9 -> excluded
    assert t[2] < 0.05       # member 2: ONE library neighbor but k=2 -> kth cos = 0.0

    # Same geometry, k=1, excluding ONLY self (callers must always exclude the
    # member's own row — production passes all artist rows): the same-artist twin
    # at row 9 now counts and rescues member 1 -> proves the artist mask (not the
    # geometry) drove the failure above.
    t_selfonly = compute_pier_bridgeability(X, [1], [1], k=1,
                                            calib_center=c, calib_scale=s, calib_gain=g)
    assert t_selfonly[0] > 0.9


def test_bridgeability_empty_members_and_k_clamp():
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)
    e = np.eye(3)
    X = _unit_rows([e[0], e[0], e[1]])
    assert compute_pier_bridgeability(X, [], [0], k=10,
                                      calib_center=c, calib_scale=s, calib_gain=g).shape == (0,)
    # k larger than available non-excluded columns clamps to what's there (2 columns).
    t = compute_pier_bridgeability(X, [0], [0], k=10,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    # kth clamps to 2nd best = cos(e0, e1) = 0 -> low T, but finite (no crash/inf)
    assert np.isfinite(t[0]) and t[0] < 0.05


# ---------------------------------------------------------------------------
# Task 2: bridgeability veto + slot reallocation wired into cluster_artist_tracks
# ---------------------------------------------------------------------------

def _bridgeability_fixture():
    """Artist 'a' with 2 sonic clusters; only cluster A has library mass near it.

    Rows 0-2: artist cluster A (~e1, tiny jitter so kmeans is stable).
    Rows 3-5: artist cluster B (~e2) — isolated, no library neighbors.
    Rows 6-11: library tracks near e1 (cluster A's bridges).
    Rows 12-13: library filler on e3.
    """
    e = np.eye(4)
    jit = [0.00, 0.01, -0.01]
    a_rows = [e[0] + j * e[3] for j in jit] + [e[1] + j * e[3] for j in jit]
    lib_rows = [e[0] + j * e[3] for j in (0.02, -0.02, 0.03, -0.03, 0.04, -0.04)]
    fill = [e[2], e[2] + 0.01 * e[3]]
    X = _unit_rows(a_rows + lib_rows + fill)
    artist_keys = np.array(["a"] * 6 + ["lib"] * 8)
    track_ids = np.array([f"t{i}" for i in range(14)])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def _cfg_bridge(**kw):
    base = dict(cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
                dedupe_versions=False, pier_bridgeability_k=3,
                pier_bridgeability_floor_t=0.30)
    base.update(kw)
    return ArtistStyleConfig(**base)


def test_outlier_cluster_contributes_no_piers_and_slots_reallocate():
    bundle = _bridgeability_fixture()
    clusters, medoids, by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0,
        medoid_top_k=2, target_pier_count=4,
    )
    assert len(clusters) == 2                      # cluster membership untouched
    assert sorted(len(m) for m in by_cluster) == [0, 3]   # B empty; A bumped to all 3
    assert all(m in {0, 1, 2} for m in medoids)    # every pier from cluster A
    assert len(medoids) == 3                       # ceil(4/1)=4 capped by cluster size


def test_bridgeability_all_fail_falls_back_unchecked():
    # Shrink the library to filler only: no artist track has 3 neighbors anywhere.
    e = np.eye(4)
    X = _unit_rows([e[0], e[0] + 0.01 * e[3], e[1], e[1] + 0.01 * e[3], e[2], e[3]])
    bundle = DummyBundle(X_sonic=X, artist_keys=np.array(["a"] * 4 + ["lib"] * 2),
                         track_ids=np.array([f"t{i}" for i in range(6)]))
    checked = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0, medoid_top_k=1)
    unchecked = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0, medoid_top_k=1)
    assert checked[1] == unchecked[1]              # same medoids: never-fail fallback


def test_bridgeability_no_veto_is_byte_identical():
    bundle = _bridgeability_fixture()
    # Floor 0.0 => nothing vetoed => identical to disabled.
    on = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_floor_t=0.0), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    off = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    assert on[1] == off[1] and on[2] == off[2]


def test_artist_style_config_has_bridgeability_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.pier_bridgeability_enabled is True   # live default (activate-fixes rule)
    assert cfg.pier_bridgeability_floor_t == 0.30
    assert cfg.pier_bridgeability_k == 10


def test_bridgeability_config_keys_parse():
    """Guards the exact key names playlist_generator.py parses (both sites)."""
    raw = {"pier_bridgeability_enabled": False,
           "pier_bridgeability_floor_t": 0.42,
           "pier_bridgeability_k": 7}
    cfg = ArtistStyleConfig(
        pier_bridgeability_enabled=bool(raw.get("pier_bridgeability_enabled", True)),
        pier_bridgeability_floor_t=float(raw.get("pier_bridgeability_floor_t", 0.30)),
        pier_bridgeability_k=int(raw.get("pier_bridgeability_k", 10)),
    )
    assert (cfg.pier_bridgeability_enabled, cfg.pier_bridgeability_floor_t,
            cfg.pier_bridgeability_k) == (False, 0.42, 7)


# ---------------------------------------------------------------------------
# Task 5: genre-relevant neighbor set for pier bridgeability
# ---------------------------------------------------------------------------

def test_eligible_mask_excludes_ineligible_high_cosine_neighbor():
    # member 0 = e0. Two perfect e0 neighbors exist (rows 1,2) but are marked
    # INELIGIBLE by the mask; the only ELIGIBLE neighbors are far (e2). So the
    # masked kth-T must collapse even though ungated it would be ~1.0.
    from src.playlist.artist_style import compute_pier_bridgeability
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)
    e = np.eye(3)
    X = _unit_rows([e[0], e[0], e[0], e[2], e[2]])  # rows 1,2 = perfect nbrs of 0; 3,4 = far
    mask = np.array([True, False, False, True, True])  # rows 1,2 ineligible
    ungated = compute_pier_bridgeability(X, [0], [0], k=1,
                                         calib_center=c, calib_scale=s, calib_gain=g)
    gated = compute_pier_bridgeability(X, [0], [0], k=1, eligible_mask=mask,
                                       calib_center=c, calib_scale=s, calib_gain=g)
    assert ungated[0] > 0.9      # perfect neighbor rescues it when ungated
    assert gated[0] < 0.05       # only eligible neighbors are e2 (cos 0) -> fails


def test_eligible_mask_none_is_identical_to_no_mask():
    from src.playlist.artist_style import compute_pier_bridgeability
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)
    e = np.eye(3)
    X = _unit_rows([e[0], e[0], e[1], e[2]])
    a = compute_pier_bridgeability(X, [0], [0], k=1,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    b = compute_pier_bridgeability(X, [0], [0], k=1, eligible_mask=None,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    assert a[0] == b[0]


def test_seed_genre_relevance_mask_basic():
    # 4 genres. Artist tracks (rows 0,1) genre [1,0,0,0]/[1,1,0,0] -> profile ~ genre0/1.
    # Library rows: 2 = [1,0,0,0] compatible; 3 = [0,0,1,0] incompatible; 4 = zero-genre.
    Xg = np.array([[1.,0,0,0],[1.,1,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,0]])
    mask = seed_genre_relevance_mask(Xg, [0, 1], genre_floor=0.30)
    assert mask is not None
    assert bool(mask[2]) is True      # compatible
    assert bool(mask[3]) is False     # orthogonal genre
    assert bool(mask[4]) is False     # zero-genre row is never an eligible neighbor


def test_seed_genre_relevance_mask_none_paths():
    assert seed_genre_relevance_mask(None, [0, 1], 0.3) is None          # no genre data
    Xg = np.array([[0.,0.],[0.,0.]])
    assert seed_genre_relevance_mask(Xg, [0, 1], 0.3) is None            # empty seed profile
    assert seed_genre_relevance_mask(np.eye(3), [], 0.3) is None         # no artist rows


def test_seed_genre_relevance_mask_zero_genre_ineligible_at_zero_floor():
    # row 2 has NO genre data; even at genre_floor=0.0 it must NOT be an eligible neighbor.
    Xg = np.array([[1., 0., 0.], [1., 1., 0.], [0., 0., 0.]])
    mask = seed_genre_relevance_mask(Xg, [0, 1], genre_floor=0.0)
    assert mask is not None
    assert bool(mask[2]) is False
    assert bool(mask[0]) is True   # a real on-genre row is still eligible at floor 0.0


def _genre_bridge_fixture():
    """Artist 'a': cluster A (~e0) has genre-COMPATIBLE library neighbors; the outlier
    cluster B (~e1) is sonically close ONLY to genre-INCOMPATIBLE near-silent junk
    (mirrors First Jam's MuQ collapse)."""
    e = np.eye(4)
    jit = [0.0, 0.01, -0.01]
    a_rows = [e[0] + j*e[3] for j in jit] + [e[1] + j*e[3] for j in jit]  # A~e0, B~e1
    lib_compat = [e[0] + j*e[3] for j in (0.02, -0.02, 0.03, -0.03)]      # near A, on-genre
    lib_junk = [e[1] + j*e[3] for j in (0.02, -0.02, 0.03, -0.03, 0.04)]  # near B, OFF-genre
    X = _unit_rows(a_rows + lib_compat + lib_junk)
    artist_keys = np.array(["a"]*6 + ["lib"]*9)
    track_ids = np.array([f"t{i}" for i in range(15)])
    b = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    # Genre: shoegaze=col0. Artist tracks + compatible lib are shoegaze; junk lib is col2.
    G = np.zeros((15, 4))
    G[list(range(6)) + [6, 7, 8, 9], 0] = 1.0   # artist + compatible-lib on shoegaze
    G[[10, 11, 12, 13, 14], 2] = 1.0            # junk-lib on an unrelated genre
    b.X_genre_smoothed = G
    return b


def test_genre_gate_vetoes_collapse_outlier_that_library_wide_missed():
    bundle = _genre_bridge_fixture()
    # genre_floor gates the junk. Cluster B (~e1) is sonically ~1.0 with junk rows
    # (10-14) but those are genre-incompatible -> masked -> B has no eligible
    # neighbors -> vetoed; cluster A keeps its on-genre neighbors -> survives.
    clusters, medoids, by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_genre_floor=0.30), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    assert len(clusters) == 2
    assert all(m in {0, 1, 2} for m in medoids)     # every pier from cluster A (~e0)
    # Control: WITHOUT the genre gate (floor 0.0 => all eligible), B's junk neighbors
    # rescue it -> both clusters contribute (proves the GATE, not the geometry, vetoed B).
    _, _, by_ungated, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_genre_floor=0.0), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    assert sum(1 for m in by_ungated if len(m) > 0) == 2


def test_artist_style_config_has_genre_floor_default():
    cfg = ArtistStyleConfig()
    assert cfg.pier_bridgeability_genre_floor == 0.30


# ---------------------------------------------------------------------------
# Corridor Phase 2 Task 3: support-aware pier demotion + arc-aware ordering
# ---------------------------------------------------------------------------

def test_artist_style_config_has_support_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.pier_support_enabled is True          # live default (activate-fixes rule)
    assert cfg.pier_support_k == 10
    assert cfg.pier_support_demotion_strength == 1.0
    assert cfg.pier_support_floor == 0.50
    assert cfg.pier_support_terminal_avoidance is True


def test_support_penalty_zero_when_typical_or_above():
    assert _support_penalty(1.0, strength=1.0) == 0.0
    assert _support_penalty(1.5, strength=1.0) == 0.0   # above-typical never promoted


def test_support_penalty_grows_linearly_below_typical():
    assert _support_penalty(0.7, strength=1.0) == pytest.approx(0.3)
    assert _support_penalty(0.5, strength=1.0) == pytest.approx(0.5)
    # scaled by strength
    assert _support_penalty(0.5, strength=0.5) == pytest.approx(0.25)


def test_support_penalty_inert_when_strength_zero():
    assert _support_penalty(0.1, strength=0.0) == 0.0


def test_compute_within_artist_support_uniform_cluster_near_one():
    # 4 tightly-clustered same-artist tracks: everyone's kNN density is close to
    # everyone else's -> relative support ~= 1.0 for all (typical for itself).
    e = np.eye(3)
    rows = [e[0], e[0] * 0.98 + e[1] * 0.02, e[0] * 0.97 - e[1] * 0.02, e[0] * 0.99 + e[1] * 0.01]
    Xn = _unit_rows(rows)
    support = compute_within_artist_support(Xn, [0, 1, 2, 3], k=2)
    for v in support.values():
        assert v == pytest.approx(1.0, abs=0.05)


def test_compute_within_artist_support_flags_off_character_track():
    # 4 same-artist tracks tightly clustered near e0, 1 off-character track (idx 4)
    # near e1 -- sonically far from the REST of this artist's own catalog (the
    # off-character-EP-track shape from the probe: Parquet Courts "Into the
    # garden", Swirlies "His Life of Academic Freedom").
    e = np.eye(3)
    rows = [e[0], e[0] * 0.95 + e[1] * 0.1, e[0] * 0.9 - e[1] * 0.1, e[0] * 0.92 + e[1] * 0.05, e[1]]
    Xn = _unit_rows(rows)
    support = compute_within_artist_support(Xn, [0, 1, 2, 3, 4], k=2)
    assert support[4] < 0.5   # far sparser than its artist-typical (~1.0) peers
    assert all(support[i] > 0.9 for i in (0, 1, 2, 3))


def test_compute_within_artist_support_single_track_inert():
    Xn = _unit_rows([[1.0, 0.0]])
    assert compute_within_artist_support(Xn, [0], k=10) == {0: 1.0}


def test_medoids_for_cluster_support_demotes_sparse_candidate():
    # Candidate 1 scores HIGHER on raw sim-to-centroid than candidate 0, but has
    # low support -- with support demotion on, candidate 0 should win instead.
    X = _unit_rows([[1.0, 0.02], [1.0, 0.0]])   # both near-identical sim to centroid
    centroid = X[1]  # favors candidate 1 slightly (exact match)
    rng = np.random.default_rng(0)
    without = _medoids_for_cluster(
        X, [0, 1], centroid, np.array(["t0", "t1"]), per_cluster=1, rng=rng, top_k=1,
        support_weight=0.0, support_values=np.array([1.0, 0.1]),
    )
    rng2 = np.random.default_rng(0)
    with_demotion = _medoids_for_cluster(
        X, [0, 1], centroid, np.array(["t0", "t1"]), per_cluster=1, rng=rng2, top_k=1,
        support_weight=5.0, support_values=np.array([1.0, 0.1]),
    )
    assert without == [1]            # without demotion: raw-similarity winner (candidate 1)
    assert with_demotion == [0]      # candidate 1's low support (0.1) demotes it below 0


def test_medoids_for_cluster_support_weight_zero_is_byte_identical():
    X = _unit_rows([[1.0, 0.02], [1.0, 0.0]])
    centroid = X[1]
    rng_a = np.random.default_rng(0)
    a = _medoids_for_cluster(
        X, [0, 1], centroid, np.array(["t0", "t1"]), per_cluster=1, rng=rng_a, top_k=1,
    )
    rng_b = np.random.default_rng(0)
    b = _medoids_for_cluster(
        X, [0, 1], centroid, np.array(["t0", "t1"]), per_cluster=1, rng=rng_b, top_k=1,
        support_weight=0.0, support_values=np.array([0.01, 1.0]),
    )
    assert a == b


def test_medoids_for_cluster_support_misaligned_array_warns_and_skips(caplog):
    X = _unit_rows([[1.0, 0.02], [1.0, 0.0]])
    centroid = X[1]
    rng = np.random.default_rng(0)
    with caplog.at_level("WARNING"):
        result = _medoids_for_cluster(
            X, [0, 1], centroid, np.array(["t0", "t1"]), per_cluster=1, rng=rng, top_k=1,
            support_weight=5.0, support_values=np.array([0.1]),   # wrong length
        )
    assert result == [1]   # falls back to unpenalized ranking, never crashes
    assert any("support_values" in r.message for r in caplog.records)


def _order_fixture():
    """4 piers on a rough path X-Y-Z-W (X_norm rows) -- see the probe/report for
    the derivation. Default greedy walk from X seats W (deliberately the
    lowest-support pier) at the closing (terminal) seat; an alternate start
    (Z) finds a walk that seats W in the interior instead."""
    def unit(v):
        v = np.array(v, dtype=float)
        return v / np.linalg.norm(v)
    X = unit([1, 0, 0, 0])
    Y = unit([0.9, 0.44, 0, 0])
    Z = unit([0, 0.3, 0.85, 0])
    W = unit([0, 0, 0.9, 0.1])
    return np.stack([X, Y, Z, W])


def test_order_clusters_start_param_defaults_to_first_backward_compatible():
    Xn = _order_fixture()
    assert order_clusters([0, 1, 2, 3], Xn) == order_clusters([0, 1, 2, 3], Xn, start=None)
    assert order_clusters([0, 1, 2, 3], Xn)[0] == 0


def test_reorder_avoiding_low_support_terminal_moves_worst_off_end():
    Xn = _order_fixture()
    order = order_clusters([0, 1, 2, 3], Xn)
    assert order[-1] == 3   # today's (pre-Task-3) behavior: W lands terminal
    support = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.3}   # W (idx 3) is the outlier pier
    result, changed = reorder_avoiding_low_support_terminal(order, support, Xn)
    assert changed is True
    assert result[0] != 3 and result[-1] != 3
    assert set(result) == set(order)   # same pier set, no piers dropped or duplicated


def test_reorder_avoiding_low_support_terminal_noop_when_already_interior():
    Xn = _order_fixture()
    order = order_clusters([0, 1, 2, 3], Xn)   # [0, 1, 2, 3]
    support = {0: 1.0, 1: 0.2, 2: 1.0, 3: 1.0}   # worst (idx 1) already interior
    result, changed = reorder_avoiding_low_support_terminal(order, support, Xn)
    assert changed is False
    assert result == order


def test_reorder_avoiding_low_support_terminal_too_few_piers_noop():
    Xn = _order_fixture()[:2]
    support = {0: 1.0, 1: 0.1}
    result, changed = reorder_avoiding_low_support_terminal([0, 1], support, Xn)
    assert changed is False
    assert result == [0, 1]


def test_reorder_avoiding_low_support_terminal_no_freedom_keeps_original():
    # 3-node case: C isolated from both A and B -> no Hamiltonian re-walk over
    # this SAME greedy algorithm can avoid seating C terminal. Never fails --
    # falls back to the original order.
    def unit(v):
        v = np.array(v, dtype=float)
        return v / np.linalg.norm(v)
    A = unit([1, 0, 0])
    B = unit([0.9, 0.44, 0])
    C = unit([0, 0, 1])
    Xn = np.stack([A, B, C])
    order = order_clusters([0, 1, 2], Xn)
    support = {0: 1.0, 1: 1.0, 2: 0.2}
    result, changed = reorder_avoiding_low_support_terminal(order, support, Xn)
    assert changed is False
    assert result == order


def test_reorder_avoiding_low_support_terminal_tie_break_deterministic():
    Xn = _order_fixture()
    order = order_clusters([0, 1, 2, 3], Xn)
    # 0 and 3 tie at the lowest support; tie breaks to the EARLIER position (0).
    support = {0: 0.2, 1: 1.0, 2: 1.0, 3: 0.2}
    result, changed = reorder_avoiding_low_support_terminal(order, support, Xn)
    assert changed is True
    assert result[0] != 0 and result[-1] != 0


def _member_indices_fixture():
    """12-track bundle: indices 0-5 span three artists (a/b/c) forming three
    well-separated 2-point sonic clusters; indices 6-11 belong to 'Test Artist'
    and mirror the same three clusters, for the default-path test."""
    artist_keys = np.array(
        ["a", "a", "b", "b", "c", "c"]
        + ["Test Artist"] * 6
    )
    track_ids = np.array([str(i) for i in range(12)])
    cluster_pattern = [
        [1.0, 0.0], [0.95, 0.05],
        [0.0, 1.0], [0.05, 0.95],
        [-1.0, 0.0], [-0.95, -0.05],
    ]
    X = np.array(cluster_pattern + cluster_pattern)
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def test_member_indices_overrides_artist_scoping(monkeypatch):
    """member_indices replaces artist-derived scoping; artist_name is only a label."""
    import src.playlist.artist_style as art

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("artist scoping must not run when member_indices is given")

    monkeypatch.setattr(art, "_artist_indices_in_bundle", _boom)

    bundle = _member_indices_fixture()
    cfg = art.ArtistStyleConfig(pier_bridgeability_enabled=False, dedupe_versions=False)
    clusters, medoids, _, _, _ = art.cluster_artist_tracks(
        bundle=bundle, artist_name="ignored", cfg=cfg,
        member_indices=[0, 1, 2, 3, 4, 5], target_pier_count=2,
    )
    assert called["n"] == 0
    assert set(i for c in clusters for i in c) <= {0, 1, 2, 3, 4, 5}
    assert medoids


def test_member_indices_none_is_unchanged(monkeypatch):
    """Default path still derives indices from the artist name."""
    import src.playlist.artist_style as art
    seen = {}
    real = art._artist_indices_in_bundle

    def _spy(bundle, name, **k):
        seen["name"] = name
        return real(bundle, name, **k)

    monkeypatch.setattr(art, "_artist_indices_in_bundle", _spy)

    bundle = _member_indices_fixture()
    cfg = art.ArtistStyleConfig(pier_bridgeability_enabled=False, dedupe_versions=False)
    art.cluster_artist_tracks(bundle=bundle, artist_name="Test Artist", cfg=cfg)
    assert seen["name"] == "Test Artist"


def test_member_indices_bridgeability_excludes_member_set_not_artist_name(monkeypatch):
    """The pier-bridgeability veto must exclude the caller-supplied member set,
    not re-derive from artist_name. In genre mode artist_name is a label like
    "[genre:shoegaze]" that matches no real artist key -- re-deriving from it
    silently returns [], which lets a track look bridgeable purely via its own
    (unexcluded) album-mates."""
    import src.playlist.artist_style as art

    real_compute = art.compute_pier_bridgeability
    captured = {}

    def _spy(X_norm, member_indices_arg, excluded_indices, k, **kwargs):
        captured["excluded"] = sorted(int(i) for i in excluded_indices)
        return real_compute(X_norm, member_indices_arg, excluded_indices, k, **kwargs)

    monkeypatch.setattr(art, "compute_pier_bridgeability", _spy)

    bundle = _member_indices_fixture()
    cfg = art.ArtistStyleConfig(pier_bridgeability_enabled=True, dedupe_versions=False)
    art.cluster_artist_tracks(
        bundle=bundle, artist_name="[genre:test]", cfg=cfg,
        member_indices=[0, 1, 2, 3, 4, 5], target_pier_count=2,
    )
    assert "excluded" in captured, "compute_pier_bridgeability was never called"
    assert captured["excluded"] == [0, 1, 2, 3, 4, 5]


def test_overlap_affinity_reranks_medoids_independently_of_tag_weight():
    """The overlap term must act even when medoid_tag_weight is 0 (tag steering
    off). Regression guard for the 'configured knob that can't act' failure mode.
    """
    import numpy as np
    from src.playlist.artist_style import _medoids_for_cluster

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    indices = [0, 1]
    centroid = np.array([1.0, 0.0])   # index 0 is the plain medoid
    rng = np.random.default_rng(0)

    plain = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1"], 1, rng, 1,
    )
    assert plain == [0]

    # A strong overlap pull toward index 1 must flip the choice.
    rng2 = np.random.default_rng(0)
    pulled = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1"], 1, rng2, 1,
        overlap_weight=5.0,
        overlap_values=np.array([0.0, 1.0]),
    )
    assert pulled == [1], "overlap term did not affect medoid ranking"


def test_overlap_weight_zero_is_inert():
    import numpy as np
    from src.playlist.artist_style import _medoids_for_cluster

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    centroid = np.array([1.0, 0.0])
    a = _medoids_for_cluster(
        X, [0, 1], centroid, ["t0", "t1"], 1, np.random.default_rng(0), 1,
    )
    b = _medoids_for_cluster(
        X, [0, 1], centroid, ["t0", "t1"], 1, np.random.default_rng(0), 1,
        overlap_weight=0.0, overlap_values=np.array([0.0, 1.0]),
    )
    assert a == b


# ── Hard pier duration gate (2026-07-30 fix) ──────────────────────────────
# Piers are STRUCTURAL: the whole bridge segment is built around one. A
# sub-minimum fragment (interlude/outtake/intro/skit) seated as a pier
# infects the entire segment, which is why this matters more for piers than
# for bridge tracks (the bridge/candidate-pool side already enforces the same
# floor via _build_duration_exclusions_for_ds). Distinct from the existing
# medoid_duration_weight soft outlier penalty, which only demotes -- a strong
# enough sonic-centrality score can still out-rank the penalty (this is
# exactly what let "Vocal Outtakes" (0:36) and "Gnot Up! V2" (0:41) seat as
# real piers). min_pier_duration_seconds is a hard exclusion applied before
# clustering even starts.

def test_min_pier_duration_seconds_excludes_sub_minimum_track():
    """A 30s fragment sitting dead-center of a sonic cluster -- the case that
    would win purely on sonic centrality -- must never be selected as a pier,
    or even survive into the candidate pool, once the hard duration floor is
    given."""
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],  # cluster 1 -- index 0 dead-center
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],  # cluster 2
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.durations_ms = np.array(
        [30_000, 200_000, 200_000, 200_000, 200_000, 200_000], dtype=float
    )
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
        pier_bridgeability_enabled=False, dedupe_versions=False,
    )
    clusters, medoids, medoids_by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0,
        min_pier_duration_seconds=46,
    )
    assert 0 not in medoids, "sub-minimum (30s < 46s) track seated as a pier"
    all_members = {i for c in clusters for i in c}
    assert 0 not in all_members, "sub-minimum track survived into the candidate pool at all"


def test_min_pier_duration_seconds_default_is_the_safe_floor_not_none():
    """Coordinator review 2026-07-30: the parameter's default must be a real,
    safe floor (46s -- the same playlists.min_track_duration_seconds fallback
    every other reader of that key in this codebase uses), not None. Every
    current caller passes the live config value explicitly, but a FUTURE call
    site that forgets this parameter entirely must still be gated -- that is
    exactly how the pre-existing history-mode call site went unguarded after
    the initial fix. Omitting the kwarg here must therefore still exclude the
    sub-minimum track, not pass it through."""
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.durations_ms = np.array(
        [30_000, 200_000, 200_000, 200_000, 200_000, 200_000], dtype=float
    )
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
        pier_bridgeability_enabled=False, dedupe_versions=False,
    )
    clusters, medoids, _by_cluster, _X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0,
        # min_pier_duration_seconds omitted entirely -- relying on the default.
    )
    all_members = {i for c in clusters for i in c}
    assert 0 not in all_members, (
        "omitting the parameter must still gate at the safe default (46s); "
        "a future call site that forgets it must not silently reopen this bug"
    )


def test_min_pier_duration_seconds_explicit_none_disables_the_gate():
    """The safe default does not remove the ability to opt out: passing None
    explicitly must still mean 'no duration gate', for a caller with a
    genuine, deliberate reason to skip it (none exists in this codebase
    today, but the capability stays available rather than being deleted)."""
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.durations_ms = np.array(
        [30_000, 200_000, 200_000, 200_000, 200_000, 200_000], dtype=float
    )
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
        pier_bridgeability_enabled=False, dedupe_versions=False,
    )
    clusters, medoids, _by_cluster, _X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0,
        min_pier_duration_seconds=None,
    )
    all_members = {i for c in clusters for i in c}
    assert all_members == {0, 1, 2, 3, 4, 5}, "explicit None must still disable the gate"


def test_min_pier_duration_seconds_unknown_duration_is_kept():
    """Zero/missing duration means 'unknown', not 'invalid' -- mirrors
    genre_mode.filter_member_indices_by_duration. Incomplete metadata must
    never silently shrink the pier candidate pool."""
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.durations_ms = np.array(
        [0.0, 200_000, 200_000, 200_000, 200_000, 200_000], dtype=float
    )
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
        pier_bridgeability_enabled=False, dedupe_versions=False,
    )
    clusters, _medoids, _by_cluster, _X_norm, _support = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0,
        min_pier_duration_seconds=46,
    )
    all_members = {i for c in clusters for i in c}
    assert 0 in all_members, "zero/unknown duration was treated as sub-minimum and dropped"


def test_min_pier_duration_seconds_starvation_warns_and_raises(caplog):
    """Every track in a 2-track artist catalog is sub-minimum -> zero eligible
    candidates. Must never silently relax the gate and never crash with no
    context: log a WARNING naming the artist and the counts, then raise the
    existing too-few-tracks ValueError (the caller's thin-artist handling
    decides what happens next -- single-artist: propagates and the outer
    fallback-to-legacy path takes over; multi-artist: the per-group
    except-ValueError catch)."""
    import logging
    artist_keys = np.array(["a", "a"])
    track_ids = np.array(["0", "1"])
    X = np.array([[1.0, 0.0], [0.9, 0.1]])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.durations_ms = np.array([30_000, 35_000], dtype=float)
    cfg = ArtistStyleConfig(
        cluster_k_min=3, enabled=True,
        pier_bridgeability_enabled=False, dedupe_versions=False,
    )
    with caplog.at_level(logging.WARNING, logger="src.playlist.artist_style"):
        with pytest.raises(ValueError, match="Not enough tracks to cluster for artist A"):
            cluster_artist_tracks(
                bundle=bundle, artist_name="A", cfg=cfg, random_seed=0,
                min_pier_duration_seconds=46,
            )
    starvation_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Pier duration gate" in r.message and "A" in r.message
    ]
    assert starvation_records, (
        f"expected a WARNING naming the artist and counts; got: "
        f"{[r.message for r in caplog.records]}"
    )
    assert "0 eligible" in starvation_records[0].message
    assert "2 sub-46s" in starvation_records[0].message
