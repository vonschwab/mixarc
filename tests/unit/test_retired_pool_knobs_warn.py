"""
Phase 0 Task 3: loud startup warnings for retired pool knobs.

"A configured knob that can't act is a startup error, not a silent no-op."
(CLAUDE.md, "Project-specific gotchas"). Task 1 deleted the per-artist cap
walk + never-starve backstop in src/playlist/candidate_pool.py; Task 2
deleted the min_pool_size preset/resolution chains that fed both the
CandidatePoolConfig.min_pool_size field and the reviewer-confirmed-dead
PierBridgeTuning.min_pool_size field. This test locks in that
_warn_retired_keys finds every yaml key that used to reach one of those
now-deleted code paths, in both locations they can appear:
  - playlists.ds_pipeline.candidate_pool.* (candidates_per_artist,
    target_artists, max_pool_size, min_pool_size, seed_artist_bonus)
  - playlists.ds_pipeline.pier_bridge.min_pool_size and its cohesion_mode-
    suffixed variants (min_pool_size_strict/_narrow/_dynamic/_discover)

Phase 1 Task 8 (the flip) added four more retired families, all locked in
by the tests below: playlists.ds_pipeline.pier_bridge.{pooling,
collapse_segment_pool_by_artist} (top-level), .infeasible_handling.* (the
legacy per-segment relaxation ladder -- guarantee_feasible/greedy_genre_weight
stay live and must NOT warn), .dj_bridging.pooling.{k_local,k_toward,k_genre,
k_union_max,step_stride,debug_compare_baseline} (the dj_union segment-pool
strategy's implementation -- strategy/cache_enabled stay live), and
playlists.ds_pipeline.artist_style.{per_cluster_candidate_pool_size,
pool_balance_mode} (build_balanced_candidate_pool's config).
"""

from src.playlist_gui.worker import _warn_retired_keys, _WARNED_RETIRED_KEYS


def test_dedup_warnings_per_process(caplog):
    """Verify that retired-key warnings are deduped per process: same key
    only logs once even if _warn_retired_keys is called multiple times.
    Both calls must still return the found key."""
    _WARNED_RETIRED_KEYS.clear()
    cfg = {"playlists": {"ds_pipeline": {"candidate_pool": {
        "min_pool_size": 0,
    }}}}

    with caplog.at_level("WARNING"):
        found1 = _warn_retired_keys(cfg)
        found2 = _warn_retired_keys(cfg)

    # Both calls must return the key
    assert "candidate_pool.min_pool_size" in found1
    assert "candidate_pool.min_pool_size" in found2

    # Warning text must appear exactly once across both calls
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    min_pool_size_warnings = [r for r in warning_records
                               if "candidate_pool.min_pool_size" in r.message]
    assert len(min_pool_size_warnings) == 1


def test_retired_candidate_pool_knobs_warn():
    cfg = {"playlists": {"ds_pipeline": {"candidate_pool": {
        "candidates_per_artist": 6,
        "target_artists": 22,
        "max_pool_size": 2400,
        "min_pool_size": 0,
        "seed_artist_bonus": 2,
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "candidate_pool.candidates_per_artist",
        "candidate_pool.target_artists",
        "candidate_pool.max_pool_size",
        "candidate_pool.min_pool_size",
        "candidate_pool.seed_artist_bonus",
    }


def test_retired_pier_bridge_min_pool_size_family_warns():
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {
        "min_pool_size": 20,
        "min_pool_size_strict": 12,
        "min_pool_size_narrow": 16,
        "min_pool_size_dynamic": 20,
        "min_pool_size_discover": 24,
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "pier_bridge.min_pool_size",
        "pier_bridge.min_pool_size_strict",
        "pier_bridge.min_pool_size_narrow",
        "pier_bridge.min_pool_size_dynamic",
        "pier_bridge.min_pool_size_discover",
    }


def test_live_pool_gate_keys_do_not_warn():
    """Keys that still act (e.g. the still-live segment_pool_max / bridge_floor
    family) must never be flagged -- a false positive here is exactly the
    noise the task brief warns against."""
    cfg = {"playlists": {"ds_pipeline": {
        "candidate_pool": {
            "similarity_floor": 0.2,
            "max_artist_fraction": 0.125,
        },
        "pier_bridge": {
            "bridge_floor_dynamic": 0.02,
            "bridge_floor_narrow": 0.05,
            "bridge_floor_strict": 0.1,
            "corridor_width_percentile": 0.85,
            "infeasible_handling": {
                "guarantee_feasible": True,
                "greedy_genre_weight": 0.5,
            },
            "dj_bridging": {
                "pooling": {
                    "strategy": "baseline",
                    "cache_enabled": True,
                },
            },
        },
        "artist_style": {
            "piers_per_cluster": 1,
            "cluster_k_min": 3,
        },
    }}}
    assert _warn_retired_keys(cfg) == []


def test_retired_pier_bridge_top_level_keys_warn():
    """Phase 1 Task 8 (the flip): the pooling dev flag + the legacy scored-pool
    collapse_segment_pool_by_artist knob."""
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {
        "pooling": "corridor",
        "collapse_segment_pool_by_artist": False,
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "pier_bridge.pooling",
        "pier_bridge.collapse_segment_pool_by_artist",
    }


def test_retired_infeasible_handling_family_warns():
    """Phase 1 Task 8: the legacy per-segment relaxation ladder -- corridor's
    own widening ladder is the sole segment-level recovery mechanism now."""
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {"infeasible_handling": {
        "enabled": True,
        "strategy": "backoff",
        "min_bridge_floor": 0.0,
        "backoff_steps": [0.08, 0.06],
        "max_attempts_per_segment": 8,
        "widen_search_on_backoff": True,
        "extra_neighbors_m": 200,
        "extra_bridge_helpers": 100,
        "extra_beam_width": 50,
        "extra_expansion_attempts": 2,
        "transition_floor_relaxation_enabled": True,
        "min_transition_floor": 0.0,
        "genre_arc_relaxation_enabled": True,
        "min_genre_arc_percentile": 0.0,
        # NOT retired -- must not appear in `found` below.
        "guarantee_feasible": True,
        "greedy_genre_weight": 0.5,
    }}}}}
    found = _warn_retired_keys(cfg)
    expected = {
        f"pier_bridge.infeasible_handling.{k}" for k in (
            "enabled", "strategy", "min_bridge_floor", "backoff_steps",
            "max_attempts_per_segment", "widen_search_on_backoff",
            "extra_neighbors_m", "extra_bridge_helpers", "extra_beam_width",
            "extra_expansion_attempts", "transition_floor_relaxation_enabled",
            "min_transition_floor", "genre_arc_relaxation_enabled",
            "min_genre_arc_percentile",
        )
    }
    assert set(found) == expected
    assert "pier_bridge.infeasible_handling.guarantee_feasible" not in found
    assert "pier_bridge.infeasible_handling.greedy_genre_weight" not in found


def test_retired_dj_pooling_keys_warn():
    """Phase 1 Task 8: the dj_union segment-pool strategy's implementation
    died with the legacy segment-scored pool builder."""
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {"dj_bridging": {"pooling": {
        "strategy": "dj_union",  # NOT itself retired -- value, not key
        "k_local": 200,
        "k_toward": 80,
        "k_genre": 40,
        "k_union_max": 900,
        "step_stride": 1,
        "debug_compare_baseline": True,
        # NOT retired -- must not appear in `found` below.
        "cache_enabled": True,
    }}}}}}
    found = _warn_retired_keys(cfg)
    expected = {
        f"pier_bridge.dj_bridging.pooling.{k}" for k in (
            "k_local", "k_toward", "k_genre", "k_union_max",
            "step_stride", "debug_compare_baseline",
        )
    } | {"pier_bridge.dj_bridging.pooling.strategy=dj_union"}
    assert set(found) == expected
    assert "pier_bridge.dj_bridging.pooling.strategy" not in found
    assert "pier_bridge.dj_bridging.pooling.cache_enabled" not in found


def test_retired_dj_union_strategy_value_warns():
    """Value-conditional: strategy: dj_union is now a silent no-op (its
    implementation died with the legacy segment-scored pool builder) even
    though `strategy` itself is not a retired KEY (baseline is fine)."""
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {"dj_bridging": {"pooling": {
        "strategy": "dj_union",
    }}}}}}
    found = _warn_retired_keys(cfg)
    assert "pier_bridge.dj_bridging.pooling.strategy=dj_union" in found

    cfg_baseline = {"playlists": {"ds_pipeline": {"pier_bridge": {"dj_bridging": {"pooling": {
        "strategy": "baseline",
    }}}}}}
    assert _warn_retired_keys(cfg_baseline) == []


def test_retired_artist_style_keys_warn():
    """Phase 1 Task 8: build_balanced_candidate_pool deleted -- corridor's
    eligible universe replaces the artist-mode external pool."""
    cfg = {"playlists": {"ds_pipeline": {"artist_style": {
        "per_cluster_candidate_pool_size": 400,
        "pool_balance_mode": "equal",
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "artist_style.per_cluster_candidate_pool_size",
        "artist_style.pool_balance_mode",
    }


def test_retired_multi_artist_alternation_bonus_warns():
    """Forced-interleaving rewrite (2026-07-30): order_with_alternation now
    forces the theoretical max alternation instead of scoring a preference,
    so multi_artist.alternation_bonus has no weight left to read -- a
    leftover value in config.yaml must warn loudly, not silently do nothing."""
    cfg = {"playlists": {"ds_pipeline": {"multi_artist": {
        "alternation_bonus": 0.15,
        # NOT retired -- must not appear in `found` below.
        "overlap_weight": 0.6,
        "enabled": True,
    }}}}
    found = _warn_retired_keys(cfg)
    assert "multi_artist.alternation_bonus" in found
    assert "multi_artist.overlap_weight" not in found
    assert "multi_artist.enabled" not in found


def test_clean_config_warns_nothing():
    assert _warn_retired_keys({"playlists": {}}) == []


def test_empty_config_warns_nothing():
    assert _warn_retired_keys({}) == []
