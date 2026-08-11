"""
Unit tests for the GUI Policy Layer.

Tests the UIStateModel and PolicyLayer derive_runtime_config function
to ensure policy rules are correctly implemented.

Created: Phase 1 of GUI "Just Works" implementation
"""
from __future__ import annotations

import pytest

from src.playlist_gui.ui_state import UIStateModel
from src.playlist_gui.policy import (
    SPACING_MAP,
    derive_runtime_config,
    merge_overrides,
    _get_nested,
    _set_nested,
)


def _has_nested(data: dict, path: str) -> bool:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


# ─────────────────────────────────────────────────────────────────────────────
# UIStateModel Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUIStateModel:
    """Tests for UIStateModel dataclass."""

    def test_defaults(self):
        """Test default values are correct."""
        state = UIStateModel()
        assert state.mode == "artist"
        assert state.track_count == 30
        assert state.recency_enabled is True
        assert state.recency_days == 14
        assert state.recency_plays_threshold == 1
        assert state.artist_spacing == "normal"
        assert state.artist_queries == []
        assert state.artist_presence == "medium"
        assert state.artist_variety == "balanced"
        assert state.history_window_days == 30
        assert state.seed_track_ids == []
        assert state.seed_auto_order is True

    def test_primary_artist_empty(self):
        """Test primary_artist returns None when no artists."""
        state = UIStateModel()
        assert state.primary_artist() is None

    def test_primary_artist_single(self):
        """Test primary_artist returns first artist."""
        state = UIStateModel(artist_queries=["Radiohead"])
        assert state.primary_artist() == "Radiohead"

    def test_primary_artist_multiple(self):
        """Test primary_artist returns first of multiple artists."""
        state = UIStateModel(artist_queries=["Radiohead", "Portishead", "Massive Attack"])
        assert state.primary_artist() == "Radiohead"

    def test_seed_count(self):
        """Test seed_count method."""
        state = UIStateModel()
        assert state.seed_count() == 0

        state = UIStateModel(seed_track_ids=["id1", "id2", "id3"])
        assert state.seed_count() == 3


# ─────────────────────────────────────────────────────────────────────────────
# Mode Pass-Through Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModePassThrough:
    """Tests for explicit genre_mode/sonic_mode/pace_mode pass-through."""

    def test_explicit_off_modes_pass_through(self):
        """Test explicit off modes are preserved for CLI parity."""
        state = UIStateModel(genre_mode="off", sonic_mode="off")
        decisions = derive_runtime_config(state)
        assert _get_nested(decisions.overrides, "playlists.genre_mode") == "off"
        assert _get_nested(decisions.overrides, "playlists.sonic_mode") == "off"

    def test_pace_mode_defaults_to_dynamic(self):
        state = UIStateModel()
        decisions = derive_runtime_config(state)

        assert _get_nested(decisions.overrides, "playlists.pace_mode") == "dynamic"

    def test_pace_mode_passes_through(self):
        state = UIStateModel(pace_mode="narrow")
        decisions = derive_runtime_config(state)

        assert _get_nested(decisions.overrides, "playlists.pace_mode") == "narrow"


# ─────────────────────────────────────────────────────────────────────────────
# Recency Override Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecencyOverrides:
    """Tests for recency filter override mapping."""

    def test_recency_defaults(self):
        """Test default recency values (on, 14 days, 1 play)."""
        state = UIStateModel()
        decisions = derive_runtime_config(state)

        overrides = decisions.overrides
        assert _get_nested(overrides, "playlists.recently_played_filter.enabled") is True
        assert _get_nested(overrides, "playlists.recently_played_filter.lookback_days") == 14
        assert _get_nested(overrides, "playlists.recently_played_filter.min_playcount_threshold") == 1

    def test_recency_disabled(self):
        """Test recency filter can be disabled."""
        state = UIStateModel(recency_enabled=False)
        decisions = derive_runtime_config(state)

        overrides = decisions.overrides
        assert _get_nested(overrides, "playlists.recently_played_filter.enabled") is False

    def test_recency_custom_values(self):
        """Test custom recency values are passed through."""
        state = UIStateModel(
            recency_enabled=True,
            recency_days=30,
            recency_plays_threshold=3,
        )
        decisions = derive_runtime_config(state)

        overrides = decisions.overrides
        assert _get_nested(overrides, "playlists.recently_played_filter.enabled") is True
        assert _get_nested(overrides, "playlists.recently_played_filter.lookback_days") == 30
        assert _get_nested(overrides, "playlists.recently_played_filter.min_playcount_threshold") == 3


# ─────────────────────────────────────────────────────────────────────────────
# DJ Bridging Gating Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDJBridgingGating:
    """Tests for DJ bridging enable/disable rules."""

    def test_artist_mode_disables_dj_bridging(self):
        """Artist mode should always disable DJ bridging."""
        state = UIStateModel(mode="artist")
        decisions = derive_runtime_config(state)
        assert decisions.dj_bridging_enabled is False
        assert any("artist" in note.lower() for note in decisions.notes)

    def test_history_mode_disables_dj_bridging(self):
        """History mode should always disable DJ bridging."""
        state = UIStateModel(mode="history")
        decisions = derive_runtime_config(state)
        assert decisions.dj_bridging_enabled is False
        assert any("history" in note.lower() for note in decisions.notes)

    def test_seeds_mode_less_than_2_seeds_disables(self):
        """Seeds mode with < 2 seeds should disable DJ bridging."""
        state = UIStateModel(mode="seeds", seed_track_ids=["id1"])
        decisions = derive_runtime_config(state, seed_artist_keys=["artist1"])
        assert decisions.dj_bridging_enabled is False
        assert any("2 seeds" in note for note in decisions.notes)

    def test_seeds_mode_no_artist_keys_disables(self):
        """Seeds mode without artist keys should conservatively disable."""
        state = UIStateModel(mode="seeds", seed_track_ids=["id1", "id2"])
        decisions = derive_runtime_config(state)  # No seed_artist_keys
        assert decisions.dj_bridging_enabled is False
        assert any("seed_artist_keys not provided" in note for note in decisions.notes)

    def test_seeds_mode_1_unique_artist_disables(self):
        """Seeds mode with < 2 unique artists should disable DJ bridging."""
        state = UIStateModel(mode="seeds", seed_track_ids=["id1", "id2", "id3"])
        # All from same artist
        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["radiohead", "radiohead", "radiohead"],
        )
        assert decisions.dj_bridging_enabled is False
        assert any("unique artists" in note.lower() for note in decisions.notes)

    def test_seeds_mode_2_seeds_2_unique_artists_enables(self):
        """Seeds mode with 2+ seeds from 2+ unique artists enables DJ bridging."""
        state = UIStateModel(mode="seeds", seed_track_ids=["id1", "id2"])
        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["radiohead", "portishead"],
        )
        assert decisions.dj_bridging_enabled is True
        assert any("enabled" in note.lower() and "bridging" in note.lower() for note in decisions.notes)

    def test_seeds_mode_multiple_seeds_multiple_artists_enables(self):
        """Seeds mode with multiple seeds from multiple artists enables DJ bridging."""
        state = UIStateModel(
            mode="seeds",
            seed_track_ids=["id1", "id2", "id3", "id4"],
        )
        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["slowdive", "beach house", "deerhunter", "helvetia"],
        )
        assert decisions.dj_bridging_enabled is True

    def test_artist_key_case_insensitive(self):
        """Artist key comparison should be case-insensitive."""
        state = UIStateModel(mode="seeds", seed_track_ids=["id1", "id2"])
        # Same artist, different cases
        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["Radiohead", "RADIOHEAD"],
        )
        assert decisions.dj_bridging_enabled is False  # Only 1 unique artist


# ─────────────────────────────────────────────────────────────────────────────
# Genre Pool Gating Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenrePoolGating:
    """Tests for genre pool enable/disable rules."""

    def test_genre_pool_requires_dj_bridging_for_effect(self):
        """
        Genre pool is desired at discover, but requires DJ bridging to work.

        When DJ bridging is disabled, genre pool has no effect on pooling strategy.
        """
        # Discover + artist mode (no DJ bridging)
        state = UIStateModel(mode="artist", cohesion_mode="discover", genre_mode="discover", sonic_mode="discover")
        decisions = derive_runtime_config(state)

        assert decisions.genre_pool_enabled is True
        assert decisions.dj_bridging_enabled is False
        # Pooling strategy should be baseline (no dj_union without DJ bridging)
        strategy = _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy"
        )
        assert strategy == "baseline"
        assert any("unavailable" in note.lower() for note in decisions.notes)

    def test_genre_pool_with_dj_bridging_sets_dj_union(self):
        """When both genre pool and DJ bridging are enabled, use dj_union strategy."""
        state = UIStateModel(
            mode="seeds",
            cohesion_mode="discover",
            genre_mode="discover",
            sonic_mode="discover",
            seed_track_ids=["id1", "id2"],
        )
        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["slowdive", "beach house"],
        )

        assert decisions.genre_pool_enabled is True
        assert decisions.dj_bridging_enabled is True

        strategy = _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy"
        )
        k_genre = _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre"
        )
        assert strategy == "dj_union"
        assert k_genre == 80  # Default genre pool size


# ─────────────────────────────────────────────────────────────────────────────
# Seed Ordering Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSeedOrdering:
    """Tests for seed ordering mapping."""

    def test_seed_auto_order_true_maps_to_auto(self):
        """seed_auto_order=True should map to 'auto' (optimize)."""
        state = UIStateModel(seed_auto_order=True)
        decisions = derive_runtime_config(state)
        assert decisions.seed_ordering_value == "auto"
        ordering = _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering"
        )
        assert ordering == "auto"

    def test_seed_auto_order_false_maps_to_fixed(self):
        """seed_auto_order=False should map to 'fixed' (preserve)."""
        state = UIStateModel(seed_auto_order=False)
        decisions = derive_runtime_config(state)
        assert decisions.seed_ordering_value == "fixed"
        ordering = _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering"
        )
        assert ordering == "fixed"


# ─────────────────────────────────────────────────────────────────────────────
# Artist Spacing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestArtistSpacing:
    """Tests for artist spacing → min_gap mapping."""

    def test_spacing_map_completeness(self):
        """Ensure all spacing levels are mapped."""
        expected_levels = {"loose", "normal", "strong", "very_strong"}
        assert set(SPACING_MAP.keys()) == expected_levels

    def test_normal_spacing(self):
        """Normal spacing should map to min_gap=6."""
        state = UIStateModel(artist_spacing="normal")
        decisions = derive_runtime_config(state)
        assert decisions.min_gap == 6
        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.constraints.min_gap"
        ) == 6

    def test_loose_spacing(self):
        """Loose spacing should map to min_gap=3."""
        state = UIStateModel(artist_spacing="loose")
        decisions = derive_runtime_config(state)
        assert decisions.min_gap == 3
        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.constraints.min_gap"
        ) == 3

    def test_strong_spacing(self):
        """Strong spacing should map to min_gap=9."""
        state = UIStateModel(artist_spacing="strong")
        decisions = derive_runtime_config(state)
        assert decisions.min_gap == 9
        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.constraints.min_gap"
        ) == 9

    def test_very_strong_spacing(self):
        """Very strong spacing should map to min_gap=12."""
        state = UIStateModel(artist_spacing="very_strong")
        decisions = derive_runtime_config(state)
        assert decisions.min_gap == 12
        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.constraints.min_gap"
        ) == 12


class TestArtistDiversity:
    """Tests for diversity slider hard artist-cap policy."""

    def test_one_per_artist_diversity_sets_pier_bridge_cap(self):
        state = UIStateModel(artist_diversity_mode="one_per_artist")

        decisions = derive_runtime_config(state)

        assert _get_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
        ) == 1
        assert any("one track per non-seed artist" in note.lower() for note in decisions.notes)

    def test_weighted_diversity_does_not_emit_pier_bridge_cap_override(self):
        state = UIStateModel(artist_diversity_mode="weighted")

        decisions = derive_runtime_config(state)

        assert not _has_nested(
            decisions.overrides,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
        )

    def test_weighted_diversity_preserves_user_pier_bridge_cap_when_merged(self):
        user_overrides = {
            "playlists": {
                "ds_pipeline": {
                    "pier_bridge": {
                        "max_non_seed_tracks_per_artist": 1,
                    }
                }
            }
        }
        state = UIStateModel(artist_diversity_mode="weighted")

        decisions = derive_runtime_config(state)
        merged = merge_overrides(user_overrides, decisions.overrides)

        assert _get_nested(
            merged,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
        ) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Merge Overrides Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeOverrides:
    """Tests for merge_overrides function."""

    def test_policy_wins_for_owned_keys(self):
        """Policy values should override user values for POLICY_OWNED_KEYS."""
        user_overrides = {
            "playlists": {
                "genre_mode": "strict",  # User set this
                "other_setting": "user_value",  # User set this too
            }
        }
        policy_overrides = {
            "playlists": {
                "genre_mode": "discover",  # Policy wants this
            }
        }

        result = merge_overrides(user_overrides, policy_overrides)

        # Policy wins for genre_mode
        assert result["playlists"]["genre_mode"] == "discover"
        # User value preserved for non-policy keys
        assert result["playlists"]["other_setting"] == "user_value"

    def test_user_values_preserved_for_non_policy_keys(self):
        """User overrides should be preserved for non-policy-owned keys."""
        user_overrides = {
            "playlists": {
                "ds_pipeline": {
                    "tower_weights": {
                        "rhythm": 0.30,  # Not a policy-owned key
                    }
                }
            }
        }
        policy_overrides = {
            "playlists": {
                "genre_mode": "dynamic",
            }
        }

        result = merge_overrides(user_overrides, policy_overrides)

        # User value preserved
        assert result["playlists"]["ds_pipeline"]["tower_weights"]["rhythm"] == 0.30
        # Policy value added
        assert result["playlists"]["genre_mode"] == "dynamic"

    def test_deep_merge_nested_dicts(self):
        """Deep merge should properly merge nested dictionaries."""
        user_overrides = {
            "a": {
                "b": {
                    "user_key": "user_value",
                }
            }
        }
        policy_overrides = {
            "a": {
                "b": {
                    "policy_key": "policy_value",
                }
            }
        }

        result = merge_overrides(user_overrides, policy_overrides)

        assert result["a"]["b"]["user_key"] == "user_value"
        assert result["a"]["b"]["policy_key"] == "policy_value"


# ─────────────────────────────────────────────────────────────────────────────
# Helper Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_set_nested(self):
        """Test _set_nested creates proper nested structure."""
        d: dict = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_get_nested(self):
        """Test _get_nested retrieves nested values."""
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested(d, "a.b.c") == 42
        assert _get_nested(d, "a.b") == {"c": 42}
        assert _get_nested(d, "x.y.z") is None
        assert _get_nested(d, "x.y.z", default="default") == "default"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests combining multiple policy rules."""

    def test_full_seeds_mode_discover_flow(self):
        """Test complete flow for seeds mode with discover cohesion_mode."""
        state = UIStateModel(
            mode="seeds",
            cohesion_mode="discover",
            genre_mode="discover",
            sonic_mode="discover",
            track_count=40,
            recency_enabled=True,
            recency_days=7,
            recency_plays_threshold=2,
            artist_spacing="strong",
            seed_track_ids=["id1", "id2", "id3"],
            seed_auto_order=True,
        )

        decisions = derive_runtime_config(
            state,
            seed_artist_keys=["slowdive", "beach house", "deerhunter"],
        )

        # Check all decisions
        assert decisions.dj_bridging_enabled is True
        assert decisions.genre_pool_enabled is True
        assert decisions.seed_ordering_value == "auto"
        assert decisions.min_gap == 9

        # Check overrides
        o = decisions.overrides
        assert _get_nested(o, "playlists.genre_mode") == "discover"
        assert _get_nested(o, "playlists.sonic_mode") == "discover"
        assert _get_nested(o, "playlists.tracks_per_playlist") == 40
        assert _get_nested(o, "playlists.recently_played_filter.enabled") is True
        assert _get_nested(o, "playlists.recently_played_filter.lookback_days") == 7
        assert _get_nested(o, "playlists.recently_played_filter.min_playcount_threshold") == 2
        assert _get_nested(o, "playlists.ds_pipeline.constraints.min_gap") == 9
        assert _get_nested(
            o, "playlists.ds_pipeline.pier_bridge.dj_bridging.enabled"
        ) is True
        assert _get_nested(
            o, "playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering"
        ) == "auto"
        assert _get_nested(
            o, "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy"
        ) == "dj_union"
        assert _get_nested(
            o, "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre"
        ) == 80

    def test_artist_mode_strict_flow(self):
        """Test complete flow for artist mode with strict cohesion_mode."""
        state = UIStateModel(
            mode="artist",
            cohesion_mode="strict",
            genre_mode="strict",
            sonic_mode="strict",
            track_count=20,
            recency_enabled=False,
            artist_spacing="normal",
            artist_queries=["Radiohead"],
        )

        decisions = derive_runtime_config(state)

        # DJ bridging should be disabled for artist mode
        assert decisions.dj_bridging_enabled is False
        # Genre pool disabled when genre_mode is not discover
        assert decisions.genre_pool_enabled is False
        assert decisions.min_gap == 6

        o = decisions.overrides
        assert _get_nested(o, "playlists.genre_mode") == "strict"
        assert _get_nested(o, "playlists.sonic_mode") == "strict"
        assert _get_nested(o, "playlists.tracks_per_playlist") == 20
        assert _get_nested(o, "playlists.recently_played_filter.enabled") is False


# ─────────────────────────────────────────────────────────────────────────────
# Cohesion Mode Derivation Tests (sub-project B)
# ─────────────────────────────────────────────────────────────────────────────


class TestCohesionModeDerivation:
    """Tests that policy.derive_runtime_config writes playlists.cohesion_mode."""

    def test_cohesion_mode_in_policy_owned_keys(self):
        from src.playlist_gui.policy import POLICY_OWNED_KEYS

        assert "playlists.cohesion_mode" in POLICY_OWNED_KEYS

    @pytest.mark.parametrize(
        "mode", ["strict", "narrow", "dynamic", "discover"]
    )
    def test_each_valid_mode_writes_through(self, mode):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel(cohesion_mode=mode)
        decisions = derive_runtime_config(state)
        assert (
            decisions.overrides["playlists"]["cohesion_mode"] == mode
        ), f"Expected {mode}, got {decisions.overrides['playlists'].get('cohesion_mode')}"

    def test_invalid_cohesion_mode_falls_back_to_dynamic(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel()
        # Bypass dataclass type-checking by setting attr directly
        object.__setattr__(state, "cohesion_mode", "garbage")
        decisions = derive_runtime_config(state)
        assert decisions.overrides["playlists"]["cohesion_mode"] == "dynamic"

    def test_cohesion_mode_default_is_dynamic(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel()  # default cohesion_mode = "dynamic"
        decisions = derive_runtime_config(state)
        assert decisions.overrides["playlists"]["cohesion_mode"] == "dynamic"


# ─────────────────────────────────────────────────────────────────────────────
# OOPS Popularity Mode Tests (Task 6)
# ─────────────────────────────────────────────────────────────────────────────


class TestOopsPopularityModeOverride:
    """Tests for OOPS popularity_mode sonic/pace baseline override."""

    def test_oops_overrides_sonic_and_pace_baseline_not_genre(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel
        ui = UIStateModel(
            genre_mode="strict", sonic_mode="strict", pace_mode="strict",
            popularity_mode="oops",
        )
        pol = derive_runtime_config(ui)
        pl = pol.overrides["playlists"]
        assert pl["sonic_mode"] == "dynamic"   # OOPS owns sonic
        assert pl["pace_mode"] == "dynamic"    # OOPS owns pace
        assert pl["genre_mode"] == "strict"    # user still owns genre

    def test_on_and_off_do_not_override_modes(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel
        for m in ("on", "off"):
            ui = UIStateModel(genre_mode="strict", sonic_mode="strict",
                              pace_mode="strict", popularity_mode=m)
            pl = derive_runtime_config(ui).overrides["playlists"]
            assert pl["sonic_mode"] == "strict"
            assert pl["pace_mode"] == "strict"


# ─────────────────────────────────────────────────────────────────────────────
# Live-album exclusion (Task 4)
# ─────────────────────────────────────────────────────────────────────────────


def test_exclude_live_defaults_true_and_reaches_overrides():
    from src.playlist_gui.ui_state import UIStateModel
    from src.playlist_gui.policy import derive_runtime_config
    state = UIStateModel(mode="artist")
    assert state.exclude_live is True
    d = derive_runtime_config(state)
    assert d.overrides["playlists"]["ds_pipeline"]["candidate_pool"]["exclude_live_albums"] is True
    d_off = derive_runtime_config(UIStateModel(mode="artist", exclude_live=False))
    assert d_off.overrides["playlists"]["ds_pipeline"]["candidate_pool"]["exclude_live_albums"] is False
