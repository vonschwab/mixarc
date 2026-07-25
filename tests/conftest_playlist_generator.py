"""Shared synthetic factory helpers for PlaylistGenerator smoke goldens (Tier-3.2)."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.playlist_generator import PlaylistGenerator

# 15 tracks across 3 artists (5 tracks per artist), all with 3-minute duration.
# artist_key uses spaces to match what normalize_artist_key("Artist N") returns.
# is_valid_duration() reads the 'duration' field in milliseconds.
SYNTHETIC_TRACKS: List[Dict[str, Any]] = [
    {
        "rating_key": f"t{i}",
        "title": f"Track {i}",
        "artist": f"Artist {i // 5}",
        "artist_key": f"artist {i // 5}",
        "genres": [f"genre_{i % 2}"],
        "duration": 180_000,  # milliseconds — consumed by filtering.is_valid_duration
        "duration_ms": 180_000,
        "file": f"/music/track_{i}.mp3",
        "album": f"Album {i // 5}",
    }
    for i in range(15)
]


def make_synthetic_generator() -> PlaylistGenerator:
    """
    Create a PlaylistGenerator bypassing __init__ with minimal mocked state.

    Returns a factory-built instance ready for monkeypatching. The caller is
    responsible for monkeypatching _maybe_generate_ds_playlist (and optionally
    _post_order_validate_ds_output / _print_playlist_report) before invoking
    any orchestrator method.
    """
    gen = PlaylistGenerator.__new__(PlaylistGenerator)

    # --- config mock ---
    # config.get() is used with positional keys: config.get('playlists', 'ds_pipeline', default={})
    # Return a real dict so callers can chain .get() on the result.
    cfg_mock = MagicMock()
    cfg_mock.get.return_value = {}
    # config.config is accessed as a real dict in several places:
    #   config.config.get('playlists', {}) and config.config.setdefault(...)
    cfg_mock.config = {}
    # Attribute accesses used by the orchestrators:
    cfg_mock.recently_played_lookback_days = 14
    cfg_mock.recently_played_min_playcount = 0
    gen.config = cfg_mock

    # --- library mock ---
    lib_mock = MagicMock()
    lib_mock.get_all_tracks.return_value = list(SYNTHETIC_TRACKS)
    lib_mock.get_track_by_key.side_effect = lambda k: next(
        (t for t in SYNTHETIC_TRACKS if t["rating_key"] == k), None
    )
    lib_mock.get_tracks_by_ids.return_value = list(SYNTHETIC_TRACKS[:5])
    lib_mock.get_similar_tracks.return_value = list(SYNTHETIC_TRACKS[:3])

    gen.library = lib_mock

    # --- optional clients (all None / disabled) ---
    gen.lastfm = None
    gen.matcher = None
    gen.metadata = None
    gen.pipeline_override = None

    # --- DS-related flags ---
    gen._logged_ds_artifact_warning = True   # suppress missing-artifact warning
    gen._last_ds_report = None
    gen._pb_backoff_enabled = False
    gen._audit_run_enabled = False
    gen._audit_run_dir = None

    # --- misc state ---
    gen.similarity_calc = MagicMock()
    gen.genre_similarity_cache = {}
    gen.sonic_variant = "full"

    return gen
