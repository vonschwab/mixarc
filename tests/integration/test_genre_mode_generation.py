"""Acceptance tests for genre-mode playlist generation
(spec docs/superpowers/specs/2026-07-24-genre-mode-design.md).

Genre-mode seed selection is DB-clustering (artist-style pier discovery via sonic
clustering + Last.fm-cache-backed prominence), which
``tests/support/gui_fidelity.py``'s ``generate_like_gui`` explicitly does NOT cover
(seeds-mode only -- see the playlist-testing skill's "What this harness does NOT
cover": "DB-clustering (artist-style pier discovery) and Last.fm recency -- those
need the full worker entry"). These tests instead drive
``PlaylistGenerator.create_playlist_for_genre`` directly -- the real production entry
point (``src/playlist_gui/worker.py``'s ``handle_generate_playlist`` calls it exactly
this way for ``mode="genre"``) -- mirroring the artist-mode precedent in
``tests/integration/test_gui_fidelity_regressions.py``'s ``_artist_generator`` /
``_artist_mode_config``: a real ``Config(config_path)`` loaded straight from
``config.yaml``, never a hand-built ``overrides`` dict for ``generate_playlist_ds``.
``cohesion_mode_override`` is passed as a genuine keyword of the production method
(the same one the worker passes from the UI's cohesion-mode slider), not a config
mutation -- so this sidesteps the "mode keys mutated after Config() construction are
silently inert" trap in the playlist-testing skill's trap catalog.

Determinism: ``PlaylistGenerator`` is constructed WITHOUT a ``lastfm_client``
(parameter defaults to ``None``), so ``self.lastfm`` is falsy and
``create_playlist_for_genre``'s ``if self.lastfm:`` guard around
``_get_lastfm_scrobbles_raw`` (``src/playlist_generator.py`` ~L2986) never fires --
no network call, no Last.fm-recency run-to-run nondeterminism. ``artist_prominence``
(cross-artist seed-scoring signal) reads the local, already-populated
``data/popularity_cache.db`` sqlite cache -- not a live API call -- so it is
deterministic given the cache's current contents. ``random_seed=0`` pins every other
RNG-touching step (clustering, tie-breaks) the same way the rest of the integration
suite does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import Config, resolve_database_path
from src.local_library_client import LocalLibraryClient
from src.playlist_generator import PlaylistGenerator
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")


def _genre_generator(config_path: str = "config.yaml") -> PlaylistGenerator:
    """A real PlaylistGenerator over the real config.yaml + live DB, with no
    lastfm_client so the run is deterministic (see module docstring). Mirrors
    test_gui_fidelity_regressions.py's _artist_generator construction (real
    Config(config_path), real LocalLibraryClient) minus the tag-steering
    machinery genre mode has no equivalent of.
    """
    cfg = Config(config_path)
    library = LocalLibraryClient(db_path=resolve_database_path(cfg))
    return PlaylistGenerator(library_client=library, config=cfg)


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
@pytest.mark.parametrize("genre,min_tracks", [
    ("shoegaze", 25),      # fat genre, 89% Last.fm coverage
    ("funk", 25),          # fat genre, 29% coverage -- exercises uncached fallback
    ("hauntology", 12),    # 9 artists -- diversity binds before track count
    ("dub_techno", 8),     # 71 tracks / 15 artists -- exercises relaxation
])
def test_genre_mode_generates_and_respects_diversity(genre, min_tracks):
    result = _genre_generator().create_playlist_for_genre(
        genre_name=genre, track_count=30, cohesion_mode_override="dynamic", random_seed=0,
    )
    assert result is not None, f"genre '{genre}' generation returned None"
    tracks = result["tracks"]
    assert len(tracks) >= min_tracks, (
        f"genre '{genre}' produced only {len(tracks)} tracks (need >= {min_tracks})"
    )
    artists = [t.get("artist") for t in tracks]
    assert len(set(artists)) >= max(4, len(tracks) // 4), (
        f"genre '{genre}' artist diversity too low: {len(set(artists))} distinct "
        f"of {len(tracks)} tracks"
    )
