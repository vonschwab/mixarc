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

GENRE IDENTITY: track-count and artist-diversity assertions alone cannot tell a
correctly-resolved genre from a wrong-but-populous one -- this is exactly how the
funk -> funk_metal misresolution (fixed in canonical_genre_search, see
tests/unit/test_genre_mode_resolution.py) slipped past manual generation until it
happened to starve down to 1 pier and return None; a less-starved misresolution would
have produced a plausible-looking playlist under the original track/artist-count-only
assertions. Every case here therefore also asserts (1) the resolved display name
(``result["genres"]``) matches the expected canonical name, catching a
canonical_genre_search-ordering regression by NAME identity, and (2) a meaningful
number of the realized tracks are independently verifiable exact members of the
EXPECTED genre_id via ``release_effective_genres`` (the authority table, read directly
-- NOT re-derived through ``genre_mode.resolve_genre_query``/``canonical_genre_search``,
so this check cannot pass merely because the buggy resolver agreed with itself),
catching a case where the display name is right but track admission has silently
decoupled from it. shoegaze/funk/hauntology all query the exact genre_id string
(case-identical to the DB), so the display-name and genre_id are equal for those three;
dub_techno's canonical display name has a space ("dub techno") where the query/genre_id
uses an underscore -- both are asserted explicitly per case below.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.config_loader import Config, resolve_database_path
from src.local_library_client import LocalLibraryClient
from src.playlist.genre_mode import genre_family_ids, seed_member_track_ids
from src.playlist_generator import PlaylistGenerator
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

# Below this, an overlap could plausibly be noise/mistagging rather than evidence the
# pool actually drew from the expected genre; observed pier counts across all four
# live cases are 5-6 (a pier is always an exact-genre member and pier positions are
# never displaced by repair/tail-DP), so this stays well under the true floor while
# still being high enough that a wrong-genre pool (funk_metal has ZERO overlap with
# funk's exact members, measured live 2026-07-24) cannot pass by chance.
_MIN_EXACT_GENRE_OVERLAP = 3


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


def _exact_genre_member_ids(genre_id: str, db_path: str) -> set[str]:
    """Independent authority re-read: member track ids for a HARDCODED ``genre_id``
    (never one recovered via ``resolve_genre_query``/``canonical_genre_search``), so
    this cannot pass merely because the resolver agreed with itself. Mirrors the
    independent-re-read pattern in test_gui_fidelity_regressions.py's
    ``_authority_on_tag_ids``.

    Membership is the genre's transitive ``is_a`` family, matching what generation
    now seeds from. The negative control still holds: `funk_metal` is not a funk
    descendant, so a funk_metal pool still has zero overlap with funk.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return seed_member_track_ids(con, genre_family_ids(con, genre_id))
    finally:
        con.close()


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
@pytest.mark.parametrize("genre,expected_genre_id,expected_name,min_tracks", [
    ("shoegaze", "shoegaze", "shoegaze", 25),      # fat genre, 89% Last.fm coverage
    ("funk", "funk", "funk", 25),                  # fat genre, 29% coverage -- exercises uncached fallback
    ("hauntology", "hauntology", "hauntology", 12),  # 9 artists -- diversity binds before track count
    ("dub_techno", "dub_techno", "dub techno", 8),   # 71 tracks / 15 artists -- exercises relaxation
])
def test_genre_mode_generates_and_respects_diversity(
    genre, expected_genre_id, expected_name, min_tracks,
):
    generator = _genre_generator()
    result = generator.create_playlist_for_genre(
        genre_name=genre, track_count=30, cohesion_mode_override="dynamic", random_seed=0,
    )
    assert result is not None, f"genre '{genre}' generation returned None"

    # (1) Genre IDENTITY: the resolved display name must be the one we asked for,
    # not a more-specific (or otherwise unrelated) neighbour that happened to
    # substring-match the query.
    assert result["genres"] == [expected_name], (
        f"genre '{genre}' resolved to {result['genres']!r}, expected [{expected_name!r}] "
        f"-- a canonical_genre_search ordering regression would surface here"
    )

    tracks = result["tracks"]
    assert len(tracks) >= min_tracks, (
        f"genre '{genre}' produced only {len(tracks)} tracks (need >= {min_tracks})"
    )
    artists = [t.get("artist") for t in tracks]
    assert len(set(artists)) >= max(4, len(tracks) // 4), (
        f"genre '{genre}' artist diversity too low: {len(set(artists))} distinct "
        f"of {len(tracks)} tracks"
    )

    # (2) Genre MEMBERSHIP: a meaningful number of the realized tracks must be
    # independently verifiable exact members of the EXPECTED genre_id (read straight
    # from release_effective_genres, bypassing the resolver entirely) -- proves the
    # pool actually drew from the right genre, not just that the display name says so.
    db_path = resolve_database_path(generator.config)
    expected_members = _exact_genre_member_ids(expected_genre_id, db_path)
    track_ids = {str(t.get("rating_key") or t.get("track_id") or "") for t in tracks}
    overlap = track_ids & expected_members
    assert len(overlap) >= _MIN_EXACT_GENRE_OVERLAP, (
        f"genre '{genre}': only {len(overlap)} of {len(tracks)} realized tracks are "
        f"independently-verified exact members of genre_id={expected_genre_id!r} "
        f"(need >= {_MIN_EXACT_GENRE_OVERLAP}) -- the pool may have drawn from the "
        f"wrong genre despite the display name matching"
    )
