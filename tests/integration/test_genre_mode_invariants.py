"""Pass 1 invariant tests, through the real ``create_playlist_for_genre`` path.

Deliberately drives the production entry point rather than the ``gui_fidelity``
harness: that harness is SEEDS-mode (it calls ``generate_playlist_ds`` with explicit
pier ids) and its own skill doc states it does not cover "DB-clustering (artist-style
pier discovery)". Genre-mode seed selection IS that path, so the harness would
exercise none of the code under test here.

Determinism: the generator is built without a Last.fm client, so the scrobble fetch
inside ``create_playlist_for_genre`` is structurally unreachable and no network call
can occur.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import Config, resolve_database_path
from src.local_library_client import LocalLibraryClient
from src.playlist_generator import PlaylistGenerator
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not ART.exists(), reason=f"artifact absent: {ART}"),
]


def _genre_generator(config_path: str = "config.yaml") -> PlaylistGenerator:
    """Production wiring minus Last.fm — see the module docstring on determinism."""
    cfg = Config(config_path)
    library = LocalLibraryClient(db_path=resolve_database_path(cfg))
    return PlaylistGenerator(library_client=library, config=cfg)


def test_algorithm_selected_piers_obey_the_duration_window():
    """Genre-mode piers are chosen by the engine, not requested by the listener, so
    they get no duration exemption.

    Regression: a 77:42 cv313 track was the TOP-scoring dub-techno pier and opened
    the playlist, because the pier path lost the duration filter the bridge pool
    kept.
    """
    gen = _genre_generator()
    cfg = Config("config.yaml")
    playlists_cfg = cfg.config.get("playlists", {}) or {}
    max_dur = int(playlists_cfg.get("max_track_duration_seconds", 720))
    min_dur = int(playlists_cfg.get("min_track_duration_seconds", 46))

    result = gen.create_playlist_for_genre("dub techno", track_count=30)
    assert result is not None, "dub techno failed to generate"

    over = [
        (t.get("artist"), t.get("title"), (t.get("duration_ms") or 0) / 1000)
        for t in result["tracks"]
        if t.get("is_pier") and (t.get("duration_ms") or 0) / 1000 > max_dur
    ]
    assert not over, f"pier(s) exceed the {max_dur}s window: {over}"

    titles = {str(t.get("title") or "").lower() for t in result["tracks"]}
    assert "depths of perception" not in titles, (
        "the 77:42 cv313 track is back in the playlist — the duration filter regressed"
    )
    assert min_dur <= max_dur  # guards a nonsense config rather than passing vacuously


def test_reggae_reports_its_artist_gap_breach_instead_of_hiding_it():
    """Reggae deterministically trips artist spacing, and the run must SAY so.

    The breach itself is not fixed in Pass 1 — its real cause is pier choice
    starving a segment pool onto a single artist, repaired in Pass 3. What Pass 1
    guarantees is that the run stops reporting success in silence.

    INVERT THIS TEST once Pass 3 lands: the expectation becomes `degraded is False`.
    """
    gen = _genre_generator()
    result = gen.create_playlist_for_genre("reggae", track_count=30)
    assert result is not None, "reggae failed to generate"

    assert result["degraded"] is True, (
        "reggae no longer reports degraded — either the breach was fixed (invert "
        "this test) or the invariant check stopped firing (a Pass 1 regression)"
    )
    gap = [w for w in result["warnings"] if w.startswith("artist_gap_violation")]
    assert gap, f"expected an artist_gap_violation, got {result['warnings']}"
    assert "positions=" in gap[0] and "configured_min_gap=" in gap[0], (
        f"warning must name the positions and the configured gap to be actionable: {gap[0]}"
    )


def test_a_clean_genre_is_not_flagged_degraded():
    """Guard against the opposite failure: a validator that cries wolf is as useless
    as one that stays silent. Jangle pop is the alpha review's positive control."""
    gen = _genre_generator()
    result = gen.create_playlist_for_genre("jangle pop", track_count=30)
    assert result is not None, "jangle pop failed to generate"
    assert result["degraded"] is False, (
        f"jangle pop should be clean but reported: {result['warnings']}"
    )
