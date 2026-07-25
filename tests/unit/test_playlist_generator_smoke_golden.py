"""
PlaylistGenerator smoke golden tests — safety net for Tier-3.2 extraction.

Strategy: bypass __init__ via __new__, monkeypatch _maybe_generate_ds_playlist
to return synthetic tracks, and snapshot the result structure in JSON goldens
under tests/unit/goldens/playlist_generator/.

Any extraction in PR-1 through PR-6 that silently changes the shape or content
of the returned dict will be caught here.

To re-baseline after an INTENTIONAL behaviour change, delete the relevant
golden file and re-run (the test will write a new baseline and skip).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

# Factory helpers live in the top-level tests/ package so they can be reused
# by future Tier-3.2 PRs without going through conftest.py injection.
from tests.conftest_playlist_generator import SYNTHETIC_TRACKS, make_synthetic_generator

GOLDEN_DIR = Path(__file__).parent / "goldens" / "playlist_generator"

# ---------------------------------------------------------------------------
# Genre-mode fixture: create_playlist_for_genre now resolves against the real
# genre authority (release_effective_genres) and clusters a real artifact
# bundle *before* the DS pipeline is ever called, so — unlike artist/seed
# mode — mocking _maybe_generate_ds_playlist alone is not enough to isolate
# this scenario. Without a real (if tiny) DB + bundle, config.get('library',
# 'database_path', ...) on the blanket MagicMock falls through
# resolve_database_path's default and silently opens the REAL
# data/metadata.db read-only — never a write risk, but not a hermetic unit
# test. Build a minimal on-disk genre authority DB + an in-memory
# ArtifactBundle instead, matching the schema already proven by
# tests/unit/test_genre_mode_membership.py.
# ---------------------------------------------------------------------------

_GENRE_TRACK_IDS = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
_GENRE_ARTISTS = {
    "g0": "Ambient One", "g1": "Ambient One", "g2": "Ambient One",
    "g3": "Ambient Two", "g4": "Ambient Two", "g5": "Ambient Two",
    "g6": "Ambient Three", "g7": "Ambient Three", "g8": "Ambient Three",
}


def _build_genre_smoke_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE genre_graph_canonical_genres "
        "(genre_id TEXT, name TEXT, kind TEXT, specificity_score REAL, "
        " status TEXT, taxonomy_version TEXT)"
    )
    conn.execute(
        "CREATE TABLE genre_graph_aliases "
        "(alias TEXT, canonical_genre_id TEXT, source TEXT, confidence REAL)"
    )
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT)")
    conn.execute(
        "CREATE TABLE release_effective_genres "
        "(album_id TEXT, release_key TEXT, genre_id TEXT, assignment_layer TEXT, "
        " confidence REAL, source TEXT)"
    )
    conn.execute(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        ("ambient", "ambient", "genre", 0.8, "active", "1"),
    )
    conn.executemany(
        "INSERT INTO tracks VALUES (?,?,?,?)",
        [
            (tid, artist, f"Album {artist}", f"alb_{tid}")
            for tid, artist in _GENRE_ARTISTS.items()
        ],
    )
    conn.executemany(
        "INSERT INTO release_effective_genres VALUES (?,?,?,?,?,?)",
        [
            (f"alb_{tid}", f"k_{tid}", "ambient", "observed_leaf", 0.85, "file")
            for tid in _GENRE_TRACK_IDS
        ],
    )
    conn.commit()
    conn.close()


def _build_genre_smoke_bundle():
    """A tiny real ArtifactBundle: 3 well-separated sonic clusters, one per
    artist, so cluster_artist_tracks (real k-means + pier-bridgeability veto)
    runs its actual logic instead of being mocked away."""
    from src.features.artifacts import ArtifactBundle
    from src.string_utils import normalize_artist_key

    rng = np.random.default_rng(0)
    centers = {
        "g0": [5, 0, 0], "g1": [5, 0, 0], "g2": [5, 0, 0],
        "g3": [0, 5, 0], "g4": [0, 5, 0], "g5": [0, 5, 0],
        "g6": [0, 0, 5], "g7": [0, 0, 5], "g8": [0, 0, 5],
    }
    X = np.zeros((len(_GENRE_TRACK_IDS), 3), dtype=np.float64)
    for i, tid in enumerate(_GENRE_TRACK_IDS):
        X[i] = np.asarray(centers[tid], dtype=np.float64) + rng.normal(scale=0.05, size=3)

    track_ids = np.array(_GENRE_TRACK_IDS)
    artist_names = np.array([_GENRE_ARTISTS[t] for t in _GENRE_TRACK_IDS])
    artist_keys = np.array([normalize_artist_key(a) for a in artist_names])
    titles = np.array([f"Track {t}" for t in _GENRE_TRACK_IDS])
    durations_ms = np.array([180_000] * len(_GENRE_TRACK_IDS))
    genre_vocab = np.array(["placeholder"])
    X_genre = np.zeros((len(_GENRE_TRACK_IDS), 1), dtype=np.float64)

    return ArtifactBundle(
        artifact_path=Path("synthetic-genre-bundle"),
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=artist_names,
        track_titles=titles,
        X_sonic=X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        genre_vocab=genre_vocab,
        track_id_to_index={str(t): i for i, t in enumerate(track_ids)},
        durations_ms=durations_ms,
    )


# ---------------------------------------------------------------------------
# Synthetic DS result — what _maybe_generate_ds_playlist returns
# ---------------------------------------------------------------------------

SYNTHETIC_DS_TRACKS: List[Dict[str, Any]] = list(SYNTHETIC_TRACKS[:8])


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def smoke_generator(monkeypatch, tmp_path):
    """
    Return a PlaylistGenerator with all I/O stubbed out.

    _maybe_generate_ds_playlist → SYNTHETIC_DS_TRACKS
    _post_order_validate_ds_output → no-op (we trust the golden to catch regressions)
    _compute_edge_scores_from_artifact → [] (no artifact on disk)
    _print_playlist_report → no-op (suppress console output)

    genre_basic additionally needs a real (tiny) genre-authority DB + artifact
    bundle, since create_playlist_for_genre resolves + clusters BEFORE the
    (mocked) DS call. Wired unconditionally here — harmless to artist_basic /
    seed_list_basic, which never reach config.get('library', 'database_path')
    or load_artifact_bundle (DS pipeline is fully mocked out for them too).
    """
    gen = make_synthetic_generator()

    monkeypatch.setattr(
        gen,
        "_maybe_generate_ds_playlist",
        lambda *args, **kwargs: list(SYNTHETIC_DS_TRACKS),
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_post_order_validate_ds_output",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_compute_edge_scores_from_artifact",
        lambda *args, **kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        gen,
        "_print_playlist_report",
        lambda *args, **kwargs: None,
        raising=False,
    )

    # --- genre-mode wiring (see module docstring above) ---
    genre_db_path = tmp_path / "genre_smoke.db"
    _build_genre_smoke_db(genre_db_path)

    def _cfg_get(*args, **kwargs):
        if args[:2] == ("library", "database_path"):
            return str(genre_db_path)
        if args[:2] == ("playlists", "ds_pipeline"):
            # random_seed=2 is the value empirically verified (see task-6
            # report) to split this fixture's 9 members into three clean
            # 3-member/one-artist-each clusters under the real k-means.
            return {"artifact_path": "synthetic-genre-bundle", "random_seed": 2}
        return {}

    gen.config.get.side_effect = _cfg_get

    genre_bundle = _build_genre_smoke_bundle()
    monkeypatch.setattr(
        "src.playlist_generator.load_artifact_bundle",
        lambda *args, **kwargs: genre_bundle,
    )

    # Real taxonomy steering would load data/layered_genre_taxonomy.yaml from
    # disk; genre_basic's single canonical genre never has a neighbor to
    # score (self-excluded), so a trivial stub keeps this a hermetic unit test.
    class _NullSteering:
        def similarity(self, a, b):
            return 1.0 if a == b else 0.0

    monkeypatch.setattr(
        "src.playlist.pier_bridge.taxonomy_steering.get_taxonomy_steering",
        lambda *args, **kwargs: _NullSteering(),
    )

    by_id = {t["rating_key"]: t for t in SYNTHETIC_TRACKS}
    by_id.update({
        tid: {
            "rating_key": tid,
            "artist": _GENRE_ARTISTS[tid],
            "artist_key": tid,
            "title": f"Track {tid}",
            "album": f"Album {_GENRE_ARTISTS[tid]}",
            "duration": 180_000,
            "file_path": f"/music/{tid}.mp3",
            "mbid": None,
        }
        for tid in _GENRE_TRACK_IDS
    })
    gen.library.get_tracks_by_ids.side_effect = (
        lambda ids: [by_id[str(i)] for i in ids if str(i) in by_id]
    )

    return gen


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _call_artist_basic(gen) -> Optional[Dict[str, Any]]:
    """artist_basic: 15 tracks across 3 artists (5 each); Artist 0 owns t0..t4.
    artist_key uses spaces ("artist 0") matching normalize_artist_key("Artist 0"),
    so the filter finds 5 tracks — above the ≥4 threshold — and proceeds to DS.
    """
    return gen.create_playlist_for_artist(
        artist_name="Artist 0",
        track_count=8,
        include_collaborations=False,
    )


def _call_genre_basic(gen) -> Optional[Dict[str, Any]]:
    """genre_basic: library mock returns all 12 SYNTHETIC_TRACKS for genre 'ambient'.
    All 12 have duration=180s which passes is_valid_duration(47s..720s) check.
    """
    return gen.create_playlist_for_genre(
        genre_name="ambient",
        track_count=8,
    )


def _call_seed_list_basic(gen) -> Optional[Dict[str, Any]]:
    """seed_list_basic: provide explicit track IDs so the exact-match path is taken."""
    return gen.create_playlist_from_seed_tracks(
        seed_tracks=["Track 0 - Artist 0", "Track 3 - Artist 1"],
        seed_track_ids=["t0", "t3"],
        track_count=8,
    )


SMOKE_SCENARIOS: Dict[str, Any] = {
    "artist_basic": _call_artist_basic,
    "genre_basic": _call_genre_basic,
    "seed_list_basic": _call_seed_list_basic,
}


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def _serialize_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce the orchestrator return value to a JSON-comparable snapshot.

    We capture: whether the result is None, the top-level keys present,
    the track_ids in order, and the track count.  We deliberately omit
    ds_report (internal metrics) and artists tuple (set-ordering is
    non-deterministic) to keep the golden stable.
    """
    if result is None:
        return {"result": None}

    tracks = result.get("tracks") or []
    track_ids = [
        str(t.get("rating_key") or t.get("track_id") or "")
        for t in tracks
    ]
    # Stable key listing (sorted) so golden doesn't depend on dict ordering
    top_keys = sorted(k for k in result if k != "ds_report")

    return {
        "result": "dict",
        "top_keys": top_keys,
        "track_ids": track_ids,
        "track_count": len(tracks),
    }


# ---------------------------------------------------------------------------
# The parametrized smoke golden test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", sorted(SMOKE_SCENARIOS.keys()))
def test_playlist_generator_smoke_golden(scenario_name, smoke_generator, request):
    call_fn = SMOKE_SCENARIOS[scenario_name]

    result = call_fn(smoke_generator)
    snapshot = _serialize_result(result)

    golden_path = GOLDEN_DIR / f"{scenario_name}.json"

    generate = request.config.getoption("--generate-golden", default=False)

    if generate or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if not generate:
            pytest.skip(
                f"Created golden baseline at {golden_path}; rerun to verify."
            )
        return

    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    assert snapshot["result"] == expected["result"], (
        f"[{scenario_name}] result presence changed: "
        f"got {snapshot['result']!r}, expected {expected['result']!r}"
    )

    if expected["result"] is None:
        # Both None — done.
        return

    assert snapshot["top_keys"] == expected["top_keys"], (
        f"[{scenario_name}] returned dict keys changed.\n"
        f"  Got:      {snapshot['top_keys']}\n"
        f"  Expected: {expected['top_keys']}"
    )
    assert snapshot["track_ids"] == expected["track_ids"], (
        f"[{scenario_name}] track ordering or content changed.\n"
        f"  Got:      {snapshot['track_ids']}\n"
        f"  Expected: {expected['track_ids']}"
    )
    assert snapshot["track_count"] == expected["track_count"], (
        f"[{scenario_name}] track_count changed: "
        f"{snapshot['track_count']} vs {expected['track_count']}"
    )
