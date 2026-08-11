import numpy as np
import sqlite3
from src.analyze.popularity_runner import (
    init_top_tracks_cache, cached_artist_keys,
    upsert_artist_top_tracks, get_artist_top_tracks_cached,
    build_popularity_sidecar,
)


def _make_artifact(tmp_path, track_ids):
    p = tmp_path / "data_matrices_step1.npz"
    np.savez(p, track_ids=np.array(track_ids, dtype=object))
    return str(p)


def _make_metadata(tmp_path, rows):
    # rows: (track_id, title, musicbrainz_id, artist_key[, album])
    # `album` mirrors the real schema: the sidecar build reads it so version
    # preference can demote a live take whose TITLE is clean and whose
    # live-ness lives only in the album name.
    db = str(tmp_path / "metadata.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE tracks (track_id TEXT, title TEXT, musicbrainz_id TEXT, "
            "artist_key TEXT, album TEXT)"
        )
        conn.executemany(
            "INSERT INTO tracks VALUES (?,?,?,?,?)",
            [tuple(r) + ("",) * (5 - len(r)) for r in rows],
        )
    return db


def test_build_popularity_sidecar_aligns_and_resolves(tmp_path):
    track_ids = ["a_studio", "a_live", "b_only"]
    artifact = _make_artifact(tmp_path, track_ids)
    meta = _make_metadata(tmp_path, [
        ("a_studio", "In Bloom", "", "nirvana"),
        ("a_live", "In Bloom (Live)", "", "nirvana"),
        ("b_only", "Tom Courtenay", "", "yo la tengo"),
    ])
    enrich = str(tmp_path / "enrich.db")
    init_top_tracks_cache(enrich)
    upsert_artist_top_tracks(enrich, "nirvana", "t", [{"name": "In Bloom", "mbid": "", "rank": 0}])
    # yo la tengo NOT cached -> b_only stays NaN
    out = str(tmp_path / "popularity" / "popularity_sidecar.npz")
    stats = build_popularity_sidecar(
        artifact_npz=artifact, metadata_db=meta, enrichment_db=enrich,
        out_path=out, min_artist_tracks=1,
    )
    z = np.load(out, allow_pickle=True)
    ids = [str(t) for t in z["track_ids"]]
    pop = z["popularity"]
    assert ids == track_ids                       # aligned to artifact order
    assert pop[ids.index("a_studio")] == 1.0      # studio In Bloom got popularity
    assert np.isnan(pop[ids.index("a_live")])     # live did not
    assert np.isnan(pop[ids.index("b_only")])     # uncached artist -> neutral
    assert stats["matched"] == 1


def test_cache_roundtrip_and_skip_set(tmp_path):
    db = str(tmp_path / "enrich.db")
    init_top_tracks_cache(db)
    assert cached_artist_keys(db) == set()
    rows = [{"name": "X", "playcount": 5, "mbid": "", "rank": 0}]
    upsert_artist_top_tracks(db, "nirvana", "2026-06-24T00:00:00Z", rows)
    assert cached_artist_keys(db) == {"nirvana"}
    assert get_artist_top_tracks_cached(db, "nirvana") == rows
    # upsert replaces
    upsert_artist_top_tracks(db, "nirvana", "later", [])
    assert get_artist_top_tracks_cached(db, "nirvana") == []
    assert get_artist_top_tracks_cached(db, "absent") == []
