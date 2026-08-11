"""Song-scoped live ban: banned only when a non-live version exists; live-only rescued."""
import sqlite3

from src.playlist.live_albums import build_registry, compute_live_ban


def _db(tmp_path, rows):
    p = str(tmp_path / "meta.db")
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE tracks (track_id TEXT, title TEXT, album TEXT, artist_key TEXT)")
        c.executemany("INSERT INTO tracks VALUES (?,?,?,?)", rows)
    return p


REG = build_registry([{"artist": "The Smiths", "album": "“Rank”"}])


def test_ban_only_when_studio_version_exists(tmp_path):
    db = _db(tmp_path, [
        ("t1", "The Queen Is Dead", "“Rank”", "the smiths"),
        ("t2", "The Queen Is Dead", "The Queen Is Dead", "the smiths"),
        ("t3", "The Draize Train", "“Rank”", "the smiths"),          # live-only
        ("t4", "Panic", "Louder Than Bombs", "the smiths"),                     # unmarked
    ])
    res = compute_live_ban(db, REG)
    assert res.banned_track_ids == {"t1"}
    assert ("the smiths", "The Draize Train") in res.rescued
    assert not res.unmatched_entries


def test_grouping_is_loose_title_and_artist_key_not_identity(tmp_path):
    # "The Smiths" vs a hypothetical "The Smiths Trio" are DIFFERENT artist_keys:
    # the trio's studio copy must NOT ban the marked artist's live take.
    db = _db(tmp_path, [
        ("t1", "Panic", "“Rank”", "the smiths"),
        ("t2", "Panic", "Studio Album", "the smiths trio"),
    ])
    res = compute_live_ban(db, REG)
    assert res.banned_track_ids == set()
    assert res.rescued  # Panic on Rank is live-only for THIS artist_key


def test_unmatched_entry_reported(tmp_path):
    db = _db(tmp_path, [("t1", "Song", "Album", "someone else")])
    res = compute_live_ban(db, REG)
    assert res.unmatched_entries and res.unmatched_entries[0]["album"] == "“Rank”"


def test_empty_registry_is_free(tmp_path):
    from src.playlist.live_albums import EMPTY_REGISTRY
    db = _db(tmp_path, [("t1", "Song", "Album", "a")])
    res = compute_live_ban(db, EMPTY_REGISTRY)
    assert res.banned_track_ids == set() and not res.rescued and not res.unmatched_entries


def test_unmatched_warning_not_suppressed_by_other_artists_same_album_name(tmp_path, caplog):
    """Two artists mark albums with the SAME normalized name; only one exists in
    the library. The other's entry must still warn."""
    import logging
    from src.playlist.live_albums import build_registry, compute_live_ban
    reg = build_registry([
        {"artist": "Artist A", "album": "Live"},
        {"artist": "Artist B", "album": "Live"},
    ])
    db = _db(tmp_path, [
        ("t1", "Song", "Live", "artist a"),
        ("t2", "Song", "Studio", "artist a"),
    ])
    with caplog.at_level(logging.WARNING, logger="src.playlist.live_albums"):
        res = compute_live_ban(db, reg)
    assert res.banned_track_ids == {"t1"}
    assert any("Artist B" in r.getMessage() for r in caplog.records), "B's entry matched nothing and must warn"
    assert not any("Artist A" in r.getMessage() and "NO library tracks" in r.getMessage() for r in caplog.records)
