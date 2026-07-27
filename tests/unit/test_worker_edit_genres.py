"""Tests for the handle_edit_genres worker command (authority-correction contract).

The handler reads the diff base server-side from metadata.db (release_effective_genres)
and resolves typed genres to canonical taxonomy ids; the heavy diff/materialize
logic is unit-tested in test_genre_edit.py. These tests cover the worker wiring:
config -> db_path, durable override + surgical authority write, and event emission.
"""

from __future__ import annotations

import json
import sqlite3

from src.ai_genre_enrichment.storage import SidecarStore
from src.playlist_gui.worker import handle_edit_genres


def _meta_db(tmp_path):
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        # `release_id` mirrors production (and tests/unit/test_genre_edit.py):
        # graph membership is decided by RELEASE KEY, not by a stamped album_id,
        # so a release fragment that shares the key still resolves to graph.
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(release_id TEXT, album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE genre_graph_canonical_genres (genre_id TEXT PRIMARY KEY, name TEXT, "
        " kind TEXT, specificity_score REAL, status TEXT, taxonomy_version TEXT);"
        "CREATE TABLE release_effective_genres (album_id TEXT NOT NULL, release_key TEXT, "
        " genre_id TEXT NOT NULL, assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, "
        " source TEXT NOT NULL, PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    conn.commit()
    return db, conn


def _config(tmp_path, db):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"library:\n  database_path: {db.as_posix()}\n"
        "playlists:\n  ds_pipeline:\n    genre_source: graph\n"
    )
    return cfg


def _events(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.strip().splitlines() if line.strip()]


def test_edit_genres_handler_writes_authority_and_override(tmp_path, monkeypatch, capsys):
    sidecar = tmp_path / "sidecar.db"
    SidecarStore(str(sidecar)).initialize()
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))

    db, conn = _meta_db(tmp_path)
    # Orphan album (tracks present, no `albums` row), zero genres — the Pet Grief case.
    conn.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    conn.commit()
    conn.close()
    cfg = _config(tmp_path, db)

    handle_edit_genres({
        "cmd": "edit_genres", "request_id": "r1", "base_config_path": str(cfg),
        "artist": "The  Radio Dept.", "album": "Pet Grief",
        "genres": ["dream pop", "shoegaze"],
    })

    override = SidecarStore(str(sidecar)).get_user_override("the radio dept::pet grief")
    assert override is not None
    assert len(override["genres_add"]) == 2

    c = sqlite3.connect(db)
    user_rows = c.execute(
        "SELECT genre_id FROM release_effective_genres "
        "WHERE album_id='ORPH1' AND source='user'"
    ).fetchall()
    c.close()
    assert len(user_rows) == 2

    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    assert result["no_change"] is False
    assert next(e for e in events if e["type"] == "done")["ok"] is True


def test_edit_genres_handler_reports_unknown_terms(tmp_path, monkeypatch, capsys):
    """A typed genre with no canonical mapping is reported, not saved."""
    sidecar = tmp_path / "sidecar.db"
    SidecarStore(str(sidecar)).initialize()
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(sidecar))

    db, conn = _meta_db(tmp_path)
    conn.execute("INSERT INTO tracks VALUES ('t1','Artist','Album','ALB1')")
    conn.commit()
    conn.close()
    cfg = _config(tmp_path, db)

    handle_edit_genres({
        "cmd": "edit_genres", "request_id": "r1", "base_config_path": str(cfg),
        "artist": "Artist", "album": "Album",
        "genres": ["dream pop", "zzzznotarealgenrezzz"],
    })
    result = next(e for e in _events(capsys) if e["type"] == "result")
    assert "zzzznotarealgenrezzz" in result["unknown"]
    assert "zzzznotarealgenrezzz" not in result["resolved"]


def test_edit_genres_no_artist_is_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp_path / "x.db"))
    handle_edit_genres({"cmd": "edit_genres", "artist": "", "album": "Amber", "genres": []})
    done = next(e for e in _events(capsys) if e["type"] == "done")
    assert done["ok"] is False
