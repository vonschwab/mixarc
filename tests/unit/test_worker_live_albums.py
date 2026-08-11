"""live_albums_fetch / live_album_set worker handlers round-trip the yaml."""
from unittest import mock

from src.playlist import live_albums as la


def test_live_album_set_adds_and_removes(tmp_path, monkeypatch):
    p = tmp_path / "live_albums.yaml"
    monkeypatch.setattr(la, "_DEFAULT_PATH", p)
    la.clear_cache()

    import src.playlist_gui.worker as worker
    emitted = []
    with mock.patch.object(worker, "emit_result", lambda kind, data: emitted.append((kind, data))), \
         mock.patch.object(worker, "emit_done", lambda *a, **k: None):
        worker.handle_live_album_set({"artist": "The Smiths", "album": "“Rank”", "enabled": True})
        worker.handle_live_albums_fetch({})
    assert la.get_active_registry().is_live("the smiths", "“Rank”")
    kinds = [k for k, _ in emitted]
    assert "live_albums" in kinds

    with mock.patch.object(worker, "emit_result", lambda *a, **k: None), \
         mock.patch.object(worker, "emit_done", lambda *a, **k: None):
        worker.handle_live_album_set({"artist": "The Smiths", "album": "“Rank”", "enabled": False})
    assert len(la.get_active_registry()) == 0
