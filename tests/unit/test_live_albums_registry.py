"""LiveAlbumRegistry: load/save round-trip, degradation, and normalizer agreement."""
import pytest

from src.playlist.live_albums import (
    EMPTY_REGISTRY,
    build_registry,
    read_live_albums,
    save_live_albums,
)


SMITHS = [{"artist": "The Smiths", "album": "“Rank”", "source": "manual", "marked": "2026-08-10"}]


def test_round_trip(tmp_path):
    p = tmp_path / "live_albums.yaml"
    save_live_albums(SMITHS, path=p)
    assert read_live_albums(path=p) == SMITHS
    # Re-save backs up the existing file first (mirrors save_artist_link_groups).
    save_live_albums(SMITHS, path=p)
    assert list(tmp_path.glob("live_albums.yaml.bak.*")), "existing file must be backed up before write"


def test_registry_matches_marked_album_case_insensitively():
    reg = build_registry(SMITHS)
    assert len(reg) == 1
    assert reg.is_live("the smiths", "“RANK”")          # casefolded
    assert not reg.is_live("the smiths", "The Queen Is Dead")
    assert not reg.is_live("duster", "“Rank”")          # other artist
    assert reg.album_keys_for("the smiths") == frozenset({"“rank”"})
    assert reg.album_keys_for("duster") == frozenset()


def test_normalizers_agree_with_album_blacklist():
    """The registry MUST key exactly as album_blacklist does — a divergent
    normalizer is the _release_key bug (CLAUDE.md)."""
    from src.metadata_client import _normalize_album_key
    from src.string_utils import normalize_artist_key
    reg = build_registry(SMITHS)
    ak = normalize_artist_key("The Smiths")
    alk = _normalize_album_key("“Rank”")
    assert reg.album_keys_for(ak) == frozenset({alk})


def test_missing_file_is_quietly_empty(tmp_path):
    assert read_live_albums(path=tmp_path / "absent.yaml") == []


def test_malformed_yaml_degrades_to_empty_with_warning(tmp_path, caplog):
    import logging
    p = tmp_path / "live_albums.yaml"
    p.write_text("albums: [unclosed", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="src.playlist.live_albums"):
        assert read_live_albums(path=p) == []
    assert any("live_albums" in r.getMessage() for r in caplog.records)


def test_empty_registry_is_inert():
    assert len(EMPTY_REGISTRY) == 0
    assert EMPTY_REGISTRY.album_keys_for("anyone") == frozenset()


def test_save_rejects_invalid_entries(tmp_path):
    with pytest.raises(ValueError):
        save_live_albums([{"album": "No Artist"}], path=tmp_path / "x.yaml")


def test_non_dict_entries_are_skipped_not_fatal(tmp_path, caplog):
    import logging
    from src.playlist.live_albums import build_registry
    with caplog.at_level(logging.WARNING, logger="src.playlist.live_albums"):
        reg = build_registry(["oops", None, {"artist": "A", "album": "B"}])
    assert len(reg) == 1
    assert any("non-mapping" in r.getMessage() for r in caplog.records)
