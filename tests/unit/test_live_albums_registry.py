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
    # 2, not 1: multi-indexing (xhigh review fix, 2026-08-11) also indexes the
    # entry under its identity key. resolve_artist_identity_keys's internal
    # normalizer strips a leading "The" (unlike the plain normalize_artist_key
    # used for the primary key below), so "The Smiths" contributes a second,
    # harmless "smiths" index key alongside "the smiths".
    assert len(reg) == 2
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


# --- Multi-indexing (xhigh review fix, 2026-08-11): a mark must fire under
# every key space a caller presents, not just the plain normalize_artist_key
# form the entry was authored under. ---------------------------------------

def test_registry_matches_identity_resolved_keys():
    """Pool-dedupe sites look up by ensemble-stripped identity key; a mark on
    'Bill Evans Trio' must fire for a row keyed 'bill evans'."""
    reg = build_registry([{"artist": "Bill Evans Trio", "album": "Live Album"}])
    assert reg.album_keys_for("bill evans trio")            # plain
    assert reg.album_keys_for("bill evans")                 # identity-stripped
    assert not reg.album_keys_for("bill withers")


def test_registry_matches_alias_resolved_keys(monkeypatch):
    import src.playlist.live_albums as la
    monkeypatch.setattr(la, "resolve_alias", lambda k: "alias_group:x" if k == "sigur ros" else k, raising=False)
    reg = build_registry([{"artist": "Sigur Ros", "album": "Live In Reykjavik"}])
    assert reg.album_keys_for("sigur ros")
    assert reg.album_keys_for("alias_group:x")
