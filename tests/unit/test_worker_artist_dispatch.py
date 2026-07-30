"""Pin the worker's artist-mode dispatch: the guard widening, the primary-name
fallback, and the artist_names forwarding added for the multi-artist blend.

This is the single riskiest hop in the artists round trip (API body ->
request model -> worker args -> generator call) -- a field can be accepted
at the boundary and silently dropped right here, which is exactly the bug
class this project's CLAUDE.md calls "the #1 failure mode". No existing
suite exercised this: test_worker_protocol_integration.py is an unrelated
ping/doctor smoke test, and both test_multi_artist_generation.py and
test_multi_artist.py call generator.create_playlist_for_artist(...) directly,
bypassing src/playlist_gui/worker.py entirely.

We drive the real handle_generate_playlist() and stub only the heavy
collaborators it constructs (LocalLibraryClient, TrackMatcher,
MetadataClient, PlaylistGenerator) so the test asserts on the actual
dispatch call, not a reimplementation of it. The fake generator returns
None from create_playlist_for_artist, which short-circuits the (unrelated)
track-formatting code below the call -- this test is about the call, not
the playlist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.playlist_gui import worker as worker_module


class _FakeLibrary:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeTrackMatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeMetadataClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeGenerator:
    """Captures every create_playlist_for_artist call for assertion."""

    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def create_playlist_for_artist(self, *args: Any, **kwargs: Any):
        _FakeGenerator.calls.append((args, kwargs))
        return None  # no playlist -> worker takes the cheap "no playlist" exit


@pytest.fixture(autouse=True)
def _patch_heavy_collaborators(monkeypatch):
    """Stub every class handle_generate_playlist imports via deferred
    `from module import Name` -- patching the attribute on the source module
    is picked up because the import executes at call time."""
    _FakeGenerator.calls = []
    monkeypatch.setattr(
        "src.local_library_client.LocalLibraryClient", _FakeLibrary
    )
    monkeypatch.setattr(
        "src.track_matcher.TrackMatcher", _FakeTrackMatcher
    )
    monkeypatch.setattr(
        "src.metadata_client.MetadataClient", _FakeMetadataClient
    )
    monkeypatch.setattr(
        "src.playlist_generator.PlaylistGenerator", _FakeGenerator
    )
    yield
    _FakeGenerator.calls = []


@pytest.fixture()
def minimal_config_path(tmp_path: Path) -> str:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "library:\n"
        "  database_path: unused.db\n"
        "logging:\n"
        "  playlist_logs:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    return str(config_yaml)


def _run_generate(minimal_config_path: str, args: dict) -> tuple[tuple, dict]:
    """Invoke the real handler and return the single captured
    create_playlist_for_artist call as (args, kwargs)."""
    worker_module.handle_generate_playlist(
        {
            "base_config_path": minimal_config_path,
            "overrides": {},
            "args": args,
        }
    )
    assert len(_FakeGenerator.calls) == 1, (
        f"expected exactly one create_playlist_for_artist call, got "
        f"{len(_FakeGenerator.calls)}"
    )
    return _FakeGenerator.calls[0]


def test_blend_with_no_scalar_artist_forwards_both_names(minimal_config_path):
    """2+ artists, no scalar `artist`: the guard must still fire (widened
    from `artist` alone to `artist or _artists`), and both names must reach
    the generator via artist_names -- this is the field this task exists to
    protect."""
    call_args, call_kwargs = _run_generate(
        minimal_config_path,
        {
            "mode": "artist",
            "artists": ["Brian Eno", "Harold Budd"],
            "tracks": 20,
        },
    )
    assert call_kwargs["artist_names"] == ["Brian Eno", "Harold Budd"]
    # Primary positional name must be non-None -- previously this branch
    # required a scalar `artist` and could not have been reached at all.
    assert call_args[0] == "Brian Eno"


def test_scalar_artist_only_is_unchanged_from_before_this_task(minimal_config_path):
    """No `artists` chips: single-artist behavior must be exactly what it
    was before this task -- primary name is the scalar `artist`, and
    artist_names is the empty-blend representation ([])."""
    call_args, call_kwargs = _run_generate(
        minimal_config_path,
        {
            "mode": "artist",
            "artist": "Brian Eno",
            "tracks": 20,
        },
    )
    assert call_args[0] == "Brian Eno"
    assert call_kwargs["artist_names"] == []


def test_both_scalar_and_list_set_pins_actual_behavior(minimal_config_path):
    """Both `artist` and `artists` set (e.g. a GUI that populates both for
    back-compat): pin what the code actually does -- the scalar `artist`
    wins as the primary positional name (short-circuit `artist or
    _artists[0]`), while the full `artists` list, unfiltered, still reaches
    artist_names. If this ever changes it should be a deliberate,
    reviewed decision, not an accidental regression."""
    call_args, call_kwargs = _run_generate(
        minimal_config_path,
        {
            "mode": "artist",
            "artist": "Harold Budd",
            "artists": ["Brian Eno", "Harold Budd"],
            "tracks": 20,
        },
    )
    assert call_args[0] == "Harold Budd"
    assert call_kwargs["artist_names"] == ["Brian Eno", "Harold Budd"]
