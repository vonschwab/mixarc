"""A configured knob that can't act is a bug, not a no-op: pin that the anchor
track_ids the artist block computes actually REACH build_pier_bridge_playlist.

This is the exact class of defect the 2026-06-10 dead-code audit found three
times (beam widths at half config, a dead pace gate, a silently-disabled
dj_bridging), so the hop chain gets its own test rather than trusting the diff.
"""
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from src import playlist_generator as playlist_generator_module
from src.playlist import ds_pipeline_runner
from src.playlist.pipeline import core
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist
from src.playlist_generator import PlaylistGenerator


def test_every_hop_accepts_tag_anchor_track_ids():
    for fn in (
        PlaylistGenerator._maybe_generate_ds_playlist,
        ds_pipeline_runner.generate_playlist_ds,
        core.generate_playlist_ds,
        build_pier_bridge_playlist,
    ):
        assert "tag_anchor_track_ids" in inspect.signature(fn).parameters, fn


def test_runner_forwards_anchor_ids_to_core(monkeypatch):
    seen = {}

    def _fake_core(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the handoff")

    monkeypatch.setattr(ds_pipeline_runner, "core_generate_playlist_ds", _fake_core)
    try:
        ds_pipeline_runner.generate_playlist_ds(
            artifact_path="fake.npz", seed_track_id="t0", mode="dynamic", length=10,
            random_seed=0, tag_anchor_track_ids={"k0", "k1"},
        )
    except RuntimeError:
        pass
    assert seen.get("tag_anchor_track_ids") == {"k0", "k1"}


def test_generator_forwards_anchor_ids_to_runner(monkeypatch):
    """generator -> runner hop.

    `run_ds_pipeline` is imported into `src.playlist_generator` under an
    alias (`generate_playlist_ds as run_ds_pipeline`, playlist_generator.py:15)
    -- the patch target must be the name as it lives in the IMPORTING module's
    namespace, not `ds_pipeline_runner.generate_playlist_ds` at its definition
    site, or the patch is a silent no-op and this test would pass regardless
    of whether the real code forwards the kwarg.

    `_maybe_generate_ds_playlist` does real work (artifact bundle load,
    metadata lookups) before it ever reaches `run_ds_pipeline`, so this test
    builds a `PlaylistGenerator` via `__new__` (bypassing `__init__`, same
    pattern as `tests/conftest_playlist_generator.py`) with a config mock, a
    metadata mock (fetch_* return empty -- a non-None metadata client is
    required so `_ensure_metadata_client` doesn't attempt a REAL DB
    connection), and a monkeypatched `load_artifact_bundle` returning a
    minimal fake bundle. This keeps the test hermetic (no artifact file, no
    real DB) while still exercising the real, unmocked control flow up to the
    handoff.
    """
    gen = PlaylistGenerator.__new__(PlaylistGenerator)

    cfg = MagicMock()

    def _cfg_get(*args, **kwargs):
        if args[:2] == ("playlists", "ds_pipeline"):
            return {"artifact_path": "fake_artifact.npz"}
        return kwargs.get("default")

    cfg.get.side_effect = _cfg_get
    cfg.config = {}
    cfg.recently_played_lookback_days = 0
    gen.config = cfg

    metadata = MagicMock()
    metadata.fetch_blacklisted_track_ids.return_value = []
    metadata.fetch_track_durations.return_value = {}
    gen.metadata = metadata

    gen.library = MagicMock()
    gen.lastfm = None
    gen.matcher = None
    gen._logged_ds_artifact_warning = True
    gen._last_ds_report = None
    gen._audit_run_enabled = False
    gen._audit_run_dir = None

    fake_bundle = SimpleNamespace(track_id_to_index={"t0": 0})
    monkeypatch.setattr(playlist_generator_module, "load_artifact_bundle", lambda path: fake_bundle)

    seen = {}

    def _fake_run_ds_pipeline(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the handoff")

    monkeypatch.setattr(playlist_generator_module, "run_ds_pipeline", _fake_run_ds_pipeline)

    try:
        gen._maybe_generate_ds_playlist(
            seed_track_id="t0",
            target_length=10,
            tag_anchor_track_ids={"k0", "k1"},
        )
    except RuntimeError:
        pass

    assert seen.get("tag_anchor_track_ids") == {"k0", "k1"}
