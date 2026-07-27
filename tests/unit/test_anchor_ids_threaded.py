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
from tests.unit.test_pipeline_smoke_golden import _build_smoke_fixture


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


def test_core_forwards_anchor_kwargs_to_builder(tmp_path, monkeypatch):
    """core -> builder hop.

    core.generate_playlist_ds does real, unmocked embedding + candidate-pool
    work (setup_embedding, build_candidate_pool, the banger gate) before it
    ever reaches build_pier_bridge_playlist -- there's no way to mock those
    away without re-implementing the function's control flow. Rather than
    build a new fixture, this reuses `_build_smoke_fixture` +
    `build_pier_bridge_playlist` monkeypatch pattern from
    test_pipeline_smoke_golden.py::test_pier_bridge_config_matches_golden
    (also reused by test_roam_gate_relax.py) -- a small synthetic-but-real
    30-track npz artifact whose sole purpose (per that module's docstring)
    is getting generate_playlist_ds to the pier-bridge call site, which is
    then short-circuited before the real beam search runs.

    There is no bundle validation between generate_playlist_ds's signature
    and the builder call that would reject a synthetic id set against the
    fixture's t0..t29 ids, so {"k0", "k1"} (deliberately NOT piers) is a
    clean probe for pure kwarg forwarding.
    """
    captured: dict = {}

    def _capture_and_short_circuit(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after the handoff")

    monkeypatch.setattr(core, "build_pier_bridge_playlist", _capture_and_short_circuit)

    artifact_path = _build_smoke_fixture(tmp_path)

    try:
        core.generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            num_tracks=8,
            mode="dynamic",
            random_seed=0,
            tag_anchor_track_ids={"k0", "k1"},
        )
    except RuntimeError:
        pass

    assert captured.get("tag_anchor_track_ids") == {"k0", "k1"}
    assert captured.get("tag_anchor_gap_insertion") is True
    assert captured.get("tag_anchor_min_bridge") == 0.35


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
    # allowed_track_ids is None below, so _build_duration_exclusions_for_ds
    # takes the `else` branch (playlist_generator.py ~404-409) and calls this
    # method instead of fetch_track_durations. An unconfigured MagicMock
    # would silently satisfy both the `if excluded:` truthiness check
    # (MagicMock.__bool__ defaults True) and the later `.update(duration_excluded)`
    # (MagicMock.__iter__ defaults to an empty iterator) -- stub it explicitly
    # so the test's assumption (no duration exclusions) is visible, not an
    # accident of mock-library defaults.
    metadata.fetch_track_ids_by_duration_limits.return_value = set()
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
