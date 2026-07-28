"""A configured knob that can't act is a bug, not a no-op: pin that the anchor
track_ids the artist block computes actually REACH build_pier_bridge_playlist.

This is the exact class of defect the 2026-06-10 dead-code audit found three
times (beam widths at half config, a dead pace gate, a silently-disabled
dj_bridging), so the hop chain gets its own test rather than trusting the diff.
"""
import inspect
import logging
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
            # Task 6 Part 1 narrows tag_anchor_track_ids to the pier ids actually
            # being handed down; anchor_seed_ids is the fallback pier set when
            # anchor_seed_tracks resolution doesn't run (as here), so k0/k1 must
            # appear there too for this pure-forwarding probe to stay meaningful.
            anchor_seed_ids=["k0", "k1"],
            tag_anchor_track_ids={"k0", "k1"},
        )
    except RuntimeError:
        pass

    assert seen.get("tag_anchor_track_ids") == {"k0", "k1"}


def _build_anchor_narrowing_test_generator(monkeypatch, *, blacklisted_ids):
    """Shared harness for Task 6 Part 1 tests: a PlaylistGenerator built via
    __new__ (bypassing __init__, same pattern as
    test_generator_forwards_anchor_ids_to_runner) with a fake 3-row bundle
    (seed t0, anchors k0/k1), a stubbed blacklist, and run_ds_pipeline
    monkeypatched to capture kwargs and short-circuit.

    This needs the title+artist re-resolution path (anchor_seed_tracks with
    >1 entries) to actually run, because blacklist filtering only touches
    anchor_seed_ids_resolved -- the raw anchor_seed_ids param is never
    blacklist-filtered. So the fake bundle carries track_artists/track_titles
    that resolve every supplied anchor_seed_track to a real bundle row.

    Returns (gen, anchor_seed_tracks, seen) where `seen` is populated with the
    kwargs forwarded to run_ds_pipeline once the caller invokes
    gen._maybe_generate_ds_playlist(...).
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
    metadata.fetch_track_ids_by_duration_limits.return_value = set()
    gen.metadata = metadata

    gen.library = MagicMock()
    gen.lastfm = None
    gen.matcher = None
    gen._logged_ds_artifact_warning = True
    gen._last_ds_report = None
    gen._audit_run_enabled = False
    gen._audit_run_dir = None

    fake_bundle = SimpleNamespace(
        track_id_to_index={"t0": 0, "k0": 1, "k1": 2},
        track_ids=["t0", "k0", "k1"],
        track_artists=["SeedArtist", "ForeignA", "ForeignB"],
        track_titles=["T0", "K0Title", "K1Title"],
    )
    monkeypatch.setattr(playlist_generator_module, "load_artifact_bundle", lambda path: fake_bundle)
    # Stub the blacklist source directly (per the brief) rather than the
    # metadata client.
    monkeypatch.setattr(gen, "_get_blacklisted_track_ids", lambda: set(blacklisted_ids))

    seen: dict = {}

    def _fake_run_ds_pipeline(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the handoff")

    monkeypatch.setattr(playlist_generator_module, "run_ds_pipeline", _fake_run_ds_pipeline)

    anchor_seed_tracks = [
        {"rating_key": "t0", "title": "T0", "artist": "SeedArtist"},
        {"rating_key": "k0", "title": "K0Title", "artist": "ForeignA"},
        {"rating_key": "k1", "title": "K1Title", "artist": "ForeignB"},
    ]
    return gen, anchor_seed_tracks, seen


def test_generator_narrows_tag_anchor_ids_when_blacklisted(monkeypatch, caplog):
    """Task 6 Part 1: narrow tag_anchor_track_ids to the pier ids actually being
    handed down, BEFORE the runner handoff. A blacklisted anchor legitimately
    stops being a pier (playlist_generator.py ~793-805 strips it out of
    anchor_seed_ids_resolved) while still sitting in tag_anchor_track_ids --
    that must be narrowed out here so Part 2's builder-side warning (a genuine
    anomaly warning) never fires for this benign, expected case. This PARTIAL
    loss (only k1 of {k0, k1} is blacklisted) must emit the INFO line and must
    NOT escalate to the total-loss WARNING -- crying wolf on a routine partial
    drop is exactly what the review follow-up warned against.
    """
    gen, anchor_seed_tracks, seen = _build_anchor_narrowing_test_generator(
        monkeypatch, blacklisted_ids={"k1"}
    )

    with caplog.at_level(logging.INFO, logger="src.playlist_generator"):
        try:
            gen._maybe_generate_ds_playlist(
                seed_track_id="t0",
                target_length=10,
                anchor_seed_tracks=anchor_seed_tracks,
                tag_anchor_track_ids={"k0", "k1"},
            )
        except RuntimeError:
            pass

    # k1 was blacklisted -> stripped from anchor_seed_ids_resolved upstream ->
    # Part 1 must narrow it out of tag_anchor_track_ids before the handoff.
    assert seen.get("tag_anchor_track_ids") == {"k0"}

    records = [r for r in caplog.records if r.name == "src.playlist_generator"]
    info_msgs = [
        r.message for r in records
        if r.levelno == logging.INFO and "no longer among the piers" in r.message
    ]
    warn_msgs = [
        r.message for r in records
        if r.levelno == logging.WARNING and "lost before reaching the pipeline" in r.message
    ]
    assert len(info_msgs) == 1, [r.message for r in records]
    assert "k1" in info_msgs[0]
    assert not warn_msgs, warn_msgs


def test_generator_warns_when_all_tag_anchor_ids_lost(monkeypatch, caplog):
    """Review follow-up (2026-07-27): when Part 1's narrowing empties
    tag_anchor_track_ids entirely (e.g. every candidate anchor was
    blacklisted -- reachable, since select_on_tag_anchors has no blacklist
    awareness), the builder's own gate (`if tag_anchor_gap_insertion and
    tag_anchor_track_ids:`) goes false and its "NONE matched a pier" warning
    never runs. Before Task 6 this scenario reliably WARNed (the unfiltered
    ids reached the builder and its own resolution found zero overlap); Part 1
    must not silently regress that down to an INFO line -- it has to escalate
    to WARNING itself, naming the lost ids, since it is the last site that
    still knows anchors were ever requested.
    """
    gen, anchor_seed_tracks, seen = _build_anchor_narrowing_test_generator(
        monkeypatch, blacklisted_ids={"k0", "k1"}
    )

    with caplog.at_level(logging.INFO, logger="src.playlist_generator"):
        try:
            gen._maybe_generate_ds_playlist(
                seed_track_id="t0",
                target_length=10,
                anchor_seed_tracks=anchor_seed_tracks,
                tag_anchor_track_ids={"k0", "k1"},
            )
        except RuntimeError:
            pass

    assert seen.get("tag_anchor_track_ids") == set()

    records = [r for r in caplog.records if r.name == "src.playlist_generator"]
    warn_msgs = [
        r.message for r in records
        if r.levelno == logging.WARNING and "lost before reaching the pipeline" in r.message
    ]
    assert len(warn_msgs) == 1, [r.message for r in records]
    assert "k0" in warn_msgs[0] and "k1" in warn_msgs[0]

    # The total-loss WARNING must REPLACE the partial-loss INFO, not join it.
    info_msgs = [
        r.message for r in records
        if r.levelno == logging.INFO and "no longer among the piers" in r.message
    ]
    assert not info_msgs, info_msgs
