"""A configured knob that can't act is a bug, not a no-op: pin that the anchor
track_ids the artist block computes actually REACH build_pier_bridge_playlist.

This is the exact class of defect the 2026-06-10 dead-code audit found three
times (beam widths at half config, a dead pace gate, a silently-disabled
dj_bridging), so the hop chain gets its own test rather than trusting the diff.
"""
import inspect

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
