"""Acceptance tests for the multi-artist blend
(spec docs/superpowers/specs/2026-07-29-multi-artist-blend-design.md).

Artist-mode pier discovery is DB-clustering, which tests/support/gui_fidelity.py's
generate_like_gui explicitly does NOT cover (seeds-mode only). Following the
precedent in test_gui_fidelity_regressions.py and test_genre_mode_generation.py,
these tests drive the real production entry point -- create_playlist_for_artist on
a PlaylistGenerator built from a real Config('config.yaml') -- never a hand-built
overrides dict.

Determinism: PlaylistGenerator is constructed WITHOUT a lastfm_client, so
self.lastfm is falsy and no network call happens. random_seed=0 pins clustering
and tie-breaks.

Library facts these tests rely on (verified 2026-07-29 against data/metadata.db):
  Brian Eno                  73 tracks
  Harold Budd                34 tracks
  Harold Budd and Brian Eno  21 tracks  <- the joint group
  David Bowie               162 tracks  <- no joint credit with Eno
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.features.artifacts import load_artifact_bundle
from src.playlist.artist_style import _artist_indices_in_bundle
from src.playlist.multi_artist import (
    ArtistGroup,
    MultiArtistBlendFailed,
    MultiArtistPiers,
    _blocked_artist_keys,
)
from tests.integration.test_gui_fidelity_regressions import _artist_generator, _select_piers
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

ENO = "Brian Eno"
BUDD = "Harold Budd"
BOWIE = "David Bowie"

# The pier IDs Brian Eno produced on master @ b4f9c89, before any multi-artist
# code existed. Any later task that changes this list has broken the
# single-artist path -- that is the whole point of this test.
#
# A legitimate artifact rebuild will also change it. That is an informative
# failure, not a false one: re-derive the list, confirm the change is explained
# by the rebuild and not by multi-artist code, and update the constant.
ENO_BASELINE_PIERS = [
    "b9348304f3408cb2c523af9a4f754607",
    "6fd170a37a8d57c0d4d2e504e18e54f1",
    "f44460624925361b21c6c6ac6039afd4",
    "0c283af35fbe6e8200ad44e682d678a9",
]


@_requires_artifact
def test_single_artist_piers_unchanged():
    """THE REGRESSION GATE. Multi-artist must be completely inert for one artist."""
    assert _select_piers(ENO, []) == ENO_BASELINE_PIERS


# ---------------------------------------------------------------------------
# Task 10 integration coverage: the DISPATCH logic in playlist_generator.py
# that wires select_multi_artist_piers's three outcomes (+ its two error
# paths) into create_playlist_for_artist. The orchestrator itself (partition,
# budget, clustering, ordering) is Task 3-9's territory and is already
# covered in tests/unit/test_multi_artist.py -- every test below
# monkeypatches select_multi_artist_piers to force one outcome and asserts on
# playlist_generator.py's reaction, never the orchestrator's own math. Live-
# artifact acceptance for real multi-artist blends is Task 13's job; nothing
# here duplicates it.
# ---------------------------------------------------------------------------


class _DispatchCaptured(Exception):
    """Raised by the ``_maybe_generate_ds_playlist`` stub the instant
    ``create_playlist_for_artist`` hands off piers -- lets these tests see
    exactly what reached the DS pipeline handoff (pier_ids AND the Task 10
    ``seed_artist_keys_override`` kwarg) without paying for the beam search.
    Mirrors ``test_gui_fidelity_regressions.py``'s ``_PiersCaptured``,
    extended with the one kwarg that test doesn't need.
    """

    def __init__(self, seed_track_id, anchor_seed_ids, seed_artist_keys_override):
        self.pier_ids = [str(seed_track_id)] + [str(a) for a in (anchor_seed_ids or [])]
        self.seed_artist_keys_override = seed_artist_keys_override
        super().__init__(f"dispatch captured: {self.pier_ids}")


def _capture_dispatch(generator) -> None:
    """Wire the _DispatchCaptured short-circuit onto ``generator`` in place."""
    generator._maybe_generate_ds_playlist = (
        lambda **kwargs: (_ for _ in ()).throw(
            _DispatchCaptured(
                kwargs.get("seed_track_id"),
                kwargs.get("anchor_seed_ids"),
                kwargs.get("seed_artist_keys_override"),
            )
        )
    )


def _real_indices_for(bundle, artist_name: str, n: int) -> list:
    idx = _artist_indices_in_bundle(bundle, artist_name, include_collaborations=False)
    assert len(idx) >= n, (
        f"fixture needs >= {n} {artist_name} track(s) in the live artifact, found {len(idx)}"
    )
    return list(idx[:n])


def _fake_multi_artist_piers(*, relaxations=None):
    """A structurally-real ``MultiArtistPiers`` built from REAL bundle indices
    (two Eno, two Budd) so every downstream consumer in
    ``create_playlist_for_artist`` (style_summary, pier_ids assembly, the
    genre-neighbor pool) sees genuine bundle rows instead of garbage indices.
    The clustering/budget/ordering MATH that would normally produce these
    indices is Task 3-9's territory (already covered in
    ``tests/unit/test_multi_artist.py``) -- this fixture exists only to feed
    the dispatch logic a plausible success result.

    Returns ``(piers, ordered_medoids, bundle)``.
    """
    # sonic_variant_override explicit -- tests/conftest.py's autouse
    # _reset_sonic_variant_override resets the process-wide override to None
    # before every test (same precedent as test_gui_fidelity_regressions.py's
    # direct load_artifact_bundle calls), so a bare load here would otherwise
    # fail with "Artifact missing required keys: ['X_sonic']".
    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
    eno_idx = _real_indices_for(bundle, ENO, 2)
    budd_idx = _real_indices_for(bundle, BUDD, 2)
    groups = [
        ArtistGroup(label=ENO, indices=eno_idx, is_joint=False),
        ArtistGroup(label=BUDD, indices=budd_idx, is_joint=False),
    ]
    ordered = [eno_idx[0], budd_idx[0], eno_idx[1], budd_idx[1]]
    piers = MultiArtistPiers(
        ordered_medoids=ordered,
        relaxations=list(relaxations or []),
        groups=groups,
        mean_affinity=0.5,
        blocked_artist_keys=_blocked_artist_keys(groups),
    )
    return piers, ordered, bundle


@_requires_artifact
def test_none_outcome_falls_back_to_single_artist_piers_unchanged():
    """select_multi_artist_piers -> None: fewer than two groups survived, so
    there's nothing to explain -- the single-artist path runs completely
    unchanged (same pier IDs as the zero-``artist_names`` gate above), and no
    ``seed_artist_keys_override`` reaches the DS pipeline handoff.
    """
    generator = _artist_generator([])
    _capture_dispatch(generator)
    with patch("src.playlist.multi_artist.select_multi_artist_piers", return_value=None):
        with pytest.raises(_DispatchCaptured) as excinfo:
            generator.create_playlist_for_artist(
                artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
            )
    assert excinfo.value.pier_ids == ENO_BASELINE_PIERS
    assert excinfo.value.seed_artist_keys_override is None


@_requires_artifact
def test_success_outcome_blocked_keys_reach_the_builder():
    """MultiArtistPiers success: its ordered_medoids become the exact piers
    handed to the DS pipeline, and its blocked_artist_keys reach
    seed_artist_keys_override -- the parameter
    ``pier_bridge_builder._derive_seed_artist_keys`` reads as its override.
    """
    fake_piers, ordered, bundle = _fake_multi_artist_piers()
    expected_pier_ids = [str(bundle.track_ids[i]) for i in ordered]

    generator = _artist_generator([])
    _capture_dispatch(generator)
    with patch("src.playlist.multi_artist.select_multi_artist_piers", return_value=fake_piers):
        with pytest.raises(_DispatchCaptured) as excinfo:
            generator.create_playlist_for_artist(
                artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
            )
    assert excinfo.value.pier_ids == expected_pier_ids
    assert excinfo.value.seed_artist_keys_override == fake_piers.blocked_artist_keys
    assert excinfo.value.seed_artist_keys_override is not None
    assert len(excinfo.value.seed_artist_keys_override) == 2


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_success_outcome_relaxations_merge_not_replace():
    """MultiArtistPiers.relaxations fold into the SAME warnings list
    pier_bridge_builder's own relaxations land in -- an existing warning must
    survive alongside the new multi-artist one, never be replaced by it. This
    needs a populated ``self._last_ds_report``, which only exists after a real
    DS pipeline run, so unlike the tests above it cannot use the pier-capture
    short-circuit.
    """
    fake_piers, _ordered, _bundle = _fake_multi_artist_piers(
        relaxations=[{
            "type": "relaxation", "scope": "multi_artist", "bridge": "Harold Budd",
            "relaxed": ["1 of 3 piers (only 2 tracks in your library)"], "severity": "info",
        }],
    )
    generator = _artist_generator([])
    real_ds_call = generator._maybe_generate_ds_playlist

    def _spy(**kwargs):
        tracks = real_ds_call(**kwargs)
        # Inject a pre-existing, unrelated warning exactly as pier_bridge_
        # builder's own relaxation reporting would -- proves the multi-artist
        # merge APPENDS rather than clobbers whatever is already there.
        pstats = generator._last_ds_report.setdefault("playlist_stats", {})
        playlist_stats = pstats.setdefault("playlist", {})
        playlist_stats.setdefault("warnings", []).append({
            "type": "relaxation", "scope": "pier_bridge", "bridge": "seg-0",
            "relaxed": ["a pre-existing warning"], "severity": "info",
        })
        return tracks

    generator._maybe_generate_ds_playlist = _spy
    with patch("src.playlist.multi_artist.select_multi_artist_piers", return_value=fake_piers):
        result = generator.create_playlist_for_artist(
            artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
        )
    assert result is not None
    warnings = (
        (result.get("ds_report") or {}).get("playlist_stats", {}).get("playlist", {}).get("warnings")
        or []
    )
    scopes = [w.get("scope") for w in warnings if isinstance(w, dict)]
    assert "pier_bridge" in scopes, f"pre-existing warning was lost: {warnings}"
    assert "multi_artist" in scopes, f"multi-artist relaxation never merged: {warnings}"


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_blend_failed_outcome_surfaces_relaxations_and_falls_back():
    """MultiArtistBlendFailed: caught explicitly, its relaxations surfaced
    into this run's warnings, and generation still completes via the
    single-artist fallback -- never propagates as an unhandled exception.
    """
    failure = MultiArtistBlendFailed(
        "Multi-artist: all 2 surviving group(s) were too thin to cluster.",
        [{
            "type": "relaxation", "scope": "multi_artist", "bridge": "Brian Eno",
            "relaxed": ["3 piers (only 1 track in your library)"], "severity": "info",
        }],
    )
    generator = _artist_generator([])
    with patch("src.playlist.multi_artist.select_multi_artist_piers", side_effect=failure):
        result = generator.create_playlist_for_artist(
            artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
        )
    assert result is not None and result.get("tracks")
    warnings = (
        (result.get("ds_report") or {}).get("playlist_stats", {}).get("playlist", {}).get("warnings")
        or []
    )
    matches = [
        w for w in warnings
        if isinstance(w, dict) and w.get("type") == "relaxation" and w.get("scope") == "multi_artist"
    ]
    assert matches, f"MultiArtistBlendFailed relaxations never surfaced: {warnings}"


@_requires_artifact
def test_malformed_artifact_value_error_propagates_uncaught():
    """A bare ValueError for a malformed artifact (missing artist_keys/
    X_sonic) is a data problem, not a "this pairing didn't work" problem -- it
    must escape create_playlist_for_artist, never be swallowed by the broad
    ``except Exception`` that handles ordinary artist-style fallback.
    """
    generator = _artist_generator([])
    with patch(
        "src.playlist.multi_artist.select_multi_artist_piers",
        side_effect=ValueError("Multi-artist: artifact missing X_sonic — cannot cluster any group."),
    ):
        with pytest.raises(ValueError, match="X_sonic"):
            generator.create_playlist_for_artist(
                artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
            )


@_requires_artifact
def test_disabled_config_warns_loudly_and_generates_from_first_artist(caplog):
    """``enabled: false`` with 2+ artists requested: warns loudly (not a
    silent no-op) and generates from ``artist_name`` alone -- pier IDs
    identical to the zero-``artist_names`` single-artist gate.
    """
    generator = _artist_generator([])
    generator.config.config.setdefault("playlists", {}).setdefault(
        "ds_pipeline", {}
    ).setdefault("multi_artist", {})["enabled"] = False
    _capture_dispatch(generator)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_DispatchCaptured) as excinfo:
            generator.create_playlist_for_artist(
                artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
            )
    assert excinfo.value.pier_ids == ENO_BASELINE_PIERS
    assert any(
        "multi_artist.enabled is false" in r.message for r in caplog.records
    ), "disabled-config path did not warn loudly"
