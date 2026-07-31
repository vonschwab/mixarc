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

import src.playlist.multi_artist as multi_artist_mod
from src.features.artifacts import load_artifact_bundle
from src.playlist.artist_style import _artist_indices_in_bundle
from src.playlist.multi_artist import (
    ABSOLUTE_MIN_TRANSITION_FLOOR,
    ArtistGroup,
    MultiArtistBlendFailed,
    MultiArtistPiers,
    _alternation_count,
    _blocked_artist_keys,
)
from tests.integration.test_gui_fidelity_regressions import _artist_generator, _select_piers
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

ENO = "Brian Eno"
BUDD = "Harold Budd"
BOWIE = "David Bowie"
FOXX = "John Foxx"
VEGYN = "Vegyn"
BMSR = "Black Moth Super Rainbow"

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


# ---------------------------------------------------------------------------
# xhigh review 2026-07-30 (docs/superpowers/sdd/xhigh-review-fixes)
# ---------------------------------------------------------------------------


@_requires_artifact
def test_thin_primary_guard_holds_when_blend_branch_unreachable():
    """Finding 4: the thin-primary bypass (Finding 3) must not apply when the
    multi-artist branch will not actually run this call -- here, track_title
    makes the big style/multi-artist gate skip entirely. Without the fix, a
    1-track primary chip falls through past both early returns with nothing
    to rescue it and crashes deep in seed derivation instead of returning
    None cleanly (the pre-Finding-3 behavior for a genuinely thin artist)."""
    generator = _artist_generator([])
    fake_tracks = (
        [{"rating_key": "primary-1", "artist": "Thin Chip For Finding Four", "title": "Only Song"}]
        + [
            {"rating_key": f"other-{i}", "artist": "Other Chip For Finding Four", "title": f"T{i}"}
            for i in range(10)
        ]
    )
    generator.library.get_all_tracks = lambda: fake_tracks
    result = generator.create_playlist_for_artist(
        artist_name="Thin Chip For Finding Four",
        artist_names=["Thin Chip For Finding Four", "Other Chip For Finding Four"],
        track_count=30, random_seed=0, track_title="Only Song",
    )
    assert result is None, (
        "a thin (1-track) primary chip locked to a single seed track, with "
        "the multi-artist branch unreachable (track_title set), must return "
        "None -- not fall through to a crash deep in seed derivation"
    )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_absent_primary_chip_drops_and_proceeds_from_surviving_chip():
    """Finding 5: a misspelled/absent PRIMARY chip must not hard-crash in
    seed derivation (the "has no tracks after duration/title/freshness
    exclusions" ValueError) -- it should be dropped with a relaxation and
    generation should proceed from the surviving chip. "Radiohed" (a typo)
    has zero tracks in the real library; David Bowie is real and well-
    stocked with no joint credit with the misspelled name, so with only 2
    chips requested, select_multi_artist_piers itself can only ever surface
    <2 surviving groups here and falls back to single-artist -- the fix
    under test is that the fallback targets Bowie, not the absent chip."""
    generator = _artist_generator([])
    result = generator.create_playlist_for_artist(
        artist_name="Radiohed", artist_names=["Radiohed", BOWIE],
        track_count=30, random_seed=0,
    )
    assert result is not None and result.get("tracks"), (
        "an absent primary chip must not prevent generation from the "
        "surviving, well-stocked chip"
    )
    artists = _artists_in(result)
    assert any(BOWIE in a for a in artists), f"expected {BOWIE} tracks: {set(artists)}"
    warnings = _playlist_warnings(result)
    assert any(
        "radiohed" in str(w).lower() for w in warnings
    ), f"the dropped chip must be reported to the user: {warnings}"


@_requires_artifact
def test_recency_readmission_evaluated_per_chip_not_pooled(monkeypatch):
    """Finding 7: seed_recency_exclusion_for_presence must be called once PER
    CHIP with that chip's own indices, never once against a pooled id set
    spanning every chip -- pooling let a well-stocked chip's fresh count mask
    a thin chip's total starvation (repro: Eno 50 fresh + Budd 8-all-recent
    pooled to fresh_count=50, so Budd's re-admission never triggered)."""
    import src.playlist.seed_eligibility as se
    from src.playlist.utils import safe_get_artist_key
    from src.string_utils import normalize_artist_key

    real_fn = se.seed_recency_exclusion_for_presence
    calls = []

    def _spy(artist_track_ids, recency_excluded_ids, target_piers, **kw):
        calls.append(frozenset(str(t) for t in artist_track_ids))
        return real_fn(artist_track_ids, recency_excluded_ids, target_piers, **kw)

    monkeypatch.setattr(se, "seed_recency_exclusion_for_presence", _spy)

    generator = _artist_generator([])
    generator._get_local_history = lambda: [{"dummy": True}]
    budd_key = normalize_artist_key(BUDD)
    budd_ids = {
        str(t.get("rating_key")) for t in generator.library.get_all_tracks()
        if safe_get_artist_key(t) == budd_key
    }
    assert budd_ids, "fixture assumption changed -- Harold Budd must have real tracks"
    generator._compute_excluded_from_history = lambda *a, **kw: set(budd_ids)
    _capture_dispatch(generator)

    with pytest.raises(_DispatchCaptured):
        generator.create_playlist_for_artist(
            artist_name=ENO, artist_names=[ENO, BUDD], track_count=30,
            random_seed=0, exclude_seed_tracks_from_recency=True,
        )

    assert len(calls) == 2, f"expected one seed_recency_exclusion_for_presence call per chip, got {len(calls)}: {calls}"
    assert calls[0] != calls[1], "each call must use that chip's OWN ids, never a pooled set"


# ---------------------------------------------------------------------------
# Task 13: live acceptance. No monkeypatching of select_multi_artist_piers --
# these drive the real clustering/budget/ordering math against the real
# artifact + DB, through create_playlist_for_artist exactly as the GUI would
# call it (via _artist_generator, the same GUI-fidelity config resolution the
# rest of this file already uses -- a bare Config("config.yaml") would skip
# the derive_runtime_config -> merge_overrides -> load_config_with_overrides
# chain and could silently diverge from production, per this file's own
# import of _artist_generator from test_gui_fidelity_regressions).
#
# create_playlist_for_artist's result dict has no top-level "metrics" or
# "relaxations" key -- both live under "ds_report" (== self._last_ds_report):
# "ds_report"."metrics"."min_transition" and
# "ds_report"."playlist_stats"."playlist"."warnings" (see
# test_blend_failed_outcome_surfaces_relaxations_and_falls_back above, which
# already reads warnings this way).
# ---------------------------------------------------------------------------


def _blend(names, track_count=30):
    return _artist_generator([]).create_playlist_for_artist(
        artist_name=names[0], artist_names=list(names),
        track_count=track_count, random_seed=0,
    )


def _artists_in(result):
    return [str(t.get("artist") or "") for t in result["tracks"]]


def _min_transition(result) -> float:
    return float((result.get("ds_report") or {}).get("metrics", {}).get("min_transition"))


def _playlist_warnings(result) -> list:
    return (
        (result.get("ds_report") or {})
        .get("playlist_stats", {})
        .get("playlist", {})
        .get("warnings")
        or []
    )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_eno_budd_seats_both_artists_and_the_joint_credit():
    r = _blend([ENO, BUDD])
    assert r is not None and len(r["tracks"]) > 0
    artists = _artists_in(r)
    assert any(ENO in a for a in artists), f"no {ENO} track: {set(artists)}"
    assert any(BUDD in a for a in artists), f"no {BUDD} track: {set(artists)}"
    # the joint credit exists in the library (21 tracks) and should be reachable
    assert any("Harold Budd and Brian Eno" in a for a in artists), \
        f"the jointly-credited group was never seated: {set(artists)}"


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_eno_bowie_seats_both_without_a_joint_group():
    r = _blend([ENO, BOWIE])
    artists = _artists_in(r)
    assert any(ENO in a for a in artists)
    assert any(BOWIE in a for a in artists)


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_blend_piers_lean_toward_the_other_artist():
    """The core claim: with the blend on, Eno's seated piers sit closer to Budd's
    prototype than Eno's piers from a SOLO Eno run do. Without this the feature
    is decorative.

    ``blend["tracks"]`` (not just the piers) is filtered to Eno rows because
    ``disallow_pier_artists_in_interiors`` (config.yaml, on by default) keeps a
    seed artist out of bridge interiors -- with both Eno and Budd blocked as
    seed artists, every Eno row surviving in the final playlist IS an Eno pier.
    """
    import numpy as np

    from src.playlist.multi_artist import group_prototypes, partition_artist_groups
    from src.playlist.tag_steering import sonic_global_mean

    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
    groups, _dropped = partition_artist_groups(bundle, [ENO, BUDD])
    protos = group_prototypes(bundle, groups)
    budd_proto = protos[BUDD]

    X = np.asarray(bundle.X_sonic, dtype=float)
    gmean = sonic_global_mean(X)
    row_of = bundle.track_id_to_index

    def mean_pull(track_ids, artist_substr):
        rows = [
            row_of[str(t)] for t in track_ids
            if str(t) in row_of
            and artist_substr in str(bundle.track_artists[row_of[str(t)]])
        ]
        if not rows:
            return None
        M = X[rows] / (np.linalg.norm(X[rows], axis=1, keepdims=True) + 1e-12)
        return float(np.mean((M - gmean) @ budd_proto))

    solo_piers = _select_piers(ENO, [])
    blend = _blend([ENO, BUDD])
    blend_piers = [str(t.get("rating_key")) for t in blend["tracks"]]

    solo_pull = mean_pull(solo_piers, "Eno")
    blend_pull = mean_pull(blend_piers, "Eno")
    assert solo_pull is not None and blend_pull is not None
    assert blend_pull > solo_pull, (
        f"blend did not pull Eno toward Budd: solo={solo_pull:.4f} "
        f"blend={blend_pull:.4f}"
    )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_worst_edge_stays_above_an_absolute_floor():
    """Principle 5: the worst edge defines the experience -- but a blend is NOT
    held to single-artist smoothness. A blend spans two artists' territory by
    design (that is the whole point of test_blend_piers_lean_toward_the_other_
    artist); requiring it to bridge as smoothly as one cohesive artist's own
    playlist is a bar it does not owe. This guards against a genuine collapse
    (a starved, broken bridge), not against "a blend is a little rougher than
    either solo playlist," which turned out to be true even after Change 1
    below (Eno+Budd: 0.6135, still short of the old solo-parity bar of 0.6237).

    HUMAN RULING 2026-07-30 (task-13-report.md): this replaces the original
    solo-parity comparison after Task 13 found and fixed the real defect it had
    flagged -- total_pier_budget's segment_floor clamp was track_count // 3,
    giving a 30-track/3-group blend 10 piers (9 segments averaging ~2.2 tracks
    each, almost no room to bridge). Changed to track_count // 5 (6 piers,
    ~4.8 tracks/segment); see multi_artist.py's total_pier_budget docstring.

    CALIBRATION RECORD (2026-07-30, measured AFTER the // 5 fix, track_count=30,
    random_seed=0, via the real artifact/DB through create_playlist_for_artist):
      Brian Eno + Harold Budd        min_transition = 0.6135
      Brian Eno + David Bowie        min_transition = 0.5490
      Harold Budd + John Foxx        min_transition = 0.6478
    Lowest observed = 0.5490. This project's break-glass edge repair triggers
    at T < 0.30 (genuinely broken territory) -- the floor below sits comfortably
    between the two: well clear of ordinary run-to-run variation around 0.55,
    and well above the point where the pipeline's own repair logic would
    already consider the edge broken.

    2026-07-30 (forced-interleaving rewrite, pass 1): this same value became
    ``src.playlist.multi_artist.ABSOLUTE_MIN_TRANSITION_FLOOR``, imported
    above rather than redefined here, so this test and that constant can
    never drift apart.

    2026-07-30 (forced-interleaving rewrite, pass 2 -- coordinator follow-up,
    same day): ``ABSOLUTE_MIN_TRANSITION_FLOOR`` is NO LONGER read inside
    ``order_with_alternation``. Pass 1 reused it as the ordering safety
    valve's own threshold, but that floor measures the FINAL PLAYLIST's
    ``min_transition`` (what this test asserts) -- a different and naturally
    much higher quantity than PIER-TO-PIER adjacency (piers are deliberately
    spread apart; the beam fills the gaps between them), so reusing it there
    rejected pier orders that went on to produce a perfectly healthy final
    playlist (measured: Eno+Bowie's best-available pier-adjacency order was
    0.3680, already below 0.40, yet still delivered a final `min_transition`
    of 0.5490). The ordering safety valve now has its own constant,
    ``PIER_ADJACENCY_ALTERNATION_FLOOR`` (calibrated separately on the pier-
    adjacency scale -- see that constant's own docstring for the measured
    table). This test's floor and that one are deliberately DIFFERENT
    numbers measuring DIFFERENT quantities now; do not reunify them.
    """
    for names in ([ENO, BUDD], [ENO, BOWIE], [BUDD, FOXX]):
        blend = _blend(names)
        blend_min = _min_transition(blend)
        assert blend_min >= ABSOLUTE_MIN_TRANSITION_FLOOR, (
            f"{'+'.join(names)} blend worst edge {blend_min:.4f} fell below the "
            f"absolute floor {ABSOLUTE_MIN_TRANSITION_FLOOR} -- this is a real "
            "starvation/regression, not ordinary blend roughness"
        )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_unresolvable_chip_still_generates_and_reports():
    r = _blend([ENO, "Definitely Not A Real Artist Xyzzy"])
    assert r is not None and len(r["tracks"]) > 0
    blob = str(_playlist_warnings(r))
    assert "Xyzzy" in blob, f"the dropped chip was not reported: {blob}"


# ---------------------------------------------------------------------------
# Coordinator review 2026-07-30: findings 2, 3, 4, 5, 8, 9. The architectural
# root cause -- create_playlist_for_artist grew two parallel post-pier-
# selection tails when the blend was wired in, so five single-artist steps
# (title exclusion, terminal-avoidance ordering, recency re-admission scope,
# relaxation-merge robustness, and the Popular Seeds knob) were silently
# skipped or scoped wrong for a blend. Each test below pins one of those.
# ---------------------------------------------------------------------------


@_requires_artifact
def test_title_excluded_blend_pier_is_filtered_not_fatal():
    """Review Finding 2: blend piers used to skip
    self._filter_title_excluded_bundle_indices (the call the single-artist
    tail always makes), so a blend medoid matching a user title-exclusion
    word seated as a pier and _post_order_validate_ds_output aborted the
    WHOLE generation with no recovery path (exempt_pier_track_ids only
    covers the recency check, not the title check).

    Uses a word taken directly from one of the fake blend's own pier titles,
    so it is guaranteed to match -- then asserts generation still succeeds
    with that pier gone, exactly like the single-artist path already does.
    """
    fake_piers, ordered, bundle = _fake_multi_artist_piers()
    excluded_idx = ordered[0]
    title = str(bundle.track_titles[excluded_idx] or "")
    words = [w.strip(".,!?()[]\"'") for w in title.split()]
    words = [w for w in words if len(w) >= 4]
    if not words:
        pytest.skip(f"pier title {title!r} has no word long enough to safely exclude")
    exclusion_word = words[0]
    excluded_track_id = str(bundle.track_ids[excluded_idx])

    generator = _artist_generator([])
    generator.config.config.setdefault("playlists", {}).setdefault(
        "ds_pipeline", {}
    ).setdefault("candidate_pool", {}).update({
        "title_exclusion_enabled": True,
        "title_exclusion_words": [exclusion_word],
    })
    with patch("src.playlist.multi_artist.select_multi_artist_piers", return_value=fake_piers):
        result = generator.create_playlist_for_artist(
            artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
        )
    assert result is not None and result.get("tracks"), (
        "blend generation must still succeed when a pier medoid was title-excluded, "
        "not abort via post_order_validation_failed"
    )
    result_ids = {str(t.get("rating_key")) for t in result["tracks"]}
    assert excluded_track_id not in result_ids, (
        f"the title-excluded pier {excluded_track_id} ({title!r}) must not appear "
        "in the final playlist"
    )


@_requires_artifact
def test_thin_first_chip_does_not_abort_before_blend_logic_runs(monkeypatch):
    """Review Finding 3: the >=4-track gate near the top of
    create_playlist_for_artist only ever looked at the PRIMARY chip's own
    library tracks, returning None before select_multi_artist_piers /
    partition_artist_groups (which are written to handle exactly a thin
    first chip: drop it, or build from the surviving chip plus any joint
    credit) ever got a chance to run. Swapping chip order made the identical
    pairing succeed -- that asymmetry was the bug.

    Synthetic library: 'Thin One' has 2 tracks (below the 4-track floor) and
    no collaboration credit with anyone; 'Strong Two' has 5. Neither exists
    in the real sonic artifact, so this generation cannot actually complete
    -- but the OLD code returned None (logging 'Artist has only 2 total
    tracks') right here, before the artifact/bundle was even touched. The fix
    must get PAST that point and fail later, for the unrelated reason that
    the synthetic track_ids aren't in the artifact -- never via the early
    return.
    """
    generator = _artist_generator([])

    def fake_get_all_tracks():
        # duration is milliseconds (src.playlist.filtering.is_valid_duration) --
        # 200_000ms = 200s, comfortably inside the 47-720s valid window so the
        # automatic-seed-selection step downstream doesn't reject these
        # synthetic tracks for an unrelated reason before we even reach the
        # artifact-lookup failure this test expects.
        tracks = []
        for i in range(2):
            tracks.append({
                "rating_key": f"thin{i}", "artist": "Thin One",
                "title": f"Thin song {i}", "duration": 200_000,
            })
        for i in range(5):
            tracks.append({
                "rating_key": f"strong{i}", "artist": "Strong Two",
                "title": f"Strong song {i}", "duration": 200_000,
            })
        return tracks

    monkeypatch.setattr(generator.library, "get_all_tracks", fake_get_all_tracks)

    with pytest.raises(ValueError, match="found in the current"):
        generator.create_playlist_for_artist(
            artist_name="Thin One", artist_names=["Thin One", "Strong Two"],
            track_count=30, random_seed=0,
        )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_recency_readmission_scope_covers_every_chip():
    """Review Finding 4: seed_recency_exclusion_for_presence's artist-track
    scope was built from artist_name (chip #1) alone -- chip #2..N's tracks
    were invisible to the re-admission calculation entirely, so a starved
    second chip's recently-played tracks could never be re-admitted no
    matter how badly its group needed them. Every requested chip must be
    covered SOMEWHERE across the calls made for this run, not just the
    primary.

    xhigh review Finding 7 (2026-07-30): the caller now calls
    seed_recency_exclusion_for_presence once PER CHIP (each chip's own
    indices), not once against a POOLED set spanning every chip -- pooling
    let a well-stocked chip's fresh count mask a thin chip's starvation. This
    test now captures every call (a list) and checks the UNION covers both
    chips, and that each chip got its own dedicated, chip-scoped call --
    the original "covers every chip" intent, updated for the new per-chip
    mechanism."""
    import src.playlist.seed_eligibility as seed_eligibility_mod

    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
    eno_ids = {
        str(bundle.track_ids[i])
        for i in _artist_indices_in_bundle(bundle, ENO, include_collaborations=False)
    }
    budd_ids = {
        str(bundle.track_ids[i])
        for i in _artist_indices_in_bundle(bundle, BUDD, include_collaborations=False)
    }

    generator = _artist_generator([])
    generator._get_local_history = lambda: [{"dummy": True}]
    generator._compute_excluded_from_history = lambda *a, **k: {next(iter(budd_ids))}

    captured_scopes = []
    real_fn = seed_eligibility_mod.seed_recency_exclusion_for_presence

    def _spy(artist_track_ids, recency_excluded_ids, target_piers, **kwargs):
        captured_scopes.append(set(str(t) for t in artist_track_ids))
        return real_fn(artist_track_ids, recency_excluded_ids, target_piers, **kwargs)

    with patch(
        "src.playlist.seed_eligibility.seed_recency_exclusion_for_presence",
        side_effect=_spy,
    ):
        generator.create_playlist_for_artist(
            artist_name=ENO, artist_names=[ENO, BUDD], track_count=30,
            random_seed=0, exclude_seed_tracks_from_recency=True,
        )
    assert captured_scopes, "seed_recency_exclusion_for_presence was never called"
    assert len(captured_scopes) == 2, (
        f"expected one call per chip (Eno, Budd), got {len(captured_scopes)}"
    )
    union_scope = set().union(*captured_scopes)
    assert union_scope & budd_ids, (
        f"Harold Budd's tracks must be part of the recency re-admission scope, "
        f"got union scope of size {len(union_scope)}"
    )
    assert union_scope & eno_ids, "Brian Eno's tracks must still be part of the scope"
    assert any(scope & eno_ids and not (scope & budd_ids) for scope in captured_scopes), (
        "one call must be scoped to Eno's OWN tracks, not a pooled set"
    )
    assert any(scope & budd_ids and not (scope & eno_ids) for scope in captured_scopes), (
        "one call must be scoped to Budd's OWN tracks, not a pooled set"
    )


@_requires_artifact
@pytest.mark.integration
@pytest.mark.slow
def test_blend_failed_relaxations_survive_a_failing_fallback():
    """Review Finding 5: the relaxation merge used to be nested inside
    `if using_artist_style and style_seed_track_id and style_allowed_track_ids:`.
    MultiArtistBlendFailed's relaxations are stashed when the blend fails, but
    generation continues via the single-artist artist-style fallback in the
    SAME branch -- if that fallback then ALSO raised (not just the
    pool-too-small ValueError the branch already retries), the broad
    ``except Exception`` around the whole artist-style block set
    using_artist_style=False and reran through the legacy `else:` branch
    instead, discarding every relaxation collected so far. Forces exactly
    that sequence: MultiArtistBlendFailed, then a style-clustering failure
    (a real exception from the artist-style tail, not the ValueError this
    test would otherwise have to reverse-engineer), then confirms the legacy
    fallback still completes AND the relaxations still surface.
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
        with patch(
            "src.playlist.artist_style.cluster_artist_tracks",
            side_effect=RuntimeError("forced artist-style failure for Finding 5"),
        ):
            result = generator.create_playlist_for_artist(
                artist_name=ENO, artist_names=[ENO, BUDD], track_count=30, random_seed=0,
            )
    assert result is not None and result.get("tracks"), (
        "the legacy fallback must still complete a playlist"
    )
    warnings = (
        (result.get("ds_report") or {}).get("playlist_stats", {}).get("playlist", {}).get("warnings")
        or []
    )
    matches = [
        w for w in warnings
        if isinstance(w, dict) and w.get("type") == "relaxation" and w.get("scope") == "multi_artist"
    ]
    assert matches, (
        f"MultiArtistBlendFailed's relaxations must survive even when the "
        f"artist-style fallback ALSO fails and generation falls back to the "
        f"legacy path: {warnings}"
    )


@_requires_artifact
def test_dispatched_pier_order_matches_order_with_alternations_own_decision():
    """Finding-1 fix (xhigh code review, docs/superpowers/sdd/xhigh-review-
    fixes/finding-1-report.md): ``pier_support_terminal_avoidance`` used to
    run reorder_avoiding_low_support_terminal as a SEPARATE post-hoc reorder
    in create_playlist_for_artist, AFTER select_multi_artist_piers had
    already forced the highest safe artist alternation via
    order_with_alternation. That helper's own preference is a greedy
    nearest-neighbour search -- the same heuristic that clumps same-artist
    piers by construction, the precise reason the forced-interleaving
    rewrite exists -- so it silently discarded the alternation just forced
    (measured on the real library: Vegyn + Black Moth Super Rainbow went
    from a forced 5/5 down to 2/5). The OLD ``test_terminal_avoidance_
    applies_to_blend_piers`` this replaces only asserted the terminal pier
    moved, on a hand-built ``MultiArtistPiers`` with ``select_multi_artist_
    piers`` mocked out entirely -- it could never have caught this, because
    it never exercised order_with_alternation's own decision at all.

    This test does the opposite: it runs the REAL, unmocked
    ``select_multi_artist_piers`` (and therefore the real ``order_with_
    alternation`` call inside it, with terminal-support now folded in --
    see multi_artist.py) through the real production entry point,
    ``create_playlist_for_artist``, for the exact pairing that reproduced the
    bug plus the two others from the review's manual repro. It spies on
    ``order_with_alternation`` to capture its own decision (the ordered pier
    list and the alternation it achieved) and asserts the pier order actually
    handed to the DS pipeline is BYTE-IDENTICAL to that decision -- nothing
    between select_multi_artist_piers's return and the DS-pipeline handoff
    may reorder the piers, which is exactly the class of bug a reintroduced
    post-hoc reorder would be.
    """
    real_order_with_alternation = multi_artist_mod.order_with_alternation
    captured: dict = {}

    def _spy(*args, **kwargs):
        result = real_order_with_alternation(*args, **kwargs)
        captured["medoids"] = list(args[0])
        captured["group_of_index"] = dict(args[2])
        captured["result"] = result
        return result

    for names in ([VEGYN, BMSR], [ENO, BUDD], [ENO, BOWIE]):
        captured.clear()
        generator = _artist_generator([])
        _capture_dispatch(generator)
        with patch.object(multi_artist_mod, "order_with_alternation", side_effect=_spy):
            with pytest.raises(_DispatchCaptured) as excinfo:
                generator.create_playlist_for_artist(
                    artist_name=names[0], artist_names=list(names),
                    track_count=30, random_seed=0,
                )
        assert "result" in captured, (
            f"{'+'.join(names)}: order_with_alternation was never called -- "
            f"select_multi_artist_piers must have returned None/raised instead "
            f"of a real blend; this test needs the real orchestrator to run"
        )
        decided_order, _improved, decided_alt, _max_alt = captured["result"]
        group_of_index = captured["group_of_index"]

        bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")
        dispatched_indices = [
            bundle.track_id_to_index[pid] for pid in excinfo.value.pier_ids
        ]

        assert dispatched_indices == decided_order, (
            f"{'+'.join(names)}: order_with_alternation decided {decided_order} "
            f"but the DS pipeline received {dispatched_indices} -- something "
            f"reordered the piers AFTER order_with_alternation's decision (the "
            f"exact defect finding-1 fixed: a post-hoc terminal-avoidance "
            f"reorder silently discarded the forced interleaving)."
        )
        dispatched_alt = _alternation_count(dispatched_indices, group_of_index)
        assert dispatched_alt == decided_alt, (
            f"{'+'.join(names)}: the dispatched pier order's own alternation "
            f"({dispatched_alt}) does not match what order_with_alternation "
            f"decided ({decided_alt}) for the identical index list -- "
            f"_alternation_count and order_with_alternation disagree, which "
            f"should be structurally impossible"
        )
        if names == [VEGYN, BMSR]:
            assert dispatched_alt >= 5, (
                f"Vegyn + Black Moth Super Rainbow is the exact reproduction "
                f"of the finding-1 defect (previously forced 5/5, silently "
                f"degraded to 2/5 by the post-hoc reorder) -- got {dispatched_alt}/5"
            )


@_requires_artifact
def test_popular_seeds_on_warns_loudly_during_blend(caplog):
    """Review Finding 9: select_multi_artist_piers has no popularity term and
    never receives popularity_values -- a configured knob that cannot act
    must warn loudly, not silently no-op (project gotcha). This is the
    required minimum fix; genuinely popularity-biased blend piers are a
    separate, optional enhancement."""
    fake_piers, _ordered, _bundle = _fake_multi_artist_piers()
    generator = _artist_generator([])
    _capture_dispatch(generator)
    with caplog.at_level(logging.WARNING):
        with patch("src.playlist.multi_artist.select_multi_artist_piers", return_value=fake_piers):
            with pytest.raises(_DispatchCaptured):
                generator.create_playlist_for_artist(
                    artist_name=ENO, artist_names=[ENO, BUDD], track_count=30,
                    random_seed=0, popular_seeds_mode="on",
                )
    assert any(
        "Popular Seeds" in r.message and "no effect" in r.message
        for r in caplog.records
    ), "Popular Seeds mode 'on' during a blend must warn loudly, not silently no-op"
