"""Unit tests for multi-artist blend (spec 2026-07-29-multi-artist-blend-design.md).

Pure-function tests only: synthetic numpy arrays and a stub bundle, no DB and no
artifact. Live-artifact acceptance lives in
tests/integration/test_multi_artist_generation.py.
"""
from __future__ import annotations

from dataclasses import dataclass as _dc

import numpy as np
import pytest

from src.playlist.artist_style import ArtistStyleConfig
from src.playlist.multi_artist import (
    ArtistGroup,
    MultiArtistBlendFailed,
    MultiArtistConfig,
    MultiArtistPiers,
    allocate_pier_budget,
    group_genre_profiles,
    group_prototypes,
    multi_artist_config_from_ds,
    order_with_alternation,
    overlap_affinity,
    partition_artist_groups,
    select_multi_artist_piers,
    total_pier_budget,
)


def test_config_defaults_are_live():
    """The feature ships ON (principle 22); it is inert below two chips anyway."""
    cfg = multi_artist_config_from_ds({})
    assert cfg.enabled is True
    assert cfg.overlap_weight == pytest.approx(0.6)
    assert cfg.genre_share == pytest.approx(0.25)
    assert cfg.max_artists == 4
    assert cfg.joint_pier_min_budget == 3
    assert cfg.alternation_bonus == pytest.approx(0.15)
    assert cfg.low_overlap_threshold == pytest.approx(0.15)


def test_config_reads_overrides():
    cfg = multi_artist_config_from_ds(
        {"multi_artist": {"enabled": False, "overlap_weight": 0.0, "max_artists": 2}}
    )
    assert cfg.enabled is False
    assert cfg.overlap_weight == pytest.approx(0.0)
    assert cfg.max_artists == 2
    # untouched keys keep their defaults
    assert cfg.genre_share == pytest.approx(0.25)


@_dc
class _StubBundle:
    """Minimal stand-in for the artifact bundle: the attributes
    _artist_indices_in_bundle actually reads."""
    artist_keys: list
    track_artists: list
    track_ids: list


def _bundle(artists):
    from src.string_utils import normalize_artist_key
    return _StubBundle(
        artist_keys=[normalize_artist_key(a) for a in artists],
        track_artists=list(artists),
        track_ids=[f"t{i}" for i in range(len(artists))],
    )


def test_partition_splits_exclusive_and_joint():
    b = _bundle([
        "Brian Eno", "Brian Eno",              # 0,1 -> eno
        "Harold Budd",                          # 2   -> budd
        "Harold Budd and Brian Eno",            # 3   -> joint
        "Someone Else",                         # 4   -> neither
    ])
    groups, dropped = partition_artist_groups(b, ["Brian Eno", "Harold Budd"])
    assert dropped == []
    by_label = {g.label: g for g in groups}
    assert sorted(by_label["Brian Eno"].indices) == [0, 1]
    assert sorted(by_label["Harold Budd"].indices) == [2]
    joint = [g for g in groups if g.is_joint]
    assert len(joint) == 1
    assert joint[0].indices == [3]
    assert joint[0].source_artists == ("Brian Eno", "Harold Budd")
    # index 4 belongs to nobody
    assert all(4 not in g.indices for g in groups)


def test_joint_track_never_lands_in_an_exclusive_group():
    b = _bundle(["Harold Budd and Brian Eno"])
    groups, _ = partition_artist_groups(b, ["Brian Eno", "Harold Budd"])
    exclusive = [g for g in groups if not g.is_joint]
    assert all(g.indices == [] or 0 not in g.indices for g in exclusive)


def test_no_joint_group_when_artists_never_collaborated():
    b = _bundle(["Brian Eno", "David Bowie", "David Bowie"])
    groups, _ = partition_artist_groups(b, ["Brian Eno", "David Bowie"])
    assert [g for g in groups if g.is_joint] == []
    assert len(groups) == 2


def test_unresolvable_chip_is_reported_as_dropped():
    b = _bundle(["Brian Eno", "Brian Eno"])
    groups, dropped = partition_artist_groups(b, ["Brian Eno", "Nobody At All"])
    assert dropped == ["Nobody At All"]
    assert [g.label for g in groups] == ["Brian Eno"]


def test_excluded_track_ids_are_removed_before_grouping():
    b = _bundle(["Brian Eno", "Brian Eno", "Harold Budd"])
    groups, _ = partition_artist_groups(
        b, ["Brian Eno", "Harold Budd"], excluded_track_ids={"t0"}
    )
    by_label = {g.label: g for g in groups}
    assert by_label["Brian Eno"].indices == [1]


def test_joint_only_artist_is_not_dropped_and_keeps_joint_provenance():
    """Regression for review finding 1: an artist credited only jointly with
    the other chip (never solo) must not be reported dropped, and must still
    show up in the joint group's label/source_artists -- e.g. the real
    library's "John Foxx and Harold Budd" alongside thin solo catalogs."""
    b = _bundle([
        "Alice and Bob", "Alice and Bob",   # 0,1 -> joint only, Alice has no solo rows
        "Bob",                               # 2   -> Bob exclusive
    ])
    groups, dropped = partition_artist_groups(b, ["Alice", "Bob"])
    assert dropped == []
    joint = next(g for g in groups if g.is_joint)
    assert joint == ArtistGroup(
        label="Alice & Bob",
        indices=[0, 1],
        is_joint=True,
        source_artists=("Alice", "Bob"),
    )
    exclusive = [g for g in groups if not g.is_joint]
    assert [g.label for g in exclusive] == ["Bob"]
    assert exclusive[0].indices == [2]


def test_duplicate_chip_names_are_deduped():
    """Regression for review finding 2: a repeated chip name must not produce
    two identical exclusive groups (it would double that artist's downstream
    pier budget)."""
    b = _bundle(["Brian Eno", "Brian Eno", "Harold Budd"])
    groups, dropped = partition_artist_groups(
        b, ["Brian Eno", "Brian Eno", "Harold Budd"]
    )
    assert dropped == []
    labels = [g.label for g in groups]
    assert labels.count("Brian Eno") == 1
    by_label = {g.label: g for g in groups}
    assert sorted(by_label["Brian Eno"].indices) == [0, 1]
    assert by_label["Harold Budd"].indices == [2]


def test_third_party_collaboration_excluded_by_default():
    """Regression for review finding 3: the include_collaborations=False path
    (previously uncovered) must strip a THIRD-party collaboration out of every
    group, while the two-chip joint track still lands in the joint group."""
    b = _bundle([
        "Brian Eno",                        # 0 -> Eno exclusive
        "Harold Budd",                      # 1 -> Budd exclusive
        "Harold Budd and Brian Eno",        # 2 -> joint (both chips)
        "Brian Eno & John Cale",            # 3 -> third-party collab, not a chip
    ])
    groups, dropped = partition_artist_groups(b, ["Brian Eno", "Harold Budd"])
    assert dropped == []
    assert all(3 not in g.indices for g in groups)
    joint = next(g for g in groups if g.is_joint)
    assert joint.indices == [2]
    by_label = {g.label: g for g in groups if not g.is_joint}
    assert by_label["Brian Eno"].indices == [0]
    assert by_label["Harold Budd"].indices == [1]


def test_third_party_collaboration_absorbed_when_include_collaborations():
    """include_collaborations=True must fold the third-party collab into that
    artist's exclusive group, while the two-chip joint track stays
    joint-only -- never duplicated into an exclusive group."""
    b = _bundle([
        "Brian Eno",                        # 0 -> Eno exclusive
        "Harold Budd",                      # 1 -> Budd exclusive
        "Harold Budd and Brian Eno",        # 2 -> joint (both chips)
        "Brian Eno & John Cale",            # 3 -> third-party collab, not a chip
    ])
    groups, dropped = partition_artist_groups(
        b, ["Brian Eno", "Harold Budd"], include_collaborations=True
    )
    assert dropped == []
    by_label = {g.label: g for g in groups if not g.is_joint}
    assert sorted(by_label["Brian Eno"].indices) == [0, 3]
    assert by_label["Harold Budd"].indices == [1]
    joint = next(g for g in groups if g.is_joint)
    assert joint.indices == [2]
    # the joint track is not duplicated into any exclusive group
    assert all(2 not in g.indices for g in groups if not g.is_joint)


@_dc
class _SonicBundle:
    artist_keys: list
    track_artists: list
    track_ids: list
    X_sonic: object
    X_genre_dense: object = None


def _sonic_bundle(artists, sonic, genre=None):
    from src.string_utils import normalize_artist_key
    return _SonicBundle(
        artist_keys=[normalize_artist_key(a) for a in artists],
        track_artists=list(artists),
        track_ids=[f"t{i}" for i in range(len(artists))],
        X_sonic=np.asarray(sonic, dtype=float),
        X_genre_dense=None if genre is None else np.asarray(genre, dtype=float),
    )


def test_prototype_is_unit_norm_per_group():
    b = _sonic_bundle(
        ["A", "A", "B", "B"],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    )
    groups, _ = partition_artist_groups(b, ["A", "B"])
    protos = group_prototypes(b, groups)
    assert set(protos) == {"A", "B"}
    for p in protos.values():
        assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-9)


def test_affinity_ranks_the_track_nearest_the_other_artist_highest():
    """A has two tracks: index 0 is far from B, index 1 leans toward B.
    The affinity for group A must rank index 1 above index 0."""
    b = _sonic_bundle(
        ["A", "A", "B", "B"],
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.1, 1.0]],
    )
    groups, _ = partition_artist_groups(b, ["A", "B"])
    protos = group_prototypes(b, groups)
    group_a = next(g for g in groups if g.label == "A")
    aff = overlap_affinity(b, group_a, groups, protos, {}, genre_share=0.0)
    assert aff.shape == (4,)
    assert aff[1] > aff[0], "the B-leaning track should score higher"
    # rows outside the group are untouched
    assert aff[2] == 0.0 and aff[3] == 0.0


def test_affinity_is_zero_length_safe_for_an_empty_other_side():
    b = _sonic_bundle(["A", "A"], [[1.0, 0.0], [0.0, 1.0]])
    groups, _ = partition_artist_groups(b, ["A"])
    protos = group_prototypes(b, groups)
    aff = overlap_affinity(b, groups[0], groups, protos, {}, genre_share=0.0)
    assert np.allclose(aff, 0.0), "no other artist -> no pull"


def test_genre_share_renormalizes_to_pure_sonic_without_dense_genre(caplog):
    b = _sonic_bundle(
        ["A", "A", "B"],
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
    )  # X_genre_dense is None
    groups, _ = partition_artist_groups(b, ["A", "B"])
    protos = group_prototypes(b, groups)
    profiles = group_genre_profiles(b, groups)
    assert profiles == {}
    group_a = next(g for g in groups if g.label == "A")
    with caplog.at_level("WARNING"):
        aff = overlap_affinity(b, group_a, groups, protos, profiles, genre_share=0.25)
    assert any("X_genre_dense" in r.message for r in caplog.records), \
        "a genre term that cannot act must WARN, never silently no-op"
    pure = overlap_affinity(b, group_a, groups, protos, {}, genre_share=0.0)
    assert np.allclose(aff, pure), "must renormalize to pure sonic, not scale down"


def test_affinity_ranking_flips_without_global_mean_centering():
    """Regression: none of the four tests above can detect a dropped or broken
    sonic_global_mean centering -- monkeypatching it to return an all-zero
    vector left all four passing (see task-4-report.md). This fixture is built
    so the RAW (uncentered) cosine and the CENTERED cosine disagree about which
    of group A's two candidates is closer to group B:

    - a library dominated by filler rows pointing along [1, 0] pulls the global
      mean toward [1, 0];
    - A's 'generic' candidate sits close to that generic direction;
    - A's 'specific' candidate sits close to the orthogonal, genuinely-B-like
      direction;
    - B's own rows lean toward the generic direction too, so B's RAW prototype
      favors the generic candidate -- but once the generic direction is
      subtracted, B's centered prototype favors the specific candidate instead.

    Only the intended (centered) behavior is asserted here. The monkeypatch-to-
    zero check that proves this fixture actually discriminates is a one-off
    verification run manually, not shipped test code -- see task-4-report.md.
    """
    filler = [[1.0, 0.0]] * 12
    a_generic = [0.9848, 0.1736]
    a_specific = [0.0872, 0.9962]
    b_rows = [[0.9659, 0.2588], [0.9659, 0.2588]]
    b = _sonic_bundle(
        ["Filler"] * 12 + ["A", "A", "B", "B"],
        filler + [a_generic, a_specific] + b_rows,
    )
    groups, _ = partition_artist_groups(b, ["A", "B"])
    protos = group_prototypes(b, groups)
    group_a = next(g for g in groups if g.label == "A")
    idx_generic, idx_specific = group_a.indices
    aff = overlap_affinity(b, group_a, groups, protos, {}, genre_share=0.0)
    assert aff[idx_specific] > aff[idx_generic], (
        "centering must surface the genuinely B-like candidate, not the one "
        "merely close to the library's generic direction"
    )


def test_single_group_budget_matches_todays_formula():
    """One group must reproduce max(3, round(track_count * fraction)) exactly."""
    for tracks, frac in [(30, 0.125), (12, 0.125), (50, 0.2), (20, 0.05)]:
        assert total_pier_budget(tracks, frac, 1) == max(3, round(tracks * frac))


def test_two_groups_double_the_budget_within_the_segment_floor():
    # Human ruling 2026-07-30 (task-13-report.md): the segment_floor clamp
    # changed from track_count // 3 to track_count // 5 -- // 3 was a real
    # pier-density starvation defect (10 piers on a 30-track/3-group blend
    # left ~2.2 tracks per bridge segment; see multi_artist.py's
    # total_pier_budget docstring). These two expected values are recomputed
    # by hand from the NEW formula, not loosened.
    # 30 tracks, 0.125 -> base 4; two groups -> 8; floor = 30 // 5 = 6 -> clamped to 6
    assert total_pier_budget(30, 0.125, 2) == 6
    # 15 tracks, 0.125 -> base 3; two groups -> 6; floor = 15 // 5 = 3 -> clamped to 3
    assert total_pier_budget(15, 0.125, 2) == 3


def test_budget_never_drops_below_one_per_group():
    assert total_pier_budget(6, 0.125, 4) == 4


def test_allocation_is_even_with_remainder_to_the_first_chip():
    groups = [
        ArtistGroup("A", [0, 1, 2, 3, 4, 5]),
        ArtistGroup("B", [6, 7, 8, 9, 10, 11]),
    ]
    alloc = allocate_pier_budget(groups, 7, joint_pier_min_budget=3)
    assert alloc == {"A": 4, "B": 3}


def test_joint_group_claims_one_seat_when_budget_allows():
    groups = [
        ArtistGroup("A", [0, 1, 2]),
        ArtistGroup("B", [3, 4, 5]),
        ArtistGroup("A & B", [6, 7], is_joint=True, source_artists=("A", "B")),
    ]
    alloc = allocate_pier_budget(groups, 5, joint_pier_min_budget=3)
    assert alloc["A & B"] == 1
    assert alloc["A"] + alloc["B"] == 4
    assert sum(alloc.values()) == 5


def test_joint_group_gets_no_seat_below_the_min_budget():
    groups = [
        ArtistGroup("A", [0, 1]),
        ArtistGroup("B", [2, 3]),
        ArtistGroup("A & B", [4], is_joint=True, source_artists=("A", "B")),
    ]
    alloc = allocate_pier_budget(groups, 2, joint_pier_min_budget=3)
    assert alloc.get("A & B", 0) == 0
    assert sum(alloc.values()) == 2


def test_a_thin_group_cannot_be_allocated_more_piers_than_it_has_tracks():
    groups = [
        ArtistGroup("A", list(range(20))),
        ArtistGroup("B", [20]),  # one track
    ]
    alloc = allocate_pier_budget(groups, 8, joint_pier_min_budget=3)
    assert alloc["B"] == 1, "cannot seat more piers than the group has tracks"
    assert alloc["A"] == 7, "the surplus reallocates to the group that can use it"


def test_joint_group_absorbs_surplus_when_exclusive_groups_are_saturated():
    """Regression for review finding: a dominant joint catalog (e.g. a real
    duo credited on far more tracks than either artist's thin solo work) must
    not be capped at its guaranteed one seat while surplus is silently
    dropped. Once A and B are saturated at their track counts, the remaining
    budget must flow to the joint group, up to its own capacity."""
    groups = [
        ArtistGroup("A", [0, 1]),
        ArtistGroup("B", [2, 3]),
        ArtistGroup("A & B", list(range(4, 24)), is_joint=True,
                    source_artists=("A", "B")),  # 20 tracks
    ]
    alloc = allocate_pier_budget(groups, 10, joint_pier_min_budget=3)
    assert alloc == {"A": 2, "B": 2, "A & B": 6}
    assert sum(alloc.values()) == 10, "all 10 requested piers must be seated"


def test_shortfall_against_total_capacity_is_seated_fully_and_warned(caplog):
    """When the request genuinely exceeds every group's combined track count,
    the allocation must saturate every group (never drop a seatable pier) and
    the shortfall must be logged, never inferred from a short playlist."""
    groups = [
        ArtistGroup("A", [0, 1]),
        ArtistGroup("B", [2, 3]),
        ArtistGroup("A & B", [4, 5, 6], is_joint=True, source_artists=("A", "B")),
    ]
    with caplog.at_level("WARNING"):
        alloc = allocate_pier_budget(groups, 10, joint_pier_min_budget=3)
    assert alloc == {"A": 2, "B": 2, "A & B": 3}, "every group saturates at capacity"
    assert sum(alloc.values()) == 7, "7 is the true combined capacity"
    assert any("shortfall" in r.message for r in caplog.records), (
        "an unsatisfiable total must warn, never silently under-deliver"
    )


def test_joint_group_guaranteed_seat_survives_when_no_surplus_is_needed():
    """Confirms the guaranteed-one-seat behavior (Part A's floor, unchanged)
    still holds in the ordinary case where the exclusive groups can absorb
    the rest without any redistribution."""
    groups = [
        ArtistGroup("A", [0, 1, 2, 3]),
        ArtistGroup("B", [4, 5, 6, 7]),
        ArtistGroup("A & B", [8, 9], is_joint=True, source_artists=("A", "B")),
    ]
    alloc = allocate_pier_budget(groups, 6, joint_pier_min_budget=3)
    assert alloc["A & B"] == 1
    assert alloc["A"] + alloc["B"] == 5
    assert sum(alloc.values()) == 6


def test_alternation_prefers_the_interleaved_walk_when_sonic_cost_is_equal():
    """Four piers on a symmetric square: A0 A1 B0 B1. Several greedy walks have
    identical sonic cost (every edge is a 90-degree turn, cosine 0); the
    minimum-edge floor (review finding 1) does not exclude any of them here, so
    the alternating one must win outright, including the ``improved`` flag."""
    X = np.zeros((4, 2))
    X[0] = [1.0, 0.0]     # A0
    X[1] = [0.0, 1.0]     # A1
    X[2] = [-1.0, 0.0]    # B0
    X[3] = [0.0, -1.0]    # B1
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    group_of = {0: "A", 1: "A", 2: "B", 3: "B"}
    ordered, improved = order_with_alternation(
        [0, 1, 2, 3], X, group_of, alternation_bonus=0.15
    )
    labels = [group_of[i] for i in ordered]
    changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert changes >= 2, f"expected an interleaved order, got {labels}"
    assert set(ordered) == {0, 1, 2, 3}, "ordering must be a permutation"
    assert improved is True, (
        "the winner must differ from the default order_clusters walk here -- "
        "all four candidate walks tie on sonic cost (every edge cosine is 0), "
        "so the alternation bonus decides, and it must pick an interleaved one"
    )


def test_alternation_never_regresses_the_worst_edge():
    """Seeded random-search regression for review finding 1: the alternation
    bonus is a SUM-based score, and a sum can trade one wrecked edge for
    several slightly better ones -- exactly what design principle 5 ("the
    worst edge defines the experience") forbids. A pre-fix audit at the
    shipped default alternation_bonus=0.15 found this happening in ~6% of 500
    random trials (e.g. default min edge 0.084 -> winner min edge -0.102).
    The min-edge floor in order_with_alternation must make this impossible:
    the winner's worst edge must never be below the default walk's worst edge.
    """
    from src.playlist.artist_style import order_clusters

    rng = np.random.default_rng(12345)
    trials = 500
    regressions = 0
    for _ in range(trials):
        n = int(rng.integers(4, 9))
        dim = int(rng.integers(2, 6))
        X = rng.normal(size=(n, dim))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        idx = list(range(n))
        labels = rng.choice(["A", "B"], size=n)
        group_of = {i: str(labels[i]) for i in idx}

        default = order_clusters(idx, X)
        default_min = min(
            float(np.dot(X[a], X[b])) for a, b in zip(default, default[1:])
        )
        ordered, _ = order_with_alternation(
            idx, X, group_of, alternation_bonus=0.15
        )
        winner_min = min(
            float(np.dot(X[a], X[b])) for a, b in zip(ordered, ordered[1:])
        )
        if winner_min < default_min - 1e-9:
            regressions += 1

    assert regressions == 0, (
        f"{regressions}/{trials} seeded trials regressed the worst edge below "
        "the default walk's floor -- the min-edge floor is not being applied"
    )


def test_alternation_bonus_zero_reproduces_order_clusters():
    from src.playlist.artist_style import order_clusters
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    group_of = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}
    ordered, improved = order_with_alternation(
        [0, 1, 2, 3, 4, 5], X, group_of, alternation_bonus=0.0
    )
    assert ordered == order_clusters([0, 1, 2, 3, 4, 5], X)
    assert improved is False


def test_ordering_is_a_permutation_and_handles_degenerate_input():
    X = np.eye(3)
    group_of = {0: "A", 1: "B", 2: "A"}
    assert order_with_alternation([], X, group_of, alternation_bonus=0.15) == ([], False)
    single, improved = order_with_alternation([1], X, group_of, alternation_bonus=0.15)
    assert single == [1] and improved is False


def _blend_bundle():
    """12 tracks: 5 A, 5 B, 2 joint. A's index 4 and B's index 9 lean toward
    the other side.

    The orchestrator tests below pass ``pier_bridgeability_enabled=False`` --
    that veto needs >= pier_bridgeability_k (10) same-library non-seed-artist
    neighbors at a calibrated similarity floor before a candidate can seat as
    a medoid, which no 12-row toy library can ever supply. Every other
    small-fixture test in this suite (test_artist_style.py, test_tag_first_
    piers.py) disables it the same way; it is exercised on its own terms
    elsewhere and is orthogonal to what select_multi_artist_piers tests here."""
    artists = (
        ["A"] * 5 + ["B"] * 5 + ["A and B"] * 2
    )
    sonic = [
        [1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.5, 0.6],   # A
        [0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.6, 0.5],   # B
        [0.7, 0.7], [0.72, 0.68],                                          # joint
    ]
    b = _sonic_bundle(artists, sonic)
    b.durations_ms = np.full(len(artists), 240000, dtype=float)
    return b


def test_orchestrator_raises_loudly_on_missing_x_sonic_not_as_a_thin_group():
    """Coordinator review Finding 1: cluster_artist_tracks raises ValueError
    for FOUR different reasons, only two of which (too-few-tracks, degenerate
    clustering) are genuinely per-group. A missing X_sonic/artist_keys is an
    artifact-integrity problem, not "this artist is thin", and must never be
    caught by the per-group try/except and reported to the user as a
    thin-group relaxation -- it must raise immediately, before any group is
    even attempted."""
    b = _blend_bundle()
    b.X_sonic = None
    with pytest.raises(ValueError, match="X_sonic"):
        select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"],
            style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
            ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
        )


def test_orchestrator_returns_none_below_two_groups():
    b = _blend_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A"], style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
    )
    assert out is None, "one artist must fall back to the single-artist path"


def test_orchestrator_seats_both_artists_and_the_joint_group():
    b = _blend_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"], style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
    )
    assert isinstance(out, MultiArtistPiers)
    assert len(out.ordered_medoids) >= 3
    labels = {g.label for g in out.groups}
    assert "A" in labels and "B" in labels
    assert any(g.is_joint for g in out.groups)
    # both named artists are blocked from interiors
    assert len(out.blocked_artist_keys) >= 2


def test_orchestrator_reports_the_thin_joint_group_by_name():
    """Coordinator review Finding 3: the 2-track 'A and B' joint group in
    _blend_bundle is below cluster_k_min=3 in EVERY orchestrator test above,
    so the except-ValueError branch added to keep a thin group from crashing
    the whole blend fires on every single run -- but nothing asserted on its
    output. Pin the relaxation it must produce: naming the joint group and
    explaining why it contributed no piers."""
    b = _blend_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
    )
    thin = [r for r in out.relaxations if r.get("bridge") == "A & B"]
    assert thin, f"expected a relaxation naming the joint group, got {out.relaxations}"
    assert any("too few to cluster" in s for s in thin[0]["relaxed"])


def test_orchestrator_raises_with_relaxations_when_every_group_fails_to_cluster():
    """Coordinator review Finding 2: >=2 groups survive partition (so this is
    NOT the <2-groups fallback) but every one is too thin to cluster -- e.g.
    two obscure chips with 1-2 tracks each and no joint catalog. Returning
    None here would reuse the same signal as "no attempt was made" and
    silently discard the relaxations already collected. select_multi_artist_
    piers must instead raise MultiArtistBlendFailed carrying every relaxation
    collected so far, so a caller can surface WHY before falling back."""
    artists = ["A", "A", "B"]  # A: 2 tracks, B: 1 track, no joint overlap
    sonic = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]
    b = _sonic_bundle(artists, sonic)
    b.durations_ms = np.full(len(artists), 240000, dtype=float)

    with pytest.raises(MultiArtistBlendFailed) as excinfo:
        select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"],
            style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
            ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
        )
    relaxations = excinfo.value.relaxations
    assert len(relaxations) >= 2, "both thin groups must be individually reported"
    assert any(r.get("bridge") == "A" for r in relaxations)
    assert any(r.get("bridge") == "B" for r in relaxations)


def test_orchestrator_reports_a_dropped_chip():
    b = _blend_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B", "Nobody"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False), ma_cfg=MultiArtistConfig(),
        track_count=30, max_artist_fraction=0.125,
    )
    assert any("Nobody" in str(r) for r in out.relaxations), \
        "a dropped chip must be reported to the user, not just logged"


def test_orchestrator_reports_low_overlap():
    b = _blend_bundle()
    # Threshold above anything achievable -> always reports.
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"], style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(low_overlap_threshold=99.0),
        track_count=30, max_artist_fraction=0.125,
    )
    # Coordinator review Finding 4: the relaxation renders as
    # "Relaxed to fit: {bridge} — dropped {relaxed}" in the real UI, so the
    # copy lives in "bridge"/"relaxed" as a short label + noun phrase, not a
    # free-text sentence -- check the fields the renderer actually reads.
    assert any("middle ground" in str(r).lower() or "affinity" in str(r).lower()
               for r in out.relaxations)


def _joint_only_bundle():
    """B has an exclusive catalog; A is credited ONLY jointly with B -- the
    real-library shape final-review Finding 1 protects (e.g. the real
    library's "John Foxx and Harold Budd" dominating a thin Foxx solo
    catalog). Reuses _blend_bundle's A/B sonic layout, but every "A" row is
    relabeled to the joint credit "A and B" instead of a solo "A" row -- A
    never appears alone anywhere in this library."""
    artists = ["B"] * 5 + ["A and B"] * 5
    sonic = [
        [0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.6, 0.5],   # B
        [1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.5, 0.6],   # joint (A and B)
    ]
    b = _sonic_bundle(artists, sonic)
    b.durations_ms = np.full(len(artists), 240000, dtype=float)
    return b


def test_joint_only_chip_still_blends_and_is_blocked():
    """Final-review Finding 1: a chip credited ONLY jointly with the other
    chip (no solo catalog at all) must not silently collapse the blend to
    single-artist mode. groups = [B(exclusive), Joint(A&B)] is 2 real,
    pier-producing groups -- the old `len(exclusive) < 2` gate (`exclusive`
    counted only [B]) wrongly returned None here with nothing dropped and
    nothing to explain to the user. It must now proceed, actually seat a
    joint-group pier, and block BOTH named artists from bridge interiors:
    `_blocked_artist_keys` used to also be called with `exclusive`, so A's
    key never reached the block set and A's own tracks could have seated as
    bridge filler inside its own blend."""
    from src.string_utils import normalize_artist_key

    b = _joint_only_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=15, max_artist_fraction=0.125,
    )
    assert isinstance(out, MultiArtistPiers), "must produce a blend, not collapse to None"
    joint = next((g for g in out.groups if g.is_joint), None)
    assert joint is not None, "the joint group must survive partition"
    assert any(m in set(joint.indices) for m in out.ordered_medoids), (
        "the joint group must actually seat at least one pier, not just "
        "survive partition"
    )
    assert normalize_artist_key("A") in out.blocked_artist_keys, (
        "the joint-only artist must still be blocked from bridge interiors"
    )
    assert normalize_artist_key("B") in out.blocked_artist_keys
