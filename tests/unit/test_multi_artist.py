"""Unit tests for multi-artist blend (spec 2026-07-29-multi-artist-blend-design.md).

Pure-function tests only: synthetic numpy arrays and a stub bundle, no DB and no
artifact. Live-artifact acceptance lives in
tests/integration/test_multi_artist_generation.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass as _dc

import numpy as np
import pytest

from src.playlist.artist_style import ArtistStyleConfig, _select_k
from src.playlist.multi_artist import (
    PIER_ADJACENCY_ALTERNATION_FLOOR,
    ArtistGroup,
    MultiArtistBlendFailed,
    MultiArtistConfig,
    MultiArtistPiers,
    _blocked_artist_keys,
    _enumerate_best_orders_by_level,
    _max_achievable_alternation,
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
    """The feature ships ON (principle 22); it is inert below two chips anyway.

    ``alternation_bonus`` is no longer a field (2026-07-30 forced-interleaving
    rewrite): alternation is a hard constraint in order_with_alternation now,
    not a scored preference, so there is no bonus weight left to read. A
    leftover ``multi_artist.alternation_bonus`` key in a user's config.yaml is
    caught and warned on loudly by
    src.playlist_gui.worker._RETIRED_MULTI_ARTIST_KEYS.
    """
    cfg = multi_artist_config_from_ds({})
    assert cfg.enabled is True
    assert cfg.overlap_weight == pytest.approx(0.6)
    assert cfg.genre_share == pytest.approx(0.25)
    assert cfg.max_artists == 4
    assert cfg.joint_pier_min_budget == 3
    assert cfg.low_overlap_threshold == pytest.approx(0.15)
    assert cfg.pier_adjacency_floor == pytest.approx(PIER_ADJACENCY_ALTERNATION_FLOOR)
    assert not hasattr(cfg, "alternation_bonus")


def test_config_reads_overrides():
    cfg = multi_artist_config_from_ds(
        {"multi_artist": {"enabled": False, "overlap_weight": 0.0, "max_artists": 2}}
    )
    assert cfg.enabled is False
    assert cfg.overlap_weight == pytest.approx(0.0)
    assert cfg.max_artists == 2
    # untouched keys keep their defaults
    assert cfg.genre_share == pytest.approx(0.25)


def test_config_reads_pier_adjacency_floor_override():
    cfg = multi_artist_config_from_ds(
        {"multi_artist": {"pier_adjacency_floor": 0.15}}
    )
    assert cfg.pier_adjacency_floor == pytest.approx(0.15)


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


# ---------------------------------------------------------------------------
# Forced-interleaving rewrite (2026-07-30, docs/superpowers/sdd/
# 2026-07-29-multi-artist-blend/forced-interleaving-report.md), TWO passes.
#
# Pass 1: alternation used to be a scored PREFERENCE (`sonic path cost +
# alternation_bonus * artist changes`) applied over order_clusters' own
# candidate set -- but every candidate in that set was a greedy nearest-
# neighbor walk, which structurally clumps same-artist piers, so no bonus
# weight could ever produce a genuinely interleaved order (real repro: Vegyn
# + Black Moth Super Rainbow, 6 piers, shipped 1/5 alternation). Fixed by
# forcing the theoretical maximum alternation, with an all-or-nothing safety
# valve that fell straight to the fully-unconstrained order when the maximum
# breached a floor.
#
# Pass 2 (coordinator follow-up, same date): that safety valve was a CLIFF,
# not a ladder -- it caused a real regression (Eno+Bowie went from a
# pre-rewrite 2/5 down to 1/5, the opposite of the point of this feature),
# and it reused the final-playlist floor (ABSOLUTE_MIN_TRANSITION_FLOOR,
# 0.40) against PIER-TO-PIER adjacency, a different and naturally much lower
# quantity. Fixed with (a) a LADDER: try the theoretical max, then max-1,
# max-2, ... taking the HIGHEST level that clears the floor, falling back to
# the fully-unconstrained order only when no level clears it; and (b) a
# separate ``pier_adjacency_floor`` (``PIER_ADJACENCY_ALTERNATION_FLOOR``),
# calibrated on the pier-adjacency scale, not reusing the final-playlist one.
# ``order_with_alternation`` now returns a 4-tuple
# ``(ordered, improved, achieved_alt, max_alt)`` so the caller can report a
# relaxation notice when the ladder had to step down.
#
# Tests removed as part of Pass 1 (documented, not silently dropped):
#   - test_alternation_bonus_zero_reproduces_order_clusters: tested the
#     alternation_bonus=0.0 "disable" escape hatch, which no longer exists --
#     alternation is unconditional now (governed by group-size geometry, not
#     a tunable weight).
# ---------------------------------------------------------------------------


def test_max_achievable_alternation_matches_the_closed_form():
    """No majority group -> full alternation (n-1). A majority group forces
    ``2*m - n - 1`` repeats, regardless of how the rest split across groups."""
    assert _max_achievable_alternation([3, 3]) == 5       # full alternation
    assert _max_achievable_alternation([4, 2]) == 4        # one forced repeat
    assert _max_achievable_alternation([2, 2, 2]) == 5     # no majority -> full
    assert _max_achievable_alternation([5, 1, 1]) == 4     # majority forces 2 repeats
    assert _max_achievable_alternation([1]) == 0
    assert _max_achievable_alternation([]) == 0


# Shared geometry for the ladder/floor tests below: four piers on a symmetric
# square, A0 A1 B0 B1 at 0/90/180/270 degrees. Exact per-level pier-adjacency
# table (measured via _enumerate_best_orders_by_level): alt=3 (full
# alternation) -> worst edge -1.0 (every fully-alternating order includes at
# least one directly-opposite pair); alt=2 -> 0.0; alt=1 (the clump) -> 0.0.
# alt=0 is unreachable (2 distinct groups always need >= 1 change).
def _square_geometry():
    X = np.zeros((4, 2))
    X[0] = [1.0, 0.0]     # A0
    X[1] = [0.0, 1.0]     # A1
    X[2] = [-1.0, 0.0]    # B0
    X[3] = [0.0, -1.0]    # B1
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    group_of = {0: "A", 1: "A", 2: "B", 3: "B"}
    return X, group_of


def test_forced_alternation_reaches_the_theoretical_maximum_when_the_floor_allows():
    """The square's full-alternation level (-1.0) is far below any real
    floor, so this test bypasses it (pier_adjacency_floor=-2.0) to isolate
    the forcing behavior itself; the floor's own ladder/fallback behavior is
    covered separately below.
    """
    X, group_of = _square_geometry()
    ordered, improved, achieved_alt, max_alt = order_with_alternation(
        [0, 1, 2, 3], X, group_of, pier_adjacency_floor=-2.0,
    )
    assert max_alt == 3
    assert achieved_alt == 3, f"expected full (3/3) alternation, got {achieved_alt}"
    assert set(ordered) == {0, 1, 2, 3}, "ordering must be a permutation"
    assert improved is True, (
        "the plain order_clusters default walk clumps (A0 A1 B0 B1, 1/3 "
        "alternation) -- forcing the maximum must differ from it"
    )


def test_ladder_steps_down_one_level_at_a_time_not_straight_to_the_floor():
    """THE PASS-2 REGRESSION THIS FIXES: a floor between the square's alt=3
    (-1.0, broken) and alt=2/alt=1 (both 0.0) must land on alt=2 -- the
    HIGHEST level clearing it -- never skip straight past it to the fully-
    unconstrained alt=1 clump just because alt=1 ALSO clears the same floor.
    Before the ladder fix, the safety valve only ever knew "the max" and
    "the unconstrained best"; it had no notion of an intermediate level.
    """
    X, group_of = _square_geometry()
    ordered, improved, achieved_alt, max_alt = order_with_alternation(
        [0, 1, 2, 3], X, group_of, pier_adjacency_floor=-0.5,
    )
    assert max_alt == 3
    assert achieved_alt == 2, (
        f"expected the ladder to land on the intermediate level 2 (which "
        f"clears -0.5), not skip to 1 or fail up at 3, got {achieved_alt}"
    )
    chosen_min = min(float(np.dot(X[a], X[b])) for a, b in zip(ordered, ordered[1:]))
    assert chosen_min == pytest.approx(0.0)
    assert improved is True


def test_pier_adjacency_floor_falls_back_when_no_level_clears_it_and_warns(caplog):
    """With the real default pier_adjacency_floor (0.08), every level on the
    square (-1.0, 0.0, 0.0) falls short -- the ladder must exhaust every
    level, fall back to the single best-available order (alt=1, the clump,
    worst edge 0.0), and log a WARNING naming the floor and the rejected
    top-level worst edge.
    """
    X, group_of = _square_geometry()
    with caplog.at_level(logging.WARNING):
        ordered, improved, achieved_alt, max_alt = order_with_alternation(
            [0, 1, 2, 3], X, group_of,
        )
    assert max_alt == 3
    assert achieved_alt < max_alt, (
        f"no level on this geometry clears the default floor -- must NOT "
        f"ship the forced maximum, got {achieved_alt}/{max_alt}"
    )
    chosen_min = min(float(np.dot(X[a], X[b])) for a, b in zip(ordered, ordered[1:]))
    assert chosen_min == pytest.approx(0.0), (
        f"expected the floor fallback to land on the best-available (0.0) "
        f"worst edge, got {chosen_min:.4f}"
    )
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "no alternation level" in m and "-1.0000" in m and "0.08" in m
        for m in warnings
    ), f"expected a WARNING naming both the rejected top level and the floor: {warnings}"


def test_floor_never_leaves_a_rescuable_level_unreached():
    """Seeded random-search regression: whenever SOME alternation level
    clears ``PIER_ADJACENCY_ALTERNATION_FLOOR``, the ladder must land on one
    that does (never leave a rescuable order behind); when nothing clears it,
    the chosen order must still be the single best available regardless of
    alternation. ``_enumerate_best_orders_by_level`` (the same function
    order_with_alternation itself uses for pier counts in this range) is the
    oracle for "the best achievable at every level" on each trial.
    """
    rng = np.random.default_rng(12345)
    trials = 300
    for _ in range(trials):
        n = int(rng.integers(4, 7))  # stays well inside the exact-enumeration cap
        dim = int(rng.integers(2, 6))
        X = rng.normal(size=(n, dim))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        idx = list(range(n))
        labels = rng.choice(["A", "B"], size=n)
        group_of = {i: str(labels[i]) for i in idx}

        by_level = _enumerate_best_orders_by_level(idx, X, group_of)
        unconstrained_min = max(mn for _order, mn, _cost in by_level.values())

        ordered, _improved, _achieved, _max_alt = order_with_alternation(idx, X, group_of)
        chosen_min = min(
            float(np.dot(X[a], X[b])) for a, b in zip(ordered, ordered[1:])
        )
        target = min(PIER_ADJACENCY_ALTERNATION_FLOOR, unconstrained_min)
        assert chosen_min >= target - 1e-9, (
            f"chosen worst edge {chosen_min:.4f} fell short of the achievable "
            f"target {target:.4f} (floor={PIER_ADJACENCY_ALTERNATION_FLOOR}, "
            f"unconstrained best={unconstrained_min:.4f})"
        )


def test_bounded_beam_path_still_forces_full_alternation_above_the_cap():
    """Above _FULL_ENUMERATION_PIER_CAP (8 piers), order_with_alternation
    switches to the bounded beam search -- this pins that it still reaches
    the true theoretical max alternation (not merely 'close enough') for a
    group-size split with no majority (4+3+3 of 10 -- 4 <= 10/2), where full
    alternation is achievable.
    """
    rng = np.random.default_rng(7)
    n = 10
    X = rng.normal(size=(n, 4))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    idx = list(range(n))
    label_list = ["A"] * 4 + ["B"] * 3 + ["C"] * 3
    group_of = {i: label_list[i] for i in idx}

    ordered, _improved, achieved_alt, max_alt = order_with_alternation(
        idx, X, group_of, pier_adjacency_floor=-2.0,
    )
    assert set(ordered) == set(idx), "must remain a permutation"
    assert max_alt == 9
    assert achieved_alt == 9, f"expected full alternation (9/9), got {achieved_alt}"


def test_logs_alternation_summary_at_info(caplog):
    """Requirement: log what happened on every call, not just the fallback
    case -- achieved alternation vs the theoretical max, the chosen order's
    worst edge, and the unconstrained-best worst edge."""
    X = np.eye(4)
    group_of = {0: "A", 1: "A", 2: "B", 3: "B"}
    with caplog.at_level(logging.INFO):
        order_with_alternation([0, 1, 2, 3], X, group_of)
    assert any(
        "Multi-artist ordering: alternation" in r.message
        and "theoretical max" in r.message
        for r in caplog.records
    ), f"expected an INFO summary line on every call: {[r.message for r in caplog.records]}"


def test_ordering_is_a_permutation_and_handles_degenerate_input():
    X = np.eye(3)
    group_of = {0: "A", 1: "B", 2: "A"}
    assert order_with_alternation([], X, group_of) == ([], False, 0, 0)
    single, improved, achieved_alt, max_alt = order_with_alternation([1], X, group_of)
    assert single == [1] and improved is False and achieved_alt == 0 and max_alt == 0


# ---------------------------------------------------------------------------
# Finding-1 fix (xhigh code review, docs/superpowers/sdd/xhigh-review-fixes/
# finding-1-report.md): pier_support_terminal_avoidance used to run as a
# SEPARATE post-hoc reorder_avoiding_low_support_terminal call in
# playlist_generator.py, AFTER order_with_alternation had already forced the
# highest safe artist alternation. That helper's own preference is a greedy
# nearest-neighbour search over the pier set -- the same heuristic
# order_clusters uses, which clumps same-artist piers by construction (the
# precise reason the forced-interleaving rewrite above exists) -- so it
# silently discarded the alternation just forced (measured: Vegyn + Black
# Moth Super Rainbow went from a forced 5/5 down to 2/5 on the real library).
# Fixed by folding the SAME preference into order_with_alternation's own
# per-level search, as a tie-break applied AFTER the minimax worst-edge
# criterion and BEFORE the sonic-cost tie-break, so it can change which tied
# order wins a level but never which level -- or which worst edge -- wins.
# ---------------------------------------------------------------------------


def test_terminal_avoidance_prefers_orders_that_avoid_the_lowest_support_pier():
    """Square geometry (see ``_square_geometry``): full alternation (alt=3)
    has 8 tied orders at worst edge -1.0. Among those, the pre-existing
    sonic-cost tie-break alone prefers a subset that INCLUDES orders seating
    B0 (index 2, given the lowest support here) at a terminal seat -- e.g.
    (A0, B1, A1, B0) has the same top cost (-1) as (A1, B0, A0, B1), which
    avoids B0 entirely. Folding in the terminal-support preference must pick
    one of the latter, without changing achieved alternation or worst edge.
    """
    X, group_of = _square_geometry()
    support = {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0}  # index 2 (B0) is the outlier
    ordered, _improved, achieved_alt, max_alt = order_with_alternation(
        [0, 1, 2, 3], X, group_of, pier_adjacency_floor=-2.0, support_by_index=support,
    )
    assert achieved_alt == 3 == max_alt, (
        "the terminal-support preference must never cost alternation"
    )
    chosen_min = min(float(np.dot(X[a], X[b])) for a, b in zip(ordered, ordered[1:]))
    assert chosen_min == pytest.approx(-1.0), (
        "the terminal-support preference must never cost worst edge"
    )
    assert ordered[0] != 2 and ordered[-1] != 2, (
        f"the lowest-support pier (index 2) landed at a terminal seat even "
        f"though a same-alternation-level order avoiding it exists: {ordered}"
    )


def test_terminal_avoidance_is_a_noop_without_a_support_signal():
    """``support_by_index=None`` (the default -- no support computed, or
    ``pier_support_terminal_avoidance`` off) must reproduce the exact
    pre-finding-1 output: same order, same achieved alternation."""
    X, group_of = _square_geometry()
    with_none = order_with_alternation(
        [0, 1, 2, 3], X, group_of, pier_adjacency_floor=-2.0, support_by_index=None,
    )
    without_kwarg = order_with_alternation(
        [0, 1, 2, 3], X, group_of, pier_adjacency_floor=-2.0,
    )
    assert with_none == without_kwarg


def test_terminal_avoidance_never_costs_alternation_or_worst_edge():
    """Property test (mirrors ``test_floor_never_leaves_a_rescuable_level_
    unreached``'s random-trial style): across random geometries and random
    support signals, adding a terminal-support preference must never change
    the achieved alternation or the chosen worst edge that the very same call
    without a support signal would have produced -- only WHICH tied order (if
    any) is picked."""
    rng = np.random.default_rng(2026)
    trials = 200
    for _ in range(trials):
        n = int(rng.integers(4, 7))
        dim = int(rng.integers(2, 6))
        X = rng.normal(size=(n, dim))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        idx = list(range(n))
        labels = rng.choice(["A", "B"], size=n)
        group_of = {i: str(labels[i]) for i in idx}
        support = {i: float(rng.random()) for i in idx}

        baseline_ordered, _bi, baseline_alt, baseline_max = order_with_alternation(
            idx, X, group_of,
        )
        baseline_min = min(
            float(np.dot(X[a], X[b])) for a, b in zip(baseline_ordered, baseline_ordered[1:])
        )

        supported_ordered, _si, supported_alt, supported_max = order_with_alternation(
            idx, X, group_of, support_by_index=support,
        )
        supported_min = min(
            float(np.dot(X[a], X[b])) for a, b in zip(supported_ordered, supported_ordered[1:])
        )

        assert supported_max == baseline_max
        assert supported_alt == baseline_alt, (
            f"terminal-support preference changed achieved alternation: "
            f"{baseline_alt} -> {supported_alt} (n={n}, group_of={group_of})"
        )
        assert supported_min == pytest.approx(baseline_min), (
            f"terminal-support preference changed worst edge: "
            f"{baseline_min:.4f} -> {supported_min:.4f} (n={n}, group_of={group_of})"
        )


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


# ── Hard pier duration gate on the multi-artist path (2026-07-30 fix) ────
# select_multi_artist_piers routes every group through cluster_artist_tracks
# (artist_style.py), which is where the hard min-duration floor lives -- this
# proves the gate reaches the multi-artist path too, not just single-artist.

def test_min_pier_duration_seconds_drops_sub_minimum_pier_from_blend():
    """Index 2 ("A") is a real pier in the ungated baseline. Sub-minimum it
    (30s) and re-run with the gate: it must be gone from the piers, replaced
    by a different eligible "A" candidate, and the blend must still succeed
    (this is one short track among five, not total starvation)."""
    b = _blend_bundle()
    baseline = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
    )
    assert 2 in baseline.ordered_medoids, "test fixture assumption changed -- update the index"

    b2 = _blend_bundle()
    b2.durations_ms[2] = 30_000.0
    gated = select_multi_artist_piers(
        bundle=b2, artist_names=["A", "B"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
        min_pier_duration_seconds=46,
    )
    assert 2 not in gated.ordered_medoids, "sub-minimum (30s < 46s) track seated as a pier"
    assert isinstance(gated, MultiArtistPiers), "blend must still succeed for the rest of the group"
    a_piers = [g for g in gated.groups if g.label == "A"]
    assert a_piers, "the 'A' group must still be present"


def test_min_pier_duration_seconds_starvation_warns_not_crashes(caplog):
    """Every 'A' track sub-minimum -> the group is starved to zero eligible
    tracks. Must not crash the whole blend and must not silently drop 'A': a
    WARNING names the artist and counts (artist_style's own gate) and the
    existing per-group relaxation mechanism reports it -- exactly the
    'existing thin-artist path' this fix must reuse, not replace. 'B' still
    seats piers."""
    import logging
    b = _blend_bundle()
    b.durations_ms[0:5] = 30_000.0  # all five "A" tracks sub-minimum

    with caplog.at_level(logging.WARNING):
        out = select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"],
            style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
            ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
            min_pier_duration_seconds=46,
        )
    assert isinstance(out, MultiArtistPiers), "one starved group must not crash the whole blend"
    assert not any(g.label == "A" and not g.is_joint for g in out.groups) or \
        all(m not in out.ordered_medoids for m in range(0, 5)), \
        "no 'A' track should have seated as a pier"
    b_piers = [g for g in out.groups if g.label == "B"]
    assert b_piers and any(i in out.ordered_medoids for i in b_piers[0].indices), \
        "'B' must still contribute piers"

    gate_warning = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Pier duration gate" in r.getMessage()
        and "starved" in r.getMessage()
    ]
    assert gate_warning, f"expected a starvation WARNING naming the artist; got {[r.getMessage() for r in caplog.records]}"
    thin_relaxation = [r for r in out.relaxations if r.get("bridge") == "A"]
    assert thin_relaxation, f"expected a relaxation reporting 'A' contributed no piers, got {out.relaxations}"


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


# ── Coordinator review 2026-07-30: findings 1, 6, 7, 8, 10 ──────────────────


def test_blocked_artist_keys_use_the_identity_keys_space():
    """Review Finding 1: _blocked_artist_keys used to key chip names via
    normalize_artist_key (punctuation-only), but the consumer --
    pier_bridge_builder._row_ok, via _derive_seed_artist_keys's default
    derivation -- compares candidate rows through
    identity_keys_for_index(...).artist_key, which normalizes through
    normalize_primary_artist_key (ensemble-suffix and collaboration
    stripping). "Bill Evans Trio" is in the MISMATCH class verified against
    the real library: normalize_artist_key keeps the ensemble suffix
    ("bill evans trio"), which never matches a real "Bill Evans Trio" row's
    actual identity key ("bill evans") -- silently making
    disallow_seed_artist_in_interiors inert for this chip. Every fixture name
    used elsewhere in this test module ("A", "B", "Brian Eno", ...) happens to
    be identical in both key spaces, which is why the suite stayed green while
    this was broken.
    """
    from src.playlist.identity_keys import normalize_primary_artist_key
    from src.string_utils import normalize_artist_key

    groups = [
        ArtistGroup(label="Bill Evans Trio", indices=[0, 1], is_joint=False),
        ArtistGroup(label="Miles Davis", indices=[2, 3], is_joint=False),
    ]
    blocked = _blocked_artist_keys(groups)

    assert normalize_primary_artist_key("Bill Evans Trio") in blocked
    assert "bill evans" in blocked, f"expected the row-space key, got {blocked}"
    old_wrong_key = normalize_artist_key("Bill Evans Trio")
    assert old_wrong_key not in blocked, (
        f"the OLD (wrong) key space must not appear: {old_wrong_key!r} in {blocked}"
    )


def test_k_predict_matches_the_select_k_heuristic_not_the_raw_ceiling():
    """Review Finding 6: k_predict must predict the SAME k cluster_artist_tracks
    actually uses (_select_k(len(g.indices), style_cfg)) -- min(cluster_k_max,
    len(g.indices)) directly over-predicts k for any group thinner than the
    ceiling, which under-predicts medoid_top_k and makes cluster_artist_tracks
    seat fewer piers than requested, reported back as a false 'dropped N
    piers' scarcity."""
    import math

    import src.playlist.artist_style as art_mod

    b = _blend_bundle()  # group "A" has 5 tracks
    style_cfg = ArtistStyleConfig(
        enabled=True, pier_bridgeability_enabled=False,
        cluster_k_min=3, cluster_k_max=6, cluster_k_heuristic_enabled=True,
    )
    captured = []
    real_cluster_artist_tracks = art_mod.cluster_artist_tracks

    def spy(**kwargs):
        captured.append(kwargs)
        return real_cluster_artist_tracks(**kwargs)

    import unittest.mock as mock
    with mock.patch.object(art_mod, "cluster_artist_tracks", side_effect=spy):
        select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"], style_cfg=style_cfg,
            ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
        )
    a_call = next(c for c in captured if c["artist_name"] == "A")
    expected_k = _select_k(5, style_cfg)
    assert expected_k != min(int(style_cfg.cluster_k_max), 5), (
        "fixture assumption changed -- pick cfg/group size where the two "
        "formulas actually diverge"
    )
    expected_medoid_top_k = max(1, math.ceil(a_call["target_pier_count"] / expected_k))
    assert a_call["medoid_top_k"] == expected_medoid_top_k, (
        f"k_predict must use _select_k({5}, cfg)={expected_k}, not "
        f"min(cluster_k_max, 5)={min(int(style_cfg.cluster_k_max), 5)}"
    )


def test_bridgeability_excluded_indices_spans_every_group():
    """Review Finding 7: the pier-bridgeability veto's same-artist exclusion
    set must span every group in the blend, not just the current group's own
    -- otherwise another chip's rows (which can never serve as bridge fill
    once disallow_seed_artist_in_interiors blocks every chip key) get counted
    as valid bridge neighbours."""
    import unittest.mock as mock

    import src.playlist.artist_style as art_mod

    b = _blend_bundle()  # 5 A (0-4), 5 B (5-9), 2 joint (10-11)
    captured = []
    real_cluster_artist_tracks = art_mod.cluster_artist_tracks

    def spy(**kwargs):
        captured.append(kwargs)
        return real_cluster_artist_tracks(**kwargs)

    with mock.patch.object(art_mod, "cluster_artist_tracks", side_effect=spy):
        select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"],
            style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
            ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
        )
    a_call = next(c for c in captured if c["artist_name"] == "A")
    b_call = next(c for c in captured if c["artist_name"] == "B")
    assert a_call["bridgeability_excluded_indices"] is not None
    a_excluded = set(int(i) for i in a_call["bridgeability_excluded_indices"])
    b_excluded = set(int(i) for i in b_call["bridgeability_excluded_indices"])
    # A's own exclusion set must reach beyond its own indices (0-4) into B's
    # (5-9) and the joint group's (10-11).
    assert a_excluded >= set(range(5, 12)), (
        f"A's bridgeability exclusion set must include B + joint rows, got {a_excluded}"
    )
    assert b_excluded >= set(range(0, 5)) | set(range(10, 12)), (
        f"B's bridgeability exclusion set must include A + joint rows, got {b_excluded}"
    )


def test_support_by_index_is_merged_across_groups_and_exposed():
    """Review Finding 8: cluster_artist_tracks already computes within-artist
    support for its own group's candidates -- select_multi_artist_piers used
    to discard it (`_support`). It must be merged across every group and
    exposed on MultiArtistPiers so the caller (playlist_generator.py) can run
    the same arc-aware terminal-avoidance reorder the single-artist tail
    applies (reorder_avoiding_low_support_terminal)."""
    b = _blend_bundle()
    out = select_multi_artist_piers(
        bundle=b, artist_names=["A", "B"],
        style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
        ma_cfg=MultiArtistConfig(), track_count=30, max_artist_fraction=0.125,
    )
    assert out.support_by_index, "support_by_index must be populated, not left empty"
    assert set(out.ordered_medoids) <= set(out.support_by_index), (
        "every seated pier must have a recorded support score"
    )


def test_group_medoids_are_sonic_sequenced_before_capping():
    """Review Finding 10: `chosen = list(medoids)[:want]` used to truncate in
    RAW k-means cluster order. This group's own candidate medoids must be
    ordered in sonic space FIRST (order_clusters -- the same primitive
    playlist_generator._cap_order uses for the single-artist path), THEN
    capped to `want`, so a well-connected medoid from a later cluster index
    is not dropped in favor of a poorly-connected one from an earlier index.
    """
    import unittest.mock as mock

    import src.playlist.artist_style as art_mod

    # Group "A"'s 3 candidate medoids (rows 0-2), in a raw cluster order that
    # is deliberately misleading: row 1 is FAR from row 0 (row 0's true
    # nearest neighbour is row 2, which comes LAST in raw order). Rows 3-4
    # belong to group "B".
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.95, 0.05],
        [0.0, -1.0],
        [-1.0, 0.0],
    ])
    b = _sonic_bundle(["A", "A", "A", "B", "B"], X.tolist())
    b.durations_ms = np.full(5, 240_000.0)
    X_norm_a = X / np.linalg.norm(X, axis=1, keepdims=True)

    real_cluster_artist_tracks = art_mod.cluster_artist_tracks

    def fake_cluster_artist_tracks(*, artist_name, target_pier_count, **kwargs):
        if artist_name == "A":
            medoids = [0, 1, 2]  # raw cluster order: 0, 1 (far), 2 (close-but-last)
            return [medoids], medoids, [medoids], X_norm_a, {i: 1.0 for i in medoids}
        return real_cluster_artist_tracks(
            artist_name=artist_name, target_pier_count=target_pier_count, **kwargs
        )

    with mock.patch.object(art_mod, "cluster_artist_tracks", side_effect=fake_cluster_artist_tracks):
        out = select_multi_artist_piers(
            bundle=b, artist_names=["A", "B"],
            style_cfg=ArtistStyleConfig(enabled=True, pier_bridgeability_enabled=False),
            ma_cfg=MultiArtistConfig(), track_count=16, max_artist_fraction=0.125,
        )
    a_group = next(g for g in out.groups if g.label == "A")
    assert len(a_group.indices) == 3, "fixture assumption changed -- update the test"
    a_piers = set(out.ordered_medoids) & {0, 1, 2}
    assert a_piers == {0, 2}, (
        f"expected A's sonically-nearest pair (row 0's true neighbour, row 2) to "
        f"survive the cap, got {a_piers} -- raw-cluster-order slicing would wrongly "
        f"keep {{0, 1}}"
    )
