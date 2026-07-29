"""Unit tests for multi-artist blend (spec 2026-07-29-multi-artist-blend-design.md).

Pure-function tests only: synthetic numpy arrays and a stub bundle, no DB and no
artifact. Live-artifact acceptance lives in
tests/integration/test_multi_artist_generation.py.
"""
from __future__ import annotations

from dataclasses import dataclass as _dc

import numpy as np
import pytest

from src.playlist.multi_artist import (
    ArtistGroup,
    group_genre_profiles,
    group_prototypes,
    multi_artist_config_from_ds,
    overlap_affinity,
    partition_artist_groups,
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
