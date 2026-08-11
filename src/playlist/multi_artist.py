"""Multi-artist blend: pier selection across 2+ named artists.

Spec: docs/superpowers/specs/2026-07-29-multi-artist-blend-design.md

Two or more artist chips make Artist mode build every pier from the region the
artists share. Soft-bias only: the overlap pull re-ranks medoid candidates, it
never gates or excludes, and a pairing with no shared ground still generates.

Everything here is pier SELECTION. The beam, the bridges, and the candidate pool
are untouched -- piers in the shared region mean the beam bridges through the
shared region for free. (Biasing the pool as well is what manufactured starvation
in the pre-corridor per-cluster external pool; see
docs/POOL_STARVATION_RESEARCH_2026-07-12.md.)
"""
from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The floor below which a pier-to-pier transition is a genuine break, not
# ordinary blend roughness -- shared with
# tests/integration/test_multi_artist_generation.py::
# test_worst_edge_stays_above_an_absolute_floor (which asserts real
# generations never fall below it) so the two never drift apart. This
# project's break-glass edge repair triggers at T < 0.30 (see
# src/playlist/... edge_repair); this floor sits between that and ordinary
# blend-run variation around 0.55 -- see that test's CALIBRATION RECORD for
# the measured numbers this was set against.
ABSOLUTE_MIN_TRANSITION_FLOOR = 0.40

# The floor order_with_alternation's ladder search applies to PIER-TO-PIER
# adjacency -- a different and naturally much lower quantity than the final
# playlist's ABSOLUTE_MIN_TRANSITION_FLOOR above, because piers are
# deliberately spread apart and the beam fills the gaps between them.
#
# CALIBRATION (2026-07-30, forced-interleaving ladder fix): measured the
# EXACT per-level pier-adjacency table (min edge at every alternation level
# from the theoretical max down to the lowest reachable) for three real
# pairings via `_enumerate_best_orders_by_level` on the real artifact/DB,
# `track_count=30`, `random_seed=0`:
#
#   Vegyn + Black Moth Super Rainbow (3+3, max_alt=5):
#     alt=5: 0.5133   alt=4: 0.5133   alt=3: 0.5647   alt=2: 0.5647   alt=1: 0.6028
#   Brian Eno + Harold Budd (3+2+1 incl. joint, max_alt=5):
#     alt=5: 0.0120   alt=4: 0.4768   alt=3: 0.4905   alt=2: 0.4905
#   Brian Eno + David Bowie (3+3, max_alt=5):
#     alt=5: 0.0413   alt=4: 0.0413   alt=3: 0.1022   alt=2: 0.1022   alt=1: 0.3680
#
# Two genuinely broken (near-orthogonal) levels: Eno+Budd's 5/5 (0.0120) and
# Eno+Bowie's 5/5 and 4/5 (0.0413) -- both must stay rejected. One level that
# MUST be admitted to avoid a real regression: Eno+Bowie's 3/5 and 2/5
# (0.1022) -- the pre-forced-interleaving code achieved 2/5 for this pairing,
# so a floor that rejects 0.1022 regresses it to 1/5 (worse than before the
# rewrite; this is exactly the bug this calibration fixes). `0.08` sits with
# ~2x margin above the highest rejected value (0.0413) and ~22% margin below
# the lowest value that must be admitted (0.1022) -- comfortably separated
# from both anchors without sitting on either boundary. Verified live: at
# 0.08 all three pairings' final playlist `min_transition` stayed >= 0.55,
# far clear of ABSOLUTE_MIN_TRANSITION_FLOOR (0.40) -- see
# docs/superpowers/sdd/2026-07-29-multi-artist-blend/forced-interleaving-report.md
# for the full measured table and the three post-calibration generations.
PIER_ADJACENCY_ALTERNATION_FLOOR = 0.08


@dataclass(frozen=True)
class MultiArtistConfig:
    enabled: bool = True
    overlap_weight: float = 0.6
    genre_share: float = 0.25
    max_artists: int = 4
    joint_pier_min_budget: int = 3
    low_overlap_threshold: float = 0.15
    pier_adjacency_floor: float = PIER_ADJACENCY_ALTERNATION_FLOOR


def multi_artist_config_from_ds(ds_cfg: Dict[str, Any]) -> MultiArtistConfig:
    """Build the config from the ``playlists.ds_pipeline`` dict.

    Mirrors how ArtistStyleConfig is assembled in playlist_generator.py (~L1842):
    plain .get() with the dataclass defaults as the fallback.

    ``alternation_bonus`` is intentionally NOT read here any more (2026-07-30,
    forced-interleaving rewrite): pier alternation is now a hard constraint in
    ``order_with_alternation``, not a scored preference, so there is no bonus
    weight left to tune. A ``multi_artist.alternation_bonus`` key surviving in
    a user's config.yaml is caught and warned on loudly by
    ``src.playlist_gui.worker._warn_retired_keys`` (the "configured knob that
    can't act is a startup error" rule) -- it must never silently do nothing.
    """
    raw = (ds_cfg or {}).get("multi_artist", {}) or {}
    return MultiArtistConfig(
        enabled=bool(raw.get("enabled", True)),
        overlap_weight=float(raw.get("overlap_weight", 0.6)),
        genre_share=float(raw.get("genre_share", 0.25)),
        max_artists=int(raw.get("max_artists", 4)),
        joint_pier_min_budget=int(raw.get("joint_pier_min_budget", 3)),
        low_overlap_threshold=float(raw.get("low_overlap_threshold", 0.15)),
        pier_adjacency_floor=float(
            raw.get("pier_adjacency_floor", PIER_ADJACENCY_ALTERNATION_FLOOR)
        ),
    )


@dataclass(frozen=True)
class ArtistGroup:
    """One pier-producing group: an artist's exclusive tracks, or the joint set."""
    label: str
    indices: List[int]
    is_joint: bool = False
    source_artists: tuple = ()


def partition_artist_groups(
    bundle,
    artist_names: Sequence[str],
    *,
    include_collaborations: bool = False,
    excluded_track_ids: Optional[set] = None,
) -> tuple[List[ArtistGroup], List[str]]:
    """Split bundle rows into one exclusive group per artist + one joint group.

    A row credited to 2+ of the named artists lands in the JOINT group only --
    never in an exclusive group. Joint detection between the named artists always
    runs: a track by both chips is the overlap by definition. The caller's
    ``include_collaborations`` still governs whether an exclusive group absorbs
    that artist's collaborations with THIRD parties.

    Returns ``(groups, dropped_names)``. A named artist with zero rows is dropped
    and reported, never raised on.
    """
    from src.playlist.artist_style import _artist_indices_in_bundle
    from src.playlist.history_analyzer import is_collaboration_of

    # A duplicate chip is user input, not an error -- dedupe silently, order
    # preserved, so a repeated name never produces two identical groups.
    names = list(
        dict.fromkeys(str(a).strip() for a in artist_names if str(a).strip())
    )
    excluded = {str(t) for t in (excluded_track_ids or set())}

    # Rows matching each named artist, including cross-chip collaborations
    # (include_collaborations=True) so the joint set is always discoverable.
    raw: Dict[str, set] = {}
    for name in names:
        idx = _artist_indices_in_bundle(bundle, name, include_collaborations=True)
        if excluded:
            idx = [i for i in idx if str(bundle.track_ids[i]) not in excluded]
        raw[name] = set(int(i) for i in idx)

    # A row claimed by 2+ chips is joint.
    claim_count: Dict[int, int] = {}
    for idx in raw.values():
        for i in idx:
            claim_count[i] = claim_count.get(i, 0) + 1
    joint_indices = sorted(i for i, n in claim_count.items() if n >= 2)
    joint_set = set(joint_indices)

    # Names that actually claimed a joint-credited row -- this is the joint
    # group's real provenance, independent of whether that name also ends up
    # with a surviving exclusive group (an artist can be joint-only).
    joint_contributors = [name for name in names if raw[name] & joint_set]

    groups: List[ArtistGroup] = []
    dropped: List[str] = []
    for name in names:
        exclusive = sorted(raw[name] - joint_set)
        if not include_collaborations:
            # Drop third-party collaborations the caller did not ask for. A row
            # whose raw credit is a collaboration but which is not joint with
            # another chip is only kept when include_collaborations is on.
            exclusive = [
                i for i in exclusive
                if not is_collaboration_of(
                    collaboration_name=str(bundle.track_artists[i] or ""),
                    base_artist=name,
                )
            ]
        if not exclusive:
            # Only a name that claimed no rows at all (not even jointly) is
            # genuinely absent from the pairing -- a joint-only artist IS
            # represented, via the joint group.
            if name not in joint_contributors:
                dropped.append(name)
                logger.warning(
                    "Multi-artist: '%s' has no usable tracks in the library — "
                    "dropped from the pairing.", name,
                )
            continue
        groups.append(ArtistGroup(label=name, indices=exclusive, is_joint=False))

    if joint_indices:
        groups.append(
            ArtistGroup(
                label=" & ".join(joint_contributors) if joint_contributors else "joint",
                indices=joint_indices,
                is_joint=True,
                source_artists=tuple(joint_contributors),
            )
        )
        logger.info(
            "Multi-artist: %d jointly-credited track(s) across %s",
            len(joint_indices), joint_contributors,
        )
    return groups, dropped


def group_prototypes(bundle, groups: Sequence[ArtistGroup]) -> Dict[str, np.ndarray]:
    """label -> centered, L2-normalized MuQ prototype for that group.

    Reuses tag_steering.sonic_prototype_from_rows: its global-mean centering
    removes the generic-sonic direction, which is what makes "closest to the
    OTHER artist" mean something rather than "closest to average music".

    An exclusive group's prototype is built from its own rows PLUS the joint
    rows -- jointly-credited work is part of both artists' identity.
    """
    from src.playlist.tag_steering import sonic_global_mean, sonic_prototype_from_rows

    X = getattr(bundle, "X_sonic", None)
    if X is None:
        raise ValueError("Artifact missing X_sonic — cannot build artist prototypes.")
    X = np.asarray(X, dtype=float)
    gmean = sonic_global_mean(X)

    joint_rows: List[int] = []
    for g in groups:
        if g.is_joint:
            joint_rows.extend(g.indices)

    protos: Dict[str, np.ndarray] = {}
    for g in groups:
        rows = list(g.indices) if g.is_joint else list(g.indices) + joint_rows
        proto, cohesion, n = sonic_prototype_from_rows(X, rows, global_mean=gmean)
        if proto is None:
            logger.warning(
                "Multi-artist: degenerate sonic prototype for '%s' (n=%d) — "
                "this group contributes no pull.", g.label, n,
            )
            continue
        protos[g.label] = proto
        logger.info(
            "Multi-artist prototype: %s n=%d cohesion=%.3f", g.label, n, cohesion,
        )
    return protos


def group_genre_profiles(bundle, groups: Sequence[ArtistGroup]) -> Dict[str, np.ndarray]:
    """label -> mean dense-genre row. Empty dict when X_genre_dense is absent."""
    xgd = getattr(bundle, "X_genre_dense", None)
    if xgd is None:
        return {}
    xgd = np.asarray(xgd, dtype=float)
    profiles: Dict[str, np.ndarray] = {}
    for g in groups:
        if not g.indices:
            continue
        profiles[g.label] = xgd[list(g.indices)].mean(axis=0)
    return profiles


def overlap_affinity(
    bundle,
    group: ArtistGroup,
    groups: Sequence[ArtistGroup],
    protos: Dict[str, np.ndarray],
    genre_profiles: Dict[str, np.ndarray],
    *,
    genre_share: float,
) -> np.ndarray:
    """Bundle-aligned (N,) pull of each of ``group``'s rows toward the OTHER groups.

        affinity = (1 - genre_share) * cos(muq_centered, proto_others)
                 +      genre_share  * genre_sim(dense_genre, profile_others)

    Zero outside ``group``. Zero everywhere when there is no other group, or when
    the other groups produced no prototype.
    """
    from src.playlist.candidate_pool import _compute_genre_similarity
    from src.playlist.tag_steering import sonic_global_mean

    X = np.asarray(getattr(bundle, "X_sonic"), dtype=float)
    out = np.zeros(X.shape[0], dtype=float)
    members = list(group.indices)
    if not members:
        return out

    other_labels = [
        g.label for g in groups if g.label != group.label and g.label in protos
    ]
    if not other_labels:
        return out

    others_proto = np.mean([protos[lbl] for lbl in other_labels], axis=0)
    n = float(np.linalg.norm(others_proto))
    if n <= 1e-12:
        logger.warning(
            "Multi-artist: the other artists' prototypes cancel out — "
            "no sonic pull for '%s'.", group.label,
        )
        return out
    others_proto = others_proto / n

    gmean = sonic_global_mean(X)
    Mn = X[members] / (np.linalg.norm(X[members], axis=1, keepdims=True) + 1e-12)
    sonic_term = (Mn - gmean) @ others_proto

    share = float(genre_share)
    xgd = getattr(bundle, "X_genre_dense", None)
    other_profiles = [genre_profiles[l] for l in other_labels if l in genre_profiles]
    if share > 0.0 and (xgd is None or not other_profiles):
        logger.warning(
            "Multi-artist: genre_share=%.2f but X_genre_dense (or the other "
            "artists' genre profile) is unavailable — renormalizing to pure-sonic "
            "for this run.", share,
        )
        share = 0.0

    if share > 0.0:
        profile = np.mean(other_profiles, axis=0)
        genre_term = _compute_genre_similarity(
            profile, np.asarray(xgd, dtype=float)[members], method="cosine",
        )
        combined = (1.0 - share) * sonic_term + share * genre_term
    else:
        combined = sonic_term

    out[members] = combined
    return out


def total_pier_budget(
    track_count: int, max_artist_fraction: float, n_groups: int
) -> int:
    """Total piers across all groups.

    ``max_artist_fraction`` (the Artist-presence dial) is the share of the
    PLAYLIST occupied by seed-artist tracks IN TOTAL -- not per chip. A
    100-track playlist at High (0.20) seats 20 seed tracks; two chips split them
    10/10. ``allocate_pier_budget`` gives the remainder to the first chip, so an
    odd total rounds that chip up. A blend therefore uses exactly the same
    formula as a single artist, which is what the GUI's "target share of the
    playlist from the seed artist" label has always promised.

    HUMAN RULING 2026-08-10 (Dylan): this replaces an ``n * base`` per-chip
    reading clamped by ``track_count // 5``. Two defects, one fix:

    1. The clamp made the dial completely INERT for 2+ chips. On a 30-track
       blend every position from Very Low to Very High returned the same 6
       piers, because ``n * base`` (>= 6 for two chips at any fraction, since
       ``base`` floors at 3) always exceeded the cap. A knob that cannot act is
       this codebase's #1 failure mode -- see CLAUDE.md principle 22.
    2. The per-chip reading the clamp was defending is undeliverable anyway:
       two chips at High wants 12 piers on 30 tracks, i.e. ~1.6 tracks per
       bridge segment -- well under the ~2.2 that Task 13 (task-13-report.md)
       measured as starved when it changed the clamp from ``// 3`` to ``// 5``.

    Task 13's starvation concern is now a DIAGNOSTIC rather than a silent cap
    (``warn_if_bridge_density_thin``): the dial is honoured and a thin-bridge
    blend says so in the log. A blend is never denser than a single-artist
    playlist at the same dial position, and single-artist has never been capped.
    """
    base = max(3, round(int(track_count) * float(max_artist_fraction)))
    return max(max(1, int(n_groups)), base)


#: Bridge tracks per segment at or below which a blend is called thin.
#: Task 13 measured the starved case at 10 piers on 30 tracks -- 20 interior
#: over 9 segments = 2.22, "almost no room to bridge" -- and the healthy case
#: at 6 piers = 24/5 = 4.8. 2.5 sits above the measured-starved figure (so it
#: actually fires on it) and well below the healthy one.
#: Diagnostic threshold only -- it warns, it never caps. See total_pier_budget.
THIN_BRIDGE_TRACKS_PER_SEGMENT = 2.5


def warn_if_bridge_density_thin(track_count: int, total_piers: int) -> Optional[float]:
    """Log a WARNING when a pier budget leaves bridge segments too thin to work.

    Returns the tracks-per-segment figure (None when there are no interior
    segments to measure). Purely diagnostic: the artist-presence dial is the
    user's call, and this makes the structural cost of the top of that dial
    visible instead of silently overriding it -- the failure mode the old
    ``track_count // 5`` clamp had.
    """
    segments = max(0, int(total_piers) - 1)
    if segments <= 0:
        return None
    interior = max(0, int(track_count) - int(total_piers))
    per_segment = interior / segments
    if per_segment <= THIN_BRIDGE_TRACKS_PER_SEGMENT:
        logger.warning(
            "Multi-artist bridge density: %d piers on a %d-track playlist leaves "
            "%.1f track(s) per bridge segment (at or below the %.1f Task 13 measured "
            "as starved). Seed-artist presence is being honoured as requested; expect "
            "rougher transitions. Lower artist presence or raise the track count for "
            "more room to bridge.",
            int(total_piers), int(track_count), per_segment,
            THIN_BRIDGE_TRACKS_PER_SEGMENT,
        )
    return per_segment


def allocate_pier_budget(
    groups: Sequence[ArtistGroup], total: int, *, joint_pier_min_budget: int,
    joint_min_cluster_size: int = 3,
) -> Dict[str, int]:
    """Split ``total`` piers across groups: even, remainder to the first chip.

    The joint group claims a GUARANTEED MINIMUM of one seat when it is
    non-empty, the total reaches ``joint_pier_min_budget``, AND it has at
    least ``joint_min_cluster_size`` tracks of its own (xhigh review Finding
    8: without this last check, a joint group of e.g. one shared
    collaboration track -- a common shape, two artists whose only overlap is
    a single feature -- claimed a guaranteed seat it could never fill,
    since ``cluster_artist_tracks`` needs ``max(3, cluster_k_min)`` tracks to
    cluster at all; the reserved seat then raised ValueError, was caught, and
    the shortfall misreported as the NAMED artists being thin rather than the
    joint group being uncastable). ``joint_min_cluster_size`` should be passed
    as ``max(3, style_cfg.cluster_k_min)`` -- the same floor
    ``cluster_artist_tracks`` applies -- so this can never diverge from what
    clustering will actually accept; the default of 3 matches
    ``ArtistStyleConfig.cluster_k_min``'s own default for callers (tests)
    that construct a budget without a live style config.

    A joint group below this floor is not blocked from piers entirely: it may
    still absorb SURPLUS during redistribution below if the exclusive groups
    saturate first (rare in practice, since a joint group too thin to cluster
    is almost always too thin to be worth surplus either) -- redistribution
    doesn't gate on the cluster floor because it is capped by the group's own
    track capacity already, and a group that ends up with a `want` it still
    can't cluster is reported by the caller's existing per-group
    except-ValueError path exactly like any other thin group.

    It may additionally absorb surplus during redistribution once the
    exclusive groups are saturated, subject to its own track capacity like
    any other group -- the joint group IS the overlap between the named
    artists, so for a pairing whose collaborative output dominates thin solo
    catalogs, it is the correct place for spare budget to land, not a
    one-seat ceiling.

    No group is ever allocated more piers than it has tracks. A shortfall
    against ``total`` after full redistribution reflects a genuine capacity
    limit (``total`` exceeds every group's tracks combined) -- it is always
    logged, never silently swallowed.
    """
    total = max(0, int(total))
    if not groups or total == 0:
        return {}

    alloc: Dict[str, int] = {g.label: 0 for g in groups}
    capacity: Dict[str, int] = {g.label: len(g.indices) for g in groups}
    remaining = total

    joint = next((g for g in groups if g.is_joint), None)
    if (
        joint is not None
        and joint.indices
        and len(joint.indices) >= int(joint_min_cluster_size)
        and total >= int(joint_pier_min_budget)
    ):
        alloc[joint.label] = 1
        remaining -= 1
        logger.info(
            "Multi-artist: reserved 1 pier for the jointly-credited group '%s'",
            joint.label,
        )
    elif joint is not None and joint.indices and len(joint.indices) < int(joint_min_cluster_size):
        logger.info(
            "Multi-artist: joint group '%s' has only %d track(s), below the "
            "clustering floor (%d) -- no guaranteed pier reserved for it (it "
            "cannot cluster regardless of how many piers it were given).",
            joint.label, len(joint.indices), int(joint_min_cluster_size),
        )

    exclusive = [g for g in groups if not g.is_joint and g.indices]
    if exclusive:
        # Even split, remainder to the first chip.
        per, rem = divmod(remaining, len(exclusive))
        for i, g in enumerate(exclusive):
            alloc[g.label] += per + (1 if i < rem else 0)
    elif joint is not None and joint.indices:
        # No exclusive groups at all: the rest of the budget goes to the
        # joint group (capped below like everyone else).
        alloc[joint.label] += remaining

    # Cap by available tracks, then redistribute the surplus.
    surplus = 0
    for label, want in list(alloc.items()):
        cap = capacity.get(label, 0)
        if want > cap:
            surplus += want - cap
            alloc[label] = cap
            logger.warning(
                "Multi-artist: '%s' has only %d track(s) — allocated %d pier(s) "
                "instead of %d.", label, cap, cap, want,
            )

    # Surplus redistributes to ANY group with headroom, including the joint
    # group beyond its guaranteed seat -- see the docstring rationale.
    seatable = [g for g in groups if g.indices]
    while surplus > 0:
        takers = [g.label for g in seatable if alloc[g.label] < capacity[g.label]]
        if not takers:
            break
        for label in takers:
            if surplus == 0:
                break
            alloc[label] += 1
            surplus -= 1

    delivered = sum(alloc.values())
    if delivered < total:
        caps_desc = ", ".join(f"{g.label}={capacity[g.label]}" for g in groups)
        logger.warning(
            "Multi-artist: requested %d pier(s) but only %d could be seated "
            "(shortfall %d) — group track capacities: %s.",
            total, delivered, total - delivered, caps_desc,
        )

    return {k: v for k, v in alloc.items() if v > 0}


def _path_sonic_cost(order: Sequence[int], X_norm: np.ndarray) -> float:
    """Sum of consecutive cosine similarities. Higher is a smoother walk."""
    if len(order) < 2:
        return 0.0
    return float(
        sum(np.dot(X_norm[a], X_norm[b]) for a, b in zip(order, order[1:]))
    )


def _alternation_count(order: Sequence[int], group_of: Dict[int, str]) -> int:
    if len(order) < 2:
        return 0
    labels = [group_of.get(int(i), "") for i in order]
    return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def _min_edge(order: Sequence[int], X_norm: np.ndarray) -> float:
    """Minimum consecutive cosine similarity along the walk -- its worst edge.

    ``float("inf")`` for a path with no edges (fewer than 2 stops), so it never
    fails a "no worse than" comparison by construction.
    """
    if len(order) < 2:
        return float("inf")
    return float(min(np.dot(X_norm[a], X_norm[b]) for a, b in zip(order, order[1:])))


def _max_achievable_alternation(group_sizes: Sequence[int]) -> int:
    """Theoretical max number of adjacent-artist changes for these group sizes.

    Not ``n - 1``: perfect alternation (every adjacent pair a change) is only
    possible when no single group holds a majority. This is the classic
    "rearrange to minimize adjacent duplicates" bound -- with the largest
    group size ``m`` and total ``n``, the minimum number of forced
    SAME-adjacent pairs is ``max(0, 2*m - n - 1)`` (each of the ``m`` items
    needs its own separator to avoid repeating; there are only ``n - m``
    separators to go around), so the max number of DIFFERENT-adjacent pairs
    is ``(n - 1) - that``. E.g. 3+3 -> 5 (fully alternating); 4+2 -> 4 (one
    forced repeat); 2+2+2 -> 5 (no majority, fully alternating regardless of
    how the non-majority piers split across groups).
    """
    n = sum(group_sizes)
    if n < 2:
        return 0
    m = max(group_sizes)
    return (n - 1) - max(0, 2 * m - n - 1)


def _alt_count_from_labels(labels: Sequence[str]) -> int:
    if len(labels) < 2:
        return 0
    return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def _label_sequence_at_level(
    counts: Dict[str, int],
    *,
    target_alt: Optional[int] = None,
    first: Optional[str] = None,
) -> List[str]:
    """A label sequence with (at most) ``target_alt`` adjacent-label changes,
    or the theoretical MAXIMUM achievable when ``target_alt`` is ``None``.

    ``target_alt=None`` is the classic exchange-argument greedy for "minimize
    adjacent duplicates" -- proven optimal, matches
    ``_max_achievable_alternation``: at every step, place the remaining label
    with the highest count that is NOT the immediately-preceding one (ties
    broken by label name for determinism); only repeat when every remaining
    pier shares the previous label (unavoidable -- a group holding a strict
    majority, exactly the ``2*m - n - 1`` case).

    A numeric ``target_alt`` generalizes this with a change BUDGET: the same
    avoid-the-previous-label greedy runs until ``target_alt`` changes have
    been spent, then switches to holding the current label's run for as long
    as it has stock (repeating on purpose, rather than avoiding a repeat),
    so the ladder in ``order_with_alternation`` can ask for progressively
    LOWER alternation levels (more same-group runs, generally easier to keep
    a healthy worst edge) without hand-building a separate construction per
    level. Not exact for arbitrary targets (a majority label running out
    mid-sequence can force one extra, unavoidable change) -- callers must
    measure the ACTUAL alternation of the result, never assume it equals
    ``target_alt``.

    ``first`` optionally forces the opening label (falls through to the
    natural greedy pick if absent or already exhausted) -- used to generate a
    handful of distinct label sequences per level for sonic variety.
    """
    remaining = {k: v for k, v in counts.items() if v > 0}
    result: List[str] = []
    prev: Optional[str] = None
    changes = 0
    forced_first = first if (first is not None and remaining.get(first, 0) > 0) else None
    while remaining:
        if not result and forced_first is not None:
            pick = forced_first
        elif target_alt is not None and changes >= target_alt and prev in remaining:
            pick = prev  # budget spent -- hold the current run rather than switch
        else:
            candidates = [lbl for lbl in remaining if lbl != prev]
            pool = candidates or list(remaining)  # forced repeat -- majority group only
            pick = max(pool, key=lambda lbl: (remaining[lbl], lbl))
        if prev is not None and pick != prev:
            changes += 1
        result.append(pick)
        remaining[pick] -= 1
        if remaining[pick] == 0:
            del remaining[pick]
        prev = pick
    return result


# Full-permutation search is exact and cheap through 8 piers (8! = 40,320
# orders, well under a second to score). 9! is already 362,880 -- real cost at
# generation time -- so above this cap order_with_alternation switches to the
# bounded beam search below. Pier counts today run ~4-10 (multi_artist.py's
# total_pier_budget docstring), so this cap covers the common case exactly
# and only degrades to the approximate beam search at the high end.
_FULL_ENUMERATION_PIER_CAP = 8
_BEAM_WIDTH = 6


def _avoids_terminal(order: Sequence[int], worst_support_pier: Optional[int]) -> bool:
    """True when ``order`` does not seat ``worst_support_pier`` at either
    terminal seat (index 0 or -1). ``worst_support_pier=None`` (no support
    signal, or ``pier_support_terminal_avoidance`` off) makes this vacuously
    True for every order -- a neutral element in the ``(mn, avoids, cost)``
    comparison key below, so passing ``support_by_index=None`` reproduces the
    pre-terminal-avoidance-folding behavior exactly.
    """
    if worst_support_pier is None:
        return True
    # With < 3 piers there is no interior seat (index 0 and -1 are the same
    # two positions), so this is naturally always False when the worst pier
    # is one of them -- mirrors reorder_avoiding_low_support_terminal's own
    # "too few piers" early return without a separate branch for it.
    return order[0] != worst_support_pier and order[-1] != worst_support_pier


def _lowest_support_pier(
    idx: Sequence[int], support_by_index: Optional[Dict[int, float]],
) -> Optional[int]:
    """The pier ``reorder_avoiding_low_support_terminal`` would call "worst":
    lowest ``support_by_index`` value among ``idx``, ties broken to whichever
    appears earliest in ``idx`` for determinism. Returns ``None`` when no
    support signal was given at all (``support_by_index`` empty/``None``) --
    the caller must treat that as "no preference", not "pier idx[0] is worst".
    """
    if not support_by_index:
        return None
    idx = list(idx)
    return min(idx, key=lambda i: (support_by_index.get(i, 1.0), idx.index(i)))


def _enumerate_best_orders_by_level(
    idx: List[int],
    X_norm: np.ndarray,
    group_of_index: Dict[int, str],
    worst_support_pier: Optional[int] = None,
) -> Dict[int, Tuple[List[int], float, float]]:
    """Exact search over every permutation of ``idx`` (assumed small -- see
    ``_FULL_ENUMERATION_PIER_CAP``), grouped by achieved alternation count.

    Returns ``{alt_count: (order, min_edge, sonic_cost)}`` -- the best order
    at EVERY alternation level some permutation of these piers actually
    reaches, not just the theoretical maximum. Ranked ``(min_edge, avoids_
    terminal, sonic_cost)`` -- minimax worst edge governs first (unchanged),
    ``avoids_terminal`` (whether the order seats ``worst_support_pier``
    somewhere other than index 0/-1) is a preference applied only to break
    ties in worst edge, and total sonic cost breaks any tie still remaining.
    Because ``min_edge`` is always compared first, folding in the terminal
    preference can never change which ``min_edge`` wins a given level -- it
    only changes WHICH order (among those already tied for that level's best
    worst edge) is reported (2026-07-30, finding-1 fix: this preference used
    to live in a SEPARATE post-hoc ``reorder_avoiding_low_support_terminal``
    call in playlist_generator.py, a greedy nearest-neighbour search that
    clumps same-artist piers by construction and silently discarded the
    alternation this function had just forced -- see
    docs/superpowers/sdd/xhigh-review-fixes/finding-1-report.md).
    ``order_with_alternation``'s ladder search walks this dict from the
    highest key down, taking the first level whose worst edge clears the
    pier-adjacency floor -- the "unconstrained best" (what the pre-ladder
    safety valve fell back to) is simply the entry with the best ``(min_edge,
    avoids_terminal, cost)`` across every key, since every permutation
    belongs to exactly one level and each level's own best is already
    tracked here.
    """
    n = len(idx)
    Xn = X_norm[idx]
    sim = Xn @ Xn.T
    labels = [group_of_index.get(m, "") for m in idx]

    by_level: Dict[int, Tuple[Tuple[int, ...], float, float, bool]] = {}
    for perm in itertools.permutations(range(n)):
        edges = [float(sim[perm[i], perm[i + 1]]) for i in range(n - 1)]
        mn = min(edges)
        cost = sum(edges)
        alt = sum(1 for i in range(n - 1) if labels[perm[i]] != labels[perm[i + 1]])
        order_ids = [idx[k] for k in perm]
        avoids = _avoids_terminal(order_ids, worst_support_pier)
        key = (mn, avoids, cost)
        existing = by_level.get(alt)
        if existing is None or key > (existing[1], existing[3], existing[2]):
            by_level[alt] = (perm, mn, cost, avoids)

    return {
        alt: ([idx[k] for k in perm], mn, cost)
        for alt, (perm, mn, cost, _avoids) in by_level.items()
    }


def _assign_medoids_to_labels(
    label_seq: Sequence[str],
    medoids_by_label: Dict[str, List[int]],
    X_norm: np.ndarray,
    *,
    beam_width: int,
) -> List[Tuple[List[int], float, float]]:
    """Beam search filling ``label_seq``'s slots with specific medoid ids.

    Scored minimax-first (worst edge so far, ties broken by running sonic-
    cost total) at every step -- the same objective the exact search uses,
    kept to ``beam_width`` survivors per step so it stays cheap above
    ``_FULL_ENUMERATION_PIER_CAP``. Returns up to ``beam_width`` completed
    ``(order, min_edge, sonic_cost)`` tuples, unsorted beyond the last step's
    ranking (best first).
    """
    first_label = label_seq[0]
    beam: List[Tuple[List[int], Dict[str, List[int]], float, float]] = []
    for m in medoids_by_label.get(first_label, [])[:beam_width]:
        remaining = {lbl: list(ids) for lbl, ids in medoids_by_label.items()}
        remaining[first_label].remove(m)
        beam.append(([m], remaining, float("inf"), 0.0))

    for label in label_seq[1:]:
        expanded: List[Tuple[List[int], Dict[str, List[int]], float, float]] = []
        for order, remaining, run_min, run_cost in beam:
            last = order[-1]
            for m in remaining.get(label, []):
                edge = float(np.dot(X_norm[last], X_norm[m]))
                new_remaining = {lbl: list(ids) for lbl, ids in remaining.items()}
                new_remaining[label].remove(m)
                expanded.append((
                    order + [m], new_remaining,
                    min(run_min, edge), run_cost + edge,
                ))
        expanded.sort(key=lambda c: (-c[2], -c[3]))
        beam = expanded[:beam_width]

    return [(order, mn, cost) for order, _remaining, mn, cost in beam]


def _bounded_best_orders_by_level(
    idx: List[int],
    X_norm: np.ndarray,
    group_of_index: Dict[int, str],
    max_alt: int,
    counts: Dict[str, int],
    worst_support_pier: Optional[int] = None,
) -> Dict[int, Tuple[List[int], float, float]]:
    """Above the exact-enumeration cap: the per-level table
    ``_enumerate_best_orders_by_level`` produces exactly, approximated.

    For every level from ``max_alt`` down to 0, try a handful of label
    sequences targeting that level (``_label_sequence_at_level``, varied by
    forced opening label for sonic diversity), beam-search each one's medoid
    assignment (``_assign_medoids_to_labels``), and file every resulting
    order under its ACTUAL achieved alternation count (never the target it
    was asked for -- ``_label_sequence_at_level`` is not exact for arbitrary
    targets). Also folds in the pure multi-start greedy nearest-neighbor walk
    (ignoring alternation entirely) so the table's global best-across-levels
    entry still matches what the old unconstrained fallback used to compute.

    ``worst_support_pier``, like ``_enumerate_best_orders_by_level``'s own
    parameter, is a tie-break PREFERENCE among the candidates this function
    already generates for a level -- ranked ``(min_edge, avoids_terminal,
    cost)`` in ``_consider`` below, so it can change which of THIS level's
    generated candidates is kept but never which level's ``min_edge`` gets
    reported. It is not a beam-search objective: ``_assign_medoids_to_labels``
    itself still ranks its partial paths by ``(min_edge, cost)`` alone (the
    interior/terminal distinction is largely undecidable mid-search), so this
    is an approximation of the exact path's guarantee, consistent with this
    whole function already being the approximate one (see the "not guaranteed
    to populate every level" note below).

    Not guaranteed to populate every level an exact search would find (this
    is the approximate path, only reached above ``_FULL_ENUMERATION_PIER_CAP``
    piers) -- the ladder in ``order_with_alternation`` tolerates gaps by
    walking whatever keys are present.
    """
    medoids_by_label: Dict[str, List[int]] = {}
    for i in idx:
        medoids_by_label.setdefault(group_of_index.get(i, ""), []).append(i)

    levels: Dict[int, Tuple[List[int], float, float, bool]] = {}

    def _consider(order: List[int], mn: float, cost: float) -> None:
        alt = _alternation_count(order, group_of_index)
        avoids = _avoids_terminal(order, worst_support_pier)
        key = (mn, avoids, cost)
        existing = levels.get(alt)
        if existing is None or key > (existing[1], existing[3], existing[2]):
            levels[alt] = (order, mn, cost, avoids)

    seen_seqs: set = set()
    for target in range(max_alt, -1, -1):
        for first in (None, *counts.keys()):
            seq = _label_sequence_at_level(dict(counts), target_alt=target, first=first)
            key = tuple(seq)
            if key in seen_seqs:
                continue
            seen_seqs.add(key)
            for order, mn, cost in _assign_medoids_to_labels(
                seq, medoids_by_label, X_norm, beam_width=_BEAM_WIDTH,
            ):
                _consider(order, mn, cost)

    unconstrained, unconstrained_min = _best_unconstrained_order_bounded(idx, X_norm)
    _consider(unconstrained, unconstrained_min, _path_sonic_cost(unconstrained, X_norm))

    return {alt: (order, mn, cost) for alt, (order, mn, cost, _avoids) in levels.items()}


def _best_unconstrained_order_bounded(
    idx: List[int], X_norm: np.ndarray,
) -> Tuple[List[int], float]:
    """Above the exact-enumeration cap: the best-available order ignoring
    alternation entirely -- a multi-start greedy nearest-neighbor sweep (the
    same heuristic ``order_clusters`` itself uses), picked by worst edge.
    """
    from src.playlist.artist_style import order_clusters

    best: Optional[List[int]] = None
    best_min: Optional[float] = None
    for start in idx:
        cand = order_clusters(idx, X_norm, start=start)
        mn = _min_edge(cand, X_norm)
        if best_min is None or mn > best_min:
            best, best_min = cand, mn
    assert best is not None and best_min is not None, "idx is non-empty by construction"
    return best, best_min


def order_with_alternation(
    medoids: Sequence[int],
    X_norm: np.ndarray,
    group_of_index: Dict[int, str],
    *,
    pier_adjacency_floor: float = PIER_ADJACENCY_ALTERNATION_FLOOR,
    support_by_index: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], bool, int, int]:
    """Order piers to FORCE the highest artist alternation level that keeps a
    healthy pier-to-pier worst edge -- alternation is a hard constraint now,
    not a scored preference (2026-07-30 forced-interleaving rewrite; the
    earlier ``alternation_bonus``-weighted search over ``order_clusters``'
    own candidate set could not produce an interleaved order at all, because
    a greedy nearest-neighbour walk clumps same-artist piers by construction
    -- see
    docs/superpowers/sdd/2026-07-29-multi-artist-blend/forced-interleaving-report.md).

    LADDER SEARCH (2026-07-30, second pass -- the first cut's safety valve
    was an all-or-nothing cliff: it fell straight from the theoretical
    maximum to the fully-unconstrained order the instant the maximum
    breached the floor, which made a real pairing (Eno+Bowie) LESS
    interleaved than the pre-rewrite code, the opposite of the point of this
    feature). For each alternation level from the theoretical maximum
    (``_max_achievable_alternation``) down to 0, computes the best order at
    that EXACT level (highest minimum consecutive-cosine edge -- minimax,
    design principle 5 -- tie-broken by highest total sonic path cost; exact
    via full permutation search through ``_FULL_ENUMERATION_PIER_CAP`` piers,
    a bounded beam search above that), and takes the HIGHEST level whose
    worst edge clears ``pier_adjacency_floor``. Only when NO level clears it
    does this fall back to the single best order across every level
    regardless of alternation (the same "unconstrained best" the pre-ladder
    code always used) -- which by construction can never be less alternating
    than that same unconstrained order, since the ladder already tried and
    rejected every level above it.

    ``pier_adjacency_floor`` measures PIER-TO-PIER adjacency, a different and
    naturally much lower quantity than the final playlist's ``min_transition``
    (``ABSOLUTE_MIN_TRANSITION_FLOOR``) -- piers are deliberately spread apart
    and the beam fills the gaps between them, so this floor is its own
    constant, calibrated separately (see ``PIER_ADJACENCY_ALTERNATION_FLOOR``
    below for the measured table this was set from). Reusing the final-
    playlist floor here was the first cut's second bug: it rejected pier
    orders that went on to produce a perfectly healthy final playlist.

    ``support_by_index``, when given (i.e. ``pier_support_terminal_avoidance``
    is on and within-artist support was computed), folds the single-artist
    tail's arc-aware terminal-avoidance preference (``reorder_avoiding_low_
    support_terminal``'s reasoning: weakest edges cluster at anchor/terminal
    seats) directly into this per-level search, as a tie-break applied AFTER
    the minimax worst-edge criterion and BEFORE the sonic-cost tie-break --
    see ``_avoids_terminal`` / ``_lowest_support_pier``. This replaces a
    separate post-hoc reorder that used to run after this function returned
    (a greedy nearest-neighbour search, which clumps same-artist piers by
    construction and silently discarded the alternation just forced above --
    finding-1 of the xhigh review, docs/superpowers/sdd/xhigh-review-fixes/
    finding-1-report.md). Passing ``None`` (the default) reproduces this
    function's pre-finding-1 behavior exactly.

    Returns ``(ordered, improved, achieved_alt, max_alt)``: ``improved`` is
    True when the returned order differs from the plain ``order_clusters``
    default walk; ``achieved_alt`` / ``max_alt`` let the caller report a
    relaxation notice when forcing had to step down from the theoretical
    maximum.
    """
    from src.playlist.artist_style import order_clusters

    idx = [int(m) for m in medoids]
    if len(idx) < 2:
        return list(idx), False, 0, 0

    counts = Counter(group_of_index.get(i, "") for i in idx)
    max_alt = _max_achievable_alternation(list(counts.values()))
    default = order_clusters(idx, X_norm)
    worst_support_pier = _lowest_support_pier(idx, support_by_index)

    if len(idx) <= _FULL_ENUMERATION_PIER_CAP:
        by_level = _enumerate_best_orders_by_level(
            idx, X_norm, group_of_index, worst_support_pier,
        )
    else:
        by_level = _bounded_best_orders_by_level(
            idx, X_norm, group_of_index, max_alt, dict(counts), worst_support_pier,
        )

    if not by_level:
        # max_alt is derived to be achievable by construction and the exact/
        # bounded search always considers at least the plain unconstrained
        # walk -- this should never trigger. Guard defensively rather than
        # crash pier selection over an ordering nicety.
        logger.warning(
            "Multi-artist ordering: the level search produced no candidates "
            "at all (%d pier(s)) -- falling back to the unconstrained walk; "
            "this should not happen, please report.", len(idx),
        )
        by_level = {_alternation_count(default, group_of_index): (
            default, _min_edge(default, X_norm), _path_sonic_cost(default, X_norm),
        )}

    # Cross-level ranking (below) must be BLIND to the terminal-support
    # preference: by_level's own per-level `cost` can be a WORSE (lower)
    # value than the level's true best when the terminal-avoidance tie-break
    # swapped in a same-mn, avoids-preferring order over the plain highest-
    # cost one -- fine for a single level in isolation (nothing downstream
    # reads that level's own cost once it is chosen), but poison for
    # `unconstrained_level` below, which compares (mn, cost) ACROSS levels:
    # a swapped-in lower cost at one level can flip which level wins that
    # comparison, silently degrading the achieved alternation relative to
    # what the very same call would return without a support signal --
    # exactly the invariant this fix must never violate. Recomputing the
    # avoids-BLIND per-level table (worst_support_pier=None collapses
    # ``_avoids_terminal`` to always-True, i.e. today's plain (mn, cost)
    # search) keeps that comparison stable; `by_level` itself (used just
    # below to fetch the actual RETURNED order for whichever level wins) is
    # unaffected since a level's ``mn`` is always the same in both tables --
    # ``_avoids_terminal`` only ever breaks ties AT a level's own mn, never
    # changes which mn wins that level.
    plain_by_level = by_level
    if worst_support_pier is not None:
        _plain = (
            _enumerate_best_orders_by_level(idx, X_norm, group_of_index)
            if len(idx) <= _FULL_ENUMERATION_PIER_CAP
            else _bounded_best_orders_by_level(idx, X_norm, group_of_index, max_alt, dict(counts))
        )
        if _plain:
            # Guards the defensive "level search produced no candidates at
            # all" branch above: that branch already hand-built a non-empty
            # single-entry ``by_level`` -- an empty ``_plain`` here (the same
            # "should never happen" condition, just without the support
            # signal) must not replace it with something that makes the
            # ``max()`` below crash on an empty dict.
            plain_by_level = _plain

    # The single best (min edge, then cost) order across every level -- what
    # the old pre-ladder safety valve always fell back to, and this ladder's
    # own last resort when nothing clears the floor.
    unconstrained_level = max(
        plain_by_level, key=lambda lvl: (plain_by_level[lvl][1], plain_by_level[lvl][2]),
    )
    unconstrained, unconstrained_min, _unconstrained_cost = by_level[unconstrained_level]

    floor = float(pier_adjacency_floor)
    chosen = None
    chosen_min = None
    chosen_level = None
    top_level_min = None  # the theoretical-max level's own worst edge, for the log line
    for level in sorted(by_level.keys(), reverse=True):
        order, mn, _cost = by_level[level]
        if top_level_min is None:
            top_level_min = mn
        if mn >= floor:
            chosen, chosen_min, chosen_level = order, mn, level
            break

    if chosen is None:
        # No level -- not even the fully-clumped one -- clears the floor.
        # Fall back to the single best order regardless of alternation. This
        # is never LESS alternating than "the unconstrained search would
        # have produced anyway": it IS that same order.
        chosen, chosen_min, chosen_level = unconstrained, unconstrained_min, unconstrained_level
        logger.warning(
            "Multi-artist ordering: no alternation level (max %d down to 0) "
            "cleared the pier-adjacency floor %.2f (best level's worst edge "
            "was %.4f) -- falling back to the single best-available order "
            "regardless of alternation (worst edge %.4f, %d/%d alternation).",
            max_alt, floor, top_level_min if top_level_min is not None else float("nan"),
            chosen_min, chosen_level, max_alt,
        )
    elif chosen_level < max_alt:
        logger.warning(
            "Multi-artist ordering: the alternation ladder stepped down from "
            "the theoretical max (%d, worst edge %.4f) to %d (worst edge "
            "%.4f) -- %d/%d was the highest level clearing the pier-adjacency "
            "floor %.2f.",
            max_alt, top_level_min if top_level_min is not None else float("nan"),
            chosen_level, chosen_min, chosen_level, max_alt, floor,
        )

    logger.info(
        "Multi-artist ordering: alternation %d/%d (theoretical max), chosen "
        "worst edge %.4f, unconstrained-best worst edge %.4f",
        chosen_level, max_alt, chosen_min, unconstrained_min,
    )
    improved = chosen != default
    return chosen, improved, chosen_level, max_alt


@dataclass
class MultiArtistPiers:
    ordered_medoids: List[int]
    relaxations: List[Dict[str, Any]]
    groups: List[ArtistGroup]
    mean_affinity: float
    blocked_artist_keys: frozenset
    # Merged within-artist support (compute_within_artist_support, keyed by
    # bundle index) across every group's own cluster_artist_tracks call.
    # Previously computed and thrown away per-group (review Finding 8).
    # ``select_multi_artist_piers`` already folds this into its own
    # ``order_with_alternation`` call (the terminal-support preference lives
    # THERE now, not in a caller-side post-hoc reorder -- finding-1 of the
    # xhigh review fixed a defect where that post-hoc reorder silently
    # discarded the forced alternation). Kept exposed here for callers'
    # diagnostics/tests, not because a caller needs to act on it. Defaults to
    # {} so existing call sites that construct MultiArtistPiers without it
    # keep working.
    support_by_index: Dict[int, float] = field(default_factory=dict)
    # The number of groups (from ``groups``) that actually seated >=1 pier --
    # NOT len(groups), which also counts groups that survived partition but
    # failed to cluster (review Finding 6: playlist_generator.py scales the
    # post-order per-artist cap by "the number of surviving groups", and
    # using the raw group count over-relaxes the cap by counting a group that
    # contributed nothing). ``None`` (the default, for callers/tests that
    # hand-build a MultiArtistPiers without this) tells the caller to fall
    # back to ``len(groups)`` -- the pre-fix behavior -- rather than silently
    # treating "not provided" as zero.
    seated_group_count: Optional[int] = None


def _blocked_artist_keys(groups: Sequence[ArtistGroup]) -> frozenset:
    """Artist keys for every CHIP artist, resolved through aliases.

    An exclusive group's ``label`` IS an artist key candidate. A joint
    group's ``label`` is a composite display string (e.g. ``"A & B"``) and is
    NOT an artist key -- its ``source_artists`` tuple holds the real chip
    names. Both must be walked: a chip credited ONLY jointly with another
    chip (no solo catalog at all -- e.g. the real library's "John Foxx"
    alongside "John Foxx and Harold Budd") has no exclusive group of its own,
    so reading only exclusive groups here would let that chip's tracks seat
    as bridge-interior filler inside its own blend (review Finding 1).

    Becomes pier_bridge_builder._derive_seed_artist_keys's ``override``, which
    does ``frozenset(str(k) for k in override if k)`` -- handing it a bare
    string would silently iterate CHARACTERS. Built directly as a frozenset
    comprehension, so the result is never anything else; the caller-side assert
    checks the thing that can actually go wrong (a non-string element slipping
    in), not the vacuously-true fact that a frozenset() call returns a frozenset.

    KEY SPACE (review Finding 1): the consumer of this override,
    ``pier_bridge_builder._row_ok``, compares each candidate row's key via
    ``identity_keys_for_index(bundle, idx).artist_key`` -- which normalizes
    through ``normalize_primary_artist_key`` (ensemble-suffix and
    collaboration stripping, e.g. "Bill Evans Trio" -> "bill evans",
    "Godspeed You! Black Emperor" kept intact because "!" is not
    collaboration/ensemble punctuation). The DEFAULT derivation in
    ``_derive_seed_artist_keys`` already uses that exact function. This
    override path must produce keys in the SAME space -- using the plainer
    ``normalize_artist_key`` (punctuation-only, no ensemble/collaboration
    stripping) here silently produced a DIFFERENT key for any chip name that
    ensemble/collaboration-stripping changes (e.g. "Bill Evans Trio" ->
    "bill evans trio" vs the row space's "bill evans"), which never matched
    at ``_row_ok`` and made ``disallow_seed_artist_in_interiors`` a no-op for
    that chip -- entirely inert on the fixture names ("Brian Eno", "Harold
    Budd", ...) that happen to be identical in both key spaces, which is why
    the test suite stayed green while the guarantee was silently broken.
    """
    from src.playlist.identity_keys import normalize_primary_artist_key

    chip_names: List[str] = []
    for g in groups:
        if g.is_joint:
            chip_names.extend(g.source_artists)
        else:
            chip_names.append(g.label)

    return frozenset(
        key for key in (normalize_primary_artist_key(n) for n in chip_names if n)
        if key
    )


class MultiArtistBlendFailed(RuntimeError):
    """A real multi-artist attempt failed and there is something to explain.

    Two distinct situations raise this, both carrying relaxations that must
    not vanish into a bare ``None`` return (Task 13 acceptance testing found
    the second situation was still falling through to a silent ``None`` --
    see task-13-report.md):

    1. >=2 groups survived partition, but every one was too thin to cluster
       (see the Finding 2 discussion in task-9-report.md).
    2. Fewer than two groups survived partition BECAUSE a requested name was
       dropped (no usable tracks) -- as opposed to the caller simply never
       having asked for a second artist, which has nothing to explain and
       still returns ``None``.

    Distinct from ``select_multi_artist_piers`` returning bare ``None``
    (nothing was ever dropped or attempted -- the caller silently runs the
    single-artist path with nothing to show the user). Callers should catch
    this, fold ``.relaxations`` into whatever they show the user, and then
    fall back to single-artist generation exactly as they would for a
    ``None`` return.
    """

    def __init__(self, message: str, relaxations: List[Dict[str, Any]]):
        super().__init__(message)
        self.relaxations = relaxations


def select_multi_artist_piers(
    *,
    bundle,
    artist_names: Sequence[str],
    style_cfg,
    ma_cfg: MultiArtistConfig,
    track_count: int,
    max_artist_fraction: float,
    random_seed: int = 0,
    include_collaborations: bool = False,
    excluded_track_ids: Optional[set] = None,
    metadata_db_path: Optional[str] = None,
    min_pier_duration_seconds: Optional[int] = 46,
) -> Optional[MultiArtistPiers]:
    """Pick and order the piers for a 2+ artist blend.

    Returns None when fewer than two groups survive partition AND nothing was
    dropped to cause it -- e.g. the caller only ever named one artist. There
    is genuinely nothing to explain, so the caller runs the normal
    single-artist path unchanged.

    Raises MultiArtistBlendFailed in two situations, both real attempts that
    produced zero piers and must not be silently folded into the same None
    return (that would discard the relaxations already collected and violate
    the never-fail-on-soft-axes rule: the user must still learn why their
    blend didn't happen):
      1. Two or more groups DID survive partition but every one of them was
         too thin to cluster (e.g. two obscure chips with 1-2 tracks each and
         no joint catalog). See task-9-report.md, Finding 2.
      2. Fewer than two groups survived partition BECAUSE a requested name
         had zero usable tracks and was dropped -- e.g. a typo'd or absent
         artist name collapses a 2-artist request down to 1 survivor. Found
         by Task 13 live acceptance testing (task-13-report.md): the original
         Finding-2 fix only covered outcome 1, so this path still returned a
         bare None and silently swallowed the drop notice.

    Raises ValueError immediately, before any group is touched, if the
    artifact itself is missing artist_keys or X_sonic -- that is a data
    problem, not a "this artist is thin" problem, and must never be
    misreported to the user as one (Finding 1).

    Relaxation entries mirror the shape pier_bridge_builder already emits for the
    ``RelaxationNotice`` component (``web/src/lib/types.ts::RelaxationEntry``):
    ``{"type": "relaxation", "scope": ..., "bridge": str, "relaxed": [str, ...],
    "severity": str}``. The renderer produces exactly
    "Relaxed to fit: {bridge} — dropped {', '.join(relaxed)}" -- there is no
    separate free-text "detail" field, so ``bridge`` must be a short label and
    each ``relaxed`` entry a short noun phrase that reads naturally after the
    word "dropped" (see task-9-report.md, Finding 4, for the exact rendered
    sentence of every entry this function emits).

    ``min_pier_duration_seconds`` is forwarded to every per-group
    ``cluster_artist_tracks`` call below as a HARD floor on pier candidacy
    (sub-minimum fragments -- outtakes, intros, skits -- must never seat as a
    pier). A group starved below the clustering floor by this gate hits the
    same per-group ``except ValueError`` path as any other thin group: it
    contributes no piers and is reported via the relaxation list, never a
    silent drop and never a crash of the whole blend. Defaults to ``46``
    (mirroring ``cluster_artist_tracks``'s own default -- see its docstring
    for the full reasoning) rather than ``None``, so a caller of this function
    that forgets the parameter is still gated.
    """
    import math

    from src.playlist.artist_style import (
        cluster_artist_tracks, order_clusters, _select_k, _post_filter_eligible_count,
    )

    names = [str(a).strip() for a in artist_names if str(a).strip()]
    if len(names) > ma_cfg.max_artists:
        logger.warning(
            "Multi-artist: %d artists requested, capping at max_artists=%d (%s dropped)",
            len(names), ma_cfg.max_artists, names[ma_cfg.max_artists:],
        )
        names = names[: ma_cfg.max_artists]

    relaxations: List[Dict[str, Any]] = []
    groups, dropped = partition_artist_groups(
        bundle, names,
        include_collaborations=include_collaborations,
        excluded_track_ids=excluded_track_ids,
    )
    for name in dropped:
        relaxations.append({
            "type": "relaxation",
            "scope": "multi_artist",
            "bridge": "the artist pairing",
            "relaxed": [f"{name} (no usable tracks in your library)"],
            "severity": "info",
        })

    # Chip names actually represented in the pairing -- either via an
    # exclusive group or (a chip with no solo catalog at all) via the joint
    # group's source_artists. Used below for user-facing text describing the
    # whole pairing; a joint group's own `label` is a composite ("A & B"),
    # not a chip name, so it cannot be used directly for that purpose.
    present_names = [n for n in dict.fromkeys(names) if n not in dropped]

    # Gate on the number of PIER-PRODUCING groups, not the number of
    # exclusive groups (review Finding 1). A chip credited ONLY jointly with
    # another chip (no solo catalog at all -- e.g. the real library's "John
    # Foxx and Harold Budd" alongside a thin Foxx solo catalog) contributes
    # no exclusive group, but its joint group is a real, pier-producing
    # group: `groups = [OtherArtist(exclusive), Joint(both)]` is a genuine
    # 2-group blend and must proceed, not silently collapse to the
    # single-artist path with nothing dropped and nothing to explain.
    if len(groups) < 2:
        if relaxations:
            # Fewer than two groups survived BECAUSE a requested name was
            # dropped (no usable tracks) -- not because the caller only ever
            # asked for one artist. That drop notice must not vanish with a
            # bare None return (the same silent-discard failure mode Finding
            # 2 fixed for the "every group too thin to cluster" case above):
            # a user who typo'd or picked an absent artist deserves to know
            # why their blend became a single-artist playlist. Raise so the
            # caller's existing MultiArtistBlendFailed handler surfaces it
            # and falls back to single-artist generation, exactly as it
            # already does for the every-group-failed-to-cluster case.
            #
            # This branch can only be reached with 0 or 1 surviving groups,
            # and a joint group can never exist alone (it requires 2
            # contributing chips) -- so the lone survivor here, if any, is
            # always exclusive.
            logger.warning(
                "Multi-artist: only %d of %d requested artist(s) survived "
                "partition (dropped: %s) — falling back to single-artist "
                "generation.", len(groups), len(names), dropped,
            )
            raise MultiArtistBlendFailed(
                f"Multi-artist: only {len(groups)} of {len(names)} requested "
                "artist(s) has usable tracks in your library.",
                relaxations,
            )
        logger.info(
            "Multi-artist: only %d artist group(s) survived — falling back to the "
            "single-artist path.", len(groups),
        )
        return None

    # Artifact-integrity checks, once, up front -- NOT inside the per-group
    # try/except below. cluster_artist_tracks raises ValueError for four
    # different reasons (artist_style.py: missing artist_keys L852, missing
    # X_sonic L855, too-few-tracks L916, degenerate-clustering-after-retries
    # L1041). Only the last two are per-group conditions; the first two are
    # properties of the WHOLE artifact and must never be caught and reported
    # to the user as "this artist has too few tracks" (Finding 1). Checking
    # them here means they raise loudly, unconditionally, before any group is
    # touched -- group_prototypes below would also catch a missing X_sonic,
    # but not missing artist_keys, and not with this function's own voice.
    if getattr(bundle, "artist_keys", None) is None:
        raise ValueError(
            "Multi-artist: artifact missing artist_keys — cannot cluster any group."
        )
    if getattr(bundle, "X_sonic", None) is None:
        raise ValueError(
            "Multi-artist: artifact missing X_sonic — cannot cluster any group."
        )

    protos = group_prototypes(bundle, groups)
    genre_profiles = group_genre_profiles(bundle, groups)

    total = total_pier_budget(track_count, max_artist_fraction, len(groups))
    alloc = allocate_pier_budget(
        groups, total, joint_pier_min_budget=ma_cfg.joint_pier_min_budget,
        # Review Finding 8: the reservation floor must match the floor
        # cluster_artist_tracks will actually apply to this same joint group,
        # or the guaranteed seat can be reserved for a group that can never
        # cluster.
        joint_min_cluster_size=max(3, int(style_cfg.cluster_k_min)),
    )
    logger.info(
        "Multi-artist pier budget: total=%d allocation=%s (track_count=%d, "
        "max_artist_fraction=%.3f)", total, alloc, track_count, max_artist_fraction,
    )
    warn_if_bridge_density_thin(track_count, total)

    all_medoids: List[int] = []
    group_of_index: Dict[int, str] = {}
    affinity_by_index: Dict[int, float] = {}
    support_by_index: Dict[int, float] = {}
    X_norm_shared: Optional[np.ndarray] = None
    # Review Finding 7: every group's own rows must be excluded from every
    # OTHER group's pier-bridgeability neighbour count, not just the group's
    # own. The blend sets disallow_seed_artist_in_interiors=True for EVERY
    # chip key (playlist_generator.py), so a candidate whose only "eligible"
    # k-th neighbour is another chip's track (or the joint group's) is not
    # actually bridgeable -- that neighbour can never serve as bridge fill.
    # member_indices=g.indices alone only excludes THIS group's own rows from
    # cluster_artist_tracks's default derivation, leaving the other chip(s)
    # countable as valid neighbours. Passing the union of every group's
    # indices as bridgeability_excluded_indices closes that gap.
    _all_group_indices = [i for gr in groups for i in gr.indices]

    for g in groups:
        want = int(alloc.get(g.label, 0))
        if want <= 0:
            # Review Finding 3: allocate_pier_budget can legitimately hand a
            # surviving group zero seats (the joint group below its own
            # min-budget threshold; an exclusive group starved out by a tiny
            # total split across many chips) -- this is a real degradation of
            # what the user asked for, not a no-op, and must be reported like
            # every other per-group shortfall in this function (never-fail-
            # on-soft-axes: degradation must always be observable).
            if g.is_joint and total < int(ma_cfg.joint_pier_min_budget):
                _zero_reason = (
                    f"the joint pier reservation needs a total pier budget "
                    f"of {ma_cfg.joint_pier_min_budget}, but this blend's "
                    f"total is only {total}"
                )
            else:
                _zero_reason = (
                    f"the {total}-pier budget split across {len(groups)} "
                    "group(s) left no seat for this one"
                )
            logger.warning(
                "Multi-artist: '%s' was allocated 0 of %d total pier(s) (%s) "
                "-- contributes no piers.", g.label, total, _zero_reason,
            )
            relaxations.append({
                "type": "relaxation",
                "scope": "multi_artist",
                "bridge": g.label,
                "relaxed": [f"every pier ({_zero_reason})"],
                "severity": "info",
            })
            continue
        # A group thinner than the clustering floor (cluster_k_min, e.g. a joint
        # group with only 1-2 tracks) cannot be clustered -- cluster_artist_tracks
        # raises ValueError rather than degrading. That is correct for the
        # single-artist path (an artist with 2 tracks genuinely cannot seed a
        # playlist), but here it is one contributor among several groups: the
        # right behavior is to drop this group's contribution and keep going
        # (design principle 25, "edge cases get graceful fallbacks"), not to
        # crash the whole blend over one thin corner (most often the joint set).
        #
        # This catch is intentionally still ValueError, not a narrower
        # exception: with the artifact-integrity checks hoisted above, the
        # only two remaining raise sites inside cluster_artist_tracks are
        # both genuinely per-group conditions -- too-few-tracks (L916) and
        # degenerate-clustering-after-retries (L1041, all k-means retries
        # produced <2 non-empty clusters). Both are real, both are about THIS
        # group's own member data, and the two cannot be told apart without
        # matching on exception message text, which is more fragile than
        # treating them the same way here (Finding 1, accepted as documented).
        aff = overlap_affinity(
            bundle, g, groups, protos, genre_profiles, genre_share=ma_cfg.genre_share,
        )
        # Review Finding 6: predict the SAME k cluster_artist_tracks is about
        # to pick. style_cfg.cluster_k_max alone under-predicts k for any
        # group thinner than the ceiling (the single-artist path's own
        # prediction below in playlist_generator.py already uses
        # _select_k(track_count, cfg) for exactly this reason) -- using the
        # ceiling directly here over-produces medoid_top_k, which then makes
        # cluster_artist_tracks return MORE medoids than `want` while this
        # function reports a false "dropped N piers" scarcity.
        #
        # Review Finding 9: predict from the POST-filter count, not
        # len(g.indices) (pre-filter, raw partition size). cluster_artist_
        # tracks itself calls _select_k(len(artist_indices), cfg) AFTER its
        # own duration gate + version-dedupe have already shrunk the group --
        # predicting from the larger pre-filter count under-predicts k_predict
        # is fine, but the resulting medoid_top_k = ceil(want / k_predict) is
        # computed against a k that will turn out too LOW once the real
        # (smaller) k is selected, so eff_top_k*k_real medoids can fall short
        # of `want` with nothing to compensate. _post_filter_eligible_count
        # shares the exact filtering helpers cluster_artist_tracks uses
        # internally, so this can never diverge from the real value again
        # (this is the second time this exact divergence has been fixed).
        k_predict = max(1, _select_k(
            _post_filter_eligible_count(
                bundle, g.indices, style_cfg,
                min_pier_duration_seconds=min_pier_duration_seconds,
                metadata_db_path=metadata_db_path,
                artist_name=g.label,
            ),
            style_cfg,
        ))
        try:
            clusters, medoids, _by_cluster, X_norm, _support = cluster_artist_tracks(
                bundle=bundle,
                artist_name=g.label,
                cfg=style_cfg,
                random_seed=int(random_seed),
                medoid_top_k=max(1, math.ceil(want / k_predict)),
                include_collaborations=include_collaborations,
                metadata_db_path=metadata_db_path,
                target_pier_count=want,
                member_indices=list(g.indices),
                bridgeability_excluded_indices=_all_group_indices,
                overlap_affinity=aff,
                overlap_weight=ma_cfg.overlap_weight,
                min_pier_duration_seconds=min_pier_duration_seconds,
            )
        except ValueError as exc:
            logger.warning(
                "Multi-artist: '%s' (%d track(s)) could not be clustered (%s) — "
                "contributes no piers.", g.label, len(g.indices), exc,
            )
            _piers_word = "pier" if want == 1 else "piers"
            _tracks_word = "track" if len(g.indices) == 1 else "tracks"
            # A user with a real 5-track catalog whose 5 tracks are all
            # sub-minimum fragments does not "have too few tracks" -- they
            # have too few ELIGIBLE tracks. cluster_artist_tracks tags the
            # exception (not a string match) when the duration gate is what
            # actually emptied the group, so this copy doesn't misreport a
            # scarcity the artist doesn't have (coordinator review
            # 2026-07-30).
            if getattr(exc, "duration_gate_starved", False):
                _reason = (
                    f"only {len(g.indices)} {_tracks_word} in your library, all "
                    "under the minimum track length -- too few eligible to cluster"
                )
            elif hasattr(exc, "duration_gate_starved"):
                # Same raise site as above (too-few-tracks), just not
                # duration-gate-caused -- a real scarcity claim.
                _reason = f"only {len(g.indices)} {_tracks_word} in your library — too few to cluster"
            else:
                # Review Finding 10: cluster_artist_tracks has a SECOND raise
                # site with no track-count relationship at all -- every
                # k-means retry produced fewer than 2 non-empty clusters
                # ("Clustering degenerate for artist ... after retries").
                # Only the too-few-tracks raise sets `duration_gate_starved`
                # (True or False); its absence here means this exception came
                # from the degenerate-clustering site instead, so blaming
                # library size is self-contradicting -- this can fire on a
                # well-stocked artist whose sonic content just clusters
                # poorly at this k. cluster_artist_tracks's own WARNING names
                # the retries; this notice just avoids telling the user to
                # add music they already own.
                _reason = (
                    "clustering could not find distinct groups within its "
                    "tracks -- not a shortage of tracks"
                )
            relaxations.append({
                "type": "relaxation",
                "scope": "multi_artist",
                "bridge": g.label,
                "relaxed": [f"{want} {_piers_word} ({_reason})"],
                "severity": "info",
            })
            continue
        X_norm_shared = X_norm
        support_by_index.update(_support)
        # Review Finding 10: sequence this group's own candidates in sonic
        # space BEFORE capping to `want`. Slicing `medoids` (raw k-means
        # cluster-index order) can keep two medoids from adjacent, poorly-
        # connected clusters while dropping a better-connected one from a
        # cluster later in k-means's arbitrary centroid order.
        # playlist_generator._cap_order exists for exactly this reason and
        # every single-artist branch uses it before capping.
        chosen = order_clusters(list(medoids), X_norm)[:want]
        for m in chosen:
            group_of_index[int(m)] = g.label
            affinity_by_index[int(m)] = float(aff[int(m)])
        all_medoids.extend(int(m) for m in chosen)
        if len(chosen) < want:
            _short = want - len(chosen)
            _tracks_word = "track" if len(g.indices) == 1 else "tracks"
            # "N of M piers" is a ratio phrase -- "piers" stays plural even
            # when the numerator (dropped count) is 1, same as "1 of 3 items"
            # reads naturally where "1 of 3 item" would not.
            #
            # Review Finding 10: this branch runs whenever cluster_artist_
            # tracks returned fewer medoids than `want` -- which can happen
            # even when the group has PLENTY of tracks, if the pier
            # bridgeability veto thinned a cluster to zero eligible medoids
            # (cluster_artist_tracks's own WARNING: "after veto+reallocation
            # only N medoid(s) available") or k-means clustering degenerated
            # to fewer non-empty clusters than requested. Blaming "only N
            # tracks in your library" unconditionally is self-contradicting
            # on a 250-track artist -- only make that claim when the group's
            # raw track count is actually below what was requested; otherwise
            # say honestly that something other than scarcity thinned it.
            if len(g.indices) < want:
                _reason = f"only {len(g.indices)} {_tracks_word} in your library"
            else:
                _reason = (
                    "clustering/bridgeability reduced the eligible "
                    "candidates, not a shortage of tracks"
                )
            relaxations.append({
                "type": "relaxation",
                "scope": "multi_artist",
                "bridge": g.label,
                "relaxed": [f"{_short} of {want} piers ({_reason})"],
                "severity": "info",
            })

    if not all_medoids:
        # >=2 groups survived partition (checked above), so this is NOT the
        # "fewer than two groups" case -- every surviving group individually
        # failed to cluster. Returning None here would reuse the same signal
        # for two different situations and the relaxations collected so far
        # (dropped chips, every group's thin-cluster failure) would vanish
        # with it. Raise instead, carrying them, so a caller can surface WHY
        # before falling back (Finding 2; see task-9-report.md for the full
        # rationale and the alternative designs considered).
        logger.warning(
            "Multi-artist: %d group(s) survived partition but none could be "
            "clustered — refusing to silently fall back; raising with %d "
            "relaxation(s) attached.", len(groups), len(relaxations),
        )
        raise MultiArtistBlendFailed(
            f"Multi-artist: all {len(groups)} surviving group(s) were too thin "
            "to cluster.",
            relaxations,
        )

    assert X_norm_shared is not None, (
        "all_medoids is non-empty, so at least one cluster_artist_tracks call "
        "must have succeeded and set X_norm_shared -- a future refactor of the "
        "loop above must preserve this."
    )
    # Finding-1 fix (xhigh review): terminal-support avoidance is folded into
    # this SAME search (as a same-alternation-level tie-break -- see
    # order_with_alternation's docstring), not applied afterward. A separate
    # post-hoc reorder here (or in the caller) would repeat the exact defect
    # this fixed: a greedy nearest-neighbour walk that clumps same-artist
    # piers by construction, silently discarding the alternation just forced.
    # Gated the same way the old post-hoc call was
    # (``style_cfg.pier_support_terminal_avoidance``) so disabling the knob
    # reproduces the pre-Task-3 (no terminal preference at all) behavior.
    ordered, _improved, _achieved_alt, _max_alt = order_with_alternation(
        all_medoids, X_norm_shared, group_of_index,
        pier_adjacency_floor=ma_cfg.pier_adjacency_floor,
        support_by_index=(
            support_by_index if style_cfg.pier_support_terminal_avoidance else None
        ),
    )
    if _achieved_alt < _max_alt:
        # The alternation ladder had to step down from the theoretical
        # maximum to keep a healthy pier-to-pier transition (see
        # order_with_alternation's own WARNING for the rejected/chosen
        # worst-edge values) -- report it to the user the same way the
        # low-overlap notice below does, rather than silently shipping a
        # less-interleaved blend with nothing to show for it.
        bridge_label = " & ".join(present_names)
        relaxations.append({
            "type": "relaxation",
            "scope": "multi_artist",
            "bridge": bridge_label,
            "relaxed": [
                f"full artist alternation ({_achieved_alt} of {_max_alt} "
                "possible pier-to-pier changes — forcing more would have "
                "broken a pier transition)"
            ],
            "severity": "info",
        })

    mean_affinity = (
        float(np.mean([affinity_by_index.get(i, 0.0) for i in ordered]))
        if ordered else 0.0
    )
    if mean_affinity < ma_cfg.low_overlap_threshold:
        # present_names (every chip actually in the pairing, exclusive or
        # joint-only) rather than the exclusive groups' own labels -- a
        # joint-only chip has no exclusive group and must not be dropped
        # from this user-facing description of the pairing (Finding 1).
        bridge_label = " & ".join(present_names)
        relaxations.append({
            "type": "relaxation",
            "scope": "multi_artist",
            "bridge": bridge_label,
            "relaxed": [
                f"the shared middle ground (mean pier affinity "
                f"{mean_affinity:.2f} — piers kept each artist's own character)"
            ],
            "severity": "info",
        })
        logger.warning(
            "Multi-artist: %s share little sonic ground (mean pier affinity "
            "%.2f) — piers stayed characteristic rather than meeting in the "
            "middle.", bridge_label, mean_affinity,
        )

    # Review Finding 2: a group that seated NO pier (caught above by the
    # per-group except-ValueError, or skipped entirely by the want<=0 branch)
    # must not be blocked from bridge interiors -- it contributes no
    # structure, so blocking it too is excluding that chip's tracks TWICE
    # (no piers AND no fill). ``group_of_index``'s values are exactly the
    # labels of groups that placed >=1 medoid into ``all_medoids`` above.
    _seated_labels = set(group_of_index.values())
    _seated_groups = [g for g in groups if g.label in _seated_labels]
    blocked = _blocked_artist_keys(_seated_groups)
    assert all(isinstance(k, str) for k in blocked), (
        "blocked_artist_keys must contain only artist-key strings"
    )

    logger.info(
        "Multi-artist piers: %d seated across %s, mean_affinity=%.3f, order=%s",
        len(ordered), [g.label for g in groups], mean_affinity,
        [group_of_index.get(i, "?") for i in ordered],
    )
    return MultiArtistPiers(
        ordered_medoids=ordered,
        relaxations=relaxations,
        groups=groups,
        mean_affinity=mean_affinity,
        blocked_artist_keys=blocked,
        support_by_index=support_by_index,
        seated_group_count=len(_seated_groups),
    )
