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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiArtistConfig:
    enabled: bool = True
    overlap_weight: float = 0.6
    genre_share: float = 0.25
    max_artists: int = 4
    joint_pier_min_budget: int = 3
    alternation_bonus: float = 0.15
    low_overlap_threshold: float = 0.15


def multi_artist_config_from_ds(ds_cfg: Dict[str, Any]) -> MultiArtistConfig:
    """Build the config from the ``playlists.ds_pipeline`` dict.

    Mirrors how ArtistStyleConfig is assembled in playlist_generator.py (~L1842):
    plain .get() with the dataclass defaults as the fallback.
    """
    raw = (ds_cfg or {}).get("multi_artist", {}) or {}
    return MultiArtistConfig(
        enabled=bool(raw.get("enabled", True)),
        overlap_weight=float(raw.get("overlap_weight", 0.6)),
        genre_share=float(raw.get("genre_share", 0.25)),
        max_artists=int(raw.get("max_artists", 4)),
        joint_pier_min_budget=int(raw.get("joint_pier_min_budget", 3)),
        alternation_bonus=float(raw.get("alternation_bonus", 0.15)),
        low_overlap_threshold=float(raw.get("low_overlap_threshold", 0.15)),
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

    ``max_artist_fraction`` (the Artist-presence dial) is EACH seed artist's share,
    so N groups get N times the single-artist base -- clamped by a floor that keeps
    bridge segments long enough to be bridgeable, and never below one per group.
    """
    base = max(3, round(int(track_count) * float(max_artist_fraction)))
    n = max(1, int(n_groups))
    if n == 1:
        return base
    segment_floor = max(n, int(track_count) // 3)
    return max(n, min(n * base, segment_floor))


def allocate_pier_budget(
    groups: Sequence[ArtistGroup], total: int, *, joint_pier_min_budget: int
) -> Dict[str, int]:
    """Split ``total`` piers across groups: even, remainder to the first chip.

    The joint group claims a GUARANTEED MINIMUM of one seat when it is
    non-empty and the total reaches ``joint_pier_min_budget``. It may
    additionally absorb surplus during redistribution once the exclusive
    groups are saturated, subject to its own track capacity like any other
    group -- the joint group IS the overlap between the named artists, so for
    a pairing whose collaborative output dominates thin solo catalogs, it is
    the correct place for spare budget to land, not a one-seat ceiling.

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
    if joint is not None and joint.indices and total >= int(joint_pier_min_budget):
        alloc[joint.label] = 1
        remaining -= 1
        logger.info(
            "Multi-artist: reserved 1 pier for the jointly-credited group '%s'",
            joint.label,
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


def order_with_alternation(
    medoids: Sequence[int],
    X_norm: np.ndarray,
    group_of_index: Dict[int, str],
    *,
    alternation_bonus: float,
) -> tuple[List[int], bool]:
    """Order piers preferring artist alternation, never worse than the default
    walk's worst edge.

    Mirrors artist_style.reorder_avoiding_low_support_terminal: every candidate is
    an order ``order_clusters`` itself could produce (same greedy walk, different
    start node). That bounds WHICH topologies are eligible, but NOT which one
    wins -- a plain sum-of-cosines score can trade one wrecked edge for several
    slightly better ones, which design principle 5 ("the worst edge defines the
    experience") forbids.

    The guarantee is enforced with a minimum-edge floor, not just topology
    membership: ``default_min`` is the worst (minimum) consecutive-cosine edge
    of the default (``start=None``) walk. Any candidate whose worst edge is
    below ``default_min`` (past a ``1e-12`` epsilon, matching the tie margin
    used elsewhere in this function) is discarded before scoring. Only among
    the survivors -- the default walk is always one, by construction, so the
    function always has something to return -- does the alternation-bonus
    score (``sonic path cost + alternation_bonus * artist changes``) pick a
    winner. This makes the ordering never-worse on the floor metric, the same
    guarantee ``reorder_avoiding_low_support_terminal`` gives for its own floor.

    Returns ``(ordered, improved)``; ``improved`` is True when the winner differs
    from the default ``order_clusters`` walk.
    """
    from src.playlist.artist_style import order_clusters

    idx = [int(m) for m in medoids]
    if len(idx) < 2:
        return list(idx), False

    default = order_clusters(idx, X_norm)
    bonus = float(alternation_bonus)
    if bonus <= 0.0:
        return default, False

    default_min = _min_edge(default, X_norm)

    def score(order: Sequence[int]) -> float:
        return _path_sonic_cost(order, X_norm) + bonus * _alternation_count(
            order, group_of_index
        )

    best = default
    best_score = score(default)
    for start in idx:
        cand = order_clusters(idx, X_norm, start=start)
        if _min_edge(cand, X_norm) < default_min - 1e-12:
            continue  # would break the worst-edge floor -- never a legal winner
        s = score(cand)
        if s > best_score + 1e-12:
            best, best_score = cand, s

    improved = best != default
    if improved:
        logger.info(
            "Multi-artist ordering: alternation preference changed the pier walk "
            "(%d -> %d artist changes, sonic cost %.4f -> %.4f)",
            _alternation_count(default, group_of_index),
            _alternation_count(best, group_of_index),
            _path_sonic_cost(default, X_norm),
            _path_sonic_cost(best, X_norm),
        )
    return best, improved


@dataclass
class MultiArtistPiers:
    ordered_medoids: List[int]
    relaxations: List[Dict[str, Any]]
    groups: List[ArtistGroup]
    mean_affinity: float
    blocked_artist_keys: frozenset


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
) -> Optional[MultiArtistPiers]:
    """Pick and order the piers for a 2+ artist blend.

    Returns None when fewer than two groups survive -- the caller then runs the
    normal single-artist path unchanged.

    Relaxation entries mirror the shape pier_bridge_builder already emits for the
    ``RelaxationNotice`` component (``web/src/lib/types.ts::RelaxationEntry``):
    ``{"type": "relaxation", "scope": ..., "bridge": str, "relaxed": [str, ...],
    "severity": str}``. ``bridge`` and ``relaxed`` are rendered verbatim as
    "Relaxed to fit: {bridge} — dropped {', '.join(relaxed)}"; there is no
    separate free-text "detail" field in the real renderer.
    """
    import math

    from src.playlist.artist_style import cluster_artist_tracks
    from src.string_utils import normalize_artist_key
    from src.playlist.artist_aliases import resolve_alias

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

    exclusive = [g for g in groups if not g.is_joint]
    if len(exclusive) < 2:
        logger.info(
            "Multi-artist: only %d artist group(s) survived — falling back to the "
            "single-artist path.", len(exclusive),
        )
        return None

    protos = group_prototypes(bundle, groups)
    genre_profiles = group_genre_profiles(bundle, groups)

    total = total_pier_budget(track_count, max_artist_fraction, len(groups))
    alloc = allocate_pier_budget(
        groups, total, joint_pier_min_budget=ma_cfg.joint_pier_min_budget,
    )
    logger.info(
        "Multi-artist pier budget: total=%d allocation=%s (track_count=%d, "
        "max_artist_fraction=%.3f)", total, alloc, track_count, max_artist_fraction,
    )

    all_medoids: List[int] = []
    group_of_index: Dict[int, str] = {}
    affinity_by_index: Dict[int, float] = {}
    X_norm_shared: Optional[np.ndarray] = None

    for g in groups:
        want = int(alloc.get(g.label, 0))
        if want <= 0:
            continue
        # A group thinner than the clustering floor (cluster_k_min, e.g. a joint
        # group with only 1-2 tracks) cannot be clustered -- cluster_artist_tracks
        # raises ValueError rather than degrading. That is correct for the
        # single-artist path (an artist with 2 tracks genuinely cannot seed a
        # playlist), but here it is one contributor among several groups: the
        # right behavior is to drop this group's contribution and keep going
        # (design principle 25, "edge cases get graceful fallbacks"), not to
        # crash the whole blend over one thin corner (most often the joint set).
        aff = overlap_affinity(
            bundle, g, groups, protos, genre_profiles, genre_share=ma_cfg.genre_share,
        )
        k_predict = max(1, min(int(style_cfg.cluster_k_max), len(g.indices)))
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
                overlap_affinity=aff,
                overlap_weight=ma_cfg.overlap_weight,
            )
        except ValueError as exc:
            logger.warning(
                "Multi-artist: '%s' (%d track(s)) could not be clustered (%s) — "
                "contributes no piers.", g.label, len(g.indices), exc,
            )
            relaxations.append({
                "type": "relaxation",
                "scope": "multi_artist",
                "bridge": f"{g.label}'s pier budget",
                "relaxed": [
                    f"{g.label} ({len(g.indices)} track(s) — too few to cluster)"
                ],
                "severity": "info",
            })
            continue
        X_norm_shared = X_norm
        chosen = list(medoids)[:want]
        for m in chosen:
            group_of_index[int(m)] = g.label
            affinity_by_index[int(m)] = float(aff[int(m)])
        all_medoids.extend(int(m) for m in chosen)
        if len(chosen) < want:
            relaxations.append({
                "type": "relaxation",
                "scope": "multi_artist",
                "bridge": f"{g.label}'s pier budget",
                "relaxed": [
                    f"{g.label} ({len(chosen)}/{want} pier(s) seated, "
                    f"{len(g.indices)} track(s) in your library)"
                ],
                "severity": "info",
            })

    if not all_medoids:
        logger.warning("Multi-artist: no medoids produced — falling back.")
        return None

    ordered, _improved = order_with_alternation(
        all_medoids, X_norm_shared, group_of_index,
        alternation_bonus=ma_cfg.alternation_bonus,
    )

    mean_affinity = (
        float(np.mean([affinity_by_index.get(i, 0.0) for i in ordered]))
        if ordered else 0.0
    )
    if mean_affinity < ma_cfg.low_overlap_threshold:
        detail = (
            f"{' & '.join(g.label for g in exclusive)} share little sonic ground "
            f"(mean pier affinity {mean_affinity:.2f})"
        )
        relaxations.append({
            "type": "relaxation",
            "scope": "multi_artist",
            "bridge": detail,
            "relaxed": ["piers stayed characteristic rather than meeting in the middle"],
            "severity": "info",
        })
        logger.warning("Multi-artist: %s", detail)

    # blocked_artist_keys becomes pier_bridge_builder._derive_seed_artist_keys's
    # `override`, which does `frozenset(str(k) for k in override if k)` -- handing
    # it a bare string would silently iterate CHARACTERS. Built directly as a
    # frozenset comprehension here, so it is never anything else; asserted
    # defensively because that call site has no such guard of its own.
    blocked = frozenset(
        resolve_alias(normalize_artist_key(g.label))
        for g in exclusive
    )
    assert isinstance(blocked, frozenset), (
        "blocked_artist_keys must be a frozenset of artist keys, never a bare string"
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
    )
