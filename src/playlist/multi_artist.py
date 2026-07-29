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

    The joint group claims exactly one seat when it is non-empty and the total
    reaches ``joint_pier_min_budget``. No group is allocated more piers than it
    has tracks; the surplus reallocates to groups that can still use it.
    """
    total = max(0, int(total))
    if not groups or total == 0:
        return {}

    alloc: Dict[str, int] = {g.label: 0 for g in groups}
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
    if not exclusive:
        # Everything to the joint group, capped by its track count.
        if joint is not None and joint.indices:
            alloc[joint.label] = min(total, len(joint.indices))
        return {k: v for k, v in alloc.items() if v > 0}

    # Even split, remainder to the first chip.
    per, rem = divmod(remaining, len(exclusive))
    for i, g in enumerate(exclusive):
        alloc[g.label] += per + (1 if i < rem else 0)

    # Cap by available tracks, then redistribute the surplus.
    capacity = {g.label: len(g.indices) for g in groups}
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
    while surplus > 0:
        takers = [g.label for g in exclusive if alloc[g.label] < capacity[g.label]]
        if not takers:
            break
        for label in takers:
            if surplus == 0:
                break
            alloc[label] += 1
            surplus -= 1

    return {k: v for k, v in alloc.items() if v > 0}
