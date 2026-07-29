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

    names = [str(a).strip() for a in artist_names if str(a).strip()]
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

    groups: List[ArtistGroup] = []
    dropped: List[str] = []
    for name in names:
        exclusive = sorted(raw[name] - set(joint_indices))
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
            dropped.append(name)
            logger.warning(
                "Multi-artist: '%s' has no usable tracks in the library — "
                "dropped from the pairing.", name,
            )
            continue
        groups.append(ArtistGroup(label=name, indices=exclusive, is_joint=False))

    if joint_indices:
        kept = [g.label for g in groups]
        groups.append(
            ArtistGroup(
                label=" & ".join(kept) if kept else "joint",
                indices=joint_indices,
                is_joint=True,
                source_artists=tuple(kept),
            )
        )
        logger.info(
            "Multi-artist: %d jointly-credited track(s) across %s",
            len(joint_indices), kept,
        )
    return groups, dropped
