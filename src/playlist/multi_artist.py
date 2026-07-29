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
from typing import Any, Dict

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
