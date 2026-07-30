"""
UI State Model - Single source of truth for generation UI state.

This dataclass captures all user-facing controls for playlist generation,
independent of the underlying config format. The PolicyLayer derives
runtime configuration from this state.

Created: Phase 1 of GUI "Just Works" implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


ModeValue = Literal["strict", "narrow", "dynamic", "discover", "off"]
PaceModeValue = Literal["strict", "narrow", "dynamic", "off"]


@dataclass
class UIStateModel:
    """
    Single source of truth for "Generate" UI state.

    All fields have sensible defaults matching the target UX.
    The PolicyLayer reads this to derive runtime config overrides.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Top-level mode
    # ─────────────────────────────────────────────────────────────────────────
    mode: Literal["artist", "genre", "seeds", "history"] = "artist"

    # ─────────────────────────────────────────────────────────────────────────
    # Genre/Sonic modes (strictness controls for matching)
    # ─────────────────────────────────────────────────────────────────────────
    cohesion_mode: Literal["strict", "narrow", "dynamic", "discover"] = "dynamic"
    """
    Overall cohesion (pier-bridge beam tightness). Independent of
    genre_mode/sonic_mode/pace_mode (which control pool composition).
    Single writer: policy.derive_runtime_config() → playlists.cohesion_mode.
    """

    genre_mode: ModeValue = "narrow"
    sonic_mode: ModeValue = "narrow"
    pace_mode: PaceModeValue = "dynamic"

    # ─────────────────────────────────────────────────────────────────────────
    # Track count
    # ─────────────────────────────────────────────────────────────────────────
    track_count: int = 30

    # Diversity bonus (soft)
    # ─────────────────────────────────────────────────────────────────────────
    diversity_gamma: float = 0.04
    artist_diversity_mode: Literal["weighted", "one_per_artist"] = "weighted"
    """
    Controls whether diversity is only a soft scoring bonus or a hard cap.
    - weighted: use diversity_gamma only
    - one_per_artist: allow at most one non-seed track per artist
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Recency filter (hard exclude; never relaxed by policy)
    # Semantics: exclude tracks played >= recency_plays_threshold times
    #            in the last recency_days days
    # ─────────────────────────────────────────────────────────────────────────
    recency_enabled: bool = True
    recency_days: int = 14
    recency_plays_threshold: int = 1
    exclude_seed_tracks_from_recency: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Instrumental lean (soft penalty; demotes vocal-classified tracks)
    # ─────────────────────────────────────────────────────────────────────────
    instrumental: bool = False

    history_window_days: int = 30

    genre_query: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Artist spacing
    # ─────────────────────────────────────────────────────────────────────────
    artist_spacing: Literal["loose", "normal", "strong", "very_strong"] = "normal"

    # ─────────────────────────────────────────────────────────────────────────
    # Artist(s) mode specific
    # ─────────────────────────────────────────────────────────────────────────
    artist_queries: List[str] = field(default_factory=list)
    """
    Artist name chips. 2+ entries activate the multi-artist blend
    (spec 2026-07-29-multi-artist-blend); the first chip remains the
    primary artist for logging and single-artist fallback.
    """

    artist_presence: Literal[
        "very_low",
        "low",
        "medium",
        "high",
        "very_high",
    ] = "medium"
    """
    Target percentage of playlist from the seed artist(s):
    - very_low: ~5%
    - low: ~10%
    - medium: ~12.5%
    - high: ~20%
    - very_high: ~33%
    """

    artist_variety: Literal["focused", "balanced", "sprawling"] = "balanced"
    """
    Controls internal artist seed dispersion (clustering variance).
    - focused: tight style clustering
    - balanced: moderate diversity
    - sprawling: wide stylistic spread
    """

    include_collaborations: bool = False
    """
    When True, collaboration tracks (e.g. "Miles Davis Quintet",
    "Greg Foat & Art Themen", "Miles Davis & John Coltrane") are mixed
    into the seed candidate pool alongside solo tracks. When False,
    only exact-artist tracks are used unless the artist has < 4 solo
    tracks (in which case collaborations are added as a fallback).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Seed(s) mode specific
    # ─────────────────────────────────────────────────────────────────────────
    seed_track_ids: List[str] = field(default_factory=list)
    """
    Stable track IDs for seeds. These should be database IDs, not display strings.
    The Seeds UI must resolve track names to IDs before populating this field.
    """

    steering_tags: List[str] = field(default_factory=list)
    """
    Genre-tag steering: canonical tag names selected in the GUI (≤3; the GUI
    only surfaces the picker in artist mode, but the engine honors tags in any
    mode). Empty = steering off. Single writer of the runtime knob:
    policy.derive_runtime_config() → playlists.ds_pipeline.pier_bridge.tag_steering_tags.
    """

    seed_auto_order: bool = True
    """
    When True, seeds are reordered for optimal bridging flow.
    When False, seeds are used in the order provided by the user.

    Implementation note (verified from pier_bridge_builder.py):
    - seed_auto_order=True maps to dj_seed_ordering="auto" (optimizes order)
    - seed_auto_order=False maps to dj_seed_ordering="fixed" (preserves order)
    """

    popularity_mode: str = "off"  # Oops All Bangers: off / on / oops

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def primary_artist(self) -> Optional[str]:
        """
        Return the first artist chip, or None if empty.

        Used for single-artist mode until multi-artist journeys are implemented.
        """
        if self.artist_queries:
            return self.artist_queries[0]
        return None

    def seed_count(self) -> int:
        """Return the number of seed tracks."""
        return len(self.seed_track_ids)
