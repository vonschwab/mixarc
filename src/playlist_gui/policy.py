"""
Policy Layer - Derives runtime configuration from UI state.

This module implements a pure policy derivation layer that translates
user-facing controls (UIStateModel) into backend configuration overrides.

Key design principles:
1. Pure functions: derive_runtime_config has no side effects
2. Policy owns certain keys: merge_overrides ensures policy wins for these
3. Transparent decisions: PolicyDecisions includes human-readable notes

Created: Phase 1 of GUI "Just Works" implementation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .ui_state import UIStateModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy-owned keys
# ─────────────────────────────────────────────────────────────────────────────

POLICY_OWNED_KEYS: Set[str] = {
    # Mode-derived settings
    "playlists.genre_mode",
    "playlists.sonic_mode",
    "playlists.pace_mode",
    "playlists.cohesion_mode",
    # Recency settings (controlled by simplified UI)
    "playlists.recently_played_filter.enabled",
    "playlists.recently_played_filter.lookback_days",
    "playlists.recently_played_filter.min_playcount_threshold",
    # Artist spacing
    "playlists.ds_pipeline.constraints.min_gap",
    # Track count
    "playlists.tracks_per_playlist",
    # Diversity bonus
    "playlists.ds_pipeline.scoring.gamma",
    "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
    # DJ bridging core settings (gated by policy rules)
    "playlists.ds_pipeline.pier_bridge.dj_bridging.enabled",
    "playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering",
    # Pooling strategy (tied to DJ bridging gating)
    "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy",
    "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre",
    # Artist presence (seed artist share)
    "playlists.ds_pipeline.candidate_pool.max_artist_fraction",
    # Instrumental lean (soft penalty; demotes vocal-classified tracks)
    "playlists.ds_pipeline.pier_bridge.instrumental_enabled",
    # Live-album ban (song-scoped; live-only songs rescued)
    "playlists.ds_pipeline.candidate_pool.exclude_live_albums",
}
"""
Keys that policy must always win for, even if Advanced Panel has values.
This ensures the simplified UI controls take precedence.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Genre/Sonic mode validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_MODES: Set[str] = {"strict", "narrow", "dynamic", "discover", "off"}
VALID_PACE_MODES: Set[str] = {"strict", "narrow", "dynamic", "off"}
VALID_COHESION_MODES: Set[str] = {"strict", "narrow", "dynamic", "discover"}

# ─────────────────────────────────────────────────────────────────────────────
# GUI dials (spec docs/superpowers/specs/2026-07-04-gui-dials-design.md)
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for the dial -> axis translation. Every detent maps
# to a grid-verified engine state (slider-differentiation re-eval 2026-07-04).
# Engine 'off' states for sonic/genre, cohesion 'narrow'/'strict', and pace 'strict'
# are deliberately NOT reachable from the GUI (config-only). cohesion 'strict' (the
# former Flow "Journey") was pulled 2026-07-06: on tested seeds it gave no measurable
# benefit over Normal/Smooth (same coverage/variety, only rougher seams) AND it
# hot-loops on spread-out seeds. Flow is now a two-position smoothness control; reach
# strict via config (playlists.cohesion_mode) only. See the 2026-07-06 spec amendment.
DIAL_DEFAULTS: Dict[str, str] = {"range": "open", "flow": "normal", "pace": "natural"}

DIAL_TO_AXES: Dict[str, Dict[str, Dict[str, str]]] = {
    "range": {
        "home":   {"sonic_mode": "strict",   "genre_mode": "strict"},
        "close":  {"sonic_mode": "narrow",   "genre_mode": "narrow"},
        "open":   {"sonic_mode": "dynamic",  "genre_mode": "dynamic"},
        "wander": {"sonic_mode": "discover", "genre_mode": "discover"},
    },
    "flow": {
        "normal": {"cohesion_mode": "dynamic"},     # default blend
        "smooth": {"cohesion_mode": "discover"},    # pure-T beam: smoothest joins
    },
    "pace": {
        "locked_in": {"pace_mode": "narrow"},  # over strict: same discipline, lower crater risk
        "natural":   {"pace_mode": "dynamic"},
        "free":      {"pace_mode": "off"},
    },
}


def resolve_dial_axes(
    range_dial: Optional[str], flow_dial: Optional[str], pace_dial: Optional[str]
) -> Dict[str, str]:
    """Translate the three GUI dials into the four engine axis modes.

    An unrecognized detent value falls back to the dial's default with a
    logged warning — a dial position must never be a silent no-op
    (Langer/placebo-control rule). A None/absent dial silently uses the
    default; that means "no dial sent" and is not a warning-worthy event.
    """
    axes: Dict[str, str] = {}
    for dial, value in (("range", range_dial), ("flow", flow_dial), ("pace", pace_dial)):
        detents = DIAL_TO_AXES[dial]
        v = str(value).strip().lower() if value else DIAL_DEFAULTS[dial]
        if v not in detents:
            logger.warning("Unknown %s dial detent %r; using %r", dial, value, DIAL_DEFAULTS[dial])
            v = DIAL_DEFAULTS[dial]
        axes.update(detents[v])
    return axes


# ─────────────────────────────────────────────────────────────────────────────
# Artist spacing mapping
# ─────────────────────────────────────────────────────────────────────────────

SPACING_MAP: Dict[str, int] = {
    "loose": 3,
    "normal": 6,
    "strong": 9,
    "very_strong": 12,
}

# Artist presence mapping (seed artist target share via per-artist cap)
PRESENCE_MAP: Dict[str, float] = {
    "very_low": 0.05,
    "low": 0.10,
    "medium": 0.125,
    "high": 0.20,
    "very_high": 0.33,
}

# Oops, All Bangers OOPS baseline: OOPS owns sonic + pace (loosened "radio" tier),
# the user still owns genre. Tunable; calibrate by ear (see spec §10).
_OOPS_SONIC_MODE = "dynamic"
_OOPS_PACE_MODE = "dynamic"


# ─────────────────────────────────────────────────────────────────────────────
# PolicyDecisions dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyDecisions:
    """
    Result of policy derivation from UIStateModel.

    Contains both the overrides dict (for passing to worker) and
    explicit decision flags (for logging/display).
    """

    overrides: Dict[str, Any] = field(default_factory=dict)
    """
    Nested override dictionary to merge with base config.
    Structure matches config.yaml hierarchy.
    """

    dj_bridging_enabled: bool = False
    """Whether DJ bridging is enabled based on policy rules."""

    genre_pool_enabled: bool = False
    """
    Whether genre pool enrichment is desired.

    Note: Genre pool (S3 in dj_union) exists ONLY inside DJ bridging today.
    If dj_bridging_enabled is False, genre_pool_enabled has no effect.
    This decision is tracked for future use when/if genre pool becomes
    independently available.
    """

    seed_ordering_value: str = "fixed"
    """
    The actual config enum value for seed ordering.
    - "auto": optimize order (when ui.seed_auto_order=True)
    - "fixed": preserve user order (when ui.seed_auto_order=False)
    """

    min_gap: int = 6
    """The artist spacing min_gap value."""

    notes: List[str] = field(default_factory=list)
    """
    Human-readable notes about policy decisions.
    Useful for logging and debugging.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _set_nested(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a nested dictionary value using dot notation.

    Example: _set_nested(d, "a.b.c", 42) sets d["a"]["b"]["c"] = 42
    """
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested(d: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested dictionary value using dot notation.

    Example: _get_nested(d, "a.b.c") returns d["a"]["b"]["c"]
    """
    keys = key_path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _count_unique_artists(artist_keys: List[str]) -> int:
    """Count unique artists from a list of artist keys."""
    return len(set(k.lower().strip() for k in artist_keys if k))


# ─────────────────────────────────────────────────────────────────────────────
# Merge function
# ─────────────────────────────────────────────────────────────────────────────

def merge_overrides(
    user_overrides: Dict[str, Any],
    policy_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge user overrides (from Advanced Panel) with policy overrides.

    Policy ALWAYS wins for POLICY_OWNED_KEYS, even if user set values.
    For other keys, user overrides are preserved.

    Args:
        user_overrides: Overrides from Advanced Panel (nested dict)
        policy_overrides: Overrides from PolicyLayer (nested dict)

    Returns:
        Merged overrides dictionary
    """
    # Deep copy user overrides
    import copy
    result = copy.deepcopy(user_overrides)

    # Deep merge policy overrides
    def deep_merge(base: Dict, overlay: Dict) -> None:
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(result, policy_overrides)

    # For policy-owned keys, ensure policy value wins even if user had set it
    # This is already handled by deep_merge since policy comes second,
    # but we explicitly verify for clarity
    for key_path in POLICY_OWNED_KEYS:
        policy_value = _get_nested(policy_overrides, key_path)
        if policy_value is not None:
            _set_nested(result, key_path, policy_value)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main policy derivation function
# ─────────────────────────────────────────────────────────────────────────────

def derive_runtime_config(
    ui: UIStateModel,
    *,
    seed_artist_keys: Optional[List[str]] = None,
) -> PolicyDecisions:
    """
    Derive runtime configuration from UI state.

    This is a pure function that translates user-facing controls into
    backend configuration. All policy rules are implemented here.

    Args:
        ui: The UI state model
        seed_artist_keys: Optional list of artist keys for each seed track.
            Required for accurate DJ bridging gating (unique artist check).
            If None, DJ bridging is conservatively disabled for seeds mode.

    Returns:
        PolicyDecisions with overrides and decision flags
    """
    decisions = PolicyDecisions()
    overrides: Dict[str, Any] = {}
    notes: List[str] = []

    # ─────────────────────────────────────────────────────────────────────
    # 1. Genre/Sonic modes
    # ─────────────────────────────────────────────────────────────────────
    genre_mode = ui.genre_mode if ui.genre_mode in VALID_MODES else "dynamic"
    sonic_mode = ui.sonic_mode if ui.sonic_mode in VALID_MODES else "dynamic"
    _set_nested(overrides, "playlists.genre_mode", genre_mode)
    _set_nested(overrides, "playlists.sonic_mode", sonic_mode)
    pace_mode = ui.pace_mode if getattr(ui, "pace_mode", "dynamic") in VALID_PACE_MODES else "dynamic"
    _set_nested(overrides, "playlists.pace_mode", pace_mode)
    cohesion_mode = getattr(ui, "cohesion_mode", "dynamic")
    if cohesion_mode not in VALID_COHESION_MODES:
        cohesion_mode = "dynamic"
    _set_nested(overrides, "playlists.cohesion_mode", cohesion_mode)
    steering_tags = [
        str(t).strip() for t in getattr(ui, "steering_tags", []) if str(t).strip()
    ][:3]
    if steering_tags:
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.tag_steering_tags",
            steering_tags,
        )
        notes.append(f"Tag steering: {', '.join(steering_tags)}")
    notes.append(f"Genre mode: {genre_mode}")
    notes.append(f"Sonic mode: {sonic_mode}")
    notes.append(f"Pace mode: {pace_mode}")
    notes.append(f"Cohesion mode: {cohesion_mode}")

    # Oops, All Bangers: OOPS rests looser on sonic/pace ("FM radio" baseline);
    # genre_mode is left as the user's umbrella width.
    if getattr(ui, "popularity_mode", "off") == "oops":
        _set_nested(overrides, "playlists.sonic_mode", _OOPS_SONIC_MODE)
        _set_nested(overrides, "playlists.pace_mode", _OOPS_PACE_MODE)
        notes.append(f"OOPS baseline: sonic_mode->{_OOPS_SONIC_MODE}, pace_mode->{_OOPS_PACE_MODE} (genre untouched)")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Track count
    # ─────────────────────────────────────────────────────────────────────
    _set_nested(overrides, "playlists.tracks_per_playlist", ui.track_count)

    # ─────────────────────────────────────────────────────────────────────
    # 2b. Diversity bonus (soft)
    # ─────────────────────────────────────────────────────────────────────
    _set_nested(
        overrides,
        "playlists.ds_pipeline.scoring.gamma",
        float(ui.diversity_gamma),
    )
    notes.append(f"Diversity gamma: {float(ui.diversity_gamma):.3f}")
    if getattr(ui, "artist_diversity_mode", "weighted") == "one_per_artist":
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.max_non_seed_tracks_per_artist",
            1,
        )
        notes.append("Artist diversity: one track per non-seed artist")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Recency filter (never relaxed by policy)
    # ─────────────────────────────────────────────────────────────────────
    _set_nested(overrides, "playlists.recently_played_filter.enabled", ui.recency_enabled)
    _set_nested(overrides, "playlists.recently_played_filter.lookback_days", ui.recency_days)
    _set_nested(
        overrides,
        "playlists.recently_played_filter.min_playcount_threshold",
        ui.recency_plays_threshold,
    )
    if ui.recency_enabled:
        notes.append(
            f"Recency filter: exclude tracks played >= {ui.recency_plays_threshold}x "
            f"in last {ui.recency_days} days"
        )
    else:
        notes.append("Recency filter: disabled")

    # ─────────────────────────────────────────────────────────────────────
    # 3b. Instrumental lean (soft penalty; demotes vocal-classified tracks)
    # ─────────────────────────────────────────────────────────────────────
    _set_nested(overrides, "playlists.ds_pipeline.pier_bridge.instrumental_enabled", ui.instrumental)
    if ui.instrumental:
        notes.append("Instrumental lean: demote vocal-classified tracks")

    # ─────────────────────────────────────────────────────────────────────
    # 3c. Live-album ban (song-scoped; live-only songs rescued)
    # ─────────────────────────────────────────────────────────────────────
    _set_nested(overrides, "playlists.ds_pipeline.candidate_pool.exclude_live_albums", ui.exclude_live)
    if not ui.exclude_live:
        notes.append("Live albums: allowed (marked-live tracks may appear)")

    # ─────────────────────────────────────────────────────────────────────
    # 4. Artist spacing → min_gap
    # ─────────────────────────────────────────────────────────────────────
    min_gap = SPACING_MAP.get(ui.artist_spacing, 6)

    # Override: For same-artist seed playlists, disable artist diversity
    # to allow repeated use of the seed artist in interior positions
    if seed_artist_keys and len(set(seed_artist_keys)) == 1:
        min_gap = 0
        notes.append(f"Artist spacing '{ui.artist_spacing}' → min_gap={min_gap} (overridden: all seeds same artist)")
    else:
        notes.append(f"Artist spacing '{ui.artist_spacing}' → min_gap={min_gap}")

    _set_nested(overrides, "playlists.ds_pipeline.constraints.min_gap", min_gap)
    decisions.min_gap = min_gap

    # ─────────────────────────────────────────────────────────────────────
    # 4b. Artist presence (artist mode only)
    # ─────────────────────────────────────────────────────────────────────
    if ui.mode == "artist":
        max_artist_fraction = PRESENCE_MAP.get(ui.artist_presence, 0.125)
        _set_nested(
            overrides,
            "playlists.ds_pipeline.candidate_pool.max_artist_fraction",
            max_artist_fraction,
        )
        notes.append(
            f"Artist presence '{ui.artist_presence}' → "
            f"max_artist_fraction={max_artist_fraction:.3f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # 5. DJ bridging gating
    # ─────────────────────────────────────────────────────────────────────
    dj_bridging_enabled = False

    if ui.mode == "artist":
        # DJ bridging disabled in artist mode
        notes.append("DJ bridging disabled: mode is 'artist' (single-artist mode)")
    elif ui.mode == "seeds":
        # Evaluate DJ bridging eligibility for seeds mode
        seed_count = ui.seed_count()

        if seed_count < 2:
            notes.append(f"DJ bridging disabled: need >= 2 seeds, have {seed_count}")
        elif seed_artist_keys is None:
            # Conservative: cannot verify unique artists
            notes.append(
                "DJ bridging disabled: seed_artist_keys not provided "
                "(Phase 1 limitation; Seeds UI must resolve track IDs to artist keys)"
            )
        else:
            # Check unique artists
            unique_artists = _count_unique_artists(seed_artist_keys)
            if unique_artists < 2:
                notes.append(
                    f"DJ bridging disabled: need >= 2 unique artists, "
                    f"have {unique_artists} from {seed_count} seeds"
                )
            else:
                # All conditions met
                dj_bridging_enabled = True
                notes.append(
                    f"DJ bridging enabled: {seed_count} seeds from "
                    f"{unique_artists} unique artists"
                )
    else:
        notes.append(f"DJ bridging disabled: unsupported mode '{ui.mode}'")

    decisions.dj_bridging_enabled = dj_bridging_enabled
    _set_nested(
        overrides,
        "playlists.ds_pipeline.pier_bridge.dj_bridging.enabled",
        dj_bridging_enabled,
    )

    # ─────────────────────────────────────────────────────────────────────
    # 6. Seed ordering
    # ─────────────────────────────────────────────────────────────────────
    # Verified from pier_bridge_builder.py:3720-3741:
    # - "auto" = optimize order (calls _order_seeds_by_bridgeability)
    # - "fixed" = preserve user order
    if ui.seed_auto_order:
        seed_ordering_value = "auto"
        notes.append("Seed ordering: auto (optimize for bridging)")
    else:
        seed_ordering_value = "fixed"
        notes.append("Seed ordering: fixed (preserve user order)")

    decisions.seed_ordering_value = seed_ordering_value
    _set_nested(
        overrides,
        "playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering",
        seed_ordering_value,
    )

    # ─────────────────────────────────────────────────────────────────────
    # 7. Genre pool gating
    # ─────────────────────────────────────────────────────────────────────
    # Genre pool (S3 in dj_union strategy) is desired when genre_mode == "discover"
    genre_pool_enabled = genre_mode == "discover"
    decisions.genre_pool_enabled = genre_pool_enabled

    if genre_pool_enabled:
        notes.append("Genre pool: enabled (genre_mode is 'discover')")
    else:
        notes.append(f"Genre pool: disabled (genre_mode is '{genre_mode}')")

    # Genre pool requires DJ bridging to actually work
    # (verified: dj_union pooling strategy is DJ-only)
    if genre_pool_enabled and dj_bridging_enabled:
        # Enable dj_union pooling with genre pool
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy",
            "dj_union",
        )
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre",
            80,  # Default k_genre for genre pool
        )
        notes.append("Pooling strategy: dj_union with k_genre=80")
    elif genre_pool_enabled and not dj_bridging_enabled:
        notes.append(
            "Genre pool desired but unavailable: "
            "dj_union pooling requires DJ bridging to be enabled"
        )
        # Set baseline pooling as fallback
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy",
            "baseline",
        )
    else:
        # Genre pool not desired or DJ bridging disabled
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy",
            "baseline",
        )
        _set_nested(
            overrides,
            "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_genre",
            0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────────
    decisions.overrides = overrides
    decisions.notes = notes

    return decisions
