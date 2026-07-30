"""
Tier-3.1 PR-8: beam search engine.

Extracted verbatim from pier_bridge_builder.py:
  BeamState              — dataclass carrying one beam-search path
  _beam_search_segment   — constrained beam search from pier_a to pier_b
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.cancellation import raise_if_cancelled
from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    resolve_artist_identity_keys,
)
from src.playlist.artist_aliases import sibling_group_of
from src.playlist.pier_bridge.config import PierBridgeConfig, _compute_transition_score
from src.playlist.pier_bridge.seed_character import anti_center_penalty
from src.playlist.layered_genre_scoring import score_layered_transition
from src.playlist.title_quality import compute_title_artifact_penalty
from src.playlist.pier_bridge.genre import (
    _compute_coverage,
    _compute_coverage_bonus,
    _extract_top_genres,
)
from src.playlist.pier_bridge.percentiles import floor_at_percentile
from src.playlist.pier_bridge.corridors import corridor_penalty
from src.playlist.pier_bridge.metrics import (
    _progress_arc_loss_value,
    _progress_target_curve,
    _step_fraction,
)
from src.playlist.transition_metrics import (
    TransitionMetricContext,
    is_broken_transition,
    score_transition_edge,
)

logger = logging.getLogger(__name__)


def _local_sonic_penalty_value(
    *,
    edge_cos: float,
    threshold: float,
    strength: float,
    scale: float,
    mode: str,
) -> float:
    """Compute the per-edge local-sonic penalty.

    legacy (default): penalty = strength * max(0, threshold - edge_cos)
    scaled:           penalty = scale    * max(0, threshold - edge_cos)
    Any other mode value falls back to legacy.
    """
    if not (edge_cos < threshold):
        return 0.0
    gap = float(threshold) - float(edge_cos)
    if str(mode).strip().lower() == "scaled":
        return float(max(0.0, float(scale) * gap))
    return float(max(0.0, float(strength) * gap))


def _layered_bundle_matrices(bundle: Optional[ArtifactBundle]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if bundle is None:
        return None
    matrices = (
        getattr(bundle, "X_genre_leaf_idf", None),
        getattr(bundle, "X_genre_family", None),
        getattr(bundle, "X_genre_bridge", None),
        getattr(bundle, "X_facet", None),
    )
    if any(matrix is None for matrix in matrices):
        return None
    leaf = np.asarray(matrices[0], dtype=float)
    family = np.asarray(matrices[1], dtype=float)
    bridge = np.asarray(matrices[2], dtype=float)
    facet = np.asarray(matrices[3], dtype=float)
    if any(matrix.ndim != 2 for matrix in (leaf, family, bridge, facet)):
        return None
    if bridge.shape[1] != leaf.shape[1]:
        return None
    row_count = leaf.shape[0]
    if any(matrix.shape[0] != row_count for matrix in (family, bridge, facet)):
        return None
    return leaf, family, bridge, facet


def _coerce_edge_score(value: Any, *, fallback: Any, default: float) -> float:
    raw = value if isinstance(value, (int, float)) else fallback
    if not isinstance(raw, (int, float)):
        return float(default)
    if raw != raw:
        return float(default)
    return max(0.0, min(1.0, float(raw)))


def _state_min_edge(s) -> float:
    """The weakest single-edge transition score along a beam state's path.

    Reads each edge's beam transition score (``T``, falling back to
    ``trans_score_in_beam``); returns -1e18 for an edge-less state. Shared by the
    final min-edge selection and the roam-corridors per-step minimax prune.
    """
    edges = getattr(s, "edge_components", None) or []
    vals = [
        float(e.get("T", e.get("trans_score_in_beam", -1e18)))
        for e in edges
        if e is not None
    ]
    return min(vals) if vals else -1e18


def _select_best_beam_state(states, *, objective: str = "total_score", min_edge_band: float = 0.0):
    """Pick the winning beam state from a (possibly empty) list.

    objective='total_score' (default): return state with highest state.score.
    objective='min_edge': lexicographic (highest min trans_score_in_beam,
        ties broken by highest total score). Optimizes for 'no broken moments'
        rather than 'good on average'.

    ``min_edge_band`` > 0 relaxes the strict min_edge lexicographic tie-break into a
    TOLERANCE BAND: among states whose worst edge is within ``band`` of the best, pick
    the highest-scoring one. This lets score-borne preferences (the tag-steering term)
    trade a bounded sliver of worst-edge for genre adherence. band=0 is the strict
    minimax default (score is a never-firing tie-break -- see BEAM_CONTRACT I1).
    """
    if not states:
        return None
    if str(objective).strip().lower() == "min_edge":
        band = float(min_edge_band)
        if band > 0.0:
            best_me = max(_state_min_edge(s) for s in states)
            near = [s for s in states if _state_min_edge(s) >= best_me - band]
            return max(near, key=lambda s: float(getattr(s, "score", 0.0)))
        return max(states, key=lambda s: (_state_min_edge(s), float(getattr(s, "score", 0.0))))
    return max(states, key=lambda s: float(getattr(s, "score", 0.0)))


@dataclass
class BeamState:
    """State for beam search."""
    path: List[int]
    score: float
    used: Set[int]
    used_artists: Set[str] = field(default_factory=set)
    used_sibling_groups: Set[str] = field(default_factory=set)
    last_progress: float = 0.0
    edge_components: List[dict] = field(default_factory=list)


def _popularity_factor(p: float, strength: float) -> float:
    """Graded multiplicative popularity demotion for a bridge candidate ("Oops, All Bangers").

    `p` = per-artist Last.fm popularity in [0,1] (1 = the artist's #1 track), or NaN if the
    track isn't charting / the artist is unknown. Returns a factor in (0, 1] to multiply into
    the candidate's score: a banger (p=1) -> 1.0 (no demotion); a deep cut or unknown -> the
    full 1 - strength. NaN is treated as a deep cut (ruthless), graded by `strength`."""
    if strength <= 0.0:
        return 1.0
    d = 1.0 - (p if p == p else 0.0)   # `p == p` is False only for NaN
    return 1.0 - strength * d


def _beam_search_segment(
    pier_a: int,
    pier_b: int,
    interior_length: int,
    candidates: List[int],
    X_full: np.ndarray,
    X_full_norm: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    beam_width: int,
    *,
    X_genre_norm_idf: Optional[np.ndarray] = None,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    X_genre_dense: Optional[np.ndarray] = None,
    genre_idf: Optional[np.ndarray] = None,
    genre_vocab: Optional[np.ndarray] = None,
    artist_key_by_idx: Optional[Dict[int, str]] = None,
    seed_artist_keys: frozenset = frozenset(),
    recent_global_artists: Optional[List[str]] = None,
    durations_ms: Optional[np.ndarray] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
    bundle: Optional[ArtifactBundle] = None,
    arc_stats: Optional[Dict[str, Any]] = None,
    genre_cache_stats: Optional[Dict[str, int]] = None,
    g_targets_override: Optional[List[np.ndarray]] = None,
    waypoint_stats: Optional[Dict[str, Any]] = None,
    local_sonic_stats: Optional[Dict[str, Any]] = None,
    edge_components_out: Optional[Dict[str, Any]] = None,
    transition_metric_context: Optional[TransitionMetricContext] = None,
    perceptual_bpm: Optional[np.ndarray] = None,
    tempo_stability: Optional[np.ndarray] = None,
    onset_rate: Optional[np.ndarray] = None,
    pair_sim_provider: Optional[Any] = None,
    energy_matrix: Optional[np.ndarray] = None,
    voice_prob: Optional[np.ndarray] = None,
    duration_penalty_values: Optional[np.ndarray] = None,
    roam_detour_sonic: Optional[np.ndarray] = None,
    roam_dev_genre: Optional[np.ndarray] = None,
    roam_dev_energy: Optional[np.ndarray] = None,
    popularity_values: Optional[np.ndarray] = None,
    sonic_tag_affinity: Optional[np.ndarray] = None,   # bundle-aligned (N,) centered MuQ tag affinity
    sonic_tag_beam_weight: float = 0.0,
    tag_steering_worst_edge_band: float = 0.0,  # worst-edge tolerance for the tag term to act (see BEAM_CONTRACT I1)
) -> Tuple[Optional[List[int]], int, int, Optional[str]]:
    """
    Constrained beam search to find path from pier_a to pier_b.

    Returns interior track indices (not including piers) or None if no path found.

    Args:
        recent_global_artists: Artist keys from the last min_gap positions of the
            global playlist prefix (from previous segments). Used to enforce
            cross-segment min_gap constraints during generation. If artist_identity_cfg
            is enabled, these are identity keys (collapsing ensemble variants).
        durations_ms: Optional array of track durations in milliseconds (parallel to bundle.track_ids)
        artist_identity_cfg: Optional config for artist identity resolution (ensemble collapsing)
        bundle: Optional bundle for resolving artist strings to identity keys
        waypoint_stats: Optional dict to populate with waypoint influence diagnostics
    """

    genre_penalty_hits = 0
    duration_penalty_hits = 0
    edges_scored = 0
    penalty_strength = float(cfg.genre_penalty_strength)
    if not math.isfinite(penalty_strength):
        penalty_strength = 0.0
    penalty_strength = float(max(0.0, min(1.0, penalty_strength)))
    popularity_penalty_strength = float(getattr(cfg, "popularity_penalty_strength", 0.0))
    if not math.isfinite(popularity_penalty_strength):
        popularity_penalty_strength = 0.0
    popularity_penalty_strength = float(max(0.0, min(1.0, popularity_penalty_strength)))
    penalty_threshold = float(cfg.genre_penalty_threshold)
    genre_tie_break_band = cfg.genre_tie_break_band
    if isinstance(genre_tie_break_band, (int, float)) and math.isfinite(float(genre_tie_break_band)):
        genre_tie_break_band = float(genre_tie_break_band)
        if genre_tie_break_band <= 0:
            genre_tie_break_band = None
    else:
        genre_tie_break_band = None

    local_sonic_penalty_enabled = bool(cfg.local_sonic_edge_penalty_enabled)
    local_sonic_penalty_threshold = float(cfg.local_sonic_edge_penalty_threshold)
    if not math.isfinite(local_sonic_penalty_threshold):
        local_sonic_penalty_threshold = 0.0
    local_sonic_penalty_strength = float(cfg.local_sonic_edge_penalty_strength)
    if not math.isfinite(local_sonic_penalty_strength):
        local_sonic_penalty_strength = 0.0
    local_sonic_penalty_strength = float(max(0.0, local_sonic_penalty_strength))
    local_sonic_penalty_mode = str(cfg.local_sonic_edge_penalty_mode or "legacy").strip().lower()
    local_sonic_penalty_scale = float(cfg.local_sonic_edge_penalty_scale)
    if not math.isfinite(local_sonic_penalty_scale):
        local_sonic_penalty_scale = 1.0
    local_sonic_penalty_scale = float(max(0.0, local_sonic_penalty_scale))
    local_sonic_floor = cfg.local_sonic_edge_floor
    if isinstance(local_sonic_floor, (int, float)) and math.isfinite(float(local_sonic_floor)):
        local_sonic_floor = float(local_sonic_floor)
    else:
        local_sonic_floor = None
    local_sonic_policy_active = (
        (local_sonic_penalty_enabled and local_sonic_penalty_strength > 0.0)
        or local_sonic_floor is not None
    )
    local_sonic_penalty_hits = 0
    local_sonic_edges_scored = 0
    local_sonic_gate_rejected = 0
    local_sonic_penalty_total = 0.0
    local_sonic_min_edge: Optional[float] = None

    def _apply_local_sonic_edge_policy(
        score: float,
        a_idx: int,
        b_idx: int,
    ) -> Optional[float]:
        nonlocal local_sonic_penalty_hits
        nonlocal local_sonic_edges_scored
        nonlocal local_sonic_gate_rejected
        nonlocal local_sonic_penalty_total
        nonlocal local_sonic_min_edge

        if not local_sonic_policy_active:
            return float(score)

        edge_sonic = float(np.dot(X_full_norm[int(a_idx)], X_full_norm[int(b_idx)]))
        if not math.isfinite(edge_sonic):
            edge_sonic = 0.0
        local_sonic_edges_scored += 1
        local_sonic_min_edge = (
            edge_sonic
            if local_sonic_min_edge is None
            else min(float(local_sonic_min_edge), edge_sonic)
        )

        if local_sonic_floor is not None and edge_sonic < float(local_sonic_floor):
            local_sonic_gate_rejected += 1
            return None

        if local_sonic_penalty_enabled:
            penalty = _local_sonic_penalty_value(
                edge_cos=edge_sonic,
                threshold=local_sonic_penalty_threshold,
                strength=local_sonic_penalty_strength,
                scale=local_sonic_penalty_scale,
                mode=local_sonic_penalty_mode,
            )
            if penalty > 0.0:
                local_sonic_penalty_hits += 1
                local_sonic_penalty_total += float(penalty)
                return float(score) - float(penalty)

        return float(score)

    def _record_local_sonic_stats() -> None:
        if local_sonic_stats is None:
            return
        local_sonic_stats.update(
            {
                "local_sonic_penalty_enabled": bool(local_sonic_penalty_enabled),
                "local_sonic_penalty_threshold": float(local_sonic_penalty_threshold),
                "local_sonic_penalty_strength": float(local_sonic_penalty_strength),
                "local_sonic_penalty_mode": str(local_sonic_penalty_mode),
                "local_sonic_penalty_scale": float(local_sonic_penalty_scale),
                "local_sonic_edge_floor": (
                    float(local_sonic_floor) if local_sonic_floor is not None else None
                ),
                "local_sonic_edges_scored": int(local_sonic_edges_scored),
                "local_sonic_penalty_hits": int(local_sonic_penalty_hits),
                "local_sonic_gate_rejected": int(local_sonic_gate_rejected),
                "local_sonic_penalty_total": float(local_sonic_penalty_total),
                "local_sonic_min_edge": (
                    float(local_sonic_min_edge)
                    if local_sonic_min_edge is not None
                    else None
                ),
            }
        )

    layered_transition_weight = float(getattr(cfg, "layered_transition_weight", 0.0) or 0.0)
    if not math.isfinite(layered_transition_weight):
        layered_transition_weight = 0.0
    layered_transition_weight = max(0.0, layered_transition_weight)
    layered_transition_mode = str(getattr(cfg, "layered_transition_mode", "dynamic") or "dynamic").strip().lower()
    layered_matrices = (
        _layered_bundle_matrices(bundle)
        if bool(getattr(cfg, "layered_transition_scoring_enabled", False)) and layered_transition_weight > 0.0
        else None
    )
    if layered_matrices is not None:
        X_genre_norm = None
        X_genre_norm_idf = None
        X_genre_raw = None
        X_genre_smoothed = None
        X_genre_dense = None
        genre_idf = None
        penalty_strength = 0.0
        genre_tie_break_band = None

    def _layered_transition_delta(a_idx: int, b_idx: int, edge_metric: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        if layered_matrices is None:
            return 0.0, {}
        leaf, family, bridge, facet = layered_matrices
        if int(a_idx) >= leaf.shape[0] or int(b_idx) >= leaf.shape[0]:
            return 0.0, {}
        sonic_similarity = _coerce_edge_score(edge_metric.get("S"), fallback=edge_metric.get("T"), default=0.0)
        transition_quality = _coerce_edge_score(edge_metric.get("T"), fallback=edge_metric.get("S"), default=0.0)
        decision = score_layered_transition(
            from_leaf=leaf[int(a_idx)],
            to_leaf=leaf[int(b_idx)],
            from_family=family[int(a_idx)],
            to_family=family[int(b_idx)],
            from_bridge=bridge[int(a_idx)],
            to_bridge=bridge[int(b_idx)],
            from_facet=facet[int(a_idx)],
            to_facet=facet[int(b_idx)],
            sonic_similarity=sonic_similarity,
            transition_quality=transition_quality,
            mode=layered_transition_mode,
        )
        delta = float(layered_transition_weight) * float(decision.score)
        return delta, {
            "layered_transition_delta": float(delta),
            "layered_transition_score": float(decision.score),
            "layered_transition_reason": str(decision.reason),
            "layered_transition_explained": bool(decision.explained),
        }

    waypoint_enabled = bool(cfg.dj_bridging_enabled) and X_genre_norm is not None
    waypoint_weight = float(cfg.dj_waypoint_weight)
    if not math.isfinite(waypoint_weight):
        waypoint_weight = 0.0
    waypoint_floor = float(cfg.dj_waypoint_floor)
    if not math.isfinite(waypoint_floor):
        waypoint_floor = 0.0
    waypoint_penalty = float(cfg.dj_waypoint_penalty)
    if not math.isfinite(waypoint_penalty):
        waypoint_penalty = 0.0
    waypoint_cap = float(cfg.dj_waypoint_cap)
    if not math.isfinite(waypoint_cap):
        waypoint_cap = 0.0
    waypoint_cap = float(max(0.0, waypoint_cap))
    waypoint_tie_break_band = cfg.dj_waypoint_tie_break_band
    if isinstance(waypoint_tie_break_band, (int, float)) and math.isfinite(float(waypoint_tie_break_band)):
        waypoint_tie_break_band = float(waypoint_tie_break_band)
        if waypoint_tie_break_band <= 0:
            waypoint_tie_break_band = None
    else:
        waypoint_tie_break_band = None

    g_targets: Optional[List[np.ndarray]] = None
    if waypoint_enabled and interior_length > 0:
        if g_targets_override is not None and len(g_targets_override) == int(interior_length):
            g_targets = g_targets_override
        else:
            g_a = X_genre_norm[pier_a]
            g_b = X_genre_norm[pier_b]
            if float(np.linalg.norm(g_a)) <= 1e-8 or float(np.linalg.norm(g_b)) <= 1e-8:
                waypoint_enabled = False
            else:
                g_targets = []
                for i in range(interior_length):
                    frac = _step_fraction(i, interior_length)
                    g = (1.0 - frac) * g_a + frac * g_b
                    norm = float(np.linalg.norm(g))
                    if norm <= 1e-12:
                        g = g_a
                    else:
                        g = g / norm
                    g_targets.append(g)
    if waypoint_weight <= 0 and waypoint_penalty <= 0:
        waypoint_enabled = False

    # Genre-arc steering targets (Task 6): the per-step interpolated pier-A->pier-B
    # genre target. Independent of the waypoint machinery above (which gates on
    # dj_bridging_enabled). Under steering, the dense g_targets arrive via override.
    _steering_cfg = bool(cfg.genre_steering_enabled)
    arc_g_targets: Optional[List[np.ndarray]] = None
    if _steering_cfg and g_targets_override is not None and len(g_targets_override) == int(interior_length):
        arc_g_targets = g_targets_override
    if _steering_cfg and arc_g_targets is None:
        # Demoted to debug: the beam is invoked many times per segment (relaxation
        # cascade), so warning here floods the log. The segment builder logs this
        # condition once per segment at WARNING (see pier_bridge_builder.py).
        logger.debug(
            "genre_steering_enabled but no usable g_targets (interior_length=%d) — "
            "genre arc inactive for this segment",
            int(interior_length),
        )
    arc_floor_percentile = float(cfg.genre_arc_floor_percentile)
    if not math.isfinite(arc_floor_percentile):
        arc_floor_percentile = 0.0

    # Phase 3: Waypoint delta mode configuration
    waypoint_delta_mode = str(cfg.dj_waypoint_delta_mode or "absolute").strip().lower()
    waypoint_centered_baseline_method = str(cfg.dj_waypoint_centered_baseline or "median").strip().lower()
    waypoint_squash_mode = str(cfg.dj_waypoint_squash or "none").strip().lower()
    waypoint_squash_alpha = float(cfg.dj_waypoint_squash_alpha)

    def _waypoint_delta(sim: Optional[float], sim0: float = 0.0) -> float:
        """
        Compute waypoint delta with optional centering and squashing.

        Phase 3 modes:
        - absolute (legacy): delta = weight * sim, clamped to [-cap, +cap]
        - centered: delta = weight * (sim - sim0), clamped to [-cap, +cap]

        Squashing (optional):
        - none (legacy): hard clamp at waypoint_cap
        - tanh: smooth squashing delta = cap * tanh(alpha * raw / cap)

        Args:
            sim: Waypoint similarity (candidate vs target)
            sim0: Baseline similarity for centered mode (median/mean of step candidates)

        Returns:
            delta ∈ [-waypoint_cap, +waypoint_cap]
        """
        if sim is None or not math.isfinite(float(sim)):
            return 0.0

        # Compute raw delta (centered or absolute)
        if waypoint_delta_mode == "centered":
            raw = float(waypoint_weight) * (float(sim) - float(sim0))
        else:  # absolute (legacy)
            raw = float(waypoint_weight) * float(sim)

        # Apply squashing or hard clamp
        if waypoint_squash_mode == "tanh" and waypoint_cap > 0:
            # Smooth squashing: delta = cap * tanh(alpha * raw / cap)
            normalized_raw = waypoint_squash_alpha * raw / waypoint_cap if waypoint_cap > 0 else 0.0
            delta = waypoint_cap * math.tanh(normalized_raw)
            # Clamp for safety (should be redundant with tanh)
            delta = float(max(-waypoint_cap, min(waypoint_cap, delta)))
        else:  # none (hard cap, legacy)
            delta = raw
            if waypoint_cap > 0:
                delta = float(max(-waypoint_cap, min(waypoint_cap, delta)))

        # Legacy penalty (only for absolute mode with floor)
        if waypoint_penalty > 0 and waypoint_floor > 0 and float(sim) < waypoint_floor:
            penalty = waypoint_penalty * (waypoint_floor - float(sim))
            if waypoint_cap > 0:
                penalty = min(waypoint_cap, penalty)
            delta -= float(penalty)

        return float(delta)

    # Coverage bonus setup (Phase 2 + Phase 3 enhancements)
    coverage_enabled = bool(cfg.dj_genre_use_coverage) and bool(cfg.dj_bridging_enabled)
    coverage_mode = str(cfg.dj_coverage_mode or "binary").strip().lower()
    coverage_presence_source = str(cfg.dj_coverage_presence_source or "same").strip().lower()
    topk_A: list[tuple[int, float]] = []
    topk_B: list[tuple[int, float]] = []
    X_genre_for_coverage: Optional[np.ndarray] = None
    X_genre_for_coverage_presence: Optional[np.ndarray] = None

    if coverage_enabled and interior_length > 0:
        # Determine which matrix to use for scoring (anchor topK extraction)
        if X_genre_norm_idf is not None:
            X_genre_for_coverage = X_genre_norm_idf
        elif X_genre_norm is not None:
            X_genre_for_coverage = X_genre_norm

        # Phase 3: Determine presence source (for coverage computation)
        if coverage_presence_source == "raw" and X_genre_raw is not None:
            # Use raw genres for presence checking (avoids smoothed inflation)
            if genre_idf is not None and bool(cfg.dj_genre_use_idf):
                # Apply IDF to raw for consistency
                X_genre_for_coverage_presence = X_genre_raw * genre_idf
                # Normalize rows
                row_norms = np.linalg.norm(X_genre_for_coverage_presence, axis=1, keepdims=True)
                X_genre_for_coverage_presence = np.divide(
                    X_genre_for_coverage_presence,
                    row_norms,
                    out=np.zeros_like(X_genre_for_coverage_presence),
                    where=row_norms > 1e-12
                )
            else:
                # Normalize raw vectors to prevent coverage >1 in weighted mode
                # (raw weights can exceed 1.0, e.g., track/album source weights are 1.2)
                row_norms = np.linalg.norm(X_genre_raw, axis=1, keepdims=True)
                X_genre_for_coverage_presence = np.divide(
                    X_genre_raw,
                    row_norms,
                    out=np.zeros_like(X_genre_raw),
                    where=row_norms > 1e-12
                )
        else:
            # Use same matrix as scoring (default, Phase 2 behavior)
            X_genre_for_coverage_presence = X_genre_for_coverage

        # Extract anchor vectors in the same space as candidates will be scored
        if X_genre_for_coverage is not None:
            vA_cov = X_genre_for_coverage[pier_a]
            vB_cov = X_genre_for_coverage[pier_b]

            # Extract top-K genres from each anchor
            topk_A = _extract_top_genres(vA_cov, int(cfg.dj_genre_coverage_top_k))
            topk_B = _extract_top_genres(vB_cov, int(cfg.dj_genre_coverage_top_k))

            if not topk_A and not topk_B:
                coverage_enabled = False  # No genres to track
        else:
            coverage_enabled = False

    # Phase 2 diagnostic logging: segment-level config
    if cfg.dj_bridging_enabled and interior_length > 0:
        target_mode = str(cfg.dj_ladder_target_mode or "onehot").strip().lower()
        logger.info("[Phase2] Segment %d→%d: mode=%s, interior_length=%d",
                    pier_a, pier_b, target_mode, interior_length)

        # Phase 3: Log genre space being used for genre_sim
        genre_space_name = "IDF" if X_genre_norm_idf is not None else "normalized"
        logger.info("  Genre space for genre_sim: %s", genre_space_name)

        # Log IDF stats if enabled
        if bool(cfg.dj_genre_use_idf) and genre_idf is not None:
            logger.info("  IDF enabled: min=%.3f median=%.3f max=%.3f (power=%.2f norm=%s)",
                        float(np.min(genre_idf)), float(np.median(genre_idf)), float(np.max(genre_idf)),
                        float(cfg.dj_genre_idf_power), str(cfg.dj_genre_idf_norm))

        # Log anchor top-K genres if coverage enabled
        if coverage_enabled and genre_vocab is not None:
            def _format_topk(topk_list: list[tuple[int, float]], label: str) -> None:
                if topk_list:
                    genre_names = [f"{genre_vocab[idx]}={weight:.3f}" for idx, weight in topk_list[:5]]
                    logger.info("  %s topK genres: %s", label, ", ".join(genre_names))
            _format_topk(topk_A, "Anchor A")
            _format_topk(topk_B, "Anchor B")

    centered_cos_floor = -0.5 if bool(cfg.center_transitions) else None

    _transition_cache: Dict[Tuple[int, int], dict] = {}

    def _score_shared_transition(prev_idx: int, cur_idx: int) -> dict:
        # Pure function of (prev_idx, cur_idx) + segment-fixed context. Beam
        # "diamonds" (states sharing a last track) re-score identical pairs, so
        # memoize per segment call. Callers only read the dict (they dict()-copy
        # before storing), so returning the shared object is bit-identical.
        _tkey = (int(prev_idx), int(cur_idx))
        _cached = _transition_cache.get(_tkey)
        if _cached is not None:
            return _cached
        if transition_metric_context is not None:
            edge = score_transition_edge(transition_metric_context, int(prev_idx), int(cur_idx))
        else:
            t_val = _compute_transition_score(
                int(prev_idx), int(cur_idx), X_full, X_start, X_mid, X_end, cfg
            )
            edge = {
                "T": float(t_val),
                "T_used": float(t_val),
                "T_raw": float(t_val),
                "T_centered_cos": None,
                "H": None,
                "S": float(np.dot(X_full_norm[int(prev_idx)], X_full_norm[int(cur_idx)])),
                "G": None,
            }
        _transition_cache[_tkey] = edge
        return edge

    def _transition_gate_failed(edge: dict) -> bool:
        return is_broken_transition(
            edge,
            transition_floor=float(cfg.transition_floor),
            centered_cos_floor=centered_cos_floor,
        )

    if interior_length == 0:
        # Check if direct transition meets floor
        direct_edge = _score_shared_transition(pier_a, pier_b)
        direct_score = float(direct_edge.get("T", float("nan")))
        edges_scored = 1
        direct_score_after_sonic = _apply_local_sonic_edge_policy(
            direct_score,
            pier_a,
            pier_b,
        )
        _record_local_sonic_stats()
        if direct_score_after_sonic is None:
            return None, 0, edges_scored, "direct local sonic edge below floor"
        if not _transition_gate_failed(direct_edge):
            return [], 0, edges_scored, None
        else:
            # Roam-only: no T floor; only the -0.5 anti-alignment safety can fail here.
            _cc = direct_edge.get("T_centered_cos")
            return None, 0, edges_scored, f"direct transition anti-aligned (centered_cos={_cc})"

    # Progress model (A→B) in sonic similarity space (X_full_norm).
    progress_active = bool(cfg.progress_enabled)
    progress_eps = float(cfg.progress_monotonic_epsilon) if math.isfinite(float(cfg.progress_monotonic_epsilon)) else 0.0
    progress_eps = float(max(0.0, progress_eps))
    progress_weight = float(cfg.progress_penalty_weight) if math.isfinite(float(cfg.progress_penalty_weight)) else 0.0
    progress_weight = float(max(0.0, progress_weight))

    progress_arc_enabled = bool(cfg.progress_arc_enabled)
    progress_arc_weight = float(cfg.progress_arc_weight)
    if not math.isfinite(progress_arc_weight):
        progress_arc_weight = 0.0
    progress_arc_weight = float(max(0.0, progress_arc_weight))
    progress_arc_shape = str(cfg.progress_arc_shape or "linear").strip().lower()
    if progress_arc_shape not in {"linear", "arc"}:
        progress_arc_shape = "linear"
    progress_arc_tolerance = float(cfg.progress_arc_tolerance)
    if not math.isfinite(progress_arc_tolerance):
        progress_arc_tolerance = 0.0
    progress_arc_tolerance = float(max(0.0, progress_arc_tolerance))
    progress_arc_loss = str(cfg.progress_arc_loss or "abs").strip().lower()
    if progress_arc_loss not in {"abs", "squared", "huber"}:
        progress_arc_loss = "abs"
    progress_arc_huber_delta = float(cfg.progress_arc_huber_delta)
    if not math.isfinite(progress_arc_huber_delta) or progress_arc_huber_delta <= 0:
        progress_arc_huber_delta = 0.10
    progress_arc_max_step = cfg.progress_arc_max_step
    if isinstance(progress_arc_max_step, (int, float)) and math.isfinite(float(progress_arc_max_step)):
        progress_arc_max_step = float(progress_arc_max_step)
        if progress_arc_max_step <= 0:
            progress_arc_max_step = None
    else:
        progress_arc_max_step = None
    progress_arc_max_step_mode = str(cfg.progress_arc_max_step_mode or "penalty").strip().lower()
    if progress_arc_max_step_mode not in {"penalty", "gate"}:
        progress_arc_max_step_mode = "penalty"
    progress_arc_max_step_penalty = float(cfg.progress_arc_max_step_penalty)
    if not math.isfinite(progress_arc_max_step_penalty):
        progress_arc_max_step_penalty = 0.0
    progress_arc_max_step_penalty = float(max(0.0, progress_arc_max_step_penalty))
    progress_arc_autoscale_enabled = bool(cfg.progress_arc_autoscale_enabled)
    progress_arc_autoscale_min_distance = float(cfg.progress_arc_autoscale_min_distance)
    if not math.isfinite(progress_arc_autoscale_min_distance):
        progress_arc_autoscale_min_distance = 0.0
    progress_arc_autoscale_min_distance = float(max(0.0, progress_arc_autoscale_min_distance))
    progress_arc_autoscale_distance_scale = float(cfg.progress_arc_autoscale_distance_scale)
    if not math.isfinite(progress_arc_autoscale_distance_scale) or progress_arc_autoscale_distance_scale <= 0:
        progress_arc_autoscale_distance_scale = 0.0
    progress_arc_autoscale_per_step_scale = bool(cfg.progress_arc_autoscale_per_step_scale)
    progress_arc_effective_weight = 0.0

    progress_by_idx: Dict[int, float] = {}
    ab_distance: Optional[float] = None
    if progress_active:
        vec_a_full = X_full_norm[pier_a]
        vec_b_full = X_full_norm[pier_b]
        d = vec_b_full - vec_a_full
        denom = float(np.dot(d, d))
        if (not math.isfinite(denom)) or denom <= 1e-12:
            progress_active = False
        else:
            ab_distance = float(math.sqrt(denom))
            progress_arc_effective_weight = float(progress_arc_weight)
            if progress_arc_enabled and progress_arc_autoscale_enabled:
                if ab_distance < progress_arc_autoscale_min_distance:
                    progress_arc_effective_weight = 0.0
                elif progress_arc_autoscale_distance_scale > 0:
                    scale = min(1.0, ab_distance / progress_arc_autoscale_distance_scale)
                    progress_arc_effective_weight *= float(scale)
            if progress_arc_enabled and progress_arc_autoscale_per_step_scale:
                progress_arc_effective_weight *= 1.0 / float(max(1, interior_length))

            progress_by_idx[pier_a] = 0.0
            progress_by_idx[pier_b] = 1.0
            for cand in candidates:
                i = int(cand)
                t_raw = float(np.dot((X_full_norm[i] - vec_a_full), d) / denom)
                t = 0.0 if not math.isfinite(t_raw) else float(max(0.0, min(1.0, t_raw)))
                progress_by_idx[i] = t
    if arc_stats is not None:
        arc_stats.update(
            {
                "enabled": bool(progress_arc_enabled and progress_active),
                "shape": str(progress_arc_shape),
                "base_weight": float(progress_arc_weight),
                "effective_weight": float(progress_arc_effective_weight),
                "tolerance": float(progress_arc_tolerance),
                "loss": str(progress_arc_loss),
                "huber_delta": float(progress_arc_huber_delta),
                "max_step": (float(progress_arc_max_step) if progress_arc_max_step is not None else None),
                "max_step_mode": str(progress_arc_max_step_mode),
                "max_step_penalty": float(progress_arc_max_step_penalty),
                "autoscale": {
                    "enabled": bool(progress_arc_autoscale_enabled),
                    "min_distance": float(progress_arc_autoscale_min_distance),
                    "distance_scale": float(progress_arc_autoscale_distance_scale),
                    "per_step_scale": bool(progress_arc_autoscale_per_step_scale),
                },
                "ab_distance": (float(ab_distance) if ab_distance is not None else None),
                "steps": int(interior_length),
            }
        )

    # ── Roam corridors: precompute per-track soft penalties once (flag-gated). ──
    # Each array is indexed by global track index; the penalty is 0 inside the
    # corridor width and grows linearly beyond it (corridors.corridor_penalty).
    _roam_pen_sonic = (
        corridor_penalty(roam_detour_sonic, cfg.roam_width_sonic, slope=cfg.roam_penalty_slope)
        if (cfg.roam_corridors_enabled and roam_detour_sonic is not None) else None
    )
    _roam_pen_genre = (
        corridor_penalty(roam_dev_genre, cfg.roam_width_genre, slope=cfg.roam_penalty_slope)
        if (cfg.roam_corridors_enabled and roam_dev_genre is not None) else None
    )
    _roam_pen_energy = (
        corridor_penalty(roam_dev_energy, cfg.roam_width_energy, slope=cfg.roam_penalty_slope)
        if (cfg.roam_corridors_enabled and roam_dev_energy is not None) else None
    )

    vec_b_full = X_full_norm[pier_b]
    sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
    sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])

    # SP2 seed-character anti-collapse precompute (once per segment; mode "off" =>
    # both stay None => the scoring loop below is byte-identical to today).
    _sc_mode = str(getattr(cfg, "seed_character_mode", "off") or "off")
    _sc_strength = float(getattr(cfg, "seed_character_strength", 0.0) or 0.0)
    _sc_center_sim: Optional[np.ndarray] = None  # global-indexed cosine to the pool centroid (mode=anti_center)
    if _sc_mode != "off" and _sc_strength > 0.0 and candidates:
        _cand_arr = np.asarray(candidates, dtype=int)
        _n_full = int(X_full_norm.shape[0])
        if _sc_mode == "anti_center":
            _pc = X_full_norm[_cand_arr].mean(axis=0)
            _pc_norm = float(np.linalg.norm(_pc))
            if _pc_norm > 1e-12:
                _sc_center_sim = np.zeros(_n_full, dtype=np.float64)
                _sc_center_sim[_cand_arr] = X_full_norm[_cand_arr] @ (_pc / _pc_norm)

    genre_cache: Dict[tuple[int, int], float] = {}
    genre_cache_hits = 0
    genre_cache_misses = 0

    # Phase 3: Use correct genre matrix for genre_sim (matching target construction)
    # Must use same source (raw vs smoothed) and IDF settings as used for g_targets
    # Genre-steering: prefer the dense PMI-SVD embedding when enabled + available.
    genre_present = None
    _steering = bool(cfg.genre_steering_enabled)
    # Pairwise genre-edge soft penalty. genre_pair_floor is the tag-level similarity
    # threshold below which an adjacent-track edge is DEMOTED (not rejected) by
    # subtracting genre_pair_penalty from its score. A hard gate here detonates the
    # infeasibility/expansion machinery on broad-genre segments (hub damping makes
    # legitimate candidates score below floor against a broad-'rock' pier), so the
    # penalty keeps the beam feasible and fast while still steering away from bad
    # edges. See PierBridgeConfig + the 2026-06-10 design note.
    _pair_floor = float(getattr(cfg, "genre_pair_floor", 0.0)) if _steering else 0.0
    _pair_penalty = float(getattr(cfg, "genre_pair_penalty", 0.0)) if _steering else 0.0
    if not math.isfinite(_pair_penalty):
        _pair_penalty = 0.0
    _pair_penalty = max(0.0, _pair_penalty)
    pair_penalty_hits = 0

    def _pair_edge_sim(a_idx: int, b_idx: int) -> Optional[float]:
        """Pair-floor similarity, None = exempt. Tag-level taxonomy provider
        when supplied (the only space that separates bad edges from good — see
        GenrePairSimProvider), else the steering-space vector cosine with
        genreless endpoints exempt."""
        if pair_sim_provider is not None:
            return pair_sim_provider.sim(int(a_idx), int(b_idx))
        if X_genre_for_sim is None:
            return None
        if genre_present is not None and not (
            bool(genre_present[int(a_idx)]) and bool(genre_present[int(b_idx)])
        ):
            return None
        return _get_genre_sim(int(a_idx), int(b_idx))

    # Steering space: dense PMI-SVD embedding (legacy) vs taxonomy genre-vocab.
    # In taxonomy mode the arc targets are genre-vocab vectors, so the arc vote must
    # score in the SAME genre-vocab space (X_genre_norm), not the 64-dim dense space.
    _steering_source = str(getattr(cfg, "genre_steering_source", "taxonomy"))
    if _steering and _steering_source != "taxonomy" and X_genre_dense is not None:
        X_genre_for_sim = X_genre_dense  # rows already L2-normalized
        genre_present = np.linalg.norm(X_genre_dense, axis=1) > 1e-9
    else:
        vector_source = str(cfg.dj_genre_vector_source or "smoothed").strip().lower()
        if vector_source == "raw" and X_genre_raw is not None:
            # Raw mode: use raw matrix, apply IDF if enabled
            if bool(cfg.dj_genre_use_idf) and genre_idf is not None:
                # Apply IDF weighting to raw matrix
                X_genre_for_sim = X_genre_raw * genre_idf
                # Normalize rows
                row_norms = np.linalg.norm(X_genre_for_sim, axis=1, keepdims=True)
                X_genre_for_sim = np.divide(
                    X_genre_for_sim,
                    row_norms,
                    out=np.zeros_like(X_genre_for_sim),
                    where=row_norms > 1e-12
                )
            else:
                # Raw mode without IDF: normalize raw matrix
                X_genre_for_sim = X_genre_raw.copy()
                row_norms = np.linalg.norm(X_genre_for_sim, axis=1, keepdims=True)
                X_genre_for_sim = np.divide(
                    X_genre_for_sim,
                    row_norms,
                    out=np.zeros_like(X_genre_for_sim),
                    where=row_norms > 1e-12
                )
        else:
            # Smoothed mode (default): use IDF-weighted if available, else normalized
            X_genre_for_sim = X_genre_norm_idf if X_genre_norm_idf is not None else X_genre_norm
        # Taxonomy steering scores in genre-vocab space: exempt genreless tracks from
        # the on-arc floor (mirrors the dense branch's genre_present).
        if _steering and _steering_source == "taxonomy" and X_genre_for_sim is not None:
            genre_present = np.linalg.norm(X_genre_for_sim, axis=1) > 1e-9

    def _get_genre_sim(a_idx: int, b_idx: int) -> Optional[float]:
        nonlocal genre_cache_hits, genre_cache_misses
        if X_genre_for_sim is None:
            return None
        key = (int(a_idx), int(b_idx))
        if key in genre_cache:
            genre_cache_hits += 1
            return genre_cache[key]
        genre_cache_misses += 1
        val = float(np.dot(X_genre_for_sim[a_idx], X_genre_for_sim[b_idx]))
        genre_cache[key] = val
        return val

    def _progress_arc_loss(cand_t: float, target_t: float) -> float:
        err0 = abs(float(cand_t) - float(target_t))
        err = max(0.0, err0 - progress_arc_tolerance)
        return _progress_arc_loss_value(err, progress_arc_loss, progress_arc_huber_delta)

    def _record_genre_cache_stats() -> None:
        if genre_cache_stats is None:
            return
        total = int(genre_cache_hits + genre_cache_misses)
        hit_rate = float(genre_cache_hits) / float(total) if total > 0 else None
        genre_cache_stats.update(
            {
                "hits": int(genre_cache_hits),
                "misses": int(genre_cache_misses),
                "hit_rate": float(hit_rate) if hit_rate is not None else None,
                "entries": int(len(genre_cache)),
            }
        )

    # Sibling-group resolver (Task 6): memoized per candidate index. No-op when
    # there are no sibling links (sibling_group_of returns None for everything).
    # Defined here, before the init block below, so it can seed the piers'
    # sibling groups into used_sibling_groups_init.
    _sibling_cache: Dict[int, Optional[str]] = {}

    def _idx_sibling_group(idx_int: int) -> Optional[str]:
        if idx_int in _sibling_cache:
            return _sibling_cache[idx_int]
        s = ""
        if bundle is not None and getattr(bundle, "track_artists", None) is not None:
            try:
                s = str(bundle.track_artists[int(idx_int)] or "")
            except Exception:
                s = ""
        if not s and artist_key_by_idx is not None:
            s = str(artist_key_by_idx.get(int(idx_int), "") or "")
        sg = sibling_group_of(s) if s else None
        _sibling_cache[idx_int] = sg
        return sg

    # Initialize beam with pier_a
    used_artists_init: Set[str] = set()

    # Add boundary context from previous segments (cross-segment min_gap enforcement)
    if recent_global_artists:
        for artist_key in recent_global_artists:
            if artist_key:
                used_artists_init.add(str(artist_key))

    used_sibling_groups_init: Set[str] = set()
    if recent_global_artists:
        for ak in recent_global_artists:
            sg = sibling_group_of(str(ak)) if ak else None
            if sg:
                used_sibling_groups_init.add(sg)
    for pier_idx in (pier_a, pier_b):
        sg = _idx_sibling_group(int(pier_idx))
        if sg:
            used_sibling_groups_init.add(sg)

    if artist_key_by_idx is not None:
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

        if cfg.disallow_pier_artists_in_interiors:
            # Add pier artist keys to used_artists_init
            for pier_idx in [pier_a, pier_b]:
                # Use raw artist string to preserve collaborations
                pier_artist_str = ""
                if bundle is not None and bundle.track_artists is not None:
                    try:
                        pier_artist_str = str(bundle.track_artists[int(pier_idx)] or "")
                    except Exception:
                        pier_artist_str = str(artist_key_by_idx.get(int(pier_idx), "") or "")
                else:
                    pier_artist_str = str(artist_key_by_idx.get(int(pier_idx), "") or "")
                if pier_artist_str:
                    if use_identity:
                        # Identity mode: add all identity keys for pier artist
                        pier_identity_keys = resolve_artist_identity_keys(pier_artist_str, artist_identity_cfg)
                        used_artists_init.update(pier_identity_keys)
                    else:
                        # Legacy mode: single artist key
                        used_artists_init.add(pier_artist_str)

        if cfg.disallow_seed_artist_in_interiors and seed_artist_keys:
            for _seed_key in seed_artist_keys:
                if use_identity:
                    # Identity mode: resolve seed artist to identity keys
                    seed_identity_keys = resolve_artist_identity_keys(_seed_key, artist_identity_cfg)
                    used_artists_init.update(seed_identity_keys)
                else:
                    # Legacy mode: single seed artist key
                    used_artists_init.add(str(_seed_key))

    initial_state = BeamState(
        path=[pier_a],
        score=0.0,
        used={pier_a, pier_b},
        used_artists=used_artists_init,
        used_sibling_groups=used_sibling_groups_init,
        last_progress=0.0,
    )
    beam = [initial_state]

    # Waypoint rank impact diagnostic (opt-in, default disabled)
    rank_impact_enabled = bool(
        cfg.dj_diagnostics_waypoint_rank_impact_enabled
        and waypoint_enabled
        and g_targets is not None
    )
    rank_impact_sampled_steps: Set[int] = set()
    rank_impact_results: List[Dict[str, Any]] = []
    if rank_impact_enabled:
        sample_count = int(cfg.dj_diagnostics_waypoint_rank_sample_steps)
        if sample_count > 0 and interior_length > 0:
            # Evenly-spaced sampling: e.g., [0, 7, 14] for length=15, sample_count=3
            if sample_count >= interior_length:
                rank_impact_sampled_steps = set(range(interior_length))
            else:
                step_interval = float(interior_length) / float(sample_count)
                rank_impact_sampled_steps = {
                    int(round(i * step_interval)) for i in range(sample_count)
                }
                # Ensure we don't exceed interior_length-1
                rank_impact_sampled_steps = {s for s in rank_impact_sampled_steps if s < interior_length}

    # TASK A: Track unique candidates that pass gates on first step
    pool_after_gating_candidates: Set[int] = set()

    # Phase 3 fix: Track actual applied waypoint deltas during beam search for correct stats
    # (Stats were using wrong sim0=0.0 instead of per-step baseline)
    chosen_waypoint_deltas: List[float] = []  # Actual applied deltas per step
    chosen_waypoint_sims: List[float] = []    # Raw sims per step (for context)
    chosen_waypoint_sim0s: List[float] = []   # Baselines per step (for debugging)

    # BPM-trust gate (onset-based). BPM is meaningless on beatless audio; if either
    # pier is beatless its BPM target is garbage, so disable the BPM band for the
    # whole segment. (Candidate-level trust is checked per-candidate below.) The
    # onset band stays active — it is the reliable beat-presence signal.
    _bpm_trust_min_onset = float(getattr(cfg, "bpm_trust_min_onset_rate", 0.0))
    _bpm_band_pier_trusted = True
    if _bpm_trust_min_onset > 0.0 and onset_rate is not None:
        _oa = float(onset_rate[int(pier_a)])
        _ob = float(onset_rate[int(pier_b)])
        _bpm_band_pier_trusted = (
            (not np.isnan(_oa)) and (not np.isnan(_ob))
            and _oa >= _bpm_trust_min_onset and _ob >= _bpm_trust_min_onset
        )

    # The energy pace penalty returns exactly 0.0 unless a strength is configured;
    # skip the per-candidate call + energy_matrix index entirely when both are 0
    # (the effective config). Bit-identical: a skipped 0.0 == an added 0.0.
    _energy_active = (
        float(getattr(cfg, "energy_step_strength", 0.0)) > 0.0
        or float(getattr(cfg, "energy_arc_strength", 0.0)) > 0.0
    )

    # Cache the artist-identity parse per candidate index. It is a pure function
    # of the candidate's artist string (bundle.track_artists[cand]), fixed for a
    # given cand, so it is identical across every beam state/step. Uncached it was
    # ~2/3 of beam time (millions of 11-delimiter regex parses). Bit-identical:
    # every call site derives cand_artist_str the same way, and the returned set
    # is only read / used with `|` (never mutated), so sharing it is safe.
    _identity_keys_cache: Dict[int, Set[str]] = {}

    def _cand_identity_keys(cand_int: int, cand_artist_str: str) -> Set[str]:
        keys = _identity_keys_cache.get(cand_int)
        if keys is None:
            keys = resolve_artist_identity_keys(cand_artist_str, artist_identity_cfg)
            _identity_keys_cache[cand_int] = keys
        return keys

    # Tag steering (Task 6): soft per-candidate sonic-tag bonus added to the
    # beam's ranking score (never to trans_score/T -- see combined_score below).
    # Fully gated on a non-None affinity AND a positive weight so a no-tag run
    # is byte-identical. Logged once per segment (not per candidate).
    _sonic_tag_active = sonic_tag_affinity is not None and float(sonic_tag_beam_weight) > 0.0
    # Worst-edge tolerance band: only relax the minimax tie-break when the tag term is
    # actually active, so default (non-steered) playlists keep strict worst-edge selection.
    _edge_band = float(tag_steering_worst_edge_band) if _sonic_tag_active else 0.0
    if _sonic_tag_active:
        logger.info(
            "Tag steering beam term: weight=%.2f worst_edge_band=%.3f active over %d candidates",
            float(sonic_tag_beam_weight), _edge_band, len(candidates),
        )

    for step in range(interior_length):
        # Cooperative cancellation: the per-step poll is what bounds cancel
        # latency inside a single long beam run (the reported segment-0 hang).
        raise_if_cancelled()
        next_beam: List[BeamState] = []
        target_t = _step_fraction(step, interior_length)
        experiment_target_t = (
            _progress_target_curve(step, interior_length, progress_arc_shape)
            if progress_arc_enabled
            else target_t
        )
        g_target = g_targets[step] if waypoint_enabled and g_targets is not None else None

        # Genre-arc steering (Task 6): per-step target + per-segment on-arc floor.
        # Precompute the arc-sim distribution over the *pool* candidates to this
        # step's target, then derive a percentile floor. Candidates below the floor
        # are gated out in the candidate loop (mirroring _transition_gate_failed).
        arc_target = (
            arc_g_targets[step]
            if (_steering_cfg and arc_g_targets is not None and step < len(arc_g_targets))
            else None
        )
        step_arc_sims: Dict[int, float] = {}
        arc_step_floor: Optional[float] = None
        if arc_target is not None and X_genre_for_sim is not None:
            for cand in candidates:
                ci = int(cand)
                if genre_present is not None and not bool(genre_present[ci]):
                    continue  # genreless: not subject to the arc floor
                step_arc_sims[ci] = float(np.dot(X_genre_for_sim[ci], arc_target))
            if arc_floor_percentile > 0.0 and step_arc_sims:
                arc_step_floor = floor_at_percentile(
                    np.array(list(step_arc_sims.values()), dtype=float),
                    arc_floor_percentile,
                )

        # Phase 3: Centered waypoint mode - collect all candidate waypoint sims for this step
        waypoint_sim0 = 0.0  # Baseline for centered mode (median or mean)
        # Phase 3 fix: Store waypoint info per candidate for this step (for stats tracking)
        step_waypoint_info: Dict[int, tuple[float, float]] = {}  # cand_idx -> (sim, delta)
        if waypoint_enabled and waypoint_delta_mode == "centered" and g_target is not None and X_genre_for_sim is not None:
            # Collect waypoint sims for all valid candidates
            step_waypoint_sims: List[float] = []
            for state in beam:
                current = state.path[-1]
                for cand in candidates:
                    if cand in state.used:
                        continue
                    # Compute waypoint sim (using IDF-weighted matrix when available)
                    waypoint_sim = float(np.dot(X_genre_for_sim[cand], g_target))
                    if math.isfinite(waypoint_sim):
                        step_waypoint_sims.append(waypoint_sim)

            # Compute baseline (sim0) from distribution
            if step_waypoint_sims:
                if waypoint_centered_baseline_method == "mean":
                    waypoint_sim0 = float(np.mean(step_waypoint_sims))
                else:  # median (default)
                    waypoint_sim0 = float(np.median(step_waypoint_sims))

        # Rank impact diagnostic: collect candidates for sampled steps
        step_is_sampled = rank_impact_enabled and step in rank_impact_sampled_steps
        step_candidates_for_ranking: List[Tuple[int, float, float, float, float]] = []  # (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)

        # Per-step pace targets — depend only on (pier_a, pier_b, step, length),
        # not on state/candidate. Hoisted out of the state+candidate loops so the
        # target + its import run once per step, not once per (state x candidate).
        # Same guard conditions as the per-candidate use below -> bit-identical.
        _bpm_target_step = None
        if (
            _bpm_band_pier_trusted
            and float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf"))) < float("inf")
            and perceptual_bpm is not None
        ):
            from src.playlist.pier_bridge.pace_gate import compute_step_log_bpm_target
            _bpm_target_step = compute_step_log_bpm_target(
                float(perceptual_bpm[int(pier_a)]),
                float(perceptual_bpm[int(pier_b)]),
                step=step,
                segment_length=interior_length,
            )
        _onset_target_step = None
        if (
            float(getattr(cfg, "onset_bridge_max_log_distance", float("inf"))) < float("inf")
            and onset_rate is not None
        ):
            from src.playlist.pier_bridge.pace_gate import compute_step_log_onset_target
            _onset_target_step = compute_step_log_onset_target(
                float(onset_rate[int(pier_a)]),
                float(onset_rate[int(pier_b)]),
                step=step,
                segment_length=interior_length,
            )

        for state in beam:
            current = state.path[-1]
            apply_tie_break = (
                (genre_tie_break_band is not None and X_genre_norm is not None and penalty_strength > 0)
                or (waypoint_enabled and waypoint_tie_break_band is not None)
            )
            cand_entries: list[tuple[int, float, float, float, Optional[float], Optional[float], float, dict, dict[str, Any]]] = []
            best_score = -float("inf")

            for cand in candidates:
                if cand in state.used:
                    continue
                # Pace bridge bands (BPM/onset): demote out-of-band candidates when a
                # soft-penalty strength is configured; otherwise legacy hard reject.
                _pace_penalty = 0.0

                # BPM bridge gate (skipped when a pier is beatless — garbage target)
                if (
                    _bpm_band_pier_trusted
                    and float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf"))) < float("inf")
                    and perceptual_bpm is not None
                ):
                    from src.playlist.bpm_axis import bpm_log_distance as _bld
                    _bpm_target = _bpm_target_step
                    _cand_bpm = float(perceptual_bpm[int(cand)])
                    _cand_stab = (
                        float(tempo_stability[int(cand)])
                        if tempo_stability is not None
                        else 1.0
                    )
                    _stab_min = float(getattr(cfg, "bpm_stability_min", 0.5))
                    # Beatless candidate -> its BPM is meaningless, bypass the band.
                    _cand_bpm_trusted = True
                    if _bpm_trust_min_onset > 0.0 and onset_rate is not None:
                        _co = float(onset_rate[int(cand)])
                        _cand_bpm_trusted = (not np.isnan(_co)) and _co >= _bpm_trust_min_onset
                    if (
                        not np.isnan(_cand_bpm)
                        and _cand_stab >= _stab_min
                        and _cand_bpm_trusted
                    ):
                        _bpm_excess = float(_bld(_cand_bpm, _bpm_target)) - float(cfg.bpm_bridge_max_log_distance)
                        if _bpm_excess > 0.0:
                            _bpm_soft = float(getattr(cfg, "bpm_bridge_soft_penalty_strength", 0.0))
                            if _bpm_soft > 0.0:
                                _pace_penalty += _bpm_soft * _bpm_excess
                            else:
                                continue

                # Onset-rate bridge band (embedding-independent). Soft penalty when
                # onset_bridge_soft_penalty_strength > 0, else legacy hard reject.
                if (
                    float(getattr(cfg, "onset_bridge_max_log_distance", float("inf"))) < float("inf")
                    and onset_rate is not None
                ):
                    from src.playlist.bpm_axis import bpm_log_distance as _old_dist

                    _onset_target = _onset_target_step
                    _cand_onset = float(onset_rate[int(cand)])
                    if not np.isnan(_cand_onset):
                        _onset_excess = float(_old_dist(_cand_onset, _onset_target)) - float(cfg.onset_bridge_max_log_distance)
                        if _onset_excess > 0.0:
                            _onset_soft = float(getattr(cfg, "onset_bridge_soft_penalty_strength", 0.0))
                            if _onset_soft > 0.0:
                                _pace_penalty += _onset_soft * _onset_excess
                            else:
                                continue

                # Energy (arousal) soft penalty — purely additive, never excludes.
                if energy_matrix is not None and _energy_active:
                    from src.playlist.pier_bridge.pace_gate import compute_energy_pace_penalty
                    _pace_penalty += compute_energy_pace_penalty(
                        energy_matrix,
                        current=int(current), cand=int(cand),
                        pier_a=int(pier_a), pier_b=int(pier_b),
                        step=step, segment_length=interior_length,
                        step_cap=float(getattr(cfg, "energy_step_cap", 0.0)),
                        step_strength=float(getattr(cfg, "energy_step_strength", 0.0)),
                        arc_band=float(getattr(cfg, "energy_arc_band", 0.0)),
                        arc_strength=float(getattr(cfg, "energy_arc_strength", 0.0)),
                    )

                # Instrumental lean: additive demotion of vocal-classified candidates.
                # Rides combined_score -= _pace_penalty below; bridge-interior-only by
                # the state.used guard above (piers are pre-seeded into state.used, so
                # this loop never scores them); NaN/None/weight<=0 -> 0.0.
                _instr_w = float(getattr(cfg, "instrumental_penalty_weight", 0.0))
                if voice_prob is not None and _instr_w > 0.0:
                    from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
                    _pace_penalty += compute_instrumental_penalty(
                        voice_prob, cand=int(cand), weight=_instr_w
                    )

                # Duration soft penalty (Phase-1 corridor pooling, C1 rehome):
                # additive demotion of over-length candidates. Corridor's
                # pool-ranking penalty (eligible_universe.py's
                # duration_rank_penalty) is never wired into `build_corridor`'s
                # `rank_scores` at all -- it's a structural placeholder
                # (always 1.0, see eligible_universe.py's module docstring),
                # not a cap-gated rarity -- so C1's SELECTION effect is
                # rehomed here instead, mirroring C10's beam half immediately
                # above. `duration_penalty_values` is a
                # precomputed per-candidate array (weight already baked in,
                # from src.playlist.candidate_pool.compute_duration_penalty)
                # threaded ONLY by the corridor call site in
                # pier_bridge_builder.py; legacy pooling never sets it, so this
                # term is a pure no-op there by construction (legacy already
                # applies its own duration soft-penalty once, during pool
                # ranking -- a second beam subtraction would double-penalize).
                if duration_penalty_values is not None:
                    _dur_pen = float(duration_penalty_values[int(cand)])
                    if _dur_pen > 0.0:
                        _pace_penalty += _dur_pen
                        duration_penalty_hits += 1

                # Artist diversity: check if candidate artist already used
                if artist_key_by_idx is not None:
                    use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

                    if use_identity:
                        # Identity mode: use raw artist string to preserve collaborations
                        # (artist_key_by_idx has collaborations stripped, which prevents detection of featured artists)
                        cand_artist_str = ""
                        if bundle is not None and bundle.track_artists is not None:
                            try:
                                cand_artist_str = str(bundle.track_artists[int(cand)] or "")
                            except Exception:
                                cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                        else:
                            cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                        if cand_artist_str:
                            cand_identity_keys = _cand_identity_keys(int(cand), cand_artist_str)
                            # Reject if ANY identity key overlaps with used_artists
                            if any(key in state.used_artists for key in cand_identity_keys):
                                continue
                    else:
                        # Legacy mode: single artist key
                        cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                        if cand_artist and cand_artist in state.used_artists:
                            continue

                # Sibling repulsion: a sibling of an already-placed (in-window) artist
                # may not be placed adjacent. No-op when there are no sibling links.
                _cand_sg = _idx_sibling_group(int(cand))
                if _cand_sg is not None and _cand_sg in state.used_sibling_groups:
                    continue

                if min(sim_to_a[cand], sim_to_b[cand]) < cfg.bridge_floor:
                    continue

                cand_t = 0.0
                if progress_active:
                    cand_t = float(progress_by_idx.get(int(cand), 0.0))
                    if cand_t < float(state.last_progress) - progress_eps:
                        continue
                    if progress_arc_enabled and progress_arc_max_step is not None:
                        step_jump = float(cand_t) - float(state.last_progress)
                        if progress_arc_max_step_mode == "gate":
                            if step_jump > float(progress_arc_max_step) + progress_eps:
                                continue

                # Compute transition score
                edge_metric = _score_shared_transition(current, cand)
                trans_score = float(edge_metric.get("T", float("nan")))

                # Anti-alignment safety only (is_broken_transition no longer T-gates; roam design)
                if _transition_gate_failed(edge_metric):
                    continue

                # TASK A: Track candidates that pass all gates (step 0 only for pool_after_gating)
                if step == 0:
                    pool_after_gating_candidates.add(int(cand))

                sim_a = float(sim_to_a[cand])
                sim_b = float(sim_to_b[cand])
                denom = sim_a + sim_b
                bridge_score = 0.0 if denom <= 1e-9 else (2 * sim_a * sim_b) / denom

                # Add heuristic pull toward destination
                dest_pull = cfg.eta_destination_pull * float(np.dot(X_full_norm[cand], vec_b_full))

                combined_score = (
                    cfg.weight_bridge * bridge_score +
                    cfg.weight_transition * trans_score
                )
                # SP2-B: demote candidates that sit closer to the local pool center
                # than to their own piers (the anti-sag scoring twin of the sag metric).
                if _sc_center_sim is not None:
                    combined_score -= anti_center_penalty(
                        float(_sc_center_sim[cand]), bridge_score, _sc_strength)
                # Pace bridge soft penalty (out-of-band BPM/onset demotion).
                if _pace_penalty > 0.0:
                    combined_score -= _pace_penalty
                if progress_active and progress_weight > 0:
                    combined_score -= progress_weight * abs(float(cand_t) - target_t)
                if progress_active and progress_arc_enabled and progress_arc_effective_weight > 0:
                    combined_score -= progress_arc_effective_weight * _progress_arc_loss(
                        float(cand_t),
                        experiment_target_t,
                    )
                if progress_active and progress_arc_enabled and progress_arc_max_step is not None:
                    step_jump = float(cand_t) - float(state.last_progress)
                    if step_jump > float(progress_arc_max_step):
                        if progress_arc_max_step_mode == "penalty" and progress_arc_max_step_penalty > 0:
                            combined_score -= progress_arc_max_step_penalty * (step_jump - float(progress_arc_max_step))

                genre_sim = None
                if X_genre_for_sim is not None and not _steering:
                    genre_sim = _get_genre_sim(int(current), int(cand))
                if _steering:
                    cand_present = (genre_present is None) or bool(genre_present[int(cand)])
                    # Genre ARC vote (first-class): closeness to this step's g_target,
                    # NOT similarity to the previous track. Plus the per-segment on-arc
                    # percentile floor (precomputed above into arc_step_floor).
                    if arc_target is not None and X_genre_for_sim is not None and cand_present:
                        arc_sim = float(step_arc_sims.get(int(cand), float(np.dot(X_genre_for_sim[int(cand)], arc_target))))
                        # Per-segment on-arc floor: drop below-floor candidates.
                        if arc_step_floor is not None and arc_sim < arc_step_floor:
                            continue
                        # Absolute safeguard floor (legacy genre_arc_floor).
                        if arc_sim < cfg.genre_arc_floor:
                            continue
                        if float(cfg.weight_genre) > 0.0:
                            combined_score += float(cfg.weight_genre) * arc_sim
                else:
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if cfg.genre_tiebreak_weight:
                            combined_score += cfg.genre_tiebreak_weight * genre_sim

                waypoint_sim = None
                if waypoint_enabled and g_target is not None and X_genre_for_sim is not None:
                    # Use IDF-weighted matrix when available to match target space
                    waypoint_sim = float(np.dot(X_genre_for_sim[cand], g_target))

                layered_delta, layered_diag = _layered_transition_delta(int(current), int(cand), edge_metric)
                combined_score += layered_delta

                # ── Roam corridors: subtract the per-dimension corridor penalty
                # (flag-gated; arrays precomputed above, indexed by track index). ──
                _roam_pen = 0.0
                if _roam_pen_sonic is not None:
                    _roam_pen += float(_roam_pen_sonic[int(cand)])
                if _roam_pen_genre is not None:
                    _roam_pen += float(_roam_pen_genre[int(cand)])
                if _roam_pen_energy is not None:
                    _roam_pen += float(_roam_pen_energy[int(cand)])
                combined_score -= _roam_pen

                combined_score_after_sonic = _apply_local_sonic_edge_policy(
                    combined_score,
                    int(current),
                    int(cand),
                )
                if combined_score_after_sonic is None:
                    continue
                combined_score = float(combined_score_after_sonic)

                # Title-artifact penalty (opt-in; zero cost when disabled)
                _title_artifact_pen = 0.0
                if bool(cfg.title_artifact_penalty_enabled) and cfg.title_artifact_penalty_weights:
                    _cand_title = ""
                    try:
                        if bundle is not None and bundle.track_titles is not None:
                            _cand_title = str(bundle.track_titles[int(cand)] or "")
                    except Exception:
                        _cand_title = ""
                    if _cand_title:
                        _title_artifact_pen = compute_title_artifact_penalty(
                            title=_cand_title,
                            weights=cfg.title_artifact_penalty_weights,
                        )
                        combined_score -= _title_artifact_pen

                # Pairwise genre-edge soft penalty: demote (never reject) the
                # candidate vs the track it is actually placed after — the arc vote
                # above scores against the waypoint target, never the real neighbor.
                if _pair_floor > 0.0 and _pair_penalty > 0.0:
                    _pair_sim = _pair_edge_sim(int(current), int(cand))
                    if (
                        _pair_sim is not None
                        and math.isfinite(_pair_sim)
                        and _pair_sim < _pair_floor
                    ):
                        combined_score -= _pair_penalty
                        pair_penalty_hits += 1

                edges_scored += 1

                if apply_tie_break:
                    cand_entries.append((
                        int(cand),
                        float(cand_t),
                        float(combined_score),
                        float(dest_pull),
                        genre_sim,
                        waypoint_sim,
                        float(trans_score),
                        dict(edge_metric),
                        dict(layered_diag),
                    ))
                    if combined_score > best_score:
                        best_score = float(combined_score)
                else:
                    base_score_for_rank = float(combined_score)
                    if (not _steering) and genre_sim is not None and math.isfinite(genre_sim):
                        if penalty_strength > 0 and genre_sim < penalty_threshold:
                            combined_score *= (1.0 - penalty_strength)
                            genre_penalty_hits += 1
                    if popularity_penalty_strength > 0.0 and popularity_values is not None:
                        combined_score *= _popularity_factor(
                            float(popularity_values[int(cand)]), popularity_penalty_strength)
                    waypoint_delta_val = 0.0
                    if waypoint_enabled:
                        waypoint_delta_val = _waypoint_delta(waypoint_sim, waypoint_sim0)
                        combined_score += waypoint_delta_val
                        # Phase 3 fix: Store for stats tracking
                        if waypoint_sim is not None:
                            step_waypoint_info[int(cand)] = (float(waypoint_sim), float(waypoint_delta_val))

                    # Coverage bonus (Phase 2 + Phase 3): reward matching anchor top-K genres with schedule decay
                    coverage_bonus_val = 0.0
                    if coverage_enabled and X_genre_for_coverage_presence is not None:
                        cand_genre_vec = X_genre_for_coverage_presence[cand]
                        coverage_A = _compute_coverage(
                            cand_genre_vec, topk_A, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_B = _compute_coverage(
                            cand_genre_vec, topk_B, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_bonus_val = _compute_coverage_bonus(
                            step, interior_length, coverage_A, coverage_B,
                            float(cfg.dj_genre_coverage_weight), float(cfg.dj_genre_coverage_power)
                        )
                        combined_score += coverage_bonus_val

                    if _sonic_tag_active:
                        combined_score += float(sonic_tag_beam_weight) * float(sonic_tag_affinity[int(cand)])

                    # Rank impact: collect (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)
                    if step_is_sampled:
                        step_candidates_for_ranking.append((
                            int(cand),
                            float(base_score_for_rank),
                            float(waypoint_delta_val),
                            float(coverage_bonus_val),
                            float(combined_score)
                        ))

                    new_score = state.score + combined_score + dest_pull
                    new_path = state.path + [cand]
                    new_used = state.used | {cand}
                    new_used_artists = state.used_artists
                    if artist_key_by_idx is not None:
                        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
                        if use_identity:
                            # Identity mode: use raw artist string to preserve collaborations
                            # (artist_key_by_idx has collaborations stripped, which prevents detection of featured artists)
                            cand_artist_str = ""
                            if bundle is not None and bundle.track_artists is not None:
                                try:
                                    cand_artist_str = str(bundle.track_artists[int(cand)] or "")
                                except Exception:
                                    cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            else:
                                cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist_str:
                                cand_identity_keys = _cand_identity_keys(int(cand), cand_artist_str)
                                new_used_artists = state.used_artists | cand_identity_keys
                        else:
                            # Legacy mode: add single artist key
                            cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist:
                                new_used_artists = state.used_artists | {cand_artist}
                    new_used_sibling_groups = state.used_sibling_groups
                    _place_sg = _idx_sibling_group(int(cand))
                    if _place_sg is not None:
                        new_used_sibling_groups = state.used_sibling_groups | {_place_sg}
                    new_last_progress = float(state.last_progress)
                    if progress_active:
                        new_last_progress = float(progress_by_idx.get(int(cand), 0.0))

                    # Build per-edge diagnostic component dict
                    _local_sonic_cos = float(np.dot(X_full_norm[int(current)], X_full_norm[int(cand)]))
                    _genre_pen_applied = 0.0
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if penalty_strength > 0 and genre_sim < penalty_threshold:
                            _genre_pen_applied = float(penalty_strength)
                    _local_pen_applied = 0.0
                    if (
                        local_sonic_penalty_enabled
                        and local_sonic_penalty_strength > 0.0
                        and _local_sonic_cos < local_sonic_penalty_threshold
                    ):
                        _local_pen_applied = float(
                            local_sonic_penalty_strength * (local_sonic_penalty_threshold - _local_sonic_cos)
                        )
                    edge_component = {
                        "from_idx": int(current),
                        "to_idx": int(cand),
                        "bridge_score": float(bridge_score),
                        "trans_score_in_beam": float(trans_score),
                        "T": float(edge_metric.get("T")) if edge_metric.get("T") is not None else None,
                        "T_raw": edge_metric.get("T_raw"),
                        "T_centered_cos": edge_metric.get("T_centered_cos"),
                        "H": edge_metric.get("H"),
                        "S": edge_metric.get("S"),
                        "G": edge_metric.get("G"),
                        "progress_t": float(cand_t) if progress_active else None,
                        "progress_jump": (float(cand_t) - float(state.last_progress)) if progress_active else None,
                        "local_sonic_raw_cos": _local_sonic_cos,
                        "local_sonic_penalty_applied": _local_pen_applied,
                        "genre_penalty_applied": _genre_pen_applied,
                        "below_transition_floor": False,
                        "title_artifact_penalty_applied": float(_title_artifact_pen),
                        **layered_diag,
                    }
                    new_edge_components = list(state.edge_components) + [edge_component]

                    next_beam.append(BeamState(
                        path=new_path,
                        score=new_score,
                        used=new_used,
                        used_artists=new_used_artists,
                        used_sibling_groups=new_used_sibling_groups,
                        last_progress=new_last_progress,
                        edge_components=new_edge_components,
                    ))

            if apply_tie_break and cand_entries:
                for cand, cand_t, base_score, dest_pull, genre_sim, waypoint_sim, trans_score, edge_metric_tb, layered_diag_tb in cand_entries:
                    base_score_for_rank = float(base_score)
                    combined_score = float(base_score)
                    if (not _steering) and genre_sim is not None and math.isfinite(genre_sim):
                        if genre_tie_break_band is not None:
                            if (best_score - combined_score) <= float(genre_tie_break_band):
                                if penalty_strength > 0 and genre_sim < penalty_threshold:
                                    combined_score *= (1.0 - penalty_strength)
                                    genre_penalty_hits += 1
                        else:
                            if penalty_strength > 0 and genre_sim < penalty_threshold:
                                combined_score *= (1.0 - penalty_strength)
                                genre_penalty_hits += 1
                    if popularity_penalty_strength > 0.0 and popularity_values is not None:
                        combined_score *= _popularity_factor(
                            float(popularity_values[int(cand)]), popularity_penalty_strength)
                    waypoint_delta_val = 0.0
                    if waypoint_enabled:
                        # Always apply waypoint scoring (tie-break band removed in Phase 2)
                        waypoint_delta_val = _waypoint_delta(waypoint_sim, waypoint_sim0)
                        combined_score += waypoint_delta_val
                        # Phase 3 fix: Store for stats tracking
                        if waypoint_sim is not None:
                            step_waypoint_info[int(cand)] = (float(waypoint_sim), float(waypoint_delta_val))

                    # Coverage bonus (Phase 2 + Phase 3): reward matching anchor top-K genres with schedule decay
                    coverage_bonus_val = 0.0
                    if coverage_enabled and X_genre_for_coverage_presence is not None:
                        cand_genre_vec = X_genre_for_coverage_presence[cand]
                        coverage_A = _compute_coverage(
                            cand_genre_vec, topk_A, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_B = _compute_coverage(
                            cand_genre_vec, topk_B, float(cfg.dj_genre_presence_threshold), coverage_mode
                        )
                        coverage_bonus_val = _compute_coverage_bonus(
                            step, interior_length, coverage_A, coverage_B,
                            float(cfg.dj_genre_coverage_weight), float(cfg.dj_genre_coverage_power)
                        )
                        combined_score += coverage_bonus_val

                    if _sonic_tag_active:
                        combined_score += float(sonic_tag_beam_weight) * float(sonic_tag_affinity[int(cand)])

                    # Title-artifact penalty (tie-break path; opt-in; zero cost when disabled)
                    _title_artifact_pen_tb = 0.0
                    if bool(cfg.title_artifact_penalty_enabled) and cfg.title_artifact_penalty_weights:
                        _cand_title_tb = ""
                        try:
                            if bundle is not None and bundle.track_titles is not None:
                                _cand_title_tb = str(bundle.track_titles[int(cand)] or "")
                        except Exception:
                            _cand_title_tb = ""
                        if _cand_title_tb:
                            _title_artifact_pen_tb = compute_title_artifact_penalty(
                                title=_cand_title_tb,
                                weights=cfg.title_artifact_penalty_weights,
                            )
                            combined_score -= _title_artifact_pen_tb

                    # Rank impact: collect (cand_idx, base_score, waypoint_delta, coverage_bonus, full_score)
                    if step_is_sampled:
                        step_candidates_for_ranking.append((
                            int(cand),
                            float(base_score_for_rank),
                            float(waypoint_delta_val),
                            float(coverage_bonus_val),
                            float(combined_score)
                        ))

                    new_score = state.score + combined_score + float(dest_pull)
                    new_path = state.path + [cand]
                    new_used = state.used | {cand}
                    new_used_artists = state.used_artists
                    if artist_key_by_idx is not None:
                        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
                        if use_identity:
                            # Identity mode: use raw artist string to preserve collaborations
                            # (artist_key_by_idx has collaborations stripped, which prevents detection of featured artists)
                            cand_artist_str = ""
                            if bundle is not None and bundle.track_artists is not None:
                                try:
                                    cand_artist_str = str(bundle.track_artists[int(cand)] or "")
                                except Exception:
                                    cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            else:
                                cand_artist_str = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist_str:
                                cand_identity_keys = _cand_identity_keys(int(cand), cand_artist_str)
                                new_used_artists = state.used_artists | cand_identity_keys
                        else:
                            cand_artist = str(artist_key_by_idx.get(int(cand), "") or "")
                            if cand_artist:
                                new_used_artists = state.used_artists | {cand_artist}
                    new_used_sibling_groups = state.used_sibling_groups
                    _place_sg = _idx_sibling_group(int(cand))
                    if _place_sg is not None:
                        new_used_sibling_groups = state.used_sibling_groups | {_place_sg}
                    new_last_progress = float(state.last_progress)
                    if progress_active:
                        new_last_progress = float(progress_by_idx.get(int(cand), 0.0))

                    # Build per-edge diagnostic component dict (tie-break path)
                    _local_sonic_cos_tb = float(np.dot(X_full_norm[int(current)], X_full_norm[int(cand)]))
                    _genre_pen_applied_tb = 0.0
                    if genre_sim is not None and math.isfinite(genre_sim):
                        if penalty_strength > 0 and genre_sim < penalty_threshold:
                            _genre_pen_applied_tb = float(penalty_strength)
                    _local_pen_applied_tb = 0.0
                    if (
                        local_sonic_penalty_enabled
                        and local_sonic_penalty_strength > 0.0
                        and _local_sonic_cos_tb < local_sonic_penalty_threshold
                    ):
                        _local_pen_applied_tb = float(
                            local_sonic_penalty_strength * (local_sonic_penalty_threshold - _local_sonic_cos_tb)
                        )
                    edge_component_tb = {
                        "from_idx": int(current),
                        "to_idx": int(cand),
                        "bridge_score": None,  # not separately tracked in tie-break path
                        "trans_score_in_beam": float(trans_score),
                        "T": float(edge_metric_tb.get("T")) if edge_metric_tb.get("T") is not None else None,
                        "T_raw": edge_metric_tb.get("T_raw"),
                        "T_centered_cos": edge_metric_tb.get("T_centered_cos"),
                        "H": edge_metric_tb.get("H"),
                        "S": edge_metric_tb.get("S"),
                        "G": edge_metric_tb.get("G"),
                        "progress_t": float(cand_t) if progress_active else None,
                        "progress_jump": (float(cand_t) - float(state.last_progress)) if progress_active else None,
                        "local_sonic_raw_cos": _local_sonic_cos_tb,
                        "local_sonic_penalty_applied": _local_pen_applied_tb,
                        "genre_penalty_applied": _genre_pen_applied_tb,
                        "below_transition_floor": False,
                        "title_artifact_penalty_applied": float(_title_artifact_pen_tb),
                        **layered_diag_tb,
                    }
                    new_edge_components_tb = list(state.edge_components) + [edge_component_tb]

                    next_beam.append(BeamState(
                        path=new_path,
                        score=new_score,
                        used=new_used,
                        used_artists=new_used_artists,
                        used_sibling_groups=new_used_sibling_groups,
                        last_progress=new_last_progress,
                        edge_components=new_edge_components_tb,
                    ))

        if not next_beam:
            _record_genre_cache_stats()
            _record_local_sonic_stats()
            return None, genre_penalty_hits, edges_scored, f"no valid continuations at step={step}"

        # Keep top beam_width states. Roam corridors: when the minimax guard is on,
        # protect the weakest edge first (lexicographic), then total score.
        if bool(getattr(cfg, "worst_edge_minimax_enabled", False)):
            if _edge_band > 0.0:
                # Band the prune too, or on-tag states get pruned before the final
                # selection can pick them: bucket near-equal worst edges, order by score.
                next_beam.sort(
                    key=lambda s: (math.floor(_state_min_edge(s) / _edge_band), s.score),
                    reverse=True,
                )
            else:
                next_beam.sort(key=lambda s: (_state_min_edge(s), s.score), reverse=True)
        else:
            next_beam.sort(key=lambda s: s.score, reverse=True)
        beam = next_beam[:beam_width]

        # Phase 3 fix: Track waypoint info for the chosen candidate (top beam state's last track)
        if beam and waypoint_enabled:
            chosen_cand = beam[0].path[-1]  # Last track in best path
            if chosen_cand in step_waypoint_info:
                sim, delta = step_waypoint_info[chosen_cand]
                chosen_waypoint_sims.append(sim)
                chosen_waypoint_deltas.append(delta)
                chosen_waypoint_sim0s.append(waypoint_sim0)
            else:
                # No waypoint info (shouldn't happen if waypoint_enabled, but handle gracefully)
                chosen_waypoint_sims.append(0.0)
                chosen_waypoint_deltas.append(0.0)
                chosen_waypoint_sim0s.append(waypoint_sim0)

        # Rank impact: compute metrics for sampled steps
        if step_is_sampled and step_candidates_for_ranking:
            # Get unique candidates (may have duplicates from different beam states)
            cand_scores_dict: Dict[int, Tuple[float, float, float, float]] = {}
            for cand_idx, base_score, waypoint_delta, coverage_bonus, full_score in step_candidates_for_ranking:
                if cand_idx not in cand_scores_dict:
                    cand_scores_dict[cand_idx] = (base_score, waypoint_delta, coverage_bonus, full_score)
                else:
                    # If duplicate, keep the one with higher full_score
                    existing_full = cand_scores_dict[cand_idx][3]
                    if full_score > existing_full:
                        cand_scores_dict[cand_idx] = (base_score, waypoint_delta, coverage_bonus, full_score)

            # Convert to list for ranking
            cand_list = [(idx, base, delta, cov, full) for idx, (base, delta, cov, full) in cand_scores_dict.items()]

            # Rank by base_score (descending)
            cand_list_by_base = sorted(cand_list, key=lambda x: x[1], reverse=True)
            # Rank by full_score (descending)
            cand_list_by_full = sorted(cand_list, key=lambda x: x[4], reverse=True)

            # Create rank maps: cand_idx -> rank (1-indexed)
            base_rank_map = {cand_idx: rank + 1 for rank, (cand_idx, _, _, _, _) in enumerate(cand_list_by_base)}
            full_rank_map = {cand_idx: rank + 1 for rank, (cand_idx, _, _, _, _) in enumerate(cand_list_by_full)}

            # Compute metrics
            winner_changed = (cand_list_by_base[0][0] != cand_list_by_full[0][0])
            topK = min(10, len(cand_list))
            topK_base_set = {cand_idx for cand_idx, _, _, _, _ in cand_list_by_base[:topK]}
            topK_reordered_count = sum(
                1 for cand_idx in topK_base_set
                if base_rank_map[cand_idx] != full_rank_map[cand_idx]
            )
            rank_deltas = [abs(base_rank_map[cand_idx] - full_rank_map[cand_idx]) for cand_idx in cand_scores_dict]
            mean_abs_rank_delta = float(np.mean(rank_deltas)) if rank_deltas else 0.0
            max_rank_jump = int(max(rank_deltas)) if rank_deltas else 0

            # Task E: Saturation metrics
            waypoint_deltas = [delta for _, _, delta, _, _ in cand_list]
            cap = float(cfg.dj_waypoint_cap)
            frac_near_cap = 0.0
            frac_at_cap = 0.0
            if waypoint_deltas and cap > 0:
                frac_near_cap = sum(1 for d in waypoint_deltas if abs(d) > 0.8 * cap) / len(waypoint_deltas)
                frac_at_cap = sum(1 for d in waypoint_deltas if abs(d) > 0.9 * cap) / len(waypoint_deltas)

            # Store step result
            step_result = {
                "step": step,
                "interior_length": interior_length,
                "winner_changed": winner_changed,
                "topK_reordered_count": topK_reordered_count,
                "topK": topK,
                "mean_abs_rank_delta": mean_abs_rank_delta,
                "max_rank_jump": max_rank_jump,
                "total_candidates": len(cand_list),
                # Task E: Saturation diagnostics
                "waypoint_sim0": waypoint_sim0,
                "waypoint_delta_mean": float(np.mean(waypoint_deltas)) if waypoint_deltas else 0.0,
                "waypoint_delta_p50": float(np.percentile(waypoint_deltas, 50)) if waypoint_deltas else 0.0,
                "waypoint_delta_p90": float(np.percentile(waypoint_deltas, 90)) if waypoint_deltas else 0.0,
                "waypoint_frac_near_cap": frac_near_cap,
                "waypoint_frac_at_cap": frac_at_cap,
                # Top-10 table for detailed logging (optional)
                "top10_table": [
                    {
                        "cand_idx": cand_idx,
                        "base_score": base,
                        "waypoint_delta": delta,
                        "coverage_bonus": cov,
                        "full_score": full,
                        "base_rank": base_rank_map[cand_idx],
                        "full_rank": full_rank_map[cand_idx],
                        "rank_delta": base_rank_map[cand_idx] - full_rank_map[cand_idx],
                    }
                    for cand_idx, base, delta, cov, full in cand_list_by_base[:topK]
                ]
            }
            rank_impact_results.append(step_result)

            # Phase 2 per-step logging: target genres and top candidates
            if cfg.dj_bridging_enabled and g_target is not None and genre_vocab is not None:
                # Log target genre distribution (top 5-8)
                target_topk = _extract_top_genres(g_target, 8)
                if target_topk:
                    genre_names = [f"{genre_vocab[idx]}={weight:.3f}" for idx, weight in target_topk[:5]]
                    logger.info("  [Step %d/%d] Target genres: %s", step, interior_length, ", ".join(genre_names))

                # Log best 3 candidates by full_score (with genre alignment)
                if X_genre_for_coverage is not None and len(cand_list_by_full) >= 3:
                    logger.info("  [Step %d/%d] Top-3 candidates by full_score:", step, interior_length)
                    for rank, (cand_idx, base, delta, cov, full) in enumerate(cand_list_by_full[:3], start=1):
                        # Compute genre alignment to target
                        cand_genre_vec = X_genre_for_coverage[cand_idx]
                        genre_sim = float(np.dot(cand_genre_vec, g_target))
                        logger.info("    #%d: idx=%d base=%.3f waypoint=%.3f coverage=%.3f full=%.3f genre_sim=%.3f",
                                    rank, cand_idx, base, delta, cov, full, genre_sim)

    # Final step: connect to pier_b
    # Per-segment on-arc floor for the final connection: derive from the arc-sims of
    # the surviving beam states' last interior tracks to the final target.
    final_arc_floor: Optional[float] = None
    if (
        _steering_cfg
        and arc_g_targets is not None
        and len(arc_g_targets) > 0
        and X_genre_for_sim is not None
        and arc_floor_percentile > 0.0
    ):
        _final_target = arc_g_targets[-1]
        _final_sims: List[float] = []
        for state in beam:
            li = int(state.path[-1])
            if genre_present is not None and not bool(genre_present[li]):
                continue
            _final_sims.append(float(np.dot(X_genre_for_sim[li], _final_target)))
        if _final_sims:
            final_arc_floor = floor_at_percentile(
                np.array(_final_sims, dtype=float), arc_floor_percentile
            )

    final_candidates: List[BeamState] = []

    for state in beam:
        last = state.path[-1]
        final_edge_metric = _score_shared_transition(last, pier_b)
        final_trans = float(final_edge_metric.get("T", float("nan")))

        # Anti-alignment safety only (is_broken_transition no longer T-gates; roam design)
        if _transition_gate_failed(final_edge_metric):
            continue

        final_edge_score = final_trans
        edges_scored += 1
        if _steering:
            # Pairwise genre-edge soft penalty on the pier-adjacent edge (last ->
            # pier_b): the highest-stakes edge in the segment. Demote, never reject,
            # so the floor can't brick the final connection. Genreless endpoints exempt.
            if _pair_floor > 0.0 and _pair_penalty > 0.0:
                _pair_sim_final = _pair_edge_sim(int(last), int(pier_b))
                if (
                    _pair_sim_final is not None
                    and math.isfinite(_pair_sim_final)
                    and _pair_sim_final < _pair_floor
                ):
                    final_edge_score -= _pair_penalty
                    pair_penalty_hits += 1
            # Genre ARC vote at the final connection: closeness of the last interior
            # track to the final target g_targets[-1] (NOT prev-track similarity to
            # pier_b). Same per-segment on-arc floor applies (computed from the final
            # step's pool distribution, stored in final_arc_floor below).
            if arc_g_targets is not None and len(arc_g_targets) > 0 and X_genre_for_sim is not None:
                final_target = arc_g_targets[-1]
                last_present = (genre_present is None) or bool(genre_present[int(last)])
                if last_present:
                    final_arc_sim = float(np.dot(X_genre_for_sim[int(last)], final_target))
                    if final_arc_floor is not None and final_arc_sim < final_arc_floor:
                        continue  # this beam state cannot legally connect to pier_b
                    if final_arc_sim < cfg.genre_arc_floor:
                        continue
                    if float(cfg.weight_genre) > 0.0:
                        final_edge_score += float(cfg.weight_genre) * final_arc_sim
        elif X_genre_norm is not None:
            genre_sim = _get_genre_sim(int(last), int(pier_b))
            if genre_sim is not None and math.isfinite(genre_sim):
                if cfg.genre_tiebreak_weight:
                    final_edge_score += cfg.genre_tiebreak_weight * genre_sim
                if (
                    penalty_strength > 0
                    and genre_sim < penalty_threshold
                ):
                    final_edge_score *= (1.0 - penalty_strength)
                    genre_penalty_hits += 1

        final_layered_delta, final_layered_diag = _layered_transition_delta(
            int(last),
            int(pier_b),
            final_edge_metric,
        )
        final_edge_score += final_layered_delta

        final_edge_score_after_sonic = _apply_local_sonic_edge_policy(
            final_edge_score,
            int(last),
            int(pier_b),
        )
        if final_edge_score_after_sonic is None:
            continue
        final_edge_score = float(final_edge_score_after_sonic)

        total_score = state.score + final_edge_score

        # Build edge component for the final edge
        _final_local_sonic_cos = float(np.dot(X_full_norm[int(last)], X_full_norm[int(pier_b)]))
        _final_genre_pen_applied = 0.0
        if (not _steering) and X_genre_norm is not None:
            genre_sim = _get_genre_sim(int(last), int(pier_b))
            if genre_sim is not None and math.isfinite(genre_sim):
                if penalty_strength > 0 and genre_sim < penalty_threshold:
                    _final_genre_pen_applied = float(penalty_strength)
        _final_local_pen_applied = 0.0
        if (
            local_sonic_penalty_enabled
            and local_sonic_penalty_strength > 0.0
            and _final_local_sonic_cos < local_sonic_penalty_threshold
        ):
            _final_local_pen_applied = float(
                local_sonic_penalty_strength * (local_sonic_penalty_threshold - _final_local_sonic_cos)
            )
        final_edge_component = {
            "from_idx": int(last),
            "to_idx": int(pier_b),
            "bridge_score": None,
            "trans_score_in_beam": float(final_trans),
            "T": float(final_edge_metric.get("T")) if final_edge_metric.get("T") is not None else None,
            "T_raw": final_edge_metric.get("T_raw"),
            "T_centered_cos": final_edge_metric.get("T_centered_cos"),
            "H": final_edge_metric.get("H"),
            "S": final_edge_metric.get("S"),
            "G": final_edge_metric.get("G"),
            "progress_t": None,
            "progress_jump": None,
            "local_sonic_raw_cos": _final_local_sonic_cos,
            "local_sonic_penalty_applied": _final_local_pen_applied,
            "genre_penalty_applied": _final_genre_pen_applied,
            "below_transition_floor": False,
            "title_artifact_penalty_applied": 0.0,
            **final_layered_diag,
        }

        # Create a new state with the final edge appended for selection purposes
        # Note: path remains unchanged (only interior tracks), final edge is tracked separately
        final_state_with_edge = BeamState(
            path=state.path,
            score=total_score,
            used=state.used,
            used_artists=state.used_artists,
            last_progress=state.last_progress,
            edge_components=list(state.edge_components) + [final_edge_component],
        )
        final_candidates.append(final_state_with_edge)

    if not final_candidates:
        _record_genre_cache_stats()
        _record_local_sonic_stats()
        return None, genre_penalty_hits, edges_scored, "no valid final connection to destination"

    best_final = _select_best_beam_state(
        final_candidates,
        objective=(
            "min_edge"
            if bool(getattr(cfg, "worst_edge_minimax_enabled", False))
            else str(getattr(cfg, "min_edge_objective", "total_score") or "total_score")
        ),
        min_edge_band=_edge_band,
    )

    # Compute waypoint diagnostics for chosen path
    if waypoint_stats is not None and waypoint_enabled and g_targets is not None and X_genre_norm is not None:
        # Phase 3 fix: Use tracked actual applied deltas instead of recomputing with wrong sim0
        # The old code recomputed deltas with sim0=0.0 (default), which is wrong for centered mode.
        # Now we use the actual deltas that were applied during beam search.
        if chosen_waypoint_sims and chosen_waypoint_deltas:
            waypoint_sims = chosen_waypoint_sims
            waypoint_deltas = chosen_waypoint_deltas
        else:
            # Fallback: recompute (shouldn't happen if waypoint_enabled, but handle gracefully)
            full_path = best_final.path  # includes pier_a at start
            interior_path = full_path[1:]  # interior tracks only
            waypoint_sims: List[float] = []
            waypoint_deltas: List[float] = []
            for step_idx, track_idx in enumerate(interior_path):
                if step_idx < len(g_targets):
                    g_target = g_targets[step_idx]
                    waypoint_sim = float(np.dot(X_genre_norm[track_idx], g_target))
                    # Note: This fallback still has the sim0=0.0 issue, but it's rare
                    waypoint_delta = _waypoint_delta(waypoint_sim)
                    waypoint_sims.append(waypoint_sim)
                    waypoint_deltas.append(waypoint_delta)

        if waypoint_sims:
            waypoint_stats["waypoint_enabled"] = True
            waypoint_stats["waypoint_sims"] = waypoint_sims
            waypoint_stats["waypoint_deltas"] = waypoint_deltas
            waypoint_stats["mean_waypoint_sim"] = float(np.mean(waypoint_sims))
            waypoint_stats["p50_waypoint_sim"] = float(np.percentile(waypoint_sims, 50))
            waypoint_stats["p90_waypoint_sim"] = float(np.percentile(waypoint_sims, 90))
            waypoint_stats["min_waypoint_sim"] = float(np.min(waypoint_sims))
            waypoint_stats["max_waypoint_sim"] = float(np.max(waypoint_sims))
            waypoint_stats["waypoint_delta_applied_count"] = int(sum(1 for d in waypoint_deltas if abs(d) > 1e-9))
            waypoint_stats["mean_waypoint_delta"] = float(np.mean(waypoint_deltas))
        else:
            waypoint_stats["waypoint_enabled"] = True
            waypoint_stats["waypoint_sims"] = []
            waypoint_stats["mean_waypoint_sim"] = 0.0

        # Add rank impact results if computed
        if rank_impact_results:
            waypoint_stats["rank_impact_results"] = rank_impact_results
    elif waypoint_stats is not None:
        waypoint_stats["waypoint_enabled"] = False

    # TASK A: Add pool_after_gating count to waypoint_stats
    if waypoint_stats is not None:
        waypoint_stats["pool_after_gating_count"] = len(pool_after_gating_candidates)

    # Expose winning beam's per-edge component dicts via out-param (diagnostic only)
    if edge_components_out is not None:
        edge_components_out["components"] = list(best_final.edge_components)

    # Return interior tracks (exclude pier_a which is path[0])
    _record_genre_cache_stats()
    _record_local_sonic_stats()
    if pair_penalty_hits > 0:
        logger.info(
            "Pairwise genre-edge soft penalty: floor=%.2f penalty=%.2f demoted %d of %d scored edges",
            _pair_floor,
            _pair_penalty,
            pair_penalty_hits,
            edges_scored,
        )
    if duration_penalty_values is not None and duration_penalty_hits > 0:
        logger.info(
            "Duration soft penalty (beam): %d candidate evaluations demoted (%d edges scored this segment)",
            duration_penalty_hits,
            edges_scored,
        )
    return best_final.path[1:], genre_penalty_hits, edges_scored, None
