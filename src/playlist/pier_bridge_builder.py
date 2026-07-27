"""
Pier + Bridge Playlist Builder
==============================

A new playlist ordering strategy where:
- Each seed track is a fixed "pier"
- Bridge segments connect consecutive piers
- No repair pass after ordering

Key features:
- Candidate pool deduped BEFORE ordering (no duplicate songs by normalized artist+title)
- Genre gating stays enabled with hard floors (no relaxation)
- Global used_track_ids prevents duplicates across segments
- One track per artist per segment enforced during beam search
- Cross-segment min_gap enforced during generation via boundary-aware constraints
- No post-order filtering or dropping (guarantees exact length)
- Single seed mode: seed acts as both start AND end pier, creating an arc structure
- Seed artist is allowed in bridges with same constraints as other artists
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from src.features.artifacts import ArtifactBundle
from src.cancellation import raise_if_cancelled
from src.string_utils import sanitize_for_logging
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    resolve_artist_identity_keys,
)
from src.playlist.run_audit import InfeasibleHandlingConfig, RunAuditConfig, RunAuditEvent, now_utc_iso

# Tier-3.1 PR-1: 14 pure genre helpers extracted to pier_bridge/genre.py.
# Re-exported here for backward compatibility (callers + tests still import
# these names from src.playlist.pier_bridge_builder).
from src.playlist.pier_bridge.genre import (  # noqa: F401  (re-exported for tests/back-compat)
    _apply_idf_weighting,
    _compute_coverage,
    _compute_coverage_bonus,
    _compute_genre_idf as _compute_genre_idf_impl,
    _ensure_genre_similarity_overrides_loaded,
    _extract_top_genres,
    _genre_similarity_score,
    _genre_vocab_map,
    _label_to_genre_vector,
    _label_to_smoothed_vector,
    _load_genre_similarity_graph,
    _normalize_vec,
    _select_top_genre_labels,
    _shortest_genre_path,
)


# Tier-3.1 PR-2: vector math + transition scoring.
from src.playlist.pier_bridge.vec import (  # noqa: F401
    _cosine_sim,
    _l2_normalize_rows,
    _compute_transition_score as _compute_transition_score_impl,
    _compute_transition_score_raw_and_transformed as _compute_transition_score_raw_and_transformed_impl,
)

# Tier-3.1 PR-2: progress + diagnostic metrics.
from src.playlist.pier_bridge.metrics import (  # noqa: F401
    _compute_chosen_source_counts,
    _compute_pool_overlap_metrics,
    _compute_progress_tracking_metrics,
    _dist,
    _progress_arc_loss_value,
    _progress_target_curve,
    _step_fraction,
)

# Tier-3.1 PR-3: config dataclasses + resolver + back-compat wrappers.
from src.playlist.pier_bridge.config import (  # noqa: F401  (re-exported for callers + tests)
    PierBridgeConfig,
    PierBridgeResult,
    SegmentDiagnostics,
    resolve_pier_bridge_tuning,
    _compute_genre_idf,
    _compute_transition_score,
    _compute_transition_score_raw_and_transformed,
)

# Tier-3.1 PR-4: audit summary + bridgeability scoring.
from src.playlist.pier_bridge.audit_summary import (  # noqa: F401
    _compute_bridgeability_score,
    _summarize_candidates_for_audit,
)

# Tier-3.1 PR-5: micro-pier + relaxation subsystem.
from src.playlist.pier_bridge.micro_pier import (  # noqa: F401
    _attempt_micro_pier_split,
    _build_dj_relaxation_attempts,
    _micro_pier_candidate_pool,
    _score_micro_pier_candidates,
    _select_micro_pier_candidates,
    _should_attempt_micro_pier,
)

# Tier-3.1 PR-6: seed ordering + pool prep helpers.
from src.playlist.pier_bridge.seeds import (  # noqa: F401
    _dedupe_candidate_pool,
    _order_seeds_by_bridgeability,
    _segment_far_stats,
    _select_connector_candidates,
)

# Tier-3.1 PR-7: bridge-score kernel. (_build_segment_candidate_pool_legacy
# and _build_segment_candidate_pool_scored deleted -- Phase 1 Task 8.)
from src.playlist.pier_bridge.pool import (  # noqa: F401  (re-exported for tests/back-compat)
    _compute_bridge_score,
)

# Tier-3.1 PR-8: genre waypoint target builder.
from src.playlist.pier_bridge.genre_targets import (  # noqa: F401  (re-exported for tests/back-compat)
    _build_genre_targets,
    _fallback_genre_vector,
    build_dense_genre_targets,
)

# Tier-3.1 PR-8: beam search engine + duration penalty.
from src.playlist.pier_bridge.beam import (  # noqa: F401  (re-exported for tests/back-compat)
    BeamState,
    _beam_search_segment,
)
from src.playlist.pier_bridge.roam import (
    segment_sonic_detour as _segment_sonic_detour,
    energy_band_deviation as _energy_band_deviation,
)
from src.playlist.transition_metrics import (
    TransitionMetricContext,
    resolve_transition_calib,
    score_transition_edge,
)
from src.playlist.pier_bridge.tail_dp import optimize_segment_tail
from src.playlist.pier_bridge.repair_triggers import compute_relative_trigger_floor

# Phase 1 Task 3: corridor segment pooling (dev flag, default off -- see
# PierBridgeConfig.pooling). build_corridor / build_eligible_universe are pure
# and are NOT modified by this task; the glue lives entirely here.
from src.playlist.pier_bridge.corridor import (
    CorridorWidenDecision,
    build_corridor,
    corridor_widen_decision,
    resolve_corridor_width_percentile,
)
from src.playlist.pier_bridge.eligible_universe import build_eligible_universe, EligibleUniverse

# Phase 1 Task 4: genre-mode-keyed relevance mask, reused from the pier-veto's
# own genre gate (artist_style.py) so the corridor path's mask is computed by
# the exact same function, not a reimplementation.
from src.playlist.artist_style import seed_genre_relevance_mask

# Phase 1 Task 5: Oops-All-Bangers reseat onto the corridor universe -- reuses
# candidate_pool.py's own rank-cutoff gate (imported, not copied) so both
# pooling strategies filter identically.
from src.playlist.candidate_pool import (
    _apply_popularity_gate,
    compute_duration_penalty,
    compute_seed_reference_duration_ms,
)


logger = logging.getLogger(__name__)

# Hard wall-clock ceiling for the per-segment floor-relaxation tiers (transition
# floor + genre-arc floor). On a starved/infeasible pool these tiers sweep a large
# transition x genre-arc cross-product that can run for minutes (observed 274s on
# a narrow+narrow Charli XCX run) and blow the 90s generation budget. Once the
# segment-build phase has spent this long cumulatively, stop entering/continuing
# relaxation tiers and fall through to the guaranteed-fill fallback placement,
# which still produces a valid playlist. Set generously above healthy build times
# (good MERT runs complete segment-build well under this) so only pathological
# grinds are cut. Kept well below the 90s hard ceiling to leave room for the
# pre-loop overhead (Last.fm recency, candidate pool, seed-order permutations)
# that this anchor does not count. TODO: promote to a config knob
# (playlists.pier_bridge.*).
_SEGMENT_RELAXATION_BUDGET_S = 40.0


def _corridor_genre_relevance_floor(genre_mode: Optional[str]) -> Optional[float]:
    """Corridor relevance-mask genre-similarity floor, keyed by genre_mode.

    Phase 1 provisional mapping (spec docs/superpowers/specs/2026-07-12-
    corridor-first-pooling-design.md section 4; NOT yet Phase-2-calibrated):
      - "off" (or unspecified/unrecognized) -> None: mask disabled, corridor
        universe ungated by genre relevance. Unspecified is treated the same
        as "off" deliberately -- see build_pier_bridge_playlist's genre_mode
        docstring for why the production caller doesn't wire this yet; a
        missing signal must never be guessed into an active gate.
      - "strict"/"narrow" -> 0.30, matching today's
        PierBridgeConfig.pier_bridgeability_genre_floor default (the pier-
        veto's own tight-mode floor) -- same order of strictness, reused
        rather than inventing a new constant.
      - "dynamic"/"discover" -> 0.20, a looser floor for the two modes that
        already admit more genre variation upstream.
    """
    if genre_mode is None:
        return None
    m = str(genre_mode).strip().lower()
    if m in ("strict", "narrow"):
        return 0.30
    if m in ("dynamic", "discover"):
        return 0.20
    return None  # "off" or unrecognized -> mask disabled


def _compute_edge_scores(
    path: List[int],
    X_full: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    metric_context: Optional[TransitionMetricContext] = None,
) -> Tuple[float, float]:
    """Compute worst and mean edge scores for a path."""
    if len(path) < 2:
        return (1.0, 1.0)

    scores = []
    for i in range(len(path) - 1):
        if metric_context is not None:
            score = float(score_transition_edge(metric_context, path[i], path[i + 1]).get("T"))
        else:
            score = _compute_transition_score(
                path[i], path[i + 1], X_full, X_start, X_mid, X_end, cfg
            )
        scores.append(score)

    return (min(scores), sum(scores) / len(scores))


def _order_avoiding_adjacent_artist(
    entries: list[tuple[int, float, float]],
    artist_key_fn: "Callable[[int], Set[str]]",
    blocked_artist_keys: Optional[Set[str]],
) -> list[int]:
    """Greedily order (idx, score, sb) tuples (already in progress order) so that no
    two adjacent tracks share an artist identity key, including against the previous
    segment's boundary artists (``blocked_artist_keys`` seeds the "previous" slot).
    Best-effort and never-fail: if a same-artist pick is unavoidable it is placed
    anyway — a track is never dropped."""
    remaining = list(entries)
    result: list[int] = []
    prev_keys: Set[str] = set(blocked_artist_keys or ())
    while remaining:
        pick = 0
        for j, entry in enumerate(remaining):
            ks = artist_key_fn(entry[0]) or set()
            if not (ks & prev_keys):
                pick = j
                break
        chosen = remaining.pop(pick)
        result.append(chosen[0])
        prev_keys = artist_key_fn(chosen[0]) or set()
    return result


def _greedy_terminal_path(
    candidates: list[int], global_used: "Set[int]", pier_a: int, pier_b: int,
    interior_len: int, X_full_norm: np.ndarray,
    X_genre_norm: Optional[np.ndarray] = None, genre_weight: float = 0.0,
    artist_key_fn: "Optional[Callable[[int], Set[str]]]" = None,
    blocked_artist_keys: Optional[Set[str]] = None,
) -> Optional[list[int]]:
    """Last-resort placement that cannot fail while >= interior_len usable tracks exist.

    Picks the interior_len tracks with the best blended cosine to both piers, then
    orders them by increasing sonic similarity to pier_b so the segment progresses
    toward B. The selection score is
        (1 - genre_weight) * mean_sonic_cos(piers) + genre_weight * mean_genre_cos(piers)
    so an infeasible segment is no longer filled on sonic similarity alone (MERT cosine
    is perceptually unreliable across dissimilar anchors). With genre_weight <= 0 or no
    genre matrix this reduces to the legacy sonic-only behavior, preserving the
    never-fail guarantee (selection never drops candidates, only reweights them).
    Excludes already-used tracks, the piers, duplicates, out-of-range, and NaN/inf-scored
    candidates (zero-vector rows) so the sort can never raise.

    When ``artist_key_fn`` is given, selection enforces artist DIVERSITY — diversity is
    a hard constraint, so the terminal fallback must honor it too: a per-artist cap is
    applied (starting at 1) and escalated only as far as needed to fill interior_len,
    and the result is ordered to avoid adjacent same-artist tracks (and against the
    previous segment's ``blocked_artist_keys``). This is what stops the fallback from
    clustering e.g. three tracks by one artist back-to-back. With ``artist_key_fn=None``
    the legacy top-score ordering is preserved exactly.
    """
    if interior_len <= 0:
        return []
    n = int(X_full_norm.shape[0])
    a_vec = X_full_norm[pier_a]
    b_vec = X_full_norm[pier_b]
    w = float(min(max(genre_weight, 0.0), 1.0))
    use_genre = (
        w > 0.0
        and X_genre_norm is not None
        and int(X_genre_norm.shape[0]) == n
    )
    ga_vec = X_genre_norm[pier_a] if use_genre else None
    gb_vec = X_genre_norm[pier_b] if use_genre else None
    pool: list[tuple[int, float, float]] = []
    seen: set[int] = set()
    for raw in candidates:
        i = int(raw)
        if i in seen or i in global_used or i == pier_a or i == pier_b or not (0 <= i < n):
            continue
        sa = float(np.dot(X_full_norm[i], a_vec))
        sb = float(np.dot(X_full_norm[i], b_vec))
        if not (math.isfinite(sa) and math.isfinite(sb)):
            continue
        sonic_blend = 0.5 * sa + 0.5 * sb
        score = sonic_blend
        if use_genre:
            ga = float(np.dot(X_genre_norm[i], ga_vec))
            gb = float(np.dot(X_genre_norm[i], gb_vec))
            genre_blend = (0.5 * ga + 0.5 * gb) if (math.isfinite(ga) and math.isfinite(gb)) else 0.0
            score = (1.0 - w) * sonic_blend + w * genre_blend
        seen.add(i)
        pool.append((i, score, sb))
    if len(pool) < interior_len:
        return None
    pool.sort(key=lambda t: t[1], reverse=True)

    if artist_key_fn is None:
        # Legacy: top-interior_len by score, ordered by progress toward B.
        chosen = pool[:interior_len]
        chosen.sort(key=lambda t: t[2])  # ascending sonic sim-to-pier_b = progress toward B
        return [t[0] for t in chosen]

    # Artist-diverse selection: apply a per-artist cap (diversity is a hard
    # constraint), escalating it only as far as needed to fill interior_len so the
    # never-fail guarantee holds even when too few distinct artists exist.
    selected: list[tuple[int, float, float]] = []
    for cap in range(1, interior_len + 1):
        picked: list[tuple[int, float, float]] = []
        counts: Dict[str, int] = {}
        for entry in pool:
            ks = artist_key_fn(entry[0]) or set()
            if ks and any(counts.get(k, 0) >= cap for k in ks):
                continue
            picked.append(entry)
            for k in ks:
                counts[k] = counts.get(k, 0) + 1
            if len(picked) >= interior_len:
                break
        if len(picked) >= interior_len:
            selected = picked
            break
    if len(selected) < interior_len:
        selected = pool[:interior_len]  # diversity unsatisfiable -> never-fail

    selected.sort(key=lambda t: t[2])  # progress toward B, then break same-artist adjacency
    return _order_avoiding_adjacent_artist(selected, artist_key_fn, blocked_artist_keys)


def _require_usable_genre_steering(
    cfg: "PierBridgeConfig", X_genre_dense: Optional[np.ndarray]
) -> None:
    """Fail loudly when a configured genre-steering source cannot act — a configured
    knob that can't act is an error, not a silently dead segment.

    'taxonomy' (the canonical default) steers on the in-artifact X_genre_raw and can
    always act. 'dense' requires the dim64 sidecar embedding (X_genre_dense), which is
    absent when the sidecar vocabulary does not match the artifact; the old code then
    silently produced zero arc targets on every segment ("no usable g_targets").
    """
    if not bool(getattr(cfg, "genre_steering_enabled", False)):
        return
    source = str(getattr(cfg, "genre_steering_source", "taxonomy"))
    if source == "dense" and X_genre_dense is None:
        raise ValueError(
            "genre_steering_source='dense' but the dense genre embedding is unavailable "
            "(X_genre_dense is None — the dim64 sidecar vocabulary does not match the "
            "artifact). Use genre_steering_source='taxonomy' (the default) or rebuild "
            "the dense sidecar via scripts/build_genre_embedding.py."
        )


def _pier_nominal_positions(segment_lengths: "Sequence[int]") -> list[int]:
    """Cumulative NOMINAL (pre-``variable_bridge_length``-flex) position of every
    pier in ``ordered_seeds``, index-aligned (``len(segment_lengths) + 1`` entries:
    one per pier, including both endpoints of every segment).

    Used only to estimate forward pier-to-pier distance for cross-segment min_gap
    enforcement (see ``_forward_pier_gap_block_indices``); ``variable_bridge_length``
    only ever LENGTHENS a segment (add-only), so nominal positions are a safe
    (conservative-short) lower bound on the real distance -- using them can only
    over-block a borderline-safe candidate, never miss a real violation.
    """
    positions = [0]
    for length in segment_lengths:
        positions.append(positions[-1] + int(length) + 1)
    return positions


def _forward_pier_gap_block_indices(
    pier_positions: "Sequence[int]", seg_idx: int, min_gap: int,
) -> list[int]:
    """Indices of UPCOMING piers (beyond this segment's own two boundary piers)
    whose nominal position lands within ``min_gap`` of this segment's own start.

    Root-cause fix (Phase 1 min_gap investigation, 2026-07-18): cross-segment
    min_gap enforcement (``recent_boundary_artists``, fixed 2026-06-01) only ever
    looked BACKWARD at already-placed tracks. ``ordered_seeds`` -- the full pier
    sequence, including mini-pier waypoints (SP3, 2026-06-30) -- is entirely fixed
    BEFORE the segment loop starts, so a segment built early can freely place an
    interior track by the same artist as a not-yet-built pier/waypoint only a few
    positions later; nothing ever re-checked that placement once the later pier
    was committed, because piers are placed unconditionally (never gated on
    novelty). This is a pre-existing hole (predates corridor pooling entirely --
    ``recent_boundary_artists`` and mini-piers are both older than the
    corridor-phase1-pooling branch), exposed here for the first time by a
    5-seed fixture whose mini-pier subdivision packs waypoints close enough
    together to collide with an earlier segment's own interior pick.

    ``seg_idx + 2`` is the first index NOT already covered by
    ``disallow_pier_artists_in_interiors`` (which bans this segment's own
    ``ordered_seeds[seg_idx]``/``ordered_seeds[seg_idx + 1]`` locally).
    """
    if min_gap <= 0:
        return []
    start = int(pier_positions[seg_idx])
    blocked: list[int] = []
    for j in range(seg_idx + 2, len(pier_positions)):
        if int(pier_positions[j]) - start >= int(min_gap):
            break  # positions strictly increase -- every later pier is further still
        blocked.append(j)
    return blocked


def build_pier_bridge_playlist(
    *,
    seed_track_ids: List[str],
    total_tracks: int,
    bundle: ArtifactBundle,
    candidate_pool_indices: List[int],
    cfg: Optional[PierBridgeConfig] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_method: str = "ensemble",
    internal_connector_indices: Optional[Set[int]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    allowed_track_ids_set: Optional[set[str]] = None,
    infeasible_handling: Optional[InfeasibleHandlingConfig] = None,
    audit_config: Optional[RunAuditConfig] = None,
    audit_events: Optional[list[RunAuditEvent]] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
    perceptual_bpm: Optional[np.ndarray] = None,
    tempo_stability_arr: Optional[np.ndarray] = None,
    onset_rate: Optional[np.ndarray] = None,
    energy_matrix: Optional[np.ndarray] = None,
    voice_prob: Optional[np.ndarray] = None,
    popularity_values: Optional[np.ndarray] = None,
    min_gap: int = 1,
    deadline: Optional[float] = None,
    sonic_tag_affinity: Optional[np.ndarray] = None,
    sonic_tag_beam_weight: float = 0.0,
    tag_steering_worst_edge_band: float = 0.0,
    tag_steering_relax_bridge_admission: bool = False,
    on_tag_guarantee_ids: Optional[set[str]] = None,
    on_tag_segment_guarantee_max: int = 0,
    on_tag_segment_guarantee_per_artist: int = 0,
    genre_mode: Optional[str] = None,
    sonic_mode: Optional[str] = None,
    popularity_ranks: Optional[np.ndarray] = None,
    popularity_rank_cutoff: Optional[int] = None,
    # Tag-steering ANCHOR placement (spec 2026-07-27). The bundle track_ids of the
    # on-tag anchors playlist_generator injected into the pier list. EXPLICIT --
    # never derived from "artist != seed artist", because this builder is shared
    # with genre mode, whose piers are cross-artist by design. Empty/None => the
    # pier set and its ordering are byte-identical to the pre-2026-07-27 path.
    tag_anchor_track_ids: Optional[set[str]] = None,
    # Rollback lever (config: playlists.ds_pipeline.pier_bridge.
    # tag_steering_anchor_gap_insertion). False = legacy append + global re-order.
    tag_anchor_gap_insertion: bool = True,
    # Floor a placed anchor's weaker flanking edge must clear; mirrors the
    # selection-time tag_steering_anchor_min_bridge.
    tag_anchor_min_bridge: float = 0.35,
) -> PierBridgeResult:
    """
    Build playlist using pier + bridge strategy.

    Args:
        seed_track_ids: List of seed track IDs (will become piers)
        total_tracks: Target total playlist length
        bundle: Artifact bundle with sonic features
        candidate_pool_indices: Pre-filtered candidate pool indices
        cfg: Configuration (uses defaults if None)
        X_genre_smoothed: Genre vectors for gating
        genre_method: Genre similarity method
        genre_mode: The run's genre_mode axis ("off"/"strict"/"narrow"/"dynamic"/
            "discover"), CORRIDOR-POOLING ONLY (Phase 1 Task 4): keys the
            relevance-mask floor fed to build_eligible_universe (see
            ``_corridor_genre_relevance_floor`` below). Not consumed anywhere
            else in this function. Phase 1 Task 5 closed the production gap
            noted here through Task 4: ``pipeline.core.generate_playlist_ds``'s
            ``_run_pier_bridge`` closure now threads the raw genre_mode string
            (resolved via ``genre_ds_params.resolve_genre_ds_params``, which
            reads ``playlists_cfg.get("genre_mode")`` -- the same raw value
            ``mode_presets.apply_mode_presets`` consumes but never deletes)
            all the way from ``playlist_generator.py`` through
            ``ds_pipeline_runner.generate_playlist_ds`` to here. Corridor-mode
            generations now get the real slider value.
        sonic_mode: The run's sonic_mode axis ("off"/"strict"/"narrow"/
            "dynamic"/"discover"), CORRIDOR-POOLING ONLY (per-mode corridor
            width, spec section 4, pulled forward from Phase 2 by Dylan's
            2026-07-18 decision): resolves the per-segment corridor width
            percentile via ``resolve_corridor_width_percentile`` (see
            ``pier_bridge.corridor``), same seam pattern as ``genre_mode``
            above -- threaded from ``playlist_generator.py`` through
            ``genre_ds_params.resolve_genre_ds_params`` (which reads
            ``playlists_cfg.get("sonic_mode")``) and
            ``ds_pipeline_runner.generate_playlist_ds`` to here. Resolved
            ONCE, near the top of this function, into a concrete
            ``cfg.corridor_width_percentile`` float (via ``replace(cfg,
            ...)``) that every downstream corridor call site in this function
            already reads unchanged -- so this is the ONLY place sonic_mode
            is consumed. Before this task, sonic_mode was a near-dead axis
            for corridor generation (it only ever reached the largely-
            vestigial legacy candidate pool, which the corridor path ignores
            except through the disabled-by-default dj_bridging path and the
            last-resort terminal-greedy fallback).
        popularity_ranks: Optional (N,) array of per-bundle-row 0-based
            Last.fm popularity ranks, CORRIDOR-POOLING ONLY (Phase 1 Task 5
            reseat): when given together with ``popularity_rank_cutoff``,
            gates the corridor eligible universe to bangers once at universe
            build (mirrors ``candidate_pool.py``'s ``_apply_popularity_gate``,
            imported not copied). Same array core.py's ``_banger_gate_inputs``
            already resolves for the legacy pool (aligned to the full bundle,
            not just ``candidate_pool_indices``). None = gate inactive.
        popularity_rank_cutoff: 0-based rank cutoff paired with
            ``popularity_ranks`` (keep rank in ``[0, cutoff)``). None = gate
            inactive. No relax-to-fill cascade under corridor pooling -- the
            widening ladder is the sole relaxation mechanism for this path.

    Returns:
        PierBridgeResult with ordered track IDs and diagnostics
    """
    if cfg is None:
        cfg = PierBridgeConfig()
    # Variant-aware transition calibration. The rescale sigmoid's center/scale
    # must track the ACTIVE sonic variant's cosine band or the transition
    # score saturates. Resolve once here, from the variant the artifact
    # actually loaded (bundle.sonic_variant), so every downstream consumer
    # (the transition context, the edge-score wrappers, the beam) reads the
    # correct band. muq is the sole registered variant post-SP-B; unknown/
    # unregistered variants (including the retired mert/tower path) raise —
    # a configured sonic space the transition rescale can't calibrate is a
    # startup error, not a silent fallback.
    # sonic_variant is an Optional bundle field (defensive only — the real
    # ArtifactBundle always carries it).
    _cal_c, _cal_s, _cal_g = resolve_transition_calib(getattr(bundle, "sonic_variant", None))
    cfg = replace(
        cfg,
        transition_calib_center=_cal_c,
        transition_calib_scale=_cal_s,
        transition_calib_gain=_cal_g,
    )
    # Per-mode corridor width (spec section 4, pulled forward from Phase 2 by
    # Dylan's 2026-07-18 decision). Resolved ONCE here, before any corridor
    # code below runs, into a concrete float written back onto
    # cfg.corridor_width_percentile -- every corridor call site in this
    # function (the segment-pool closure, the widening ladder, the universe
    # log line) already reads that field unchanged, so this single
    # replace() is the entire seam; see resolve_corridor_width_percentile's
    # docstring for the priority order (explicit override > sonic_mode > the
    # "dynamic" fallback for unspecified/unrecognized modes; "off" -> 0.0).
    cfg = replace(
        cfg,
        corridor_width_percentile=resolve_corridor_width_percentile(
            sonic_mode,
            override=cfg.corridor_width_percentile,
            strict=cfg.corridor_width_percentile_strict,
            narrow=cfg.corridor_width_percentile_narrow,
            dynamic=cfg.corridor_width_percentile_dynamic,
            discover=cfg.corridor_width_percentile_discover,
        ),
    )
    # Genre steering supersedes the older dj_bridging waypoint system. They are two
    # overlapping genre-arc implementations (dense 64-dim arc vote vs. 893-dim waypoint
    # pooling) and were never meant to run together. When steering is on, force
    # dj_bridging off so only the steering path drives genre routing.
    if bool(cfg.genre_steering_enabled) and bool(cfg.dj_bridging_enabled):
        logger.info(
            "genre_steering_enabled=True supersedes dj_bridging_enabled; disabling dj_bridging for this run."
        )
        cfg = replace(cfg, dj_bridging_enabled=False)
    if infeasible_handling is None:
        infeasible_handling = InfeasibleHandlingConfig()
    if audit_config is None:
        audit_config = RunAuditConfig()
    audit_enabled = bool(audit_config.enabled) and audit_events is not None
    top_k = int(audit_config.include_top_k) if audit_enabled else 0

    num_seeds = len(seed_track_ids)
    if num_seeds == 0:
        raise ValueError("At least one seed is required")
    if num_seeds > total_tracks:
        raise ValueError(f"Number of seeds ({num_seeds}) exceeds total_tracks ({total_tracks})")

    # Resolve seed indices
    seed_indices: List[int] = []
    for tid in seed_track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            raise ValueError(f"Seed track not found in bundle: {tid}")
        seed_indices.append(idx)

    # Remove duplicates while preserving order
    seed_indices = list(dict.fromkeys(seed_indices))
    num_seeds = len(seed_indices)
    seed_id_set = {str(bundle.track_ids[i]) for i in seed_indices}

    logger.info("Pier+Bridge: %d seeds, target %d tracks", num_seeds, total_tracks)

    # Deduplicate candidate pool by artist+title
    deduped_pool, _ = _dedupe_candidate_pool(candidate_pool_indices, bundle)

    # Exclude seed indices from candidate pool
    seed_set = set(seed_indices)
    universe = [idx for idx in deduped_pool if idx not in seed_set]

    logger.info("Pier+Bridge: universe size after dedupe and seed exclusion: %d", len(universe))

    # Resolve on-tag guarantee ids -> indices once (mirrors allowed_set_indices
    # below). Steering-gated: only non-empty when the caller passed
    # on_tag_guarantee_ids (tag steering resolved on-tag authority rows);
    # None/empty means bridge_admission_relaxed below stays False and the
    # segment pool is byte-identical to the pre-Phase-A path. Moved here
    # (review fix, Task 5, CRITICAL): must be resolved BEFORE the corridor
    # branch's bangers popularity gate runs, so that gate can exempt
    # guarantee ids the same way it exempts seeds -- see the bangers-gate
    # comment below for why this ordering is load-bearing, not cosmetic.
    on_tag_guarantee_indices: Optional[Set[int]] = None
    if on_tag_guarantee_ids:
        on_tag_guarantee_indices = set()
        for tid in on_tag_guarantee_ids:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                on_tag_guarantee_indices.add(idx)
    _bridge_admission_relaxed = bool(
        tag_steering_relax_bridge_admission and on_tag_guarantee_indices
    )

    # ── Corridor pooling (Phase 1 Task 3, dev flag) ────────────────────────────
    # cfg.pooling == "corridor" swaps the per-segment pool builders below for a
    # library-wide eligible universe (build_eligible_universe) narrowed per
    # segment by a self-calibrating min-sim corridor (build_corridor), instead
    # of `universe` above (deduped_pool minus seeds) + the legacy KNN-union /
    # segment-scored builders. `legacy` (default) is completely untouched --
    # this block only runs when the flag is on.
    #
    # The corridor's eligible universe is intentionally WIDER than `universe`:
    # it bypasses the upstream candidate-pool's sonic/genre-mode gating
    # entirely (that gating never runs for corridor mode in Phase 1). Task 4
    # wires it back in via the mode-keyed relevance mask below
    # (`seed_genre_relevance_mask`, floor from `_corridor_genre_relevance_floor`)
    # -- see docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md
    # section 4. Seeds/piers are exempt from the mask by construction
    # (build_eligible_universe never excludes a seed index, mask or no mask).
    #
    # Phase 1 leaves one of build_eligible_universe's hard-exclusion inputs
    # deliberately inert:
    #   - `excluded_track_ids=set()`: recency/blacklist exclusion is already
    #     baked into `bundle` itself by pipeline.core's restrict_bundle() call
    #     BEFORE bundle reaches this function, so re-passing it here would be
    #     redundant, not missing.
    #
    # Task 7 C1 fix: `duration_reference_ms`/`duration_penalty_weight` and
    # `title_hard_exclude_flags` are now wired for real. cfg.duration_*
    # fields (see PierBridgeConfig) are populated by core.py's
    # generate_playlist_ds from cfg.candidate -- the SAME CandidatePoolConfig
    # fields the legacy pool reads -- via `replace(pb_cfg, ...)`, so a
    # generation reads one config source regardless of pooling mode. The
    # reference duration itself is the seed-median of positive seed
    # durations, computed by `compute_seed_reference_duration_ms` (imported
    # from candidate_pool.py, not re-derived, so the math can't drift from
    # legacy's identical computation). `cfg.duration_penalty_enabled=False`
    # (the PierBridgeConfig default, e.g. for a PierBridgeConfig built
    # directly by a test without threading through core.py) keeps
    # `duration_reference_ms=None`/`duration_penalty_weight=0.0` -- the
    # module's documented "off" state -- so the Task 3 dead-knob-trap
    # regression (tests/integration/test_corridor_pooling.py) stays green.
    # `title_hard_exclude_flags` has no separate enabled bool (mirrors
    # legacy: candidate_pool.py applies it unconditionally from
    # cfg.title_hard_exclude_flags) -- an empty frozenset (the default) is
    # itself the off-state.
    # Phase 1 Task 8: corridor pooling is THE sole path -- the "legacy"/
    # "corridor" dev flag (PierBridgeConfig.pooling) and every legacy
    # KNN-union / segment-scored pool builder it used to select between were
    # deleted. `pooling_mode` stays as a local constant (rather than being
    # excised at every call site) so the corridor-only code below reads
    # identically to how it did behind the flag -- every remaining
    # `pooling_mode == "corridor"` check is now always-true by construction.
    pooling_mode = "corridor"
    corridor_universe: Optional[EligibleUniverse] = None
    corridor_genre_dense_universe: Optional[np.ndarray] = None
    corridor_artist_keys: List[str] = []
    corridor_track_keys: List[Tuple[str, str]] = []
    # C1 beam rehome (controller decision, Task 7 follow-up): a precomputed
    # per-candidate duration-penalty array, threaded into the beam.
    corridor_duration_penalty_values: Optional[np.ndarray] = None
    if pooling_mode == "corridor":
        # Genre-mode-keyed relevance mask (Task 4). Computed once per
        # generation from the run's seed/pier indices -- seed_genre_relevance_mask
        # max-pools their genre profile, same function the pier-veto uses
        # (artist_style.py), just handed the pier set instead of one artist's
        # catalog. `_xg_for_mask` mirrors the fallback build_pier_bridge_playlist
        # applies to X_genre_smoothed later in this function (line ~700):
        # X_genre_smoothed is resolved from `bundle` here too since that later
        # fallback hasn't executed yet at this point in the function.
        _corridor_genre_floor = _corridor_genre_relevance_floor(genre_mode)
        _xg_for_mask = X_genre_smoothed if X_genre_smoothed is not None else getattr(bundle, "X_genre_smoothed", None)
        corridor_relevance_mask: Optional[np.ndarray] = None
        if _corridor_genre_floor is not None:
            corridor_relevance_mask = seed_genre_relevance_mask(
                _xg_for_mask, seed_indices, _corridor_genre_floor,
            )
            if corridor_relevance_mask is not None:
                logger.info(
                    "Corridor relevance mask: %d/%d library rows eligible "
                    "(genre_mode=%s floor=%.2f)",
                    int(corridor_relevance_mask.sum()), int(corridor_relevance_mask.size),
                    str(genre_mode), float(_corridor_genre_floor),
                )
            else:
                logger.warning(
                    "Corridor relevance mask inactive (genre_mode=%s floor=%.2f) — %s; "
                    "falling back to the ungated full universe.",
                    str(genre_mode), float(_corridor_genre_floor),
                    "X_genre_smoothed absent" if _xg_for_mask is None
                    else "seed set has no genre profile",
                )
        _corridor_duration_reference_ms: Optional[float] = None
        _corridor_duration_penalty_weight = 0.0
        if bool(cfg.duration_penalty_enabled):
            _ref = compute_seed_reference_duration_ms(
                getattr(bundle, "durations_ms", None), seed_indices,
            )
            if _ref and _ref > 0:
                _corridor_duration_reference_ms = float(_ref)
                _corridor_duration_penalty_weight = float(cfg.duration_penalty_weight)
            # else: no durations column / no seed with a known positive
            # duration -- mirrors candidate_pool.py's duration_penalty_active
            # falling back to inert when reference_duration_ms <= 0.
        corridor_universe = build_eligible_universe(
            bundle=bundle,
            seed_indices=seed_indices,
            excluded_track_ids=set(),
            relevance_mask=corridor_relevance_mask,
            duration_reference_ms=_corridor_duration_reference_ms,
            duration_cutoff_multiplier=float(cfg.duration_cutoff_multiplier),
            duration_penalty_weight=_corridor_duration_penalty_weight,
            # cfg.title_hard_exclude_flags is a tuple (JSON-safe on PierBridgeConfig,
            # see its field comment); build_eligible_universe's frozen interface
            # expects a frozenset -- convert only at this point-of-use.
            title_hard_exclude_flags=frozenset(cfg.title_hard_exclude_flags),
            instrumental_enabled=bool(cfg.instrumental_enabled),
            instrumental_penalty_weight=float(cfg.instrumental_penalty_weight),
            voice_prob=voice_prob,
        )
        logger.info(
            "Corridor duration/title hygiene: duration_penalty_enabled=%s "
            "reference_ms=%s weight=%.3f title_hard_exclude_flags=%s "
            "excluded_duration_cutoff=%d excluded_title_hygiene=%d",
            bool(cfg.duration_penalty_enabled),
            f"{_corridor_duration_reference_ms:.0f}" if _corridor_duration_reference_ms else "None",
            _corridor_duration_penalty_weight,
            sorted(cfg.title_hard_exclude_flags),
            int(corridor_universe.stats.get("excluded_duration_cutoff", 0)),
            int(corridor_universe.stats.get("excluded_title_hygiene", 0)),
        )
        # C1 beam rehome: precompute the per-candidate duration penalty over
        # the corridor-eligible indices (a superset of every candidate the
        # beam will see this generation) using the SAME reference/weight just
        # resolved above and the SAME compute_duration_penalty math legacy's
        # pool uses. Stays None (beam term off) when the penalty is disabled
        # or there is no usable seed reference -- mirrors the universe build's
        # own duration_active gate just above.
        if _corridor_duration_reference_ms is not None and _corridor_duration_penalty_weight > 0.0:
            _durations_ms_full = getattr(bundle, "durations_ms", None)
            if _durations_ms_full is not None:
                corridor_duration_penalty_values = np.zeros(len(bundle.track_ids), dtype=np.float64)
                for _idx in corridor_universe.indices:
                    _dur = float(_durations_ms_full[int(_idx)])
                    if _dur > 0.0 and _dur > _corridor_duration_reference_ms:
                        corridor_duration_penalty_values[int(_idx)] = compute_duration_penalty(
                            _dur, _corridor_duration_reference_ms, _corridor_duration_penalty_weight
                        )
        if corridor_duration_penalty_values is not None:
            logger.info(
                "Corridor beam duration penalty: active (reference_ms=%.0f weight=%.2f)",
                float(_corridor_duration_reference_ms),
                float(_corridor_duration_penalty_weight),
            )
        else:
            logger.info(
                "Corridor beam duration penalty: inactive (duration_penalty_enabled=%s)",
                bool(cfg.duration_penalty_enabled),
            )
        # Oops, All Bangers (Task 5 reseat, design spec §3: "popularity filter
        # applies to the corridor universe when enabled"). Applied ONCE here,
        # at universe build -- NOT inside eligible_universe.py (its reviewed
        # interface stays frozen this task) -- via the SAME rank-cutoff gate
        # candidate_pool.py's legacy pool uses (_apply_popularity_gate,
        # imported above, not copied). core.py's `_run_pier_bridge` closure
        # plumbs the same (popularity_ranks, popularity_rank_cutoff) pair it
        # already resolves once via `_banger_gate_inputs` for the legacy pool
        # (aligned to the FULL bundle, so it applies unchanged here). No
        # relax-to-fill cascade: unlike the legacy path's `_banger_relaxation_
        # steps` cascade (core.py), the corridor path's only relaxation
        # mechanism is the widening ladder (Task 4) -- core.py's relax-to-fill
        # loop only rebuilds the LEGACY `pool`/`universe` variable, which the
        # corridor branch never reads, so it is inert here by construction,
        # not by an explicit skip. Seeds are exempt (unioned back in), mirroring
        # every other exclusion in this universe (seeds are never dropped from
        # their own generation for failing a popularity check).
        #
        # CRITICAL review fix: on-tag guarantee ids MUST be exempted here too,
        # the same way seeds are. Legacy parity: candidate_pool.py's
        # select_pool_guarantee (:1348-1369) resolves its guarantee universe
        # bypassing _apply_popularity_gate entirely -- the tag guarantee
        # OVERRIDES bangers, not the other way around. Without this exemption,
        # a below-cutoff on-tag track would be dropped from corridor_universe
        # entirely, and _build_corridor_segment_pool's force_include lookup
        # (scoped to avail_idx, itself derived from corridor_universe.indices)
        # can only skip an id that isn't there -- it has no way to know the id
        # was supposed to be force-included. This exemption, run BEFORE that
        # lookup ever executes, is the only correct place to fix this (hence
        # on_tag_guarantee_indices was moved above to be resolved before this
        # branch runs).
        if popularity_rank_cutoff is not None and popularity_ranks is not None:
            _pre_gate_idx = corridor_universe.indices.tolist()
            _kept_list, _pop_excluded_raw = _apply_popularity_gate(
                _pre_gate_idx, np.asarray(popularity_ranks), int(popularity_rank_cutoff)
            )
            _tag_guarantee_exempt = set(int(i) for i in (on_tag_guarantee_indices or ()))
            _kept_set = set(int(i) for i in _kept_list) | seed_set | _tag_guarantee_exempt
            _keep_mask = np.fromiter(
                (int(i) in _kept_set for i in _pre_gate_idx),
                dtype=bool, count=len(_pre_gate_idx),
            )
            _pop_excluded_net = int(len(_pre_gate_idx) - int(_keep_mask.sum()))
            corridor_universe = replace(
                corridor_universe,
                indices=corridor_universe.indices[_keep_mask],
                X_norm=corridor_universe.X_norm[_keep_mask],
                duration_rank_penalty=corridor_universe.duration_rank_penalty[_keep_mask],
                stats={
                    **corridor_universe.stats,
                    "excluded_popularity_gate": _pop_excluded_net,
                    "eligible_count": int(_keep_mask.sum()),
                },
            )
            # excluded_raw = the gate's own verdict (rank outside cutoff),
            # before seed/tag-guarantee exemption is unioned back in;
            # excluded_net = before-after, i.e. what actually left the
            # universe. exempted = the reconciling delta between them (Minor
            # 2 review fix: these two numbers previously diverged whenever a
            # seed or (as of this fix) a guarantee id had an out-of-cutoff
            # rank, with no way to tell from the log alone).
            logger.info(
                "Corridor bangers gate: cutoff=top-%d before=%d after=%d "
                "excluded_raw=%d excluded_net=%d exempted=%d",
                int(popularity_rank_cutoff), len(_pre_gate_idx),
                int(corridor_universe.indices.size), int(_pop_excluded_raw),
                _pop_excluded_net, int(_pop_excluded_raw - _pop_excluded_net),
            )
        _genre_dense_full_for_corridor = getattr(bundle, "X_genre_dense", None)
        if _genre_dense_full_for_corridor is not None:
            corridor_genre_dense_universe = _genre_dense_full_for_corridor[corridor_universe.indices]
        logger.info(
            "Corridor universe: eligible=%d/%d (excluded_total=%d width_percentile=%.2f "
            "relevance_excluded=%d popularity_excluded=%d)",
            int(corridor_universe.stats.get("eligible_count", 0)),
            int(corridor_universe.stats.get("universe_size_total", 0)),
            int(corridor_universe.stats.get("excluded_total", 0)),
            float(cfg.corridor_width_percentile),
            int(corridor_universe.stats.get("excluded_relevance_mask", 0)),
            int(corridor_universe.stats.get("excluded_popularity_gate", 0)),
        )
        # Identity-key arrays, precomputed ONCE PER GENERATION, aligned to
        # corridor_universe.indices -- review fix (2026-07-17): legacy
        # (segment_pool_builder.SegmentCandidatePoolBuilder.build) runs ALL
        # structural filters, including the identity-key ones
        # (disallow_seed_artist_in_interiors, disallow_pier_artists_in_interiors,
        # track-key collision), on the full universe BEFORE scoring/capping.
        # The original Task 3 pass ran these post-cap (on build_corridor's
        # already-capped result) purely to avoid the per-segment cost of
        # identity_keys_for_index -- but that reordering is a real correctness
        # bug, not just a perf tradeoff: a segment whose corridor top-K is
        # dominated by used_track_keys collisions could starve even though
        # clean, lower-ranked candidates exist just below the cap (legacy
        # structurally can't hit this, since it filters first). Computing
        # these arrays ONCE here (not per segment) gets both: filter-before-cap
        # correctness AND no repeated identity_keys_for_index cost.
        corridor_artist_keys = []
        corridor_track_keys = []
        for _idx in corridor_universe.indices.tolist():
            try:
                _keys = identity_keys_for_index(bundle, int(_idx))
                corridor_artist_keys.append(_keys.artist_key)
                corridor_track_keys.append(_keys.track_key)
            except Exception:
                corridor_artist_keys.append("")
                corridor_track_keys.append(("", ""))

    # Get sonic matrices (raw beat3tower space)
    X_full_raw = bundle.X_sonic
    X_start_raw = bundle.X_sonic_start
    X_mid_raw = bundle.X_sonic_mid
    X_end_raw = bundle.X_sonic_end

    # Similarity space for bridge gating (full vectors): plain L2-normalized
    # cosine on the loaded sonic matrix (muq) — matches DS admission.
    X_full_norm = _l2_normalize_rows(X_full_raw)
    logger.debug("Pier+Bridge sonic sim space: dim=%d", int(X_full_norm.shape[1]))

    # Pace gating is BPM/onset-band based (the rhythm tower axis was removed
    # in SP-B; under muq it had already fallen back to BPM permanently).
    rhythm_matrix: Optional[np.ndarray] = None
    if float(getattr(cfg, "pace_bridge_floor", 0.0)) > 0.0:
        if perceptual_bpm is not None:
            from src.playlist.pier_bridge.pace_gate import bpm_fallback_max_log_distance

            _bpm_cap = float(getattr(cfg, "bpm_bridge_max_log_distance", float("inf")))
            if not np.isfinite(_bpm_cap):
                _bpm_cap = bpm_fallback_max_log_distance(float(cfg.pace_bridge_floor))
                cfg = replace(cfg, bpm_bridge_max_log_distance=_bpm_cap)
            logger.info(
                "Pace bridge gate: perceptual-BPM band (bpm_bridge_max_log_distance=%.2f)",
                float(cfg.bpm_bridge_max_log_distance),
            )
        else:
            logger.warning(
                "Pace bridge gate DISABLED: pace_bridge_floor=%.2f is set but no "
                "perceptual-BPM data is available (a configured knob that can't act).",
                float(cfg.pace_bridge_floor),
            )

    # Transition space: the raw sonic matrices (optional mean-centering below).
    X_full_tr = X_full_raw
    X_start_tr = X_start_raw
    X_mid_tr = X_mid_raw
    X_end_tr = X_end_raw

    if cfg.center_transitions:
        X_full_tr = X_full_tr - X_full_tr.mean(axis=0, keepdims=True)
        if X_start_tr is not None:
            X_start_tr = X_start_tr - X_start_tr.mean(axis=0, keepdims=True)
        if X_mid_tr is not None:
            X_mid_tr = X_mid_tr - X_mid_tr.mean(axis=0, keepdims=True)
        if X_end_tr is not None:
            X_end_tr = X_end_tr - X_end_tr.mean(axis=0, keepdims=True)

    X_full_tr_norm = _l2_normalize_rows(X_full_tr)
    X_start_tr_norm = _l2_normalize_rows(X_start_tr) if X_start_tr is not None else None
    X_mid_tr_norm = _l2_normalize_rows(X_mid_tr) if X_mid_tr is not None else None
    X_end_tr_norm = _l2_normalize_rows(X_end_tr) if X_end_tr is not None else None

    # Instrument transition saturation (sampled); compare raw vs transformed end→start
    if logger.isEnabledFor(logging.DEBUG) and X_end_raw is not None and X_start_raw is not None:
        rng = np.random.default_rng(0)
        n = int(X_full_raw.shape[0])
        sample_n = int(min(5000, n))
        prev = rng.integers(0, n, size=sample_n)
        cand = rng.integers(0, n, size=sample_n)
        end_raw = X_end_raw[prev]
        start_raw = X_start_raw[cand]
        raw_sims = np.sum(end_raw * start_raw, axis=1) / (
            (np.linalg.norm(end_raw, axis=1) * np.linalg.norm(start_raw, axis=1)) + 1e-12
        )
        end_tr = X_end_tr_norm[prev] if X_end_tr_norm is not None else None
        start_tr = X_start_tr_norm[cand] if X_start_tr_norm is not None else None
        if end_tr is not None and start_tr is not None:
            tr_sims = np.sum(end_tr * start_tr, axis=1)
            if cfg.center_transitions:
                tr_sims = (tr_sims + 1.0) / 2.0
            logger.debug(
                "Transition end→start sample: raw[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] "
                "transformed[min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f] center_transitions=%s",
                float(np.min(raw_sims)),
                float(np.percentile(raw_sims, 5)),
                float(np.percentile(raw_sims, 50)),
                float(np.percentile(raw_sims, 95)),
                float(np.max(raw_sims)),
                float(np.min(tr_sims)),
                float(np.percentile(tr_sims, 5)),
                float(np.percentile(tr_sims, 50)),
                float(np.percentile(tr_sims, 95)),
                float(np.max(tr_sims)),
                bool(cfg.center_transitions),
            )

    # For seed ordering bridgeability heuristic, prefer transition-normalized mats when present
    X_start_norm = X_start_tr_norm
    X_end_norm = X_end_tr_norm

    # Extract genre matrices from bundle (Phase 2: needed for IDF and vector mode)
    # Prefer parameter if provided, otherwise extract from bundle
    X_genre_raw = getattr(bundle, "X_genre_raw", None)
    X_genre_dense = getattr(bundle, "X_genre_dense", None)
    if X_genre_smoothed is None:
        X_genre_smoothed = getattr(bundle, "X_genre_smoothed", None)

    # No silent no-op: a configured steering source that cannot act must fail loudly.
    _require_usable_genre_steering(cfg, X_genre_dense)

    # Per-genre track counts for taxonomy waypoint mass filter (routing fix)
    _genre_vocab_list = list(np.asarray(bundle.genre_vocab, dtype=object)) if getattr(bundle, "genre_vocab", None) is not None else []
    genre_track_counts: Optional[dict[str, int]] = None
    if X_genre_raw is not None and _genre_vocab_list:
        genre_track_counts = {
            str(_genre_vocab_list[j]): int((X_genre_raw[:, j] > 0).sum())
            for j in range(len(_genre_vocab_list))
        }

    # Pairwise genre-edge floor: under taxonomy steering the gate MUST score at
    # tag level (calibration 2026-06-10: the smoothed-vector cosine cannot
    # separate bad edges from good — Sharp Pins->Springsteen 0.693 vs the
    # praised YYY->StVincent 0.677). Without the provider the beam falls back
    # to that cosine, so its absence here is a loud misconfiguration.
    pair_sim_provider = None
    if (
        bool(cfg.genre_steering_enabled)
        and str(getattr(cfg, "genre_steering_source", "taxonomy")) == "taxonomy"
        and float(getattr(cfg, "genre_pair_floor", 0.0)) > 0.0
    ):
        if X_genre_raw is not None and _genre_vocab_list:
            from src.playlist.pier_bridge.taxonomy_steering import (
                build_taxonomy_pair_provider,
                get_taxonomy_steering,
            )
            pair_sim_provider = build_taxonomy_pair_provider(
                get_taxonomy_steering(), X_genre_raw, bundle.genre_vocab
            )
            logger.info(
                "Pairwise genre-edge floor active: floor=%.2f (tag-level taxonomy provider)",
                float(cfg.genre_pair_floor),
            )
        else:
            logger.warning(
                "Pairwise genre-edge floor DEGRADED: genre_pair_floor=%.2f is set but "
                "X_genre_raw/genre_vocab unavailable — the beam will gate on the "
                "smoothed-vector cosine, which does NOT separate bad edges from good",
                float(cfg.genre_pair_floor),
            )

    # Genre similarity for soft edge penalty / tiebreak (cosine on smoothed genre vectors)
    X_genre_use = X_genre_smoothed if X_genre_smoothed is not None else None
    X_genre_norm = None
    if X_genre_use is not None:
        denom_g = np.linalg.norm(X_genre_use, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre_use / denom_g

    transition_metric_context = TransitionMetricContext(
        X_full=X_full_tr_norm,
        X_start=X_start_tr_norm,
        X_mid=X_mid_tr_norm,
        X_end=X_end_tr_norm,
        X_sonic_norm=X_full_norm,
        X_genre_norm=X_genre_norm,
        center_transitions=bool(cfg.center_transitions),
        weight_end_start=float(cfg.weight_end_start),
        weight_mid_mid=float(cfg.weight_mid_mid),
        weight_full_full=float(cfg.weight_full_full),
        calib_center=float(cfg.transition_calib_center),
        calib_scale=float(cfg.transition_calib_scale),
        calib_gain=float(cfg.transition_calib_gain),
    )

    # Compute IDF for genre vector mode (Phase 2)
    genre_idf: Optional[np.ndarray] = None
    X_genre_norm_idf: Optional[np.ndarray] = None
    if bool(cfg.dj_genre_use_idf) and bool(cfg.dj_bridging_enabled):
        if X_genre_raw is not None:
            logger.info("Computing genre IDF (power=%.2f norm=%s)...",
                        cfg.dj_genre_idf_power, cfg.dj_genre_idf_norm)
            genre_idf = _compute_genre_idf(X_genre_raw, cfg)
            logger.info("  IDF computed: min=%.3f median=%.3f max=%.3f",
                        float(np.min(genre_idf)), float(np.median(genre_idf)), float(np.max(genre_idf)))

            # Create IDF-weighted matrix for S3 pooling and beam search
            if X_genre_norm is not None:
                X_genre_norm_idf = _apply_idf_weighting(X_genre_norm, genre_idf)
        else:
            logger.warning("IDF enabled but X_genre_raw unavailable; using base genre weights")

    warnings: list[dict[str, Any]] = []
    if bool(cfg.dj_bridging_enabled):
        if X_genre_norm is None:
            warnings.append({
                "type": "genre_missing",
                "scope": "global",
                "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                "anchors_missing": int(num_seeds),
            })
        else:
            seed_vecs = X_genre_norm[seed_indices]
            norms = np.linalg.norm(seed_vecs, axis=1)
            missing_ids = [
                str(bundle.track_ids[idx])
                for idx, nval in zip(seed_indices, norms)
                if float(nval) <= 1e-8
            ]
            if missing_ids:
                warnings.append({
                    "type": "genre_missing",
                    "scope": "anchors",
                    "message": "Genre guidance reduced because metadata is missing; consider adding genres.",
                    "missing_anchor_ids": missing_ids,
                })
        if bool(cfg.dj_anchors_must_include_all) and len(seed_indices) != len(seed_track_ids):
            warnings.append({
                "type": "anchor_deduped",
                "scope": "anchors",
                "message": "Duplicate anchors were removed while must_include_all is set.",
                "requested_count": int(len(seed_track_ids)),
                "resolved_count": int(len(seed_indices)),
            })

    genre_graph: Optional[dict[str, list[tuple[str, float]]]] = None
    genre_vocab: Optional[np.ndarray] = None
    if bool(cfg.dj_bridging_enabled):
        route_shape = str(cfg.dj_route_shape or "linear").strip().lower()
        if route_shape == "ladder":
            genre_vocab = getattr(bundle, "genre_vocab", None)
            if genre_vocab is None:
                warnings.append({
                    "type": "genre_ladder_unavailable",
                    "scope": "global",
                    "message": "Genre ladder disabled; missing genre vocab.",
                })
            else:
                repo_root = Path(__file__).resolve().parents[2]
                genre_yaml = repo_root / "data" / "genre_similarity.yaml"
                if bool(cfg.dj_ladder_use_smoothed_waypoint_vectors):
                    _ensure_genre_similarity_overrides_loaded(genre_yaml)
                genre_graph = _load_genre_similarity_graph(
                    genre_yaml,
                    min_similarity=float(cfg.dj_ladder_min_similarity),
                )
                if not genre_graph:
                    warnings.append({
                        "type": "genre_ladder_unavailable",
                        "scope": "global",
                        "message": "Genre ladder disabled; similarity graph unavailable.",
                    })

    # Genre-arc steering: build the niche genre graph ONCE per run from the dense
    # genre embedding (distinct from the hand-curated YAML graph above). Used to
    # route dense per-segment g_targets when steering is enabled.
    genre_graph_arc: Optional[dict[str, list[tuple[str, float]]]] = None
    if bool(cfg.genre_steering_enabled) and getattr(bundle, "genre_emb", None) is not None and getattr(bundle, "genre_vocab", None) is not None:
        from src.playlist.pier_bridge.genre_graph import build_genre_graph
        genre_graph_arc = build_genre_graph(
            bundle.genre_emb, bundle.genre_vocab,
            k=int(getattr(cfg, "dj_ladder_top_labels", 8) or 8),
            min_cos=float(getattr(cfg, "dj_ladder_min_similarity", 0.35) or 0.35),
            hub_labels={"rock", "indie", "alternative", "pop", "indie rock", "electronic"},
        )

    # Precompute allowed indices set if caller passed allowed_track_ids_set.
    # (In style-aware runs, the bundle is often already restricted, but this still
    # acts as a hard gate for candidate admission inside pier-bridge.)
    allowed_set_indices: Optional[Set[int]] = None
    if allowed_track_ids_set is not None:
        allowed_set_indices = set()
        for tid in allowed_track_ids_set:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                allowed_set_indices.add(idx)
        # Ensure piers are always allowed
        allowed_set_indices.update(seed_indices)

    # Order seeds by bridgeability (or preserve order if fixed)
    seed_ordering = str(cfg.dj_seed_ordering or "auto").strip().lower()
    if seed_ordering not in {"auto", "fixed"}:
        warnings.append({
            "type": "seed_ordering_invalid",
            "scope": "anchors",
            "message": f"Unknown seed_ordering '{seed_ordering}', defaulting to auto.",
        })
        seed_ordering = "auto"
    # Tag-steering anchors are ordered SEPARATELY (spec 2026-07-27): the artist
    # piers alone define the sequence and its terminals, then anchors are inserted
    # into the interior gaps. Appending them and re-ordering everything as
    # co-equals is what let three anchors cluster and one take the closing seat
    # (Sonic Youth, 2026-07-27). Empty anchor set => this whole block is inert and
    # ordering is byte-identical to the legacy path.
    _anchor_indices: List[int] = []
    _artist_pier_indices: List[int] = list(seed_indices)
    if tag_anchor_gap_insertion and tag_anchor_track_ids:
        _anchor_row_set = {
            idx
            for idx in (
                bundle.track_id_to_index.get(str(tid)) for tid in tag_anchor_track_ids
            )
            if idx is not None
        }
        # Selection rank == order of appearance among the piers (playlist_generator
        # appends anchors in rank order and every hop preserves it).
        _anchor_indices = [i for i in seed_indices if i in _anchor_row_set]
        _artist_pier_indices = [i for i in seed_indices if i not in _anchor_row_set]
        if not _anchor_indices:
            logger.warning(
                "Tag steering anchor placement: %d anchor id(s) supplied but NONE "
                "matched a pier -- gap insertion cannot act; falling back to legacy "
                "co-equal ordering.",
                len(tag_anchor_track_ids),
            )
        elif len(_artist_pier_indices) < 2:
            logger.warning(
                "Tag steering anchor placement: only %d artist pier(s) -- no interior "
                "gap exists; falling back to legacy co-equal ordering for %d anchor(s).",
                len(_artist_pier_indices), len(_anchor_indices),
            )
            _anchor_indices = []
            _artist_pier_indices = list(seed_indices)

    _to_order = _artist_pier_indices

    if bool(cfg.dj_bridging_enabled) and seed_ordering == "fixed":
        ordered_seeds = list(_to_order)
    else:
        if bool(cfg.dj_bridging_enabled):
            ordered_seeds = _order_seeds_by_bridgeability(
                _to_order,
                X_full_norm,
                X_start_norm,
                X_end_norm,
                X_genre_norm,
                weight_sonic=float(cfg.dj_seed_ordering_weight_sonic),
                weight_genre=float(cfg.dj_seed_ordering_weight_genre),
                weight_bridge=float(cfg.dj_seed_ordering_weight_bridge),
                min_bottleneck=bool(cfg.roam_corridors_enabled),
            )
        else:
            ordered_seeds = _order_seeds_by_bridgeability(
                _to_order, X_full_norm, X_start_norm, X_end_norm,
                min_bottleneck=bool(cfg.roam_corridors_enabled),
            )

    if _anchor_indices:
        from src.playlist.pier_bridge.anchor_placement import place_anchors_in_gaps

        def _anchor_pair_score(a: int, b: int) -> float:
            return _compute_bridgeability_score(
                int(a), int(b), X_full_norm, X_start_norm, X_end_norm
            )

        _placement = place_anchors_in_gaps(
            ordered_seeds,
            _anchor_indices,
            _anchor_pair_score,
            min_bridge=float(tag_anchor_min_bridge),
        )
        ordered_seeds = list(_placement.sequence)
        logger.info(
            "Tag steering anchor placement: %d/%d anchor(s) placed into gaps %s "
            "(artist piers=%d)",
            len(_placement.placed), len(_anchor_indices),
            [g for _a, g in _placement.placed], len(_artist_pier_indices),
        )
        if _placement.dropped_clamped:
            logger.info(
                "Tag steering anchor placement: dropped %d anchor(s) -- only %d "
                "interior gap(s) for %d anchor(s) (K <= P-1): %s",
                len(_placement.dropped_clamped), len(_artist_pier_indices) - 1,
                len(_anchor_indices),
                [str(bundle.track_ids[i]) for i in _placement.dropped_clamped],
            )
        if _placement.dropped_unbridgeable:
            logger.info(
                "Tag steering anchor placement: dropped %d anchor(s) with no gap "
                "clearing min_bridge=%.2f: %s",
                len(_placement.dropped_unbridgeable), float(tag_anchor_min_bridge),
                [str(bundle.track_ids[i]) for i in _placement.dropped_unbridgeable],
            )
        if _placement.dropped_displaced:
            logger.info(
                "Tag steering anchor placement: dropped %d anchor(s) that cleared "
                "min_bridge=%.2f but lost their gap to a higher-scoring anchor: %s",
                len(_placement.dropped_displaced), float(tag_anchor_min_bridge),
                [str(bundle.track_ids[i]) for i in _placement.dropped_displaced],
            )

    logger.info("Pier+Bridge: seed order = %s",
               [str(bundle.track_ids[i]) for i in ordered_seeds])

    # Handle single seed as both start AND end pier (arc structure)
    # This creates a playlist that starts from seed, explores, and returns to seed-similar sounds
    is_single_seed_arc = (num_seeds == 1)
    if is_single_seed_arc:
        # Duplicate the seed as both start and end pier
        ordered_seeds = [ordered_seeds[0], ordered_seeds[0]]
        num_segments = 1
        total_interior = total_tracks - 1  # Only one seed in final output
        # Single unique seed pier, duplicated at both ends -- exactly one seed
        # reaches the final playlist. Anchor gap insertion cannot engage here
        # (its fallback requires >=2 artist piers), so this count is never
        # affected by a dropped anchor.
        _expected_seed_pier_count = 1
        logger.info("Pier+Bridge: single-seed arc mode (seed is both start and end pier)")
    else:
        _piers_before_mini = len(ordered_seeds)
        # The number of seed piers (artist piers + any anchors gap-insertion
        # actually placed) that should land in the final playlist. Captured
        # BEFORE mini-pier waypoint insertion, since those are non-seed
        # interior tracks. Review fix (spec 2026-07-27): a dropped anchor
        # previously inflated the downstream "expected" count via the
        # untouched pre-drop seed_indices/seed_id_set, firing a false "seed
        # count mismatch" warning and overcounting stats["num_seeds"].
        _expected_seed_pier_count = _piers_before_mini
        if bool(getattr(cfg, "mini_pier_enabled", False)):
            from src.playlist.pier_bridge.mini_pier_select import plan_pier_sequence
            # exclude seed/pier-artist tracks (waypoints are real piers; keep them
            # off the seed artists) using normalized track_artists.
            if bundle.track_artists is not None:
                _artists = np.array([" ".join(str(x).split()).lower()
                                      for x in bundle.track_artists])
                _pier_artists = {_artists[i] for i in ordered_seeds}
                exclude_base = frozenset(int(i) for i in np.where(
                    np.isin(_artists, list(_pier_artists)))[0])
            else:
                # Graceful degradation: no artist data — exclude only the pier tracks
                # themselves so waypoints don't land on seeds.
                exclude_base = frozenset(int(i) for i in ordered_seeds)
            ordered_seeds = plan_pier_sequence(
                ordered_seeds, total_tracks, candidate_pool_indices, X_full_norm,
                max_interior=int(cfg.mini_pier_max_interior),
                margin=float(cfg.mini_pier_smoothness_margin),
                k_broad=150, exclude_base=exclude_base,
                # SAFETY BACKSTOP: roughly one waypoint per 4 tracks.
                # Primary terminators are "interior ≤ K" and "no feasible waypoint";
                # this cap rarely binds in practice.
                max_waypoints=max(0, total_tracks // 4),
                balance_gaps=bool(getattr(cfg, "mini_pier_balance_gaps", False)),
            )
            logger.info("Mini-piers: %d waypoint(s) inserted (piers now %d)",
                        len(ordered_seeds) - _piers_before_mini, len(ordered_seeds))
        num_segments = len(ordered_seeds) - 1
        total_interior = total_tracks - len(ordered_seeds)

    # Even split with remainder distributed to earlier segments
    base_length = total_interior // num_segments
    remainder = total_interior % num_segments
    segment_lengths = [
        base_length + (1 if i < remainder else 0)
        for i in range(num_segments)
    ]

    logger.info("Pier+Bridge: segment lengths = %s (total_interior=%d)",
               segment_lengths, total_interior)

    # Build segments
    global_used: Set[int] = set(ordered_seeds)  # Seeds are already "used"
    # Track-key dedupe across the full run: prevent "same song twice" even if track_id differs.
    seed_artist_key: Optional[str] = None
    try:
        if seed_indices:
            seed_artist_key = identity_keys_for_index(bundle, int(seed_indices[0])).artist_key
    except Exception:
        seed_artist_key = None

    seed_track_keys: Set[tuple[str, str]] = set()
    for sidx in set(int(i) for i in seed_indices):
        try:
            seed_track_keys.add(identity_keys_for_index(bundle, int(sidx)).track_key)
        except Exception:
            continue
    used_track_keys: Set[tuple[str, str]] = set(seed_track_keys)

    # ── Corridor pooling (Phase 1 Task 3) per-run accumulators ──────────────
    corridor_segments_diag: List[Dict[str, Any]] = []
    corridor_logged_segments: Set[int] = set()
    # Edge-repair reseat (Task 5): accumulates every segment's FINAL corridor
    # membership (post-widening-ladder, i.e. the widened corridor if widening
    # fired -- `last_segment_candidates` at each segment's acceptance point,
    # see the main segment loop below) across the whole generation.
    # repair_playlist_edges takes ONE candidate_indices for the entire repair
    # pass (confirmed: src/playlist/repair/edge_repair.py has no per-edge
    # scoping), so per the design spec §3 ("Repair stack ... candidate
    # universe becomes the segment's corridor, including widened material")
    # this union is the sanctioned corridor-mode substitute for the legacy
    # `universe` argument -- a repaired-in candidate is guaranteed to be a
    # member of AT LEAST ONE segment's corridor, not necessarily the specific
    # edge's own segment. Empty/unused on the legacy path.
    corridor_segment_members_union: Set[int] = set()

    def _build_corridor_segment_pool(
        pier_a_idx: int,
        pier_b_idx: int,
        seg_idx_local: int,
        segment_pool_max_local: int,
        width_percentile_override: Optional[float] = None,
    ) -> Tuple[List[int], Dict[int, str], Dict[int, str], Dict[str, Any]]:
        """Corridor-mode segment pool (Phase 1 dev flag).

        Mirrors the legacy builders' output shape exactly (candidates,
        artist_key_by_idx, title_key_by_idx) so the call site + everything
        downstream (beam search) is unchanged.

        Structural filtering order mirrors legacy exactly (review fix,
        2026-07-17): segment_pool_builder.SegmentCandidatePoolBuilder.build
        runs ALL structural filters -- used_track_ids, allowed_set,
        disallow_seed_artist_in_interiors, disallow_pier_artists_in_interiors,
        track-key collision -- on the FULL universe BEFORE scoring/capping.
        This closure now does the same: every exclusion (index-set AND
        identity-key-based) folds into ONE boolean mask applied to the
        eligible universe BEFORE build_corridor runs, so its internal
        segment_pool_max cap only ever truncates the ALREADY-CLEAN candidate
        set -- exactly like legacy. A prior version of this closure ran the
        identity-key filters (artist/track-key) AFTER build_corridor's cap to
        avoid recomputing identity_keys_for_index per segment; that was a
        real correctness bug, not just a perf tradeoff -- a segment whose
        corridor top-K was dominated by used_track_keys collisions could
        starve even though clean, lower-ranked candidates existed just below
        the cap (legacy structurally can't hit this). The fix: identity keys
        are precomputed ONCE PER GENERATION (`corridor_artist_keys` /
        `corridor_track_keys`, aligned to `corridor_universe.indices`, built
        right after `corridor_universe` itself), so this closure's per-segment
        cost is array/list indexing + set membership checks, never a second
        identity_keys_for_index pass over the universe.

        NOT replicated (see .superpowers/sdd/p1-task-3-report.md "concerns for
        Task 4/5"): internal connectors, the disallow_seed_artist_in_interiors
        relaxation retry, and dj_union debug-compare-baseline -- legacy-only
        reseat/DJ-bridging features out of this task's scope. On-tag
        guarantees WERE reseated as of Task 5 (see the force_include wiring
        below) -- this docstring previously named them as not-replicated;
        that was true only through Task 4.

        ``width_percentile_override`` (Task 4): when given, overrides
        ``cfg.corridor_width_percentile`` for this one call -- the widening
        ladder's mechanism for retrying a segment at a wider (lower-
        percentile) corridor without mutating ``cfg`` itself. None (default)
        preserves the original single-width Task 3 behavior byte-for-byte.

        Health-line logging / ``corridor_segments_diag`` are NOT emitted here
        (Task 3 did this inline, gated to fire once per segment). Task 4
        moved that to the widening-ladder wrapper (``_run_corridor_widening_
        ladder``); Task 9 moved it once more, out of that wrapper and into
        the main segment loop (after the interior length is chosen), because
        the wrapper itself can be invoked more than once per seg_idx under
        variable_bridge_length (once per candidate length) -- see the
        wrapper's own docstring/comments and the "Health line (Task 9 fix)"
        comment in the segment loop. The ONE health line + ONE diagnostics
        entry per segment now reflect the ACCEPTED length's ACCEPTED
        (widening-ladder) attempt's stats -- never a provisional, un-chosen
        attempt at a different length or width. This closure still always
        returns its per-call stats as the 4th tuple element.
        """
        assert corridor_universe is not None
        idx_arr = corridor_universe.indices

        pier_a_ak: Optional[str] = None
        pier_b_ak: Optional[str] = None
        try:
            pier_a_ak = identity_keys_for_index(bundle, int(pier_a_idx)).artist_key
            pier_b_ak = identity_keys_for_index(bundle, int(pier_b_idx)).artist_key
        except Exception:
            pass
        _pier_artist_block = (
            {pier_a_ak, pier_b_ak} if cfg.disallow_pier_artists_in_interiors else None
        )
        _disallow_seed = bool(cfg.disallow_seed_artist_in_interiors) and bool(seed_artist_key)
        _allowed = allowed_set_indices

        def _row_ok(i: int, ak: str, tk: Tuple[str, str]) -> bool:
            if i in global_used:
                return False
            if _allowed is not None and i not in _allowed:
                return False
            if _disallow_seed and ak == seed_artist_key:
                return False
            if _pier_artist_block is not None and ak in _pier_artist_block:
                return False
            if tk in used_track_keys:
                return False
            return True

        mask = np.fromiter(
            (
                _row_ok(int(i), ak, tk)
                for i, ak, tk in zip(idx_arr.tolist(), corridor_artist_keys, corridor_track_keys)
            ),
            dtype=bool,
            count=len(idx_arr),
        )

        avail_idx = idx_arr[mask]
        avail_X = corridor_universe.X_norm[mask]
        avail_artist_keys = [ak for ak, keep in zip(corridor_artist_keys, mask.tolist()) if keep]
        avail_genre = (
            corridor_genre_dense_universe[mask]
            if corridor_genre_dense_universe is not None
            else None
        )

        genre_blend_weight = max(0.0, min(1.0, float(getattr(cfg, "segment_pool_genre_weight", 0.0))))
        genre_vec_a = genre_vec_b = None
        X_genre_dense_bundle = getattr(bundle, "X_genre_dense", None)
        if genre_blend_weight > 0.0 and X_genre_dense_bundle is not None:
            genre_vec_a = X_genre_dense_bundle[int(pier_a_idx)]
            genre_vec_b = X_genre_dense_bundle[int(pier_b_idx)]

        _width_percentile = (
            float(width_percentile_override)
            if width_percentile_override is not None
            else float(cfg.corridor_width_percentile)
        )
        # Tag-steering pool guarantee (Task 5 reseat, design spec §3:
        # "force-include on-tag tracks into relevant corridors"). Pass EVERY
        # on-tag guarantee index to EVERY segment's build_corridor call --
        # build_corridor's own force_include lookup is scoped to
        # `universe_indices=avail_idx` (this segment's structurally-eligible,
        # not-yet-used candidates, i.e. the `_row_ok` mask above already
        # applied), so an id not eligible HERE (used elsewhere, blocked by
        # disallow_seed/pier_artist_in_interiors, or a track-key collision)
        # is silently skipped -- that scoping IS the "relevant corridors"
        # semantics, no extra per-segment filtering needed. Unlike the legacy
        # dj_union pool's on_tag_guarantee_max / on_tag_guarantee_per_artist
        # caps, the corridor path applies no separate cap on the guarantee
        # itself (corridor.py's forced-first semantics, reviewed in Task 1,
        # never truncate the forced set -- only the ranked/non-forced portion
        # is capped by segment_pool_max). This also resolves the legacy
        # segment-relaxed-max vs beam-strict-min bridge-floor contradiction
        # (#32/#36, design spec §2) for the corridor path: there is no
        # relaxed-max special case here at all -- beam strict-min
        # (transition_floor) is the only floor check corridor segments ever
        # go through (the widening ladder is the sole recovery mechanism).
        _force_include_arr = (
            np.array(sorted(int(i) for i in on_tag_guarantee_indices), dtype=np.int64)
            if on_tag_guarantee_indices else None
        )
        result = build_corridor(
            vec_a=X_full_norm[int(pier_a_idx)],
            vec_b=X_full_norm[int(pier_b_idx)],
            X_norm=avail_X,
            universe_indices=avail_idx,
            width_percentile=_width_percentile,
            segment_pool_max=int(segment_pool_max_local),
            genre_blend_weight=genre_blend_weight,
            X_genre_dense=(avail_genre if genre_blend_weight > 0.0 else None),
            force_include=_force_include_arr,
            genre_vec_a=genre_vec_a,
            genre_vec_b=genre_vec_b,
        )

        # C1 rehome (Task 2): mean duration-rank penalty, diagnostics only.
        # The beam owns C1's SELECTION effect since f142677
        # (duration_penalty_values threaded additively into
        # _beam_search_segment's per-edge scoring, the same pattern as C10's
        # instrumental lean) -- multiplying a penalty vector into
        # `rank_scores` here too would double-apply it at the
        # segment_pool_max cap margin, violating single-enforcement (Phase 1
        # Task 8 review finding; see CLAUDE.md Layer 3 item 18's
        # "single-enforcement" discipline).
        #
        # Computed directly over THIS segment's small (final_candidates-
        # sized) accepted corridor, not the whole eligible universe:
        # `EligibleUniverse.duration_rank_penalty` used to carry a REAL
        # per-row factor computed for every eligible track in the whole
        # library (tens of thousands of rows) purely so this mean could look
        # up the handful (~2-30) that ended up here -- a dead full-universe
        # pass (final-review finding, corridor-phase1-pooling, 2026-07-18;
        # see eligible_universe.py's module docstring). The vector field
        # stays on `EligibleUniverse` (a structural placeholder, always
        # 1.0 -- nothing else reads it) but the real math now lives here,
        # where it's cheap.
        _duration_seed_set = set(int(i) for i in seed_indices)
        _result_penalties: List[float] = []
        for _i in result.indices.tolist():
            _factor = 1.0
            if int(_i) not in _duration_seed_set:
                if (
                    _corridor_duration_reference_ms is not None
                    and _corridor_duration_penalty_weight > 0.0
                    and getattr(bundle, "durations_ms", None) is not None
                ):
                    _dur = float(bundle.durations_ms[int(_i)])
                    if _dur > 0.0 and _dur > float(_corridor_duration_reference_ms):
                        _dp = compute_duration_penalty(
                            _dur, float(_corridor_duration_reference_ms),
                            float(_corridor_duration_penalty_weight),
                        )
                        if _dp > 0:
                            _factor *= max(0.0, 1.0 - _dp)
                if (
                    bool(cfg.instrumental_enabled)
                    and float(cfg.instrumental_penalty_weight) > 0.0
                    and voice_prob is not None
                ):
                    from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
                    _ip = compute_instrumental_penalty(
                        voice_prob, cand=int(_i), weight=float(cfg.instrumental_penalty_weight)
                    )
                    if _ip > 0:
                        _factor *= max(0.0, 1.0 - _ip)
            _result_penalties.append(_factor)
        _mean_duration_penalty = float(np.mean(_result_penalties)) if _result_penalties else 1.0

        _artist_key_by_avail_idx = dict(zip(avail_idx.tolist(), avail_artist_keys))
        final_candidates: List[int] = [int(i) for i in result.indices.tolist()]
        cand_artist_keys: Dict[int, str] = {
            i: _artist_key_by_avail_idx.get(i, "") for i in final_candidates
        }
        if pier_a_ak is not None:
            cand_artist_keys[int(pier_a_idx)] = pier_a_ak
        if pier_b_ak is not None:
            cand_artist_keys[int(pier_b_idx)] = pier_b_ak

        support_a = float(result.stats.get("anchor_support_a", 0.0))
        support_b = float(result.stats.get("anchor_support_b", 0.0))
        # Tag-guarantee summary count (Task 5): how many of the on-tag
        # guarantee ids actually landed in THIS segment's corridor -- a
        # diagnostics-route summary count, never a member-id list (worker
        # NDJSON line-size trap, same discipline as every other corridor_
        # segments field).
        _forced_included = (
            len(set(int(i) for i in on_tag_guarantee_indices) & set(final_candidates))
            if on_tag_guarantee_indices else 0
        )
        seg_diag: Dict[str, Any] = {
            "seg": int(seg_idx_local),
            "size": int(len(final_candidates)),
            "threshold": float(result.threshold),
            "width": float(result.width_percentile),
            "support_a": support_a,
            "support_b": support_b,
            "capped": bool(result.capped),
            # Filled in by the widening-ladder wrapper for the ACCEPTED
            # attempt (0 = the initial, un-widened width succeeded).
            "widened": 0,
            "forced_included": int(_forced_included),
            # C1 rehome parity (see comment above): mean duration-rank
            # penalty applied to this segment's corridor members
            # (diagnostics only -- the beam is the sole enforcer).
            "mean_duration_penalty": _mean_duration_penalty,
        }

        return final_candidates, cand_artist_keys, {}, seg_diag

    all_segments: List[List[int]] = []
    all_beam_components: List[dict] = []  # flat per-edge component dicts for audit
    diagnostics: List[SegmentDiagnostics] = []
    soft_genre_penalty_hits_total = 0
    soft_genre_penalty_edges_scored_total = 0
    local_sonic_penalty_hits_total = 0
    local_sonic_edges_scored_total = 0
    local_sonic_gate_rejected_total = 0
    local_sonic_penalty_total = 0.0
    segment_bridge_floors_used: list[float] = []
    segment_backoff_attempts_used: list[int] = []

    # Boundary context tracking for cross-segment min_gap enforcement.
    # Each segment's beam enforces one-track-per-artist WITHIN the segment, but
    # repeats across a segment boundary were previously only blocked 1 position
    # back (hardcoded), so e.g. an artist at the end of segment N and the start of
    # segment N+1 sat far closer than the configured min_gap. Use the resolved
    # min_gap so the last `min_gap` placed artists are carried into the next
    # segment's blocked set. Segments are typically <= min_gap long, so a flat
    # block of the boundary window matches min_gap semantics without over-blocking.
    MIN_GAP_GLOBAL = max(1, int(min_gap))  # Cross-segment min_gap constraint
    recent_boundary_artists: List[str] = []
    global_non_seed_artist_counts: Dict[str, int] = {}
    # Forward-looking half of cross-segment min_gap enforcement (see
    # _forward_pier_gap_block_indices's docstring): ordered_seeds (all piers,
    # including mini-pier waypoints) is fixed before this loop starts, so nominal
    # pier positions can be computed once, up front.
    _pier_positions_nominal = _pier_nominal_positions(segment_lengths)

    # Tail-DP endgame (spec 2026-07-02) summary counters.
    tail_dp_attempted_segments = 0
    tail_dp_applied_segments = 0

    def _artist_keys_for_cap(idx: int) -> Set[str]:
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
        if use_identity:
            artist_str = ""
            if bundle is not None and bundle.track_artists is not None:
                try:
                    artist_str = str(bundle.track_artists[int(idx)] or "")
                except Exception:
                    artist_str = ""
            if artist_str:
                return set(resolve_artist_identity_keys(artist_str, artist_identity_cfg))
        try:
            artist_key = identity_keys_for_index(bundle, int(idx)).artist_key
        except Exception:
            artist_key = ""
        return {str(artist_key)} if artist_key else set()

    def _recent_artists_for_segment(seg_idx: int) -> Optional[List[str]]:
        artists: List[str] = []
        if seg_idx > 0:
            artists.extend(recent_boundary_artists)
        cap = cfg.max_non_seed_tracks_per_artist
        if isinstance(cap, int) and cap > 0:
            artists.extend(
                artist_key
                for artist_key, count in global_non_seed_artist_counts.items()
                if int(count) >= int(cap)
            )
        # Forward half of cross-segment min_gap enforcement: block any UPCOMING
        # pier (real seed or mini-pier waypoint) whose fixed position lands
        # within min_gap of this segment (see _forward_pier_gap_block_indices).
        # This segment's own two boundary piers are already banned locally via
        # disallow_pier_artists_in_interiors; this only reaches further ahead.
        for _j in _forward_pier_gap_block_indices(
            _pier_positions_nominal, seg_idx, MIN_GAP_GLOBAL
        ):
            artists.extend(_artist_keys_for_cap(int(ordered_seeds[_j])))
        return list(dict.fromkeys(artists)) or None

    def _bridge_floor_attempts(initial_floor: float) -> list[float]:
        # Phase 1 Task 8: the legacy infeasible_handling bridge_floor backoff
        # cascade was deleted -- the corridor widening ladder
        # (_run_corridor_widening_ladder) is THE sole segment-level recovery
        # mechanism now (this was already true behind the flag: this function
        # forced single-attempt whenever pooling_mode == "corridor", which is
        # now unconditional). Retired config family:
        # playlists.ds_pipeline.pier_bridge.infeasible_handling.{enabled,
        # strategy,min_bridge_floor,backoff_steps,max_attempts_per_segment,
        # widen_search_on_backoff,extra_neighbors_m,extra_bridge_helpers,
        # extra_beam_width,extra_expansion_attempts,
        # transition_floor_relaxation_enabled,min_transition_floor,
        # genre_arc_relaxation_enabled,min_genre_arc_percentile} -- warned via
        # src.playlist_gui.worker._warn_retired_keys. `.guarantee_feasible`
        # and `.greedy_genre_weight` remain live (terminal fallback
        # placement, see the guarantee_feasible check later in this
        # function) and are NOT retired.
        return [float(initial_floor)]

    def _run_segment_backoff_attempts(
        *,
        cfg_attempt_base: PierBridgeConfig,
        segment_allow_detours: bool,
        segment_g_targets: Optional[list[np.ndarray]],
        segment_g_targets_dense: Optional[list[np.ndarray]] = None,
        pier_a: int,
        pier_b: int,
        interior_len: int,
        pier_a_id: str,
        pier_b_id: str,
        seg_idx: int,
        recent_boundary_artists: Optional[List[str]],
        transition_floor_override: Optional[float] = None,
        genre_arc_floor_percentile_override: Optional[float] = None,
        corridor_width_override: Optional[float] = None,
    ) -> dict[str, Any]:
        cfg = cfg_attempt_base
        if transition_floor_override is not None:
            cfg = replace(cfg, transition_floor=float(transition_floor_override))
        if genre_arc_floor_percentile_override is not None:
            cfg = replace(cfg, genre_arc_floor_percentile=float(genre_arc_floor_percentile_override))
        segment_path: Optional[List[int]] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts = _bridge_floor_attempts(float(cfg.bridge_floor))
        backoff_used_count = 0
        widened_search_used = False
        last_failure_reason: Optional[str] = None

        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0
        local_sonic_stats_segment: Dict[str, Any] = {}
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_waypoint_stats: Dict[str, Any] = {}
        last_edge_components: List[dict] = []
        # Initialized pre-loop so an entry-time deadline break (below) still
        # returns a bound value — the return dict references it unconditionally.
        # Without this, a generation that crosses the deadline on the FIRST
        # floor attempt (before the in-loop assignment) crashes with
        # UnboundLocalError (observed: Khruangbin, 2026-06-27).
        last_pool_diag: Dict[str, Any] = {}
        # Corridor-mode per-call pool stats (Task 4): the 4th return value of
        # _build_corridor_segment_pool, captured here so the widening ladder
        # can pick the ACCEPTED attempt's stats for the once-per-segment
        # health line + diagnostics entry. Empty/unused on the legacy path.
        last_corridor_pool_diag: Dict[str, Any] = {}

        for floor_attempt_idx, bridge_floor in enumerate(backoff_attempts):
            # Shared generation deadline: if a deadline was passed from core.py,
            # stop further relaxation attempts and bail to fallback placement.
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    "Generation deadline exceeded in bridge-floor backoff "
                    "(floor_attempt=%d) — bailing to fallback placement",
                    floor_attempt_idx,
                )
                break
            backoff_used_count = floor_attempt_idx + 1
            # `widened` (infeasible_handling.widen_search_on_backoff) could only
            # ever fire on floor_attempt_idx > 0, but `_bridge_floor_attempts`
            # always returns a single-element list (Phase 1 Task 8 retired the
            # bridge_floor backoff cascade -- the corridor widening ladder is
            # the sole segment-level recovery mechanism now), so
            # floor_attempt_idx is always 0 and the resource-expansion branch
            # that used to key off `widened` (extra pool/beam-width/expansion
            # budget) was unreachable dead code, deleted (final-review minor,
            # corridor-phase1-pooling, 2026-07-18). `widened` itself stays as
            # an always-False constant -- it's still read below for
            # per-attempt logging/audit-event diagnostics.
            widened = False
            widened_search_used = widened_search_used or widened

            cfg_attempt = replace(cfg, bridge_floor=float(bridge_floor))

            segment_pool_max = int(cfg.segment_pool_max)
            beam_width = cfg.initial_beam_width
            max_expansion_attempts = cfg.max_expansion_attempts

            expansions = 0
            pool_size_initial = 0
            pool_size_final = 0
            soft_genre_penalty_hits_segment = 0
            soft_genre_penalty_edges_scored_segment = 0
            local_sonic_stats_segment = {}
            last_failure_reason = None
            expansion_attempts_used = 0
            last_pool_diag: Dict[str, Any] = {}
            last_segment_candidates = []
            last_candidate_artist_keys = {}
            last_arc_stats: Dict[str, Any] = {}
            last_genre_cache_stats: Dict[str, int] = {}
            last_waypoint_stats = {}  # Reset for this backoff attempt
            segment_pool_cache: Optional[Dict[str, Any]] = (
                {} if bool(cfg.dj_pooling_cache_enabled) else None
            )
            last_segment_pool_max = int(segment_pool_max)
            last_beam_width = int(beam_width)

            for attempt in range(max_expansion_attempts):
                # Shared generation deadline: bail this expansion attempt loop if
                # the total generation budget has been exceeded.
                if deadline is not None and time.monotonic() > deadline:
                    logger.warning(
                        "Generation deadline exceeded in expansion attempt %d "
                        "(bridge_floor=%.3f) — bailing to fallback placement",
                        attempt, float(bridge_floor),
                    )
                    break
                # Cooperative cancellation: poll between beam-run attempts so a
                # narrow-mode backoff/expansion cascade cannot grind on after the
                # user requests cancel. raise_if_cancelled() is a no-op unless a
                # hook is registered (worker generation path).
                raise_if_cancelled()
                pool_diag: Dict[str, Any] = {}
                cand_artist_keys: Dict[int, str] = {}
                arc_stats_segment: Dict[str, Any] = {}
                genre_cache_stats_segment: Dict[str, int] = {}
                waypoint_stats_segment: Dict[str, Any] = {}
                pool_strategy = str(cfg.segment_pool_strategy).strip().lower()
                dj_pooling_strategy = str(cfg.dj_pooling_strategy or "baseline").strip().lower()
                if bool(cfg.dj_bridging_enabled) and dj_pooling_strategy == "dj_union":
                    pool_strategy = "dj_union"

                segment_internal_connectors = internal_connector_indices
                segment_connector_cap = int(internal_connector_max_per_segment)
                if bool(cfg.dj_bridging_enabled) and bool(cfg.dj_connector_bias_enabled) and segment_allow_detours:
                    adventurous = str(cfg.dj_route_shape or "linear").strip().lower() in {"arc", "ladder"}
                    dj_connector_cap = (
                        int(cfg.dj_connector_max_per_segment_adventurous)
                        if adventurous
                        else int(cfg.dj_connector_max_per_segment_linear)
                    )
                    dj_connector_cap = max(0, dj_connector_cap)
                    if dj_connector_cap > 0:
                        available = [int(i) for i in universe if int(i) not in global_used]
                        if allowed_set_indices is not None:
                            allowed = set(int(i) for i in allowed_set_indices)
                            available = [int(i) for i in available if int(i) in allowed]
                        if available:
                            dj_connectors = _select_connector_candidates(
                                available,
                                X_full_norm,
                                pier_a,
                                pier_b,
                                dj_connector_cap,
                            )
                        else:
                            dj_connectors = []
                        pool_diag["dj_connectors_selected"] = int(len(dj_connectors))
                        pool_diag["dj_connectors_injected_count"] = int(len(dj_connectors))
                        if dj_connectors:
                            try:
                                pool_diag["dj_connectors_preview"] = [
                                    str(bundle.track_ids[int(i)])
                                    for i in dj_connectors[:5]
                                ]
                            except Exception:
                                pool_diag["dj_connectors_preview"] = [
                                    str(int(i)) for i in dj_connectors[:5]
                                ]
                            if segment_pool_cache is not None:
                                segment_pool_cache["dj_connectors"] = set(
                                    int(i) for i in dj_connectors
                                )
                        if dj_connectors:
                            if segment_internal_connectors:
                                segment_internal_connectors = set(segment_internal_connectors) | set(dj_connectors)
                            else:
                                segment_internal_connectors = set(dj_connectors)
                            segment_connector_cap = max(segment_connector_cap, dj_connector_cap)

                # Phase 1 Task 8: corridor pooling is the sole segment-pool
                # strategy -- the legacy KNN-union (_build_segment_candidate_pool_legacy)
                # and segment-scored (_build_segment_candidate_pool_scored,
                # incl. its disallow_seed_artist_in_interiors relaxed retry
                # and its dj_union debug_compare_baseline comparison) builders
                # were deleted along with the elif/else selector that chose
                # between them and this corridor branch.
                pool_diag["pool_strategy"] = "corridor"
                segment_candidates, cand_artist_keys, _cand_title_keys, _corridor_pool_diag = _build_corridor_segment_pool(
                    pier_a,
                    pier_b,
                    seg_idx,
                    int(segment_pool_max),
                    width_percentile_override=corridor_width_override,
                )
                last_corridor_pool_diag = dict(_corridor_pool_diag)

                if segment_candidates and recent_boundary_artists:
                    blocked_artist_keys = set(str(k) for k in recent_boundary_artists if k)
                    if blocked_artist_keys:
                        before_artist_gate = len(segment_candidates)
                        segment_candidates = [
                            int(i)
                            for i in segment_candidates
                            if not (_artist_keys_for_cap(int(i)) & blocked_artist_keys)
                        ]
                        removed_by_artist_gate = before_artist_gate - len(segment_candidates)
                        if removed_by_artist_gate > 0:
                            pool_diag["removed_by_global_artist_gate"] = int(
                                removed_by_artist_gate
                            )

                if segment_candidates:
                    last_segment_candidates = list(segment_candidates)
                    last_candidate_artist_keys = dict(cand_artist_keys or {})
                last_pool_diag.update(pool_diag)

                # TASK A: Track pool_before_gating (after merge, before gates)
                pool_size_initial = len(segment_candidates) if segment_candidates else 0

                if not segment_candidates or len(segment_candidates) < int(interior_len):
                    last_failure_reason = f"pool_after_gate {len(segment_candidates)} < interior_len {interior_len}"
                    pool_size_final = 0
                else:
                    _edge_components_buf: Dict[str, Any] = {}
                    # ── Roam corridors: per-segment on-manifold deviations (flag-gated). ──
                    _roam_detour_sonic = None
                    _roam_dev_energy = None
                    if cfg_attempt.roam_corridors_enabled:
                        _roam_detour_sonic = _segment_sonic_detour(
                            pier_a, pier_b, segment_candidates, X_full_norm,
                            k=int(cfg_attempt.roam_knn_k),
                            mutual_proximity=bool(cfg_attempt.roam_mutual_proximity),
                        )
                        _roam_dev_energy = _energy_band_deviation(energy_matrix, [pier_a, pier_b])
                    segment_path, soft_genre_penalty_hits_segment, soft_genre_penalty_edges_scored_segment, beam_failure_reason = _beam_search_segment(
                        pier_a,
                        pier_b,
                        interior_len,
                        segment_candidates,
                        X_full_tr_norm,
                        X_full_norm,
                        X_start_tr_norm,
                        X_mid_tr_norm,
                        X_end_tr_norm,
                        X_genre_norm,
                        cfg_attempt,
                        beam_width,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        X_genre_dense=X_genre_dense,
                        genre_idf=genre_idf,
                        genre_vocab=genre_vocab,
                        artist_key_by_idx=(cand_artist_keys if cand_artist_keys else None),
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=recent_boundary_artists if seg_idx > 0 else None,
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        arc_stats=arc_stats_segment,
                        genre_cache_stats=genre_cache_stats_segment,
                        # Beam arc vote uses the 64-dim dense targets when steering is on;
                        # falls back to the 893-dim dj-bridging targets otherwise. Keeping
                        # these separate from the pooling's genre_targets prevents a
                        # 893-vs-64 dimension crash when both systems are enabled.
                        g_targets_override=(
                            segment_g_targets_dense
                            if segment_g_targets_dense is not None
                            else segment_g_targets
                        ),
                        waypoint_stats=waypoint_stats_segment,
                        local_sonic_stats=local_sonic_stats_segment,
                        edge_components_out=_edge_components_buf,
                        transition_metric_context=transition_metric_context,
                        perceptual_bpm=perceptual_bpm,
                        tempo_stability=tempo_stability_arr,
                        onset_rate=onset_rate,
                        pair_sim_provider=pair_sim_provider,
                        energy_matrix=energy_matrix,
                        voice_prob=voice_prob,
                        duration_penalty_values=corridor_duration_penalty_values,
                        roam_detour_sonic=_roam_detour_sonic,
                        roam_dev_energy=_roam_dev_energy,
                        popularity_values=popularity_values,
                        sonic_tag_affinity=sonic_tag_affinity,
                        sonic_tag_beam_weight=sonic_tag_beam_weight,
                        tag_steering_worst_edge_band=tag_steering_worst_edge_band,
                    )
                    last_failure_reason = beam_failure_reason
                    # ── Roam corridors: log realized sonic roam of the chosen interior. ──
                    if (
                        cfg_attempt.roam_corridors_enabled
                        and segment_path is not None
                        and _roam_detour_sonic is not None
                    ):
                        _dets = [
                            float(_roam_detour_sonic[int(t)])
                            for t in segment_path
                            if math.isfinite(float(_roam_detour_sonic[int(t)]))
                        ]
                        if _dets:
                            logger.info(
                                "Roam[seg %d]: sonic detour mean=%.3f max=%.3f "
                                "(width_sonic=%.3f width_energy=%.3f k=%d)",
                                seg_idx, float(np.mean(_dets)), float(np.max(_dets)),
                                float(cfg_attempt.roam_width_sonic),
                                float(cfg_attempt.roam_width_energy),
                                int(cfg_attempt.roam_knn_k),
                            )
                    if segment_path is not None:
                        last_edge_components = list(_edge_components_buf.get("components") or [])
                        # Capture waypoint stats for successful path
                        last_waypoint_stats = dict(waypoint_stats_segment)

                        # TASK A: Extract pool_after_gating count from waypoint_stats
                        pool_size_final = int(waypoint_stats_segment.get("pool_after_gating_count", 0))

                        baseline_pool = None
                        if (
                            segment_pool_cache is not None
                            and "dj_baseline_pool" in segment_pool_cache
                        ):
                            baseline_pool = segment_pool_cache.get(
                                "dj_baseline_pool"
                            )
                        elif pool_strategy != "dj_union":
                            baseline_pool = set(int(i) for i in segment_candidates)
                        sources = None
                        if segment_pool_cache is not None:
                            sources = segment_pool_cache.get("dj_pool_sources")
                        last_pool_diag.update(
                            _compute_chosen_source_counts(
                                segment_path,
                                sources=sources,
                                baseline_pool=baseline_pool,
                                log_per_track=bool(cfg.dj_diagnostics_pool_verbose),  # Task C
                            )
                        )
                        if segment_pool_cache is not None and "dj_connectors" in segment_pool_cache:
                            connector_set = segment_pool_cache.get("dj_connectors", set())
                            chosen_connectors = [
                                int(i) for i in segment_path if int(i) in connector_set
                            ]
                            last_pool_diag["dj_connectors_chosen_count"] = int(
                                len(chosen_connectors)
                            )
                            if chosen_connectors:
                                try:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(bundle.track_ids[int(i)])
                                        for i in chosen_connectors[:5]
                                    ]
                                except Exception:
                                    last_pool_diag["dj_connectors_chosen_preview"] = [
                                        str(int(i)) for i in chosen_connectors[:5]
                                    ]

                if segment_path is not None:
                    break

                expansions += 1
                # Phase 1 Task 8: corridor is the only pooling strategy, so this
                # doubling is now unconditional (was already unconditional for
                # corridor before the legacy elif/else pool builders were deleted).
                segment_pool_max = min(int(segment_pool_max) * 2, int(cfg.max_segment_pool_max))
                beam_width = min(int(beam_width) * 2, int(cfg.max_beam_width))

                if infeasible_handling and infeasible_handling.enabled:
                    if str(cfg.segment_pool_strategy).strip().lower() == "legacy":
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d strategy=legacy beam=%d)",
                            seg_idx,
                            attempt + 1,
                            beam_width,
                        )
                    else:
                        logger.debug(
                            "Segment %d: expanding search (expansion_attempt=%d segment_pool_max=%d beam=%d)",
                            seg_idx,
                            attempt + 1,
                            int(segment_pool_max),
                            beam_width,
                        )

            if infeasible_handling and infeasible_handling.enabled:
                logger.info(
                    "Segment %d attempt %d: bridge_floor=%.2f widened=%s pool_after_gate=%d",
                    seg_idx,
                    floor_attempt_idx + 1,
                    float(bridge_floor),
                    widened,
                    int(len(last_segment_candidates)),
                )

            if audit_enabled:
                top_rows, dists = _summarize_candidates_for_audit(
                    candidates=last_segment_candidates,
                    pier_a=pier_a,
                    pier_b=pier_b,
                    X_full_norm=X_full_norm,
                    X_full_tr_norm=X_full_tr_norm,
                    X_start_tr_norm=X_start_tr_norm,
                    X_mid_tr_norm=X_mid_tr_norm,
                    X_end_tr_norm=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_attempt,
                    bundle=bundle,
                    internal_connector_indices=internal_connector_indices,
                    top_k=top_k,
                )
                audit_events.append(
                    RunAuditEvent(
                        kind="segment_attempt",
                        ts_utc=now_utc_iso(),
                        payload={
                            "segment_index": int(seg_idx),
                            "segment_header": f"{pier_a_id} -> {pier_b_id} (interior={interior_len})",
                            "attempt_number": int(floor_attempt_idx + 1),
                            "expansion_attempts": int(expansion_attempts_used),
                            "bridge_floor": float(bridge_floor),
                            "widened": bool(widened),
                            "segment_pool_strategy": str(
                                last_pool_diag.get("pool_strategy", cfg.segment_pool_strategy)
                            ),
                            "segment_pool_max": int(last_segment_pool_max),
                            "beam_width": int(last_beam_width),
                            "pool_counts": dict(last_pool_diag),
                            "pool_size_initial": int(pool_size_initial),
                            "pool_size_final": int(pool_size_final),
                            "distributions": dists,
                            "soft_genre_penalty": {
                                "edges_scored": int(soft_genre_penalty_edges_scored_segment),
                                "hits": int(soft_genre_penalty_hits_segment),
                                "threshold": float(cfg_attempt.genre_penalty_threshold),
                                "strength": float(cfg_attempt.genre_penalty_strength),
                            },
                            "local_sonic_edge_penalty": dict(local_sonic_stats_segment),
                            "genre_cache": dict(last_genre_cache_stats),
                            "progress_arc": dict(last_arc_stats),
                        },
                    )
                )

            if segment_path is not None:
                chosen_bridge_floor = float(bridge_floor)
                beam_width_used = int(last_beam_width)
                break

            last_segment_pool_cache = segment_pool_cache

        return {
            "segment_path": segment_path,
            "chosen_bridge_floor": float(chosen_bridge_floor),
            "backoff_attempts": [float(x) for x in backoff_attempts],
            "backoff_used_count": int(backoff_used_count),
            "widened_search_used": bool(widened_search_used),
            "expansions": int(expansions),
            "pool_size_initial": int(pool_size_initial),
            "pool_size_final": int(pool_size_final),
            "beam_width_used": int(beam_width_used),
            "soft_genre_penalty_hits_segment": int(soft_genre_penalty_hits_segment),
            "soft_genre_penalty_edges_scored_segment": int(soft_genre_penalty_edges_scored_segment),
            "local_sonic_stats_segment": dict(local_sonic_stats_segment),
            "last_failure_reason": (str(last_failure_reason) if last_failure_reason else None),
            "last_segment_candidates": list(last_segment_candidates),
            "last_candidate_artist_keys": dict(last_candidate_artist_keys),
            "segment_pool_cache": dict(last_segment_pool_cache or {}),
            "last_waypoint_stats": dict(last_waypoint_stats),
            "last_pool_diag": dict(last_pool_diag),
            "edge_components": list(last_edge_components),
            "corridor_pool_diag": dict(last_corridor_pool_diag),
        }

    def _segment_min_edge_t(attempt_result: dict[str, Any]) -> Optional[float]:
        """Worst (minimum) per-edge transition T along a segment's chosen path.

        Source: the beam's own minimax path tracking -- ``edge_components``
        (populated by ``_beam_search_segment`` via its ``edge_components_out``
        param, one dict per chosen edge with a ``"T"`` key) is exactly the
        per-edge score list the beam itself used to pick the path, so this is
        a read of the beam's own bookkeeping, not a recomputation.

        Returns None when no path was found (nothing to score). Returns
        +inf for a path with zero scoreable edges (e.g. a 0-interior segment
        -- trivially satisfies any floor, nothing to violate).
        """
        if attempt_result.get("segment_path") is None:
            return None
        comps = attempt_result.get("edge_components") or []
        t_vals = [
            float(c["T"]) for c in comps
            if isinstance(c, dict) and c.get("T") is not None
        ]
        return min(t_vals) if t_vals else float("inf")

    def _run_corridor_widening_ladder(
        *,
        cfg_base: PierBridgeConfig,
        segment_allow_detours: bool,
        segment_g_targets: Optional[list[np.ndarray]],
        segment_g_targets_dense: Optional[list[np.ndarray]],
        pier_a: int,
        pier_b: int,
        interior_len: int,
        pier_a_id: str,
        pier_b_id: str,
        seg_idx: int,
        recent_boundary_artists: Optional[List[str]],
    ) -> dict[str, Any]:
        """Quality-triggered corridor width-widening ladder (Phase 1 Task 4,
        spec section 4). THE segment-level recovery mechanism for
        pooling=="corridor" -- replaces the legacy bridge_floor-backoff /
        transition-floor-relaxation / genre-arc-floor-relaxation tiers for
        corridor segments (those tiers are gated off for pooling=="corridor"
        at their own call sites below; see the double-ladder comments there
        and on ``_bridge_floor_attempts``).

        ::

            corridor = build(width)          # initial (un-widened) width
            path = beam(corridor)
            if path is None:
                widen unconditionally, to the full attempt budget   # hard
                                                # infeasibility -- never gated
            elif path.min_edge_T < transition_floor:
                widen attempt 1 unconditionally      # trigger firing is signal enough
                while attempts < corridor_widen_attempts:
                    if this attempt's improvement over the prior best
                       > corridor_widen_improvement_epsilon:
                        width -= corridor_widen_step   # percentile down = wider
                        corridor = build(width)         # capped at 0.0 = whole universe
                        path = beam(corridor)
                    else:
                        STOP widening               # empirically not paying for itself;
                                                     # accept best-seen path, hand to repair stack
            if still failing: accept best-min-edge path seen; log LOUDLY;
                              below_floor reporting + repair stack unchanged.

        Deterministic: the width sequence is a fixed arithmetic ladder from
        ``cfg_base.corridor_width_percentile`` (same inputs -> same widths,
        same attempts, same result). Each attempt still runs
        ``_run_segment_backoff_attempts`` (single bridge_floor attempt, see
        above) so beam invocation itself is completely untouched -- this
        function only decides which width to try next and which attempt to
        keep.

        Task 6 remediation, iteration 2 (empirical continue-gate; replaces
        iteration 1's predictive anchor-support gate, retired -- see
        ``corridor_widen_decision``'s docstring and
        .superpowers/sdd/p1-task6-remediation-report.md's "Iteration 2
        appendix" for why the prediction was falsified by real evidence,
        Alex G/home segment 1 specifically). The ladder always tries the
        first widen attempt unconditionally once the quality trigger fires;
        after that, each further widen is gated on whether the PREVIOUS
        attempt actually improved the best-seen min_edge_T by more than
        ``corridor_widen_improvement_epsilon`` (``corridor_widen_decision``
        in ``src.playlist.pier_bridge.corridor``). A non-improving attempt
        means widening further is empirically not paying off -- the ladder
        stops early, accepts the best-seen path, and tags
        ``widen_stopped_early: true`` in the segment's corridor diagnostics.
        Hard infeasibility (no path found) always widens to the full attempt
        budget regardless of this gate.
        """
        width = float(cfg_base.corridor_width_percentile)
        step = max(0.0, float(cfg_base.corridor_widen_step))
        max_widen_attempts = max(0, int(cfg_base.corridor_widen_attempts))
        floor = float(cfg_base.transition_floor)

        best_result: Optional[dict[str, Any]] = None
        best_comparable = float("-inf")
        best_widened = 0
        best_width = width

        epsilon = float(cfg_base.corridor_widen_improvement_epsilon)
        widen_stopped_early = False

        attempt = 0
        while True:
            # Snapshot the best-seen comparable value from strictly BEFORE
            # this attempt runs -- attempt 0's snapshot is the loop's -inf
            # init (unused: corridor_widen_decision ignores `improvement` at
            # attempt_index == 0). For attempt >= 1 this is the best value
            # after the PRIOR attempt's own update below, i.e. exactly what
            # "did the previous attempt improve on the best-seen-before-it"
            # needs.
            prior_best = best_comparable

            result = _run_segment_backoff_attempts(
                cfg_attempt_base=cfg_base,
                segment_allow_detours=segment_allow_detours,
                segment_g_targets=segment_g_targets,
                segment_g_targets_dense=segment_g_targets_dense,
                pier_a=pier_a,
                pier_b=pier_b,
                interior_len=interior_len,
                pier_a_id=pier_a_id,
                pier_b_id=pier_b_id,
                seg_idx=seg_idx,
                recent_boundary_artists=recent_boundary_artists,
                corridor_width_override=width,
            )
            min_edge_t = _segment_min_edge_t(result)
            quality_ok = min_edge_t is not None and min_edge_t >= floor
            comparable = min_edge_t if min_edge_t is not None else float("-inf")

            if best_result is None or comparable > best_comparable:
                best_result = result
                best_comparable = comparable
                best_widened = attempt
                best_width = width

            if quality_ok:
                if attempt > 0:
                    logger.info(
                        "CorridorWiden[seg %d]: recovered at attempt %d (width=%.2f "
                        "min_edge_T=%.3f >= floor=%.3f)",
                        seg_idx, attempt, width, float(min_edge_t), floor,
                    )
                break

            # Empirical continue-gate (Task 6 remediation, iteration 2):
            # attempt 0 always widens unconditionally (nothing to compare
            # yet); attempt >= 1 widens further only if THIS attempt just
            # improved on the best-seen value from before it ran by more
            # than epsilon -- see corridor_widen_decision's docstring. Hard
            # infeasibility (segment_path is None) always resolves to WIDEN
            # regardless of improvement, so the gate never blocks the
            # pre-existing no-path-found recovery behavior.
            # `comparable - prior_best` is well-defined even when prior_best
            # is -inf (a strictly-prior attempt found NO path at all): Python
            # float arithmetic gives +inf, which trivially clears epsilon --
            # correctly treating "went from infeasible to a real (if weak)
            # path" as unambiguous improvement, not a null "nothing to
            # compare" case. Only guard the genuinely undefined case (this
            # attempt itself found no path, so there is no min_edge_t to
            # subtract with) -- but that case is moot anyway, since
            # path_found=False resolves to WIDEN before `improvement` is
            # ever consulted.
            improvement = (
                (comparable - prior_best) if (attempt > 0 and min_edge_t is not None) else None
            )
            decision = corridor_widen_decision(
                path_found=result.get("segment_path") is not None,
                min_edge_t=min_edge_t,
                floor=floor,
                attempt_index=attempt,
                improvement=improvement,
                epsilon=epsilon,
            )
            if decision == CorridorWidenDecision.STOP:
                widen_stopped_early = True
                logger.info(
                    "CorridorWiden[seg %d] STOPPED after attempt %d: improvement "
                    "%.3f < epsilon %.3f — handing to repair stack",
                    seg_idx, attempt,
                    (improvement if improvement is not None else float("-inf")),
                    epsilon,
                )
                break

            if attempt >= max_widen_attempts:
                _best_t_repr = "None" if best_comparable == float("-inf") else f"{best_comparable:.3f}"
                logger.warning(
                    "CorridorWiden[seg %d] EXHAUSTED after %d widen attempt(s) "
                    "(initial width=%.2f, final width=%.2f): best min_edge_T=%s "
                    "vs floor=%.3f — accepting best-effort path; below-floor "
                    "reporting + repair stack proceed unchanged.",
                    seg_idx, attempt, float(cfg_base.corridor_width_percentile),
                    width, _best_t_repr, floor,
                )
                break
            attempt += 1
            width = max(0.0, width - step)
            logger.info(
                "CorridorWiden[seg %d]: attempt %d — widening width -> %.2f "
                "(prior min_edge_T=%s, floor=%.3f)",
                seg_idx, attempt, width,
                ("None" if min_edge_t is None else f"{min_edge_t:.3f}"), floor,
            )

        assert best_result is not None  # attempt 0 always runs and is captured above

        # Health-line/diagnostics entry is NOT logged/recorded here anymore
        # (Task 9 fix). Under variable_bridge_length, choose_segment_length
        # re-invokes this wrapper multiple times for the SAME seg_idx -- once
        # per candidate interior length -- each running its OWN independent
        # widening ladder. Emitting here on first-invocation (the old
        # behavior) latched onto whichever length var-bridge happened to try
        # first (typically nominal), not necessarily the length it ultimately
        # accepts, so the recorded diagnostics/health line could describe a
        # different corridor than the one that actually supplied the
        # segment's emitted tracks. Instead, attach this attempt's stats onto
        # the returned result; the seg_idx loop emits the health line exactly
        # once, AFTER choose_segment_length has picked the accepted length,
        # using THAT accepted attempt's stats (see the "Health line (Task 9
        # fix)" comment in the main segment loop below). `corridor_pool_diag`
        # already carries the resolved `width` for this attempt; keep that
        # value if present, otherwise fall back to this attempt's own width.
        _diag = dict(best_result.get("corridor_pool_diag") or {})
        _diag.setdefault("seg", int(seg_idx))
        _diag.setdefault("width", best_width)
        _diag["widened"] = int(best_widened)
        _diag["widen_stopped_early"] = bool(widen_stopped_early)
        best_result["corridor_health_diag"] = _diag

        return best_result

    # Wall-clock anchor for the relaxation-tier budget (see
    # _SEGMENT_RELAXATION_BUDGET_S). Cumulative across all segments so the total
    # build cannot blow the generation budget on a pathological relaxation grind.
    _pb_build_start = time.monotonic()
    # When the caller disables the generation deadline (deadline=None,
    # generation_budget_s<=0), the per-build relaxation cap is disabled too — "no
    # time limit" must mean BOTH wall-clock cutoffs are off, or relaxation grinds
    # would still bail at 40s. A finite deadline keeps the legacy 40s cap.
    _relax_budget_s = float("inf") if deadline is None else _SEGMENT_RELAXATION_BUDGET_S

    # Variable bridge length (default OFF -> byte-identical to the even-split path).
    # When ON, each segment's interior length flexes within a band off the nominal
    # even-split length to maximize its worst edge (bottleneck), capped by a
    # deterministic segment count limit. See src/playlist/pier_bridge/var_bridge.py.
    _vbl = bool(getattr(cfg, "variable_bridge_length", False))
    if _vbl:
        from src.playlist.pier_bridge.var_bridge import (
            choose_segment_length,
            segment_bottleneck,
        )
        _vbl_k = int(getattr(cfg, "variable_bridge_flex", 2))
        _vbl_good = float(getattr(cfg, "variable_bridge_min_edge", 0.30))
        _vbl_eps = float(getattr(cfg, "variable_bridge_epsilon", 0.02))
        _vbl_flexed = 0    # count of segments that actually flexed (deterministic cap)
        _vbl_max_flex = int(getattr(cfg, "variable_bridge_max_flex_segments", 3))

        def _edge_T(a: int, c: int) -> float:
            return float(
                score_transition_edge(transition_metric_context, int(a), int(c)).get("T", 0.0)
            )

    for seg_idx in range(num_segments):
        raise_if_cancelled()  # cooperative cancellation at each segment boundary
        # Shared generation deadline: if the caller passed a deadline (computed
        # once before all One-Each retries in core.py), mark this segment as
        # over-budget so all relaxation tiers are skipped and the segment goes
        # straight to guaranteed-fill fallback. Without this the segment loop
        # can grind past the deadline on starved pools.
        _deadline_exceeded_at_segment_start = (
            deadline is not None and time.monotonic() > deadline
        )
        if _deadline_exceeded_at_segment_start:
            logger.warning(
                "Generation deadline exceeded at segment %d start — "
                "forcing remaining segments to fallback placement",
                seg_idx,
            )
        pier_a = ordered_seeds[seg_idx]
        pier_b = ordered_seeds[seg_idx + 1]
        interior_len = segment_lengths[seg_idx]

        pier_a_id = str(bundle.track_ids[pier_a])
        pier_b_id = str(bundle.track_ids[pier_b])

        logger.info("Building segment %d: %s -> %s (interior=%d)",
                   seg_idx, pier_a_id, pier_b_id, interior_len)

        # ── Length-INDEPENDENT per-segment setup (does NOT depend on interior_len) ──
        # Far-stats (pier-pair only) and the relaxation-attempt plan depend on the
        # piers + cfg, not on the interior length, so compute them ONCE per segment.
        # They are captured by the length-parameterized build closure below.
        segment_far_stats: Optional[dict[str, Optional[float]]] = None
        segment_is_far = False
        if bool(cfg.dj_bridging_enabled) and X_genre_norm is not None:
            segment_far_stats = _segment_far_stats(
                pier_a=pier_a,
                pier_b=pier_b,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                universe=universe,
                used_track_ids=global_used,
                bridge_floor=float(cfg.bridge_floor),
            )
            if segment_far_stats:
                sonic_sim = segment_far_stats.get("sonic_sim")
                genre_sim = segment_far_stats.get("genre_sim")
                scarcity = segment_far_stats.get("connector_scarcity")
                if sonic_sim is not None and (1.0 - float(sonic_sim)) > float(cfg.dj_far_threshold_sonic):
                    segment_is_far = True
                if genre_sim is not None and (1.0 - float(genre_sim)) > float(cfg.dj_far_threshold_genre):
                    segment_is_far = True
                if scarcity is not None and float(scarcity) < float(cfg.dj_far_threshold_connector_scarcity):
                    segment_is_far = True

        _steering_source = str(getattr(cfg, "genre_steering_source", "taxonomy"))
        segment_allow_detours = bool(cfg.dj_allow_detours_when_far) and segment_is_far
        relaxation_enabled = (
            bool(cfg.dj_bridging_enabled)
            and bool(cfg.dj_relaxation_enabled)
            and str(cfg.dj_route_shape or "linear").strip().lower() == "ladder"
        )
        cfg_base = cfg
        segment_allow_detours_base = segment_allow_detours

        def _build_segment_at(interior_len: int) -> dict[str, Any]:
            """Build one bridge attempt at a specific interior length.

            Encapsulates the LENGTH-COUPLED work: the genre-arc target build (both
            the dj-bridging pooling targets and the steering targets) and the primary
            bridge-floor backoff / relaxation-attempt loop. Everything it consumes is
            captured from the enclosing segment scope EXCEPT ``interior_len``, so the
            variable-length selector can re-run it at several lengths side-effect-free
            (it only READS ``global_used`` — the commit happens in the loop body).

            Returns a dict carrying the chosen ``segment_path`` plus every diagnostic
            and length-coupled local the loop body and fallback tiers need. When the
            variable-length feature is OFF this is called exactly once at the nominal
            length, so the off-path stays byte-identical to the even-split path.
            """
            _g_targets: Optional[list[np.ndarray]] = None
            _g_targets_dense: Optional[list[np.ndarray]] = None
            _ladder_diag: dict[str, Any] = {}
            if bool(cfg_base.dj_bridging_enabled) and X_genre_norm is not None:
                # Phase 3 fix: genre_vocab is optional for vector mode, always try to build targets
                genre_vocab = getattr(bundle, "genre_vocab", None)
                _g_targets = _build_genre_targets(
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_length=interior_len,
                    X_full_norm=X_full_norm,
                    X_genre_norm=X_genre_norm,
                    genre_vocab=genre_vocab,  # Can be None for vector mode
                    genre_graph=genre_graph,
                    cfg=cfg_base,
                    warnings=warnings,
                    ladder_diag=_ladder_diag,
                    X_genre_raw=X_genre_raw,
                    X_genre_smoothed=X_genre_smoothed,
                    genre_idf=genre_idf,
                )

            # Genre-arc steering: build per-step g_targets that feed the beam's first-class
            # arc vote (via g_targets_override). Two sources, selected by genre_steering_source:
            #   - "taxonomy": route the arc through the SP3a taxonomy graph (hub-damped);
            #     targets live in the genre-vocab space (beam scores against X_genre_norm).
            #   - "dense" (legacy): interpolate the 64-dim dense PMI-SVD vectors (beam scores
            #     against X_genre_dense). Kept SEPARATE from segment_g_targets (dj-bridging
            #     pooling) to avoid dimension clashes.
            if (
                bool(cfg_base.genre_steering_enabled)
                and _steering_source == "taxonomy"
                and getattr(bundle, "X_genre_raw", None) is not None
                and getattr(bundle, "genre_vocab", None) is not None
            ):
                from src.playlist.pier_bridge.taxonomy_steering import (
                    build_taxonomy_genre_targets,
                    get_taxonomy_steering,
                )
                _tax_diag: dict[str, Any] = {}
                _g_targets_dense = build_taxonomy_genre_targets(
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_length=interior_len,
                    X_genre_raw=bundle.X_genre_raw,
                    genre_vocab=bundle.genre_vocab,
                    steering=get_taxonomy_steering(),
                    top_labels=int(cfg_base.dj_ladder_top_labels),
                    min_label_weight=float(cfg_base.dj_ladder_min_label_weight),
                    smooth_top_k=int(cfg_base.dj_ladder_smooth_top_k),
                    smooth_min_sim=float(cfg_base.dj_ladder_smooth_min_sim),
                    max_steps=int(cfg_base.dj_ladder_max_steps),
                    genre_track_counts=genre_track_counts,
                    min_waypoint_mass=int(getattr(cfg_base, "taxonomy_waypoint_min_library_mass", 0)),
                    ladder_diag=_tax_diag,
                )
                if _g_targets_dense is not None and _tax_diag.get("taxonomy_waypoint_labels"):
                    _ladder_diag.update(_tax_diag)
                    logger.info(
                        "Genre steering [taxonomy]: %s -> %s via %s",
                        bundle.track_ids[pier_a],
                        bundle.track_ids[pier_b],
                        _tax_diag.get("taxonomy_waypoint_labels"),
                    )
                elif _g_targets_dense is None:
                    logger.info(
                        "Genre steering [taxonomy]: no taxonomy path for segment %s -> %s "
                        "(uncovered genres); genre arc inactive this segment",
                        bundle.track_ids[pier_a],
                        bundle.track_ids[pier_b],
                    )

            # Dense steering (legacy). Skipped entirely in taxonomy mode so the beam's
            # genre-vocab arc vote never receives 64-dim dense targets.
            if (
                _steering_source != "taxonomy"
                and bool(cfg_base.genre_steering_enabled)
                and getattr(bundle, "X_genre_dense", None) is not None
            ):
                labels_a = _select_top_genre_labels(
                    bundle.X_genre_raw[pier_a], bundle.genre_vocab,
                    top_n=int(cfg_base.dj_ladder_top_labels), min_weight=float(cfg_base.dj_ladder_min_label_weight),
                ) if getattr(bundle, "genre_vocab", None) is not None else None
                labels_b = _select_top_genre_labels(
                    bundle.X_genre_raw[pier_b], bundle.genre_vocab,
                    top_n=int(cfg_base.dj_ladder_top_labels), min_weight=float(cfg_base.dj_ladder_min_label_weight),
                ) if getattr(bundle, "genre_vocab", None) is not None else None
                _g_targets_dense = build_dense_genre_targets(
                    bundle.X_genre_dense[pier_a], bundle.X_genre_dense[pier_b],
                    interior_length=interior_len, route=str(cfg_base.dj_route_shape or "linear"),
                    genre_emb=getattr(bundle, "genre_emb", None),
                    genre_vocab=list(bundle.genre_vocab) if getattr(bundle, "genre_vocab", None) is not None else None,
                    genre_graph=genre_graph_arc, labels_a=labels_a, labels_b=labels_b,
                    max_steps=int(cfg_base.dj_ladder_max_steps),
                )

            # Log the "steering enabled but no usable targets" condition ONCE per
            # segment here. The per-beam-call site in beam.py is demoted to debug so
            # the relaxation cascade does not flood the log with identical warnings.
            if (
                bool(cfg_base.genre_steering_enabled)
                and _g_targets_dense is None
                and _g_targets is None
            ):
                logger.warning(
                    "Segment %d: genre steering enabled (source=%s) but no usable genre "
                    "targets — genre arc inactive this segment",
                    seg_idx, _steering_source,
                )

            _seg_relax_attempts: list[dict[str, Any]] = []
            _relax_success: Optional[int] = None
            _cfg_used = cfg_base
            _attempt_result: Optional[dict[str, Any]] = None
            # Phase 1 Task 8: corridor pooling is the only path -- the widening
            # ladder IS the segment-level recovery mechanism now (it replaced
            # the legacy relaxation-attempts loop, which tried different
            # dj-bridging cfg variants, not corridor width; that loop and the
            # `relaxation_attempts` list it consumed were deleted). One call,
            # one baseline cfg; the ladder itself handles retrying at wider
            # corridor widths internally.
            if not _deadline_exceeded_at_segment_start:
                _attempt_result = _run_corridor_widening_ladder(
                    cfg_base=cfg_base,
                    segment_allow_detours=segment_allow_detours_base,
                    segment_g_targets=_g_targets,
                    segment_g_targets_dense=_g_targets_dense,
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_len=interior_len,
                    pier_a_id=pier_a_id,
                    pier_b_id=pier_b_id,
                    seg_idx=seg_idx,
                    recent_boundary_artists=_recent_artists_for_segment(seg_idx),
                )
                _attempt_path = _attempt_result["segment_path"]
                _seg_relax_attempts.append({
                    "attempt_index": 0,
                    "label": "corridor_widening_ladder",
                    "changes": [],
                    "failure_reason": (
                        str(_attempt_result["last_failure_reason"])
                        if _attempt_path is None else None
                    ),
                })
                if _attempt_path is not None:
                    _relax_success = 0

            return {
                "attempt_result": _attempt_result,
                "segment_g_targets": _g_targets,
                "segment_g_targets_dense": _g_targets_dense,
                "segment_ladder_diag": _ladder_diag,
                "segment_relaxation_attempts": _seg_relax_attempts,
                "relaxation_success_attempt": _relax_success,
                "cfg_used_for_segment": _cfg_used,
            }

        # Select the interior length and build the primary attempt. OFF -> one build
        # at the even-split nominal (byte-identical). ON -> greedy bottleneck-maximizing
        # flex within the running-total band, capped by a deterministic segment count.
        nominal = int(segment_lengths[seg_idx])
        if not _vbl:
            _seg_build = _build_segment_at(nominal)
            interior_len = nominal
        else:
            # ADD-only: never shorten a segment. lo == nominal; may lengthen up to
            # +variable_bridge_flex. No length budget (track count is not a target).
            lo = nominal
            hi = nominal + _vbl_k
            # Deterministic cap: once max-flex-segments have flexed, force nominal.
            if _vbl_flexed >= _vbl_max_flex:
                lo = hi = nominal

            def _build_and_score(length: int) -> tuple[dict[str, Any], float]:
                r = _build_segment_at(length)
                ar = r["attempt_result"]
                p = ar["segment_path"] if ar is not None else None
                if p:
                    nodes = [int(pier_a), *[int(x) for x in p], int(pier_b)]
                    b = segment_bottleneck(nodes, _edge_T)[0]
                else:
                    b = float("-inf")
                return (r, b)

            chosen_len, _seg_build, _vbl_seg_flexed = choose_segment_length(
                nominal, lo, hi, _build_and_score,
                good_enough=_vbl_good, eps=_vbl_eps,
            )
            if _vbl_seg_flexed:
                _vbl_flexed += 1
            interior_len = int(chosen_len)
            logger.info(
                "Var-bridge seg %d: nominal=%d chosen=%d flexed=%s (%d/%d) [add-only]",
                seg_idx, nominal, chosen_len,
                _vbl_seg_flexed, _vbl_flexed, _vbl_max_flex,
            )

        # Unpack the chosen-length build into the loop-body locals the downstream
        # fallback tiers + diagnostics consume (exactly as the inline relax loop did).
        segment_g_targets = _seg_build["segment_g_targets"]
        segment_g_targets_dense = _seg_build["segment_g_targets_dense"]
        segment_ladder_diag = _seg_build["segment_ladder_diag"]
        segment_relaxation_attempts = _seg_build["segment_relaxation_attempts"]
        relaxation_success_attempt = _seg_build["relaxation_success_attempt"]
        cfg_used_for_segment = _seg_build["cfg_used_for_segment"]
        attempt_result = _seg_build["attempt_result"]

        # Health line (Task 9 fix — once-per-segment health line + diagnostics
        # entry, Task 3's F7 contract): emitted HERE, after the interior
        # length has been chosen, not inside `_run_corridor_widening_ladder`
        # itself. When variable_bridge_length is on, `choose_segment_length`
        # may have invoked that ladder several times for this same seg_idx —
        # once per candidate length, each with its own independently-run
        # widening ladder — before settling on `interior_len` above.
        # `attempt_result` here is the ACCEPTED length's own attempt, so
        # logging from its `corridor_health_diag` (set by the ladder wrapper)
        # guarantees the health line + `corridor_segments` diagnostics always
        # describe the corridor that actually supplied this segment's
        # emitted tracks, even under var-bridge re-entry. `corridor_logged_
        # segments` keeps the once-per-segment-index invariant explicit.
        if attempt_result is not None and seg_idx not in corridor_logged_segments:
            corridor_logged_segments.add(seg_idx)
            _health_diag = dict(attempt_result.get("corridor_health_diag") or {})
            _health_diag.setdefault("seg", int(seg_idx))
            logger.info(
                "Corridor[seg %d]: size=%d width=%.2f widened=%d support_a=%.2f "
                "support_b=%.2f threshold=%.3f capped=%s",
                int(seg_idx),
                int(_health_diag.get("size", 0)),
                float(_health_diag.get("width", 0.0)),
                int(_health_diag.get("widened", 0)),
                float(_health_diag.get("support_a", 0.0)),
                float(_health_diag.get("support_b", 0.0)),
                float(_health_diag.get("threshold", 0.0)),
                bool(_health_diag.get("capped", False)),
            )
            corridor_segments_diag.append(_health_diag)

        segment_path: Optional[List[int]] = None
        last_segment_candidates: List[int] = []
        last_candidate_artist_keys: Dict[int, str] = {}
        last_segment_pool_cache: Optional[Dict[str, Any]] = None
        last_failure_reason: Optional[str] = None
        chosen_bridge_floor = float(cfg.bridge_floor)
        backoff_attempts: list[float] = [float(cfg.bridge_floor)]
        backoff_used_count = 0
        widened_search_used = False
        expansions = 0
        pool_size_initial = 0
        pool_size_final = 0
        beam_width_used = cfg.initial_beam_width
        soft_genre_penalty_hits_segment = 0
        soft_genre_penalty_edges_scored_segment = 0
        local_sonic_stats_segment: Dict[str, Any] = {}
        # Pre-initialize variables that are set from attempt_result so that an early
        # break (e.g. deadline exceeded before any attempt ran) doesn't leave them
        # unbound for the downstream diagnostics code.
        last_waypoint_stats: Dict[str, Any] = {}
        last_pool_diag: Dict[str, Any] = {}
        segment_edge_components: List[dict] = []

        if attempt_result is not None:
            segment_path = attempt_result["segment_path"]
            chosen_bridge_floor = float(attempt_result["chosen_bridge_floor"])
            backoff_used_count = int(attempt_result["backoff_used_count"])
            backoff_attempts = list(attempt_result.get("backoff_attempts") or [])
            widened_search_used = bool(attempt_result["widened_search_used"])
            expansions = int(attempt_result["expansions"])
            pool_size_initial = int(attempt_result["pool_size_initial"])
            pool_size_final = int(attempt_result["pool_size_final"])
            beam_width_used = int(attempt_result["beam_width_used"])
            soft_genre_penalty_hits_segment = int(attempt_result["soft_genre_penalty_hits_segment"])
            soft_genre_penalty_edges_scored_segment = int(attempt_result["soft_genre_penalty_edges_scored_segment"])
            local_sonic_stats_segment = dict(attempt_result.get("local_sonic_stats_segment", {}))
            last_failure_reason = attempt_result["last_failure_reason"]
            last_segment_candidates = list(attempt_result["last_segment_candidates"])
            # Edge-repair reseat (Task 5): fold this segment's FINAL corridor
            # membership (post-widening-ladder -- see the accumulator's own
            # comment above) into the whole-generation union edge repair will
            # draw from.
            corridor_segment_members_union.update(int(i) for i in last_segment_candidates)
            last_candidate_artist_keys = dict(attempt_result["last_candidate_artist_keys"])
            pool_cache = attempt_result.get("segment_pool_cache")
            last_segment_pool_cache = dict(pool_cache) if pool_cache is not None else None
            last_waypoint_stats = dict(attempt_result.get("last_waypoint_stats", {}))
            last_pool_diag = dict(attempt_result.get("last_pool_diag", {}))
            segment_edge_components = list(attempt_result.get("edge_components") or [])

        cfg = cfg_base
        segment_allow_detours = segment_allow_detours_base

        # If the segment pool itself is too small to fill the interior, the
        # relaxation tiers below cannot help: they relax beam SCORING floors
        # (transition_floor / genre_arc_floor), not pool ADMISSION, so they would
        # re-run the beam across the whole transition x genre-arc cross-product
        # against the same too-small pool — the "endless cascade" symptom on a
        # starved pool. Skip them and fall straight to the guaranteed-fill
        # fallback placement below.
        pool_too_small_for_segment = (
            segment_path is None
            and len(last_segment_candidates) < int(interior_len)
        )
        if pool_too_small_for_segment:
            logger.info(
                "Segment %d: pool too small (%d < interior_len %d) after bridge-floor "
                "backoff; floor-relaxation tiers cannot grow the pool — skipping to "
                "fallback placement",
                seg_idx, len(last_segment_candidates), int(interior_len),
            )

        # Wall-clock budget guard: an infeasible-but-nonempty pool makes the beam
        # fail without the floor-relaxation tiers being able to help, and the
        # transition x genre-arc cross-product below can then grind for minutes.
        # Once cumulative build time exceeds the budget, skip the relaxation tiers
        # for this and every later segment and use the fallback placement instead.
        # The shared generation deadline (passed from core.py) takes priority when
        # set; otherwise fall back to the legacy per-build anchor check.
        _now = time.monotonic()
        over_relaxation_budget = (
            _deadline_exceeded_at_segment_start
            or (deadline is not None and _now > deadline)
            or (_now - _pb_build_start) > _relax_budget_s
        )
        if segment_path is None and over_relaxation_budget and not pool_too_small_for_segment:
            logger.warning(
                "Segment %d: floor-relaxation budget (%.0fs) exceeded — skipping "
                "relaxation tiers, using fallback placement to stay within the "
                "generation time budget",
                seg_idx, _SEGMENT_RELAXATION_BUDGET_S,
            )

        # Phase 1 Task 8: the legacy transition-floor and genre-arc-floor
        # relaxation tiers were deleted. Both were already gated
        # `pooling_mode != "corridor"` (never-fired under corridor by
        # construction, per the "double-ladder" rationale in
        # _bridge_floor_attempts above) and are now provably unreachable
        # since pooling_mode is always "corridor". The corridor widening
        # ladder (_run_corridor_widening_ladder) remains the sole
        # segment-level recovery mechanism.

        micro_pier_diag: dict[str, Any] = {}
        if relaxation_enabled and bool(cfg.dj_relaxation_emit_warnings):
            warnings.append({
                "type": "dj_relaxation_attempts",
                "scope": "segment",
                "segment_index": int(seg_idx),
                "message": "DJ relaxation attempts executed.",
                "attempts": list(segment_relaxation_attempts),
                "success_attempt": relaxation_success_attempt,
            })

        if segment_path is None:
            # Skip the micro-pier attempt if the shared generation deadline has passed —
            # go directly to the guaranteed-fill term-pool fallback below.
            _micro_deadline_ok = (deadline is None or time.monotonic() <= deadline)
            if _micro_deadline_ok and _should_attempt_micro_pier(relaxation_enabled=relaxation_enabled, segment_path=segment_path):
                if bool(cfg.dj_micro_piers_enabled):
                    candidates = _micro_pier_candidate_pool(
                        cfg.dj_micro_piers_candidate_source,
                        last_segment_candidates,
                        last_segment_pool_cache,
                    )
                    micro_path = _attempt_micro_pier_split(
                        pier_a=pier_a,
                        pier_b=pier_b,
                        interior_length=interior_len,
                        candidates=candidates,
                        X_full=X_full_tr_norm,
                        X_full_norm=X_full_norm,
                        X_start=X_start_tr_norm,
                        X_mid=X_mid_tr_norm,
                        X_end=X_end_tr_norm,
                        X_genre_norm=X_genre_norm,
                        cfg=cfg_used_for_segment,
                        beam_width=int(beam_width_used),
                        artist_key_by_idx=last_candidate_artist_keys,
                        seed_artist_key=seed_artist_key,
                        recent_global_artists=_recent_artists_for_segment(seg_idx),
                        durations_ms=bundle.durations_ms,
                        artist_identity_cfg=artist_identity_cfg,
                        bundle=bundle,
                        warnings=warnings,
                        X_genre_vocab=getattr(bundle, "genre_vocab", None),
                        genre_graph=genre_graph,
                        micro_diag=micro_pier_diag,
                        X_genre_norm_idf=X_genre_norm_idf,
                        X_genre_raw=X_genre_raw,
                        X_genre_smoothed=X_genre_smoothed,
                        genre_idf=genre_idf,
                        transition_metric_context=transition_metric_context,
                        popularity_values=popularity_values,
                    )
                    if micro_path is not None and len(micro_path) == interior_len:
                        segment_path = micro_path
                        warnings.append({
                            "type": "micro_pier_fallback",
                            "scope": "segment",
                            "segment_index": int(seg_idx),
                            "message": "Inserted micro-pier due to infeasible bridge; consider lowering genre drift, increasing effort, or adding genre metadata.",
                        })
            elif bool(cfg.dj_bridging_enabled) and bool(cfg.dj_micro_piers_enabled) and segment_allow_detours:
                candidates = _micro_pier_candidate_pool(
                    cfg.dj_micro_piers_candidate_source,
                    last_segment_candidates,
                    last_segment_pool_cache,
                )
                micro_path = _attempt_micro_pier_split(
                    pier_a=pier_a,
                    pier_b=pier_b,
                    interior_length=interior_len,
                    candidates=candidates,
                    X_full=X_full_tr_norm,
                    X_full_norm=X_full_norm,
                    X_start=X_start_tr_norm,
                    X_mid=X_mid_tr_norm,
                    X_end=X_end_tr_norm,
                    X_genre_norm=X_genre_norm,
                    cfg=cfg_used_for_segment,
                    beam_width=int(beam_width_used),
                    artist_key_by_idx=last_candidate_artist_keys,
                    seed_artist_key=seed_artist_key,
                    recent_global_artists=_recent_artists_for_segment(seg_idx),
                    durations_ms=bundle.durations_ms,
                    artist_identity_cfg=artist_identity_cfg,
                    bundle=bundle,
                    warnings=warnings,
                    X_genre_vocab=getattr(bundle, "genre_vocab", None),
                    genre_graph=genre_graph,
                    micro_diag=micro_pier_diag,
                    X_genre_norm_idf=X_genre_norm_idf,
                    X_genre_raw=X_genre_raw,
                    X_genre_smoothed=X_genre_smoothed,
                    genre_idf=genre_idf,
                    transition_metric_context=transition_metric_context,
                    popularity_values=popularity_values,
                )
                if micro_path is not None and len(micro_path) == interior_len:
                    segment_path = micro_path

            if segment_path is None and infeasible_handling and infeasible_handling.guarantee_feasible:
                _term_pool = last_segment_candidates or list(universe)
                _greedy_genre_w = float(getattr(infeasible_handling, "greedy_genre_weight", 0.0))
                # Diversity is a hard constraint even in the last-resort fallback: pass
                # the identity-key resolver + the boundary artists so the terminal
                # placement cannot cluster same-artist tracks (e.g. 3 in a row).
                _greedy_blocked = set(_recent_artists_for_segment(seg_idx) or [])
                _greedy = _greedy_terminal_path(
                    _term_pool, global_used, int(pier_a), int(pier_b), int(interior_len), X_full_norm,
                    X_genre_norm=X_genre_norm, genre_weight=_greedy_genre_w,
                    artist_key_fn=_artist_keys_for_cap, blocked_artist_keys=_greedy_blocked,
                )
                if _greedy is None:
                    _greedy = _greedy_terminal_path(
                        list(universe), global_used, int(pier_a), int(pier_b), int(interior_len), X_full_norm,
                        X_genre_norm=X_genre_norm, genre_weight=_greedy_genre_w,
                        artist_key_fn=_artist_keys_for_cap, blocked_artist_keys=_greedy_blocked,
                    )
                if _greedy is not None:
                    segment_path = _greedy
                    warnings.append({
                        "type": "relaxation",
                        "scope": "segment",
                        "segment_index": int(seg_idx),
                        "bridge": f"{pier_a_id} -> {pier_b_id}",
                        "relaxed": ["all guideline gates (terminal greedy placement)"],
                        "severity": "invariant",
                    })

            if segment_path is None:
                if audit_enabled:
                    audit_events.append(
                        RunAuditEvent(
                            kind="segment_failure",
                            ts_utc=now_utc_iso(),
                            payload={
                                "segment_index": int(seg_idx),
                                "failure_reason": str(last_failure_reason or "segment infeasible"),
                                "attempted_bridge_floors": [float(x) for x in backoff_attempts],
                            },
                        )
                    )
                if infeasible_handling and infeasible_handling.enabled:
                    failure = f"Segment {seg_idx} infeasible under bridge_floor backoff (attempted={backoff_attempts}; last_reason={last_failure_reason})"
                else:
                    failure = f"Segment {seg_idx} infeasible under bridge_floor={cfg.bridge_floor}"
                logger.error(failure)
                return PierBridgeResult(
                    track_ids=[],
                    track_indices=[],
                    seed_positions=[],
                    segment_diagnostics=[],
                    stats={},
                    success=False,
                    failure_reason=failure,
                )

        soft_genre_penalty_hits_total += int(soft_genre_penalty_hits_segment)
        soft_genre_penalty_edges_scored_total += int(soft_genre_penalty_edges_scored_segment)
        local_sonic_penalty_hits_total += int(local_sonic_stats_segment.get("local_sonic_penalty_hits", 0))
        local_sonic_edges_scored_total += int(local_sonic_stats_segment.get("local_sonic_edges_scored", 0))
        local_sonic_gate_rejected_total += int(local_sonic_stats_segment.get("local_sonic_gate_rejected", 0))
        local_sonic_penalty_total += float(local_sonic_stats_segment.get("local_sonic_penalty_total", 0.0))
        if cfg.genre_penalty_strength > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Segment %d: soft_genre_penalty_hits=%d edges_scored=%d threshold=%.2f strength=%.2f",
                seg_idx,
                int(soft_genre_penalty_hits_segment),
                int(soft_genre_penalty_edges_scored_segment),
                float(cfg.genre_penalty_threshold),
                float(cfg.genre_penalty_strength),
            )
        if (
            (cfg.local_sonic_edge_penalty_enabled and cfg.local_sonic_edge_penalty_strength > 0)
            or cfg.local_sonic_edge_floor is not None
        ) and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Segment %d: local_sonic_penalty_hits=%d edges_scored=%d gate_rejected=%d threshold=%.2f strength=%.2f floor=%s",
                seg_idx,
                int(local_sonic_stats_segment.get("local_sonic_penalty_hits", 0)),
                int(local_sonic_stats_segment.get("local_sonic_edges_scored", 0)),
                int(local_sonic_stats_segment.get("local_sonic_gate_rejected", 0)),
                float(cfg.local_sonic_edge_penalty_threshold),
                float(cfg.local_sonic_edge_penalty_strength),
                (
                    f"{float(cfg.local_sonic_edge_floor):.2f}"
                    if cfg.local_sonic_edge_floor is not None
                    else "None"
                ),
            )

        # Tail-DP endgame (spec 2026-07-02): re-open the last min(2, interior)
        # slots of the just-finalized segment and exactly maximize the window
        # min-edge (T) over the segment's own candidate pool. Runs AFTER
        # var-bridge finalizes segment_path and BEFORE full_segment is
        # assembled below, so every downstream consumer (diagnostics, the
        # global_used commit, audits) sees the re-optimized path. Never-worse
        # by construction (tail_dp.optimize_segment_tail); never raises — any
        # internal error here is caught and logged, leaving segment_path as-is.
        #
        # Corridor reseat (Task 5, design spec §3: "candidate universe becomes
        # the segment's corridor, including widened material"): `td_candidates`
        # below is filtered from `last_segment_candidates`, which for
        # pooling_mode=="corridor" already IS this segment's FINAL (post-
        # widening-ladder) corridor membership -- the ladder's accepted
        # attempt sets it (see `attempt_result["last_segment_candidates"]`
        # unpacked above, and `_run_segment_backoff_attempts`'s own
        # `last_segment_candidates = list(segment_candidates)` assignment,
        # where `segment_candidates` came straight from
        # `_build_corridor_segment_pool`). No separate plumbing was needed:
        # this reseat was already correct as of Task 3/4's existing
        # `last_segment_candidates` chain -- pinned by a regression test
        # (test_corridor_pooling.py) rather than new code.
        if bool(getattr(cfg, "tail_dp_enabled", False)) and segment_path:
            tail_dp_attempted_segments += 1
            try:
                segment_path = list(segment_path)
                td_window = min(2, len(segment_path))
                td_kept_prefix = list(segment_path[:-td_window]) if td_window else list(segment_path)

                # Interior-banned artist identities (seed/pier artists), reusing
                # the same construction the edge-repair pass uses below.
                td_banned_artist_keys: Set[str] = set()
                if bool(cfg.disallow_seed_artist_in_interiors) or bool(cfg.disallow_pier_artists_in_interiors):
                    for td_sidx in seed_indices:
                        td_banned_artist_keys |= _artist_keys_for_cap(int(td_sidx))

                # Candidate prefilter: the segment's own pool, minus used tracks
                # (cross-segment), the current segment's kept-prefix tracks, the
                # piers, and interior-banned artists. The two tail tracks being
                # replaced are intentionally NOT excluded (a candidate equal to
                # the current occupant is a harmless no-op swap).
                td_exclude = set(int(i) for i in td_kept_prefix) | {int(pier_a), int(pier_b)}
                td_candidates = [
                    int(c) for c in last_segment_candidates
                    if int(c) not in global_used
                    and int(c) not in td_exclude
                    and not (_artist_keys_for_cap(int(c)) & td_banned_artist_keys)
                ]

                # "Assembled playlist so far" = all prior committed segments
                # (all_segments has not yet been appended-to for this segment)
                # + the kept prefix of this segment. Used for the trailing
                # min_gap recency window (a rolling window of the last
                # min_gap identity keys; same pattern the deleted
                # _enforce_min_gap_global backstop used).
                td_assembled_so_far: List[int] = []
                for td_ci, td_cseg in enumerate(all_segments):
                    if td_ci == 0:
                        td_assembled_so_far.extend(td_cseg)
                    else:
                        td_assembled_so_far.extend(td_cseg[1:])
                if not td_assembled_so_far or int(td_assembled_so_far[-1]) != int(pier_a):
                    td_assembled_so_far.append(int(pier_a))

                td_recent_source = td_assembled_so_far + td_kept_prefix
                td_recent_keys: Set[str] = set()
                if int(min_gap) > 0:
                    for td_ridx in td_recent_source[-int(min_gap):]:
                        td_recent_keys |= _artist_keys_for_cap(int(td_ridx))
                # Forward-pier gap enforcement (choke point, 91e7d91): every
                # other placement mechanism -- the corridor pool's artist
                # gate, the beam's recent_global_artists, micro-pier, and the
                # terminal-greedy fallback -- reads blocked artists through
                # _recent_artists_for_segment(seg_idx), which folds in both
                # the backward boundary window AND upcoming (not-yet-reached)
                # pier/waypoint artists within min_gap. Tail-DP re-optimizes
                # AFTER those mechanisms run and built td_recent_keys from
                # only its own (more precise, since it includes this
                # segment's own kept prefix) backward window -- missing the
                # forward half entirely, so a swap could still land a
                # same-artist track within min_gap of an UPCOMING pier. Union
                # in the choke point's view rather than replace: it can only
                # ADD candidates to the blocked set, so is_allowed_pair can
                # only refuse a swap it previously would have allowed, never
                # force one that wasn't already going to be picked
                # (never-worse semantics are preserved -- optimize_segment_tail
                # falls back to the beam's own (already forward-safe) choice
                # whenever no allowed candidate beats it).
                td_recent_keys |= set(_recent_artists_for_segment(seg_idx) or [])

                def _tail_dp_is_allowed_pair(x: int, y: int) -> bool:
                    try:
                        x = int(x)
                        y = int(y)
                        two_slot = td_window == 2
                        if two_slot and x == y:
                            return False
                        x_keys = _artist_keys_for_cap(x)
                        y_keys = _artist_keys_for_cap(y) if two_slot else set()
                        if two_slot and int(min_gap) > 0 and (x_keys & y_keys):
                            return False
                        if int(min_gap) > 0:
                            if x_keys & td_recent_keys:
                                return False
                            if two_slot and (y_keys & td_recent_keys):
                                return False
                        return True
                    except Exception:
                        return False

                # Relative trigger (Phase 2 Task 2): the effective floor is
                # max(tail_dp_floor, segment_mean_T - tail_dp_relative_epsilon),
                # so a landing that clears the absolute floor but sits well
                # below its OWN segment's realized level still re-opens the
                # search (docs/corridor_baseline/phase2_mechanism_probes.md --
                # PC seg 4's 0.394 cleared tail_dp_floor=0.3 with an unused
                # ~0.7-0.8-class connector in-pool). segment_mean_T reads the
                # SAME currency (score_transition_edge's "T") as old_min inside
                # optimize_segment_tail, computed over the pre-swap segment
                # path so the reference isn't polluted by the swap it may gate.
                _td_full_segment_pre = [int(pier_a)] + [int(i) for i in segment_path] + [int(pier_b)]
                _, td_segment_mean_T = _compute_edge_scores(
                    _td_full_segment_pre, X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm,
                    X_end_tr_norm, cfg, metric_context=transition_metric_context,
                )
                td_trigger = compute_relative_trigger_floor(
                    base_floor=float(getattr(cfg, "tail_dp_floor", 0.30)),
                    reference_mean=float(td_segment_mean_T),
                    relative_epsilon=float(getattr(cfg, "tail_dp_relative_epsilon", 0.25)),
                )
                logger.debug(
                    "Tail-DP seg %d: trigger floor=%.3f (source=%s, base=%.2f, "
                    "relative_threshold=%.3f, segment_mean_T=%.3f)",
                    seg_idx, td_trigger.effective_floor, td_trigger.source,
                    float(getattr(cfg, "tail_dp_floor", 0.30)), td_trigger.relative_threshold,
                    float(td_segment_mean_T),
                )

                td_swap = optimize_segment_tail(
                    transition_metric_context,
                    segment_path=segment_path,
                    pier_a=int(pier_a),
                    pier_b=int(pier_b),
                    candidates=td_candidates,
                    epsilon=float(getattr(cfg, "tail_dp_epsilon", 0.02)),
                    is_allowed_pair=_tail_dp_is_allowed_pair,
                    floor=float(td_trigger.effective_floor),
                )
                if td_swap is not None:
                    td_old_tail = tuple(int(i) for i in segment_path[-td_window:])
                    segment_path[-td_window:] = list(td_swap.new_tail)
                    tail_dp_applied_segments += 1
                    try:
                        td_old_ids = [str(bundle.track_ids[i]) for i in td_old_tail]
                        td_new_ids = [str(bundle.track_ids[i]) for i in td_swap.new_tail]
                    except Exception:
                        td_old_ids = [str(i) for i in td_old_tail]
                        td_new_ids = [str(i) for i in td_swap.new_tail]
                    logger.info(
                        "Tail-DP seg %d: window min %.3f -> %.3f (swapped [%s] -> [%s]) "
                        "[trigger=%s floor=%.3f]",
                        seg_idx,
                        float(td_swap.old_min),
                        float(td_swap.new_min),
                        ", ".join(td_old_ids),
                        ", ".join(td_new_ids),
                        td_trigger.source,
                        float(td_trigger.effective_floor),
                    )
            except Exception:
                logger.warning(
                    "Tail-DP: internal error on segment %d; keeping original segment tail",
                    seg_idx,
                    exc_info=True,
                )

        # Compute edge scores for diagnostics
        full_segment = [pier_a] + segment_path + [pier_b]
        worst_edge, mean_edge = _compute_edge_scores(
            full_segment,
            X_full_tr_norm,
            X_start_tr_norm,
            X_mid_tr_norm,
            X_end_tr_norm,
            cfg,
            metric_context=transition_metric_context,
        )
        micro_pier_used = "micro_pier_index" in micro_pier_diag
        micro_pier_track_id = None
        if micro_pier_used:
            try:
                micro_pier_track_id = str(
                    bundle.track_ids[int(micro_pier_diag["micro_pier_index"])]
                )
            except Exception:
                micro_pier_track_id = str(micro_pier_diag.get("micro_pier_index"))

        # Record diagnostics
        diagnostics.append(SegmentDiagnostics(
            pier_a_id=pier_a_id,
            pier_b_id=pier_b_id,
            target_length=interior_len,
            actual_length=len(segment_path),
            pool_size_initial=pool_size_initial,
            pool_size_final=pool_size_final,
            expansions=expansions,
            beam_width_used=beam_width_used,
            worst_edge_score=worst_edge,
            mean_edge_score=mean_edge,
            success=segment_path is not None and len(segment_path) == interior_len,
            bridge_floor_used=float(chosen_bridge_floor),
            backoff_attempts_used=int(backoff_used_count),
            widened_search=bool(widened_search_used),
            route_shape=str(segment_ladder_diag.get("route_shape", str(cfg.dj_route_shape or "linear"))),
            ladder_waypoint_labels=list(segment_ladder_diag.get("ladder_waypoint_labels", [])),
            ladder_waypoint_count=int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
            ladder_waypoint_vector_mode=str(segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")),
            ladder_waypoint_vector_stats=list(segment_ladder_diag.get("ladder_waypoint_vector_stats", [])),
            relaxation_attempts=list(segment_relaxation_attempts),
            relaxation_success_attempt=relaxation_success_attempt,
            micro_pier_used=bool(micro_pier_used),
            micro_pier_track_id=micro_pier_track_id,
            micro_pier_metric_value=(
                float(micro_pier_diag.get("micro_pier_metric_value"))
                if micro_pier_diag.get("micro_pier_metric_value") is not None
                else None
            ),
            micro_pier_left_success=micro_pier_diag.get("left_success"),
            micro_pier_right_success=micro_pier_diag.get("right_success"),
            waypoint_enabled=bool(last_waypoint_stats.get("waypoint_enabled", False)),
            mean_waypoint_sim=last_waypoint_stats.get("mean_waypoint_sim"),
            p50_waypoint_sim=last_waypoint_stats.get("p50_waypoint_sim"),
            p90_waypoint_sim=last_waypoint_stats.get("p90_waypoint_sim"),
            min_waypoint_sim=last_waypoint_stats.get("min_waypoint_sim"),
            max_waypoint_sim=last_waypoint_stats.get("max_waypoint_sim"),
            waypoint_delta_applied_count=int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
            mean_waypoint_delta=last_waypoint_stats.get("mean_waypoint_delta"),
        ))
        segment_bridge_floors_used.append(float(chosen_bridge_floor))
        segment_backoff_attempts_used.append(int(backoff_used_count))
        logger.info(
            "Segment %d: %s -> %s bridge_floor=%.2f pool_before=%d pool_after=%d",
            seg_idx, pier_a_id, pier_b_id, float(chosen_bridge_floor), pool_size_initial, pool_size_final,
        )

        # Log waypoint influence stats (Phase 2 diagnostics)
        if last_waypoint_stats.get("waypoint_enabled"):
            logger.info(
                "  Waypoint stats: enabled=True mean_sim=%.3f p50=%.3f p90=%.3f delta_applied=%d/%d mean_delta=%.4f",
                float(last_waypoint_stats.get("mean_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p50_waypoint_sim", 0.0)),
                float(last_waypoint_stats.get("p90_waypoint_sim", 0.0)),
                int(last_waypoint_stats.get("waypoint_delta_applied_count", 0)),
                interior_len,
                float(last_waypoint_stats.get("mean_waypoint_delta", 0.0)),
            )

            # Log rank impact metrics (TASK B: opt-in diagnostic)
            rank_impact_results = last_waypoint_stats.get("rank_impact_results", [])
            if rank_impact_results:
                sampled_steps_count = len(rank_impact_results)
                winner_changed_count = sum(1 for r in rank_impact_results if r.get("winner_changed"))
                mean_reordered = float(np.mean([r.get("topK_reordered_count", 0) for r in rank_impact_results]))
                mean_topK = float(np.mean([r.get("topK", 10) for r in rank_impact_results]))
                mean_rank_delta = float(np.mean([r.get("mean_abs_rank_delta", 0.0) for r in rank_impact_results]))

                logger.info(
                    "  Waypoint rank impact: sampled_steps=%d winner_changed=%d/%d topK_reordered=%.1f/%.0f mean_rank_delta=%.1f",
                    sampled_steps_count,
                    winner_changed_count,
                    sampled_steps_count,
                    mean_reordered,
                    mean_topK,
                    mean_rank_delta,
                )

                # Task E: Saturation diagnostics
                mean_sim0 = float(np.mean([r.get("waypoint_sim0", 0.0) for r in rank_impact_results]))
                mean_delta_mean = float(np.mean([r.get("waypoint_delta_mean", 0.0) for r in rank_impact_results]))
                mean_delta_p50 = float(np.mean([r.get("waypoint_delta_p50", 0.0) for r in rank_impact_results]))
                mean_delta_p90 = float(np.mean([r.get("waypoint_delta_p90", 0.0) for r in rank_impact_results]))
                mean_frac_near_cap = float(np.mean([r.get("waypoint_frac_near_cap", 0.0) for r in rank_impact_results]))
                mean_frac_at_cap = float(np.mean([r.get("waypoint_frac_at_cap", 0.0) for r in rank_impact_results]))

                logger.info(
                    "  Waypoint saturation: sim0=%.3f delta(mean=%.4f p50=%.4f p90=%.4f) near_cap=%.1f%% at_cap=%.1f%%",
                    mean_sim0,
                    mean_delta_mean,
                    mean_delta_p50,
                    mean_delta_p90,
                    100.0 * mean_frac_near_cap,
                    100.0 * mean_frac_at_cap,
                )

                # Phase 2: Coverage bonus impact (compare base+waypoint vs full)
                if bool(cfg.dj_genre_use_coverage):
                    coverage_winner_changed_count = 0
                    coverage_mean_bonus = []
                    for r in rank_impact_results:
                        top10_table = r.get("top10_table", [])
                        if top10_table:
                            # Find winner by base+waypoint score (before coverage)
                            base_waypoint_scores = [(entry["cand_idx"], entry["base_score"] + entry["waypoint_delta"])
                                                     for entry in top10_table]
                            base_waypoint_winner = max(base_waypoint_scores, key=lambda x: x[1])[0]
                            # Find winner by full score (after coverage)
                            full_winner = top10_table[0]["cand_idx"]  # Already sorted by base_rank
                            # Actually need to re-sort by full_score to get true full_winner
                            full_scores = [(entry["cand_idx"], entry["full_score"]) for entry in top10_table]
                            full_winner = max(full_scores, key=lambda x: x[1])[0]

                            if base_waypoint_winner != full_winner:
                                coverage_winner_changed_count += 1

                            # Collect mean coverage bonus for this step
                            coverage_bonuses = [entry["coverage_bonus"] for entry in top10_table]
                            coverage_mean_bonus.append(float(np.mean(coverage_bonuses)))

                    if coverage_mean_bonus:
                        logger.info(
                            "  Coverage bonus impact: winner_changed=%d/%d mean_bonus=%.4f",
                            coverage_winner_changed_count,
                            sampled_steps_count,
                            float(np.mean(coverage_mean_bonus)),
                        )

        # Log chosen edge provenance (dj_union) - TASK A: renamed from "Pool sources"
        if last_pool_diag:
            if "chosen_from_local_count" in last_pool_diag or "dj_pool_strategy" in last_pool_diag:
                # Legacy exclusive counts (priority-based)
                logger.info(
                    "  Chosen edge provenance (exclusive): strategy=%s local=%d toward=%d genre=%d baseline_only=%d",
                    str(last_pool_diag.get("dj_pool_strategy", last_pool_diag.get("pool_strategy", "unknown"))),
                    int(last_pool_diag.get("chosen_from_local_count", 0)),
                    int(last_pool_diag.get("chosen_from_toward_count", 0)),
                    int(last_pool_diag.get("chosen_from_genre_count", 0)),
                    int(last_pool_diag.get("chosen_from_baseline_only_count", 0)),
                )
                # Phase 3: Membership-based counts (all overlaps)
                if "local_only" in last_pool_diag:
                    logger.info(
                        "  Provenance memberships (Phase3): local_only=%d toward_only=%d genre_only=%d " +
                        "local+toward=%d local+genre=%d toward+genre=%d local+toward+genre=%d baseline_only=%d",
                        int(last_pool_diag.get("local_only", 0)),
                        int(last_pool_diag.get("toward_only", 0)),
                        int(last_pool_diag.get("genre_only", 0)),
                        int(last_pool_diag.get("local+toward", 0)),
                        int(last_pool_diag.get("local+genre", 0)),
                        int(last_pool_diag.get("toward+genre", 0)),
                        int(last_pool_diag.get("local+toward+genre", 0)),
                        int(last_pool_diag.get("baseline_only", 0)),
                    )

        # TASK A: Invariant checks (log WARNINGs for inconsistencies)
        if pool_size_final > 0 and pool_size_initial == 0:
            logger.warning(
                "  WARNING: pool_before_gating=0 but pool_after_gating=%d (possible missing instrumentation)",
                pool_size_final
            )

        if last_pool_diag and "chosen_from_local_count" in last_pool_diag:
            chosen_sum = (
                int(last_pool_diag.get("chosen_from_local_count", 0)) +
                int(last_pool_diag.get("chosen_from_toward_count", 0)) +
                int(last_pool_diag.get("chosen_from_genre_count", 0)) +
                int(last_pool_diag.get("chosen_from_baseline_only_count", 0))
            )
            if chosen_sum != interior_len:
                logger.warning(
                    "  WARNING: chosen_from_* sum (%d) != interior_length (%d) (possible provenance tracking gap)",
                    chosen_sum,
                    interior_len
                )

        # Log ladder waypoint labels (route planning)
        if segment_ladder_diag and segment_ladder_diag.get("ladder_waypoint_count", 0) > 0:
            labels = segment_ladder_diag.get("ladder_waypoint_labels", [])
            mode = segment_ladder_diag.get("ladder_waypoint_vector_mode", "onehot")
            logger.info(
                "  Ladder route: shape=%s mode=%s waypoints=%d labels=%s",
                str(segment_ladder_diag.get("route_shape", "linear")),
                mode,
                int(segment_ladder_diag.get("ladder_waypoint_count", 0)),
                ", ".join(labels[:6]) if labels else "none",
            )
        # DEBUG top candidates for this segment
        if logger.isEnabledFor(logging.DEBUG):
            scores_dbg = []
            sim_to_a = np.dot(X_full_norm, X_full_norm[pier_a])
            sim_to_b = np.dot(X_full_norm, X_full_norm[pier_b])
            for cand in last_segment_candidates[: min(200, len(last_segment_candidates))]:
                sim_a = float(sim_to_a[cand])
                sim_b = float(sim_to_b[cand])
                denom = sim_a + sim_b
                hmean = 0.0 if denom <= 1e-9 else (2 * sim_a * sim_b) / denom
                trans = _compute_transition_score(
                    cand,
                    pier_b,
                    X_full_tr_norm,
                    X_start_tr_norm,
                    X_mid_tr_norm,
                    X_end_tr_norm,
                    cfg,
                )
                final_score = cfg.weight_bridge * hmean + cfg.weight_transition * trans
                scores_dbg.append((final_score, sim_a, sim_b, hmean, trans, cand))
            scores_dbg = sorted(scores_dbg, key=lambda t: t[0], reverse=True)[:10]
            dbg_rows = []
            for final_score, sim_a, sim_b, hmean, trans, cand in scores_dbg:
                keys = identity_keys_for_index(bundle, int(cand))
                artist = (
                    str(bundle.track_artists[cand])
                    if bundle.track_artists is not None
                    else (str(bundle.artist_keys[cand]) if bundle.artist_keys is not None else "")
                )
                title = str(bundle.track_titles[cand]) if bundle.track_titles is not None else ""
                dbg_rows.append({
                    "track_id": str(bundle.track_ids[cand]),
                    "artist": sanitize_for_logging(artist),
                    "title": sanitize_for_logging(title),
                    "artist_key": keys.artist_key,
                    "title_key": keys.title_key,
                    "simA": round(sim_a, 3),
                    "simB": round(sim_b, 3),
                    "hmean": round(hmean, 3),
                    "transition": round(trans, 3),
                    "final": round(final_score, 3),
                    "internal": bool(internal_connector_indices and cand in internal_connector_indices),
                })
            logger.debug("Segment %d top candidates: %s", seg_idx, dbg_rows)

        # Commit segment path to used set
        for idx in segment_path:
            global_used.add(idx)
            try:
                used_track_keys.add(identity_keys_for_index(bundle, int(idx)).track_key)
            except Exception:
                pass
            for artist_key in _artist_keys_for_cap(int(idx)):
                global_non_seed_artist_counts[artist_key] = (
                    global_non_seed_artist_counts.get(artist_key, 0) + 1
                )

        all_segments.append(full_segment)
        all_beam_components.extend(segment_edge_components)

        # Update boundary context for next segment (cross-segment min_gap enforcement)
        # Build the concatenated result so far to extract recent artists
        current_concat: List[int] = []
        for concat_seg_idx, concat_seg in enumerate(all_segments):
            if concat_seg_idx == 0:
                current_concat.extend(concat_seg)
            else:
                current_concat.extend(concat_seg[1:])  # Drop duplicate pier

        # Extract artist keys from the last MIN_GAP_GLOBAL positions
        # If artist_identity_cfg is enabled, resolve to identity keys (collapsing ensemble variants)
        recent_boundary_artists = []
        start_pos = max(0, len(current_concat) - MIN_GAP_GLOBAL)
        use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled

        for pos in range(start_pos, len(current_concat)):
            try:
                if use_identity:
                    # Identity mode: use raw artist string to preserve collaborations
                    artist_str = ""
                    if bundle is not None and bundle.track_artists is not None:
                        try:
                            artist_str = str(bundle.track_artists[int(current_concat[pos])] or "")
                        except Exception:
                            pass
                    if artist_str:
                        identity_keys_set = resolve_artist_identity_keys(artist_str, artist_identity_cfg)
                        # Add ALL identity keys to boundary tracking
                        for identity_key in identity_keys_set:
                            recent_boundary_artists.append(identity_key)
                else:
                    # Legacy mode: single artist_key
                    artist_key = identity_keys_for_index(bundle, int(current_concat[pos])).artist_key
                    if artist_key:
                        recent_boundary_artists.append(str(artist_key))
            except Exception:
                continue

    if bool(getattr(cfg, "tail_dp_enabled", False)):
        logger.info(
            "Tail-DP summary: applied=%d/%d segments",
            int(tail_dp_applied_segments),
            int(tail_dp_attempted_segments),
        )

    # Concatenate segments
    # First segment: keep full [A, ..., B]
    # Subsequent segments: drop first element (the pier) to avoid duplication
    # Single-seed arc: drop last element (the duplicated seed) to avoid repetition
    final_indices: List[int] = []
    seed_positions: List[int] = []

    if is_single_seed_arc:
        # Single-seed arc: segment is [seed, interior..., seed]
        # Output only [seed, interior...] to avoid duplicate seed at end
        segment = all_segments[0] if all_segments else [ordered_seeds[0]]
        final_indices = segment[:-1]  # Drop the trailing duplicate seed
        seed_positions = [0]  # Seed is at position 0
        logger.info("Pier+Bridge: single-seed arc output: %d tracks (seed at start, arc returns to seed-similar)", len(final_indices))
    else:
        for seg_idx, segment in enumerate(all_segments):
            if seg_idx == 0:
                final_indices.extend(segment)
                seed_positions.append(0)  # First pier
                seed_positions.append(len(final_indices) - 1)  # Second pier
            else:
                # Drop first element (the pier, already included)
                final_indices.extend(segment[1:])
                seed_positions.append(len(final_indices) - 1)  # New pier

    edge_repair_swap_log: list[dict[str, Any]] = []
    if bool(getattr(cfg, "edge_repair_enabled", False)):
        from src.playlist.repair.edge_repair import repair_playlist_edges

        disallowed_repair_artist_keys: set[str] = set()
        if bool(cfg.disallow_seed_artist_in_interiors) or bool(cfg.disallow_pier_artists_in_interiors):
            for sidx in seed_indices:
                try:
                    disallowed_repair_artist_keys.add(
                        identity_keys_for_index(bundle, int(sidx)).artist_key
                    )
                except Exception:
                    continue
        # Edge repair (Task 5 reseat, design spec §3: "candidate universe
        # becomes the segment's corridor, including widened material"):
        # repair_playlist_edges takes ONE candidate_indices for the entire
        # pass (no per-edge scoping in repair/edge_repair.py, confirmed by
        # reading its signature -- read-only, not modified by this task), so
        # the spec-sanctioned substitute for "the corridor of the segment
        # containing each repaired edge" is the UNION of every segment's
        # final corridor members (see corridor_segment_members_union's own
        # comment above).
        _repair_candidate_indices = sorted(corridor_segment_members_union)

        # Relative trigger (Phase 2 Task 2): the effective floor is
        # max(edge_repair_t_floor, playlist_mean_T - edge_repair_relative_epsilon),
        # computed ONCE over the pre-repair playlist so an edge that clears the
        # absolute floor but sits well below the playlist's own achievable
        # level is still repair-worthy (docs/corridor_baseline/
        # phase2_mechanism_probes.md -- SADE/home's 0.454 first-edge cleared
        # edge_repair_t_floor=0.3 with an unused 0.697-class connector already
        # admitted). playlist_mean_T reads the same "T" currency _needs_repair
        # compares against, via the same score_transition_edge path.
        # Intentionally non-progressive: computed once over the pre-repair
        # playlist, not recomputed after each swap inside the pass below --
        # a moving target would risk the trigger floor drifting mid-pass
        # (non-determinism) and could churn indefinitely as swaps keep
        # nudging the mean (no guaranteed termination).
        _, _er_playlist_mean_T = _compute_edge_scores(
            list(final_indices), X_full_tr_norm, X_start_tr_norm, X_mid_tr_norm,
            X_end_tr_norm, cfg, metric_context=transition_metric_context,
        )
        er_trigger = compute_relative_trigger_floor(
            base_floor=float(getattr(cfg, "edge_repair_t_floor", 0.30)),
            reference_mean=float(_er_playlist_mean_T),
            relative_epsilon=float(getattr(cfg, "edge_repair_relative_epsilon", 0.25)),
        )
        logger.debug(
            "Edge repair: trigger floor=%.3f (source=%s, base=%.2f, "
            "relative_threshold=%.3f, playlist_mean_T=%.3f)",
            er_trigger.effective_floor, er_trigger.source,
            float(getattr(cfg, "edge_repair_t_floor", 0.30)), er_trigger.relative_threshold,
            float(_er_playlist_mean_T),
        )

        repair_result = repair_playlist_edges(
            final_indices=final_indices,
            candidate_indices=_repair_candidate_indices,
            metric_context=transition_metric_context,
            bundle=bundle,
            seed_indices=seed_indices,
            pier_positions=seed_positions,
            transition_floor=float(cfg.transition_floor),
            centered_cos_floor=float(getattr(cfg, "edge_repair_centered_cos_floor", -0.5)),
            margin=float(getattr(cfg, "edge_repair_margin", 0.05)),
            allowed_indices=allowed_set_indices,
            disallowed_artist_keys=disallowed_repair_artist_keys,
            variety_guard_enabled=bool(getattr(cfg, "edge_repair_variety_guard_enabled", False)),
            variety_guard_threshold=float(getattr(cfg, "edge_repair_variety_guard_threshold", 0.85)),
            max_non_seed_tracks_per_artist=getattr(cfg, "max_non_seed_tracks_per_artist", None),
            artist_identity_cfg=artist_identity_cfg,
            t_floor=float(er_trigger.effective_floor),
            min_gap=int(min_gap),
            trigger_floor_source=er_trigger.source,
            relative_threshold=float(er_trigger.relative_threshold),
        )
        edge_repair_swap_log = list(repair_result.swap_log)
        if list(repair_result.indices) != list(final_indices):
            final_indices = list(repair_result.indices)
            all_beam_components = []

    # Remove-only last resort (repair-by-deletion): runs AFTER break-glass
    # repair, since repair is strictly non-destructive to length and should
    # get first shot at a broken edge. Only removes an interior track that is
    # NOT a pier/seed, and only when the deletion strictly improves on the
    # still-broken edge (never-worse). See
    # docs/superpowers/plans/2026-07-02-weak-edge-cascade-reorder.md.
    edge_delete_log: list[dict[str, Any]] = []
    if bool(getattr(cfg, "edge_delete_enabled", True)) and len(final_indices) >= 3:
        from src.playlist.repair.edge_delete import delete_broken_edges

        _edge_delete_protected = {int(s) for s in seed_indices}

        def _edge_delete_score(a: int, b: int) -> float:
            _edge = score_transition_edge(transition_metric_context, int(a), int(b))
            _t = _edge.get("T")
            return float(_t) if isinstance(_t, (int, float)) else 0.0

        delete_result = delete_broken_edges(
            final_indices,
            edge_score=_edge_delete_score,
            floor=float(getattr(cfg, "edge_delete_floor", 0.30)),
            protected_indices=_edge_delete_protected,
            max_deletions=int(getattr(cfg, "edge_delete_max_deletions", 4)),
            artist_key_of=lambda i: bundle.artist_keys[int(i)],
            min_gap=int(min_gap),
        )
        edge_delete_log = list(delete_result.delete_log)
        if list(delete_result.indices) != list(final_indices):
            final_indices = list(delete_result.indices)
            all_beam_components = []

    # Convert to track IDs
    # Cross-segment min_gap is enforced DURING generation (boundary-aware beam search),
    # not as a post-order filter.
    final_track_ids = [str(bundle.track_ids[i]) for i in final_indices]

    # Soft length check: variable-bridge mode intentionally produces totals in a band
    # [N-m, N+m], so an exact-N guarantee is no longer enforced here.  A mismatch on
    # the default (feature-off) path is unexpected and worth a warning, but generation
    # proceeds with whatever length was produced.
    if len(final_track_ids) != total_tracks:
        logger.warning(
            "Pier-bridge length mismatch: generated %d tracks but expected %d "
            "(variable_bridge_length=%s). Proceeding with actual length.",
            len(final_track_ids),
            total_tracks,
            getattr(cfg, "variable_bridge_length", False),
        )

    # Compute per-edge transition scores for reporting (matches builder scoring)
    edge_scores: list[dict[str, Any]] = []
    transition_vals: list[float] = []
    for i in range(1, len(final_indices)):
        prev_idx = final_indices[i - 1]
        cur_idx = final_indices[i]
        edge = score_transition_edge(transition_metric_context, prev_idx, cur_idx)
        edge["floor"] = float(cfg.transition_floor)
        t_val = float(edge.get("T"))
        transition_vals.append(float(t_val))
        # BPM columns (diagnostic only)
        bpm_a: Optional[float] = None
        bpm_b: Optional[float] = None
        bpm_log_dist: Optional[float] = None
        if perceptual_bpm is not None:
            _ba = perceptual_bpm[prev_idx]
            _bb = perceptual_bpm[cur_idx]
            if not np.isnan(_ba):
                bpm_a = float(_ba)
            if not np.isnan(_bb):
                bpm_b = float(_bb)
            if bpm_a is not None and bpm_b is not None:
                from src.playlist.bpm_axis import bpm_log_distance
                bpm_log_dist = float(bpm_log_distance(bpm_a, bpm_b))
        edge_scores.append(
            {
                "prev_id": str(bundle.track_ids[prev_idx]),
                "cur_id": str(bundle.track_ids[cur_idx]),
                "prev_idx": int(prev_idx),
                "cur_idx": int(cur_idx),
                **edge,
                "bpm_a": bpm_a,
                "bpm_b": bpm_b,
                "bpm_log_dist": bpm_log_dist,
            }
        )

    # Recompute seed positions from final track IDs for diagnostic accuracy.
    # Expected count is _expected_seed_pier_count (artist piers + placed
    # anchors, excluding anchors gap-insertion legitimately dropped), not the
    # pre-drop seed_id_set/seed_indices -- see the review-fix comment at its
    # assignment above.
    seed_positions = [idx for idx, tid in enumerate(final_track_ids) if tid in seed_id_set]
    if len(seed_positions) != _expected_seed_pier_count:
        logger.warning(
            "Pier+Bridge: seed count mismatch in final result (expected %d, found %d)",
            _expected_seed_pier_count,
            len(seed_positions),
        )

    # Compute overall stats
    actual_num_seeds = _expected_seed_pier_count
    seed_index_set = set(int(i) for i in seed_indices)
    artist_counts: Dict[str, int] = {}
    non_seed_artist_counts: Dict[str, int] = {}
    for idx in final_indices:
        artist_key = ""
        try:
            artist_key = identity_keys_for_index(bundle, int(idx)).artist_key
        except Exception:
            artist_key = ""
        if not artist_key:
            continue
        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        if int(idx) not in seed_index_set:
            non_seed_artist_counts[artist_key] = non_seed_artist_counts.get(artist_key, 0) + 1

    # BPM per-playlist summary (diagnostic)
    _bpm_summary: Optional[dict] = None
    if perceptual_bpm is not None:
        _bpms = [float(perceptual_bpm[i]) for i in final_indices if not np.isnan(perceptual_bpm[i])]
        if _bpms:
            _bpm_summary = {
                "min": float(min(_bpms)),
                "mean": float(sum(_bpms) / len(_bpms)),
                "max": float(max(_bpms)),
                "std": float(np.std(_bpms)),
                "n": len(_bpms),
                "total": len(final_indices),
            }
            logger.info(
                "BPM (perceptual): min=%.0f mean=%.0f max=%.0f std=%.0f (%d/%d tracks have data)",
                _bpm_summary["min"],
                _bpm_summary["mean"],
                _bpm_summary["max"],
                _bpm_summary["std"],
                _bpm_summary["n"],
                _bpm_summary["total"],
            )

    stats = {
        "num_seeds": actual_num_seeds,
        "single_seed_arc": is_single_seed_arc,
        "target_tracks": total_tracks,
        "actual_tracks": len(final_indices),
        "artist_counts": artist_counts,
        "non_seed_artist_counts": non_seed_artist_counts,
        "max_non_seed_tracks_per_artist": cfg.max_non_seed_tracks_per_artist,
        "universe_size": len(universe),
        # Phase 1 Task 3 corridor pooling diagnostics (empty/None on the legacy
        # path -- see PierBridgeConfig.pooling). Summary stats ONLY, never
        # member-index lists (worker NDJSON line-size trap).
        "pooling_strategy": pooling_mode,
        "corridor_segments": list(corridor_segments_diag),
        "corridor_universe_size": (
            int(len(corridor_universe.indices)) if corridor_universe is not None else None
        ),
        "segments_built": len(all_segments),
        "segments_successful": sum(1 for d in diagnostics if d.success),
        "total_expansions": sum(d.expansions for d in diagnostics),
        "edge_scores": edge_scores,
        "bpm_summary": _bpm_summary,
        "min_transition": float(np.min(transition_vals)) if transition_vals else None,
        "mean_transition": float(np.mean(transition_vals)) if transition_vals else None,
        "transition_centered": bool(cfg.center_transitions),
        "soft_genre_penalty_hits": int(soft_genre_penalty_hits_total),
        "soft_genre_penalty_edges_scored": int(soft_genre_penalty_edges_scored_total),
        "local_sonic_penalty_hits": int(local_sonic_penalty_hits_total),
        "local_sonic_edges_scored": int(local_sonic_edges_scored_total),
        "local_sonic_gate_rejected": int(local_sonic_gate_rejected_total),
        "local_sonic_penalty_total": float(local_sonic_penalty_total),
        "segment_bridge_floors_used": [float(x) for x in segment_bridge_floors_used],
        "segment_backoff_attempts_used": [int(x) for x in segment_backoff_attempts_used],
        "beam_edge_components": list(all_beam_components),
        "edge_repair_enabled": bool(getattr(cfg, "edge_repair_enabled", False)),
        "edge_repair_applied": bool(
            any(isinstance(entry, dict) and "new_idx" in entry for entry in edge_repair_swap_log)
        ),
        "edge_repair_swap_log": edge_repair_swap_log,
        "edge_delete_enabled": bool(getattr(cfg, "edge_delete_enabled", True)),
        "edge_delete_applied": bool(edge_delete_log),
        "edge_delete_log": edge_delete_log,
        "warnings": warnings,
        "config": {
            "transition_floor": cfg.transition_floor,
            "edge_repair_enabled": bool(cfg.edge_repair_enabled),
            "edge_repair_centered_cos_floor": float(cfg.edge_repair_centered_cos_floor),
            "edge_repair_margin": float(cfg.edge_repair_margin),
            "edge_repair_variety_guard_enabled": bool(cfg.edge_repair_variety_guard_enabled),
            "edge_repair_variety_guard_threshold": float(cfg.edge_repair_variety_guard_threshold),
            "initial_neighbors_m": cfg.initial_neighbors_m,
            "initial_beam_width": cfg.initial_beam_width,
            "eta_destination_pull": cfg.eta_destination_pull,
            "genre_tiebreak_weight": float(cfg.genre_tiebreak_weight),
            "genre_penalty_threshold": float(cfg.genre_penalty_threshold),
            "genre_penalty_strength": float(cfg.genre_penalty_strength),
            "genre_tie_break_band": (
                float(cfg.genre_tie_break_band) if cfg.genre_tie_break_band is not None else None
            ),
            "local_sonic_edge_penalty_enabled": bool(cfg.local_sonic_edge_penalty_enabled),
            "local_sonic_edge_penalty_threshold": float(cfg.local_sonic_edge_penalty_threshold),
            "local_sonic_edge_penalty_strength": float(cfg.local_sonic_edge_penalty_strength),
            "local_sonic_edge_floor": (
                float(cfg.local_sonic_edge_floor)
                if cfg.local_sonic_edge_floor is not None
                else None
            ),
            "bridge_floor": float(cfg.bridge_floor),
            "max_non_seed_tracks_per_artist": cfg.max_non_seed_tracks_per_artist,
            "infeasible_handling_enabled": bool(infeasible_handling and infeasible_handling.enabled),
            "experiment_bridge_scoring": {
                "enabled": bool(cfg.experiment_bridge_scoring_enabled),
                "min_weight": float(cfg.experiment_bridge_min_weight),
                "balance_weight": float(cfg.experiment_bridge_balance_weight),
            },
            "dj_bridging": {
                "enabled": bool(cfg.dj_bridging_enabled),
                "seed_ordering": str(cfg.dj_seed_ordering),
                "anchors_must_include_all": bool(cfg.dj_anchors_must_include_all),
                "route_shape": str(cfg.dj_route_shape),
                "waypoint_weight": float(cfg.dj_waypoint_weight),
                "waypoint_floor": float(cfg.dj_waypoint_floor),
                "waypoint_penalty": float(cfg.dj_waypoint_penalty),
                "waypoint_tie_break_band": (
                    float(cfg.dj_waypoint_tie_break_band) if cfg.dj_waypoint_tie_break_band is not None else None
                ),
                "waypoint_cap": float(cfg.dj_waypoint_cap),
                "seed_ordering_weights": {
                    "sonic": float(cfg.dj_seed_ordering_weight_sonic),
                    "genre": float(cfg.dj_seed_ordering_weight_genre),
                    "bridge": float(cfg.dj_seed_ordering_weight_bridge),
                },
                "pooling_strategy": str(cfg.dj_pooling_strategy),
                "pooling_k_local": int(cfg.dj_pooling_k_local),
                "pooling_k_toward": int(cfg.dj_pooling_k_toward),
                "pooling_k_genre": int(cfg.dj_pooling_k_genre),
                "pooling_k_union_max": int(cfg.dj_pooling_k_union_max),
                "pooling_step_stride": int(cfg.dj_pooling_step_stride),
                "pooling_cache_enabled": bool(cfg.dj_pooling_cache_enabled),
                "pooling_debug_compare_baseline": bool(
                    cfg.dj_pooling_debug_compare_baseline
                ),
                "allow_detours_when_far": bool(cfg.dj_allow_detours_when_far),
                "far_thresholds": {
                    "sonic": float(cfg.dj_far_threshold_sonic),
                    "genre": float(cfg.dj_far_threshold_genre),
                    "connector_scarcity": float(cfg.dj_far_threshold_connector_scarcity),
                },
                "connector_bias": {
                    "enabled": bool(cfg.dj_connector_bias_enabled),
                    "max_per_segment_linear": int(cfg.dj_connector_max_per_segment_linear),
                    "max_per_segment_adventurous": int(cfg.dj_connector_max_per_segment_adventurous),
                },
                "ladder": {
                    "top_labels": int(cfg.dj_ladder_top_labels),
                    "min_label_weight": float(cfg.dj_ladder_min_label_weight),
                    "min_similarity": float(cfg.dj_ladder_min_similarity),
                    "max_steps": int(cfg.dj_ladder_max_steps),
                    "use_smoothed_waypoint_vectors": bool(
                        cfg.dj_ladder_use_smoothed_waypoint_vectors
                    ),
                    "smooth_top_k": int(cfg.dj_ladder_smooth_top_k),
                    "smooth_min_sim": float(cfg.dj_ladder_smooth_min_sim),
                },
                "waypoint_fallback_k": int(cfg.dj_waypoint_fallback_k),
                "micro_piers": {
                    "enabled": bool(cfg.dj_micro_piers_enabled),
                    "max": int(cfg.dj_micro_piers_max),
                    "topk": int(cfg.dj_micro_piers_topk),
                    "candidate_source": str(cfg.dj_micro_piers_candidate_source),
                    "selection_metric": str(cfg.dj_micro_piers_selection_metric),
                },
                "relaxation": {
                    "enabled": bool(cfg.dj_relaxation_enabled),
                    "max_attempts": int(cfg.dj_relaxation_max_attempts),
                    "emit_warnings": bool(cfg.dj_relaxation_emit_warnings),
                    "allow_floor_relaxation": bool(cfg.dj_relaxation_allow_floor_relaxation),
                },
            },
            "progress_arc": {
                "enabled": bool(cfg.progress_arc_enabled),
                "weight": float(cfg.progress_arc_weight),
                "shape": str(cfg.progress_arc_shape),
                "tolerance": float(cfg.progress_arc_tolerance),
                "loss": str(cfg.progress_arc_loss),
                "huber_delta": float(cfg.progress_arc_huber_delta),
                "max_step": (float(cfg.progress_arc_max_step) if cfg.progress_arc_max_step is not None else None),
                "max_step_mode": str(cfg.progress_arc_max_step_mode),
                "max_step_penalty": float(cfg.progress_arc_max_step_penalty),
                "autoscale": {
                    "enabled": bool(cfg.progress_arc_autoscale_enabled),
                    "min_distance": float(cfg.progress_arc_autoscale_min_distance),
                    "distance_scale": float(cfg.progress_arc_autoscale_distance_scale),
                    "per_step_scale": bool(cfg.progress_arc_autoscale_per_step_scale),
                },
            },
        },
    }

    logger.info("Pier+Bridge complete: %d tracks, %d segments, %d successful",
               len(final_indices), len(all_segments),
               sum(1 for d in diagnostics if d.success))

    return PierBridgeResult(
        track_ids=final_track_ids,
        track_indices=final_indices,
        seed_positions=seed_positions,
        segment_diagnostics=diagnostics,
        stats=stats,
    )

