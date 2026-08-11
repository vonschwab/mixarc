from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.features.artifacts import load_artifact_bundle
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import DSPipelineConfig, default_ds_config
from src.playlist.mode_presets import resolve_pace_mode
from src.playlist.layered_bridge_diagnostics import build_layered_transition_diagnostics
from src.playlist.constructor import PlaylistResult  # Type only, no longer calling construct_playlist
from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    PierBridgeResult,
    build_pier_bridge_playlist,
)
from src.playlist.artist_identity_resolver import parse_artist_identity_config
from src.playlist.pipeline.audit_emitter import AuditEmitter
from src.playlist.pipeline.bundle_restrict import restrict_bundle
from src.playlist.pipeline.embedding_setup import setup_embedding
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides
from src.playlist.pipeline.pier_resolver import (
    dedupe_pool_by_track_key,
    resolve_pier_seeds,
)
from src.playlist.pipeline.post_validation import (
    build_failure_diagnostic,
    run_post_order_validation,
)
from src.playlist.run_audit import (
    parse_infeasible_handling_config,
    parse_run_audit_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPipelineResult:
    track_ids: list[str]
    track_indices: list[int]
    stats: Dict[str, Any]
    params_requested: Dict[str, Any]
    params_effective: Dict[str, Any]


def _apply_overrides(cfg: DSPipelineConfig, overrides: Optional[dict]) -> DSPipelineConfig:
    if not overrides:
        return cfg
    cand_cfg = cfg.candidate
    cons_cfg = cfg.construct

    if "candidate" in overrides:
        cand_updates = {**cand_cfg.__dict__, **overrides["candidate"]}
        cand_cfg = type(cand_cfg)(**cand_updates)
    if "construct" in overrides:
        cons_updates = {**cons_cfg.__dict__, **overrides["construct"]}
        cons_cfg = type(cons_cfg)(**cons_updates)

    return replace(cfg, candidate=cand_cfg, construct=cons_cfg)


def _params_from_config(cfg: DSPipelineConfig) -> Dict[str, Any]:
    return {
        "mode": cfg.mode,
        "candidate": cfg.candidate.__dict__,
        "construct": cfg.construct.__dict__,
    }


def enforce_allowed_invariant(track_ids: list[str], allowed: set[str], context: str = "") -> None:
    """Raise if any track_id is not within allowed set."""
    out = [tid for tid in track_ids if tid not in allowed]
    if out:
        raise ValueError(f"Out-of-pool tracks detected {context}: {out[:5]}")


@dataclass(frozen=True)
class _CandidateRelaxationAttempt:
    attempt: int
    candidate_cfg: Any
    min_genre_similarity: Optional[float]
    summary: Dict[str, Any]


def _banger_gate_inputs(
    bundle: Any,
    rank_cutoff: Optional[int],
    *,
    db_path: str,
    metadata_db_path: Optional[str] = None,
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Pure helper: resolve (_banger_ranks, _banger_cutoff) for the Oops-All-Bangers gate.

    Takes the already-resolved cutoff int (computed at the call site from either
    pier_bridge_config.popularity_rank_cutoff for artist mode, or pb_overrides for
    seed mode). Returns (None, None) when rank_cutoff is None (gate inactive). Otherwise
    loads per-track Last.fm ranks once over the full bundle and returns
    (rank_array, int_cutoff). Isolated into a helper so the unit test can monkeypatch
    the loader without building a real artifact bundle."""
    if rank_cutoff is None:
        return None, None
    _banger_cutoff = int(rank_cutoff)
    from src.analyze.popularity_runner import (
        popularity_cache_db_path as _edb_path,
        load_pool_popularity_ranks_cached,
    )
    _effective_db = db_path if db_path else _edb_path()
    _banger_ranks = load_pool_popularity_ranks_cached(
        bundle, list(range(len(bundle.track_ids))), db_path=_effective_db,
        metadata_db_path=metadata_db_path,
    )
    return _banger_ranks, _banger_cutoff


# Phase 2 Task 4 grep-verification (re-checks Phase 1 Task 8's own finding,
# quoted in .superpowers/sdd/p1-task-8-report.md section "core.py's _build_pool
# nested fn + banger relax-to-fill cascade -- NOT deleted"): this function, the
# banger relax-to-fill cascade below (_banger_relaxation_steps/_loosen_sonic),
# and the _build_pool nested closure (~line 810 at generate_playlist_ds's call
# site) all remain genuinely LIVE and SHARED -- confirmed fresh via a repo-wide
# grep, not re-derived from the old report. `candidate_pool_indices` (the sole
# product of `_build_pool`) still has exactly one production call site
# (`_run_pier_bridge`, generate_playlist_ds), which single-seed and multi-pier
# runs both flow through identically (there is no separate single-seed
# pipeline module on this branch -- generate_playlist_ds's own docstring says
# so). Downstream, candidate_pool_indices still feeds `_dedupe_candidate_pool`,
# mini-pier waypoint planning (`plan_pier_sequence`), and the dj_bridging
# legacy-fallback `universe` -- none replaced by corridor's own eligible-
# universe mechanism. The One-Each retry ladder below (this function's sole
# caller) is gated on `pb_cfg.max_non_seed_tracks_per_artist == 1`, a field
# still threaded end-to-end from the GUI (src/playlist_gui/policy.py) through
# pier_bridge_overrides.py to both pier_bridge_builder.py's cap enforcement and
# repair/edge_repair.py's repair-time enforcement -- reachable, not dead.
# Nothing in this area is pier-bridge-orphaned; correctly left untouched.
def _relaxed_one_each_candidate_attempts(
    candidate_cfg: Any,
    min_genre_similarity: Optional[float],
) -> list[_CandidateRelaxationAttempt]:
    """Fallback candidate gates for One Each when the first bridge pass is infeasible."""
    base_similarity = float(candidate_cfg.similarity_floor)
    base_sonic = candidate_cfg.min_sonic_similarity
    base_genre = min_genre_similarity
    steps = [
        (1, 0.05, 0.08, 0.30),
        (2, 0.10, 0.05, 0.20),
        (3, 0.15, 0.00, 0.00),
    ]
    attempts: list[_CandidateRelaxationAttempt] = []

    for attempt, similarity_drop, sonic_target, genre_target in steps:
        relaxed_similarity = max(0.0, base_similarity - similarity_drop)
        relaxed_sonic = None if base_sonic is None else min(float(base_sonic), float(sonic_target))
        relaxed_genre = None if base_genre is None else min(float(base_genre), float(genre_target))

        if (
            relaxed_similarity == base_similarity
            and relaxed_sonic == base_sonic
            and relaxed_genre == base_genre
        ):
            continue

        relaxed_cfg = replace(
            candidate_cfg,
            similarity_floor=relaxed_similarity,
            min_sonic_similarity=relaxed_sonic,
        )
        attempts.append(
            _CandidateRelaxationAttempt(
                attempt=attempt,
                candidate_cfg=relaxed_cfg,
                min_genre_similarity=relaxed_genre,
                summary={
                    "attempt": attempt,
                    "similarity_floor": {
                        "from": base_similarity,
                        "to": relaxed_similarity,
                    },
                    "min_sonic_similarity": {
                        "from": base_sonic,
                        "to": relaxed_sonic,
                    },
                    "min_genre_similarity": {
                        "from": base_genre,
                        "to": relaxed_genre,
                    },
                },
            )
        )

    return attempts


# ---------------------------------------------------------------------------
# Pace-mode override plumbing (Phase 2 Task 4 pace-plumb fix)
# ---------------------------------------------------------------------------
#
# Two config.yaml sources feed a PACE_MODE_PRESETS-shaped override dict for
# resolve_pace_mode(), mirroring the two places pace_settings values land
# downstream in this module (fixes the pre-existing dead-outlet documented in
# scripts/corridor_baseline/perturb.py: resolve_pace_mode was called with NO
# overrides param at all, so no yaml key could ever reach these fields --
# genuine per-mode pace band knobs like bpm_bridge_max_log_distance were
# preset-only):
#   - Admission-side keys (admission_floor/bridge_floor/bpm_admission_max_log_
#     distance/onset_admission_max_log_distance) already have a REAL
#     `playlists.ds_pipeline.candidate_pool.*` override consumed by
#     default_ds_config (src/playlist/config.py:615-633) -- but it was
#     silently discarded by this module's later `replace(cfg.candidate, ...)`
#     call (below), which always used the bare preset value. Sourcing them
#     here from the same candidate_pool dict restores that override instead
#     of clobbering it.
#   - Bridge-side + energy keys have no candidate_pool equivalent; they come
#     from `playlists.ds_pipeline.pier_bridge.<key>` (pb_overrides), the same
#     flat-key convention the energy-only override block (folded into this
#     function) used.
_PACE_CANDIDATE_POOL_OVERRIDE_KEYS: Dict[str, str] = {
    "admission_floor": "pace_admission_floor",
    "bridge_floor": "pace_bridge_floor",
    "bpm_admission_max_log_distance": "bpm_admission_max_log_distance",
    "onset_admission_max_log_distance": "onset_admission_max_log_distance",
}
_PACE_PIER_BRIDGE_OVERRIDE_KEYS: Tuple[str, ...] = (
    "bpm_bridge_max_log_distance",
    "bpm_trust_min_onset_rate",
    "onset_bridge_max_log_distance",
    "bpm_bridge_soft_penalty_strength",
    "onset_bridge_soft_penalty_strength",
    "energy_step_cap",
    "energy_step_strength",
    "energy_arc_band",
    "energy_arc_strength",
    "pace_rescue_k_energy",  # AUDIT (review follow-up): also silently dead pre-
    # fix (same missing-overrides= dead outlet as the rest of this list) --
    # revived as a side effect of this fix, not separately tested; verify live
    # in the final contract sweep.
)


def _resolve_pace_overrides(overrides: Optional[dict], pb_overrides: dict) -> Dict[str, float]:
    """Build the flat override dict passed as ``resolve_pace_mode(..., overrides=...)``.

    Pure function (no engine imports) -- see module comment above for the two
    config sources. Only numeric (non-bool) values are forwarded; anything
    else is left for the preset default so a malformed config value can never
    silently corrupt the resolved settings dict.
    """
    candidate_pool_overrides = (overrides or {}).get("candidate_pool", {}) if isinstance(overrides, dict) else {}
    if not isinstance(candidate_pool_overrides, dict):
        candidate_pool_overrides = {}

    pace_overrides: Dict[str, float] = {}
    for preset_key, config_key in _PACE_CANDIDATE_POOL_OVERRIDE_KEYS.items():
        value = candidate_pool_overrides.get(config_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pace_overrides[preset_key] = float(value)
    for key in _PACE_PIER_BRIDGE_OVERRIDE_KEYS:
        value = pb_overrides.get(key) if isinstance(pb_overrides, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pace_overrides[key] = float(value)
    return pace_overrides


# ---------------------------------------------------------------------------
# Oops, All Bangers: relax-to-fill cascade (spec §3.5)
# ---------------------------------------------------------------------------

@dataclass
class _BangerRelaxStep:
    candidate_cfg: Any
    genre_gate: Optional[float]
    rank_cutoff: Optional[int]
    label: str


def _loosen_sonic(cfg, level: float):
    """level in (0,1] scales the sonic floors toward open; 0.0 disables sonic gating."""
    sap = cfg.sonic_admission_percentile
    mss = cfg.min_sonic_similarity
    if level <= 0.0:
        return replace(cfg, sonic_admission_percentile=0.0, min_sonic_similarity=None)
    return replace(
        cfg,
        sonic_admission_percentile=(float(sap) * level) if sap else sap,
        min_sonic_similarity=(float(mss) * level) if mss else mss,
    )


def _loosen_pace(cfg, *, off: bool):
    """Widen the BPM/onset admission bands; off=True disables pace gating entirely."""
    if off:
        return replace(cfg, bpm_admission_max_log_distance=float("inf"),
                       onset_admission_max_log_distance=float("inf"))
    def _wider(x):
        return float("inf") if x == float("inf") else float(x) * 2.0
    return replace(cfg,
                   bpm_admission_max_log_distance=_wider(cfg.bpm_admission_max_log_distance),
                   onset_admission_max_log_distance=_wider(cfg.onset_admission_max_log_distance))


def _banger_relaxation_steps(
    base_cfg,
    base_genre_gate,
    base_cutoff,
) -> Iterator[_BangerRelaxStep]:
    """Progressively looser banger-pool admission, in the fixed priority order
    sonic -> pace -> genre -> popularity (spec §3.5). Sonic gets the most/earliest
    relaxation; popularity is the LAST rung and the ONLY one that admits a non-banger.
    Mirrors _relaxed_one_each_candidate_attempts (a deterministic generator)."""
    cfg, gate, cutoff = base_cfg, base_genre_gate, base_cutoff
    # NOTE: _loosen_sonic/_loosen_pace scale the RUNNING cfg, so notches COMPOUND —
    # e.g. sonic 0.66 then 0.33 -> ~0.22x of the original floor, not 0.33x. Intentional
    # (deeper = more aggressive); the multipliers are calibration knobs (spec §10).
    # 1 sonic notch 1, 2 pace notch 1, 3 sonic notch 2, 4 pace off, 5 sonic off
    cfg = _loosen_sonic(cfg, 0.66)
    yield _BangerRelaxStep(cfg, gate, cutoff, "sonic notch1")
    cfg = _loosen_pace(cfg, off=False)
    yield _BangerRelaxStep(cfg, gate, cutoff, "pace notch1")
    cfg = _loosen_sonic(cfg, 0.33)
    yield _BangerRelaxStep(cfg, gate, cutoff, "sonic notch2")
    cfg = _loosen_pace(cfg, off=True)
    yield _BangerRelaxStep(cfg, gate, cutoff, "pace off")
    cfg = _loosen_sonic(cfg, 0.0)
    yield _BangerRelaxStep(cfg, gate, cutoff, "sonic off")
    # 6 genre notch (one past the user), 7 genre off
    gate = (float(base_genre_gate) * 0.5) if base_genre_gate is not None else None
    yield _BangerRelaxStep(cfg, gate, cutoff, "genre notch1")
    gate = None
    yield _BangerRelaxStep(cfg, gate, cutoff, "genre off")
    # 8-10 popularity: the only purity-breaking rungs (logged loudly by the caller)
    for new_cutoff in (25, 50, None):
        cutoff = new_cutoff
        yield _BangerRelaxStep(cfg, gate, cutoff, f"popularity top-{new_cutoff}" if new_cutoff else "popularity off")


def generate_playlist_ds(
    *,
    artifact_path: str | Path,
    seed_track_id: str,
    num_tracks: int,
    mode: str,
    random_seed: int,
    pace_mode: str = "dynamic",
    overrides: Optional[dict] = None,
    allowed_track_ids: Optional[list[str]] = None,
    excluded_track_ids: Optional[set[str]] = None,
    single_artist: bool = False,
    anchor_seed_ids: Optional[List[str]] = None,
    pier_bridge_config: Optional[PierBridgeConfig] = None,
    # Hybrid-level tuning dials (not part of DSPipelineConfig)
    sonic_weight: Optional[float] = None,
    genre_weight: Optional[float] = None,
    min_genre_similarity: Optional[float] = None,
    genre_method: Optional[str] = None,
    genre_admission_percentile: Optional[float] = None,
    genre_mode: Optional[str] = None,
    sonic_mode: Optional[str] = None,
    allowed_track_ids_set: Optional[set[str]] = None,
    internal_connector_ids: Optional[List[str]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    # Tag-steering ANCHOR piers (spec 2026-07-27): bundle track_ids of the on-tag
    # anchors the artist block injected into the pier list, so the builder can
    # place them in gaps instead of re-ordering them as co-equal piers.
    tag_anchor_track_ids: Optional[set[str]] = None,
    # Multi-artist blend (Task 9/10): explicit override for the interior-block
    # artist set, threaded straight through to build_pier_bridge_playlist's
    # seed_artist_keys_override kwarg. None (default) preserves the
    # single-artist derivation byte-for-byte.
    seed_artist_keys_override: Optional[Sequence[str]] = None,
    # Multi-artist blend (Task 10 fix, human ruling): the number of surviving
    # artist groups (joint group included) when the multi-artist branch
    # produced piers; 1 (default) is a no-op. max_artist_fraction means EACH
    # seed artist's share of the playlist -- the same reading Task 5's
    # total_pier_budget = n * base already applies to the pier budget -- so
    # the post-order per-artist cap below must scale by the same n, or a
    # legitimate N-artist pier allocation self-reports as "degraded" via
    # post_order_validation's artist_cap_violation on every successful blend.
    multi_artist_group_count: int = 1,
    # Optional pier-bridge infeasible handling + audit context (CLI/GUI)
    dry_run: bool = False,
    pool_source: Optional[str] = None,
    artist_style_enabled: bool = False,
    artist_playlist: bool = False,
    audit_context_extra: Optional[Dict[str, Any]] = None,
) -> DSPipelineResult:
    """
    Orchestrate:
    - load_artifact_bundle
    - build embedding (use smoothed genres for embedding)
    - build candidate pool
    - construct playlist via pier-bridge strategy (no repair pass)
    Returns ordered track_ids and stats.

    Playlist construction uses pier-bridge strategy exclusively:
    - Multiple seeds: seeds become fixed "piers", bridge segments connect them
    - Single seed: seed acts as both start and end pier (arc structure)
    - Artist playlists can optionally enforce "seed artist = piers only"

    Optional hybrid-level tuning dials:
    - sonic_weight, genre_weight: control hybrid embedding balance
    - min_genre_similarity: gate threshold
    - genre_method: which genre similarity algorithm to use
    - genre_mode: raw playlists.genre_mode string (Phase 1 Task 5). Threaded
      straight through to build_pier_bridge_playlist's genre_mode kwarg,
      CORRIDOR-POOLING ONLY (Phase 1 Task 4's relevance mask) -- not consumed
      by anything else in this function. min_genre_similarity/genre_admission_
      percentile above are the NUMBERS this mode resolves to upstream
      (mode_presets.apply_mode_presets via genre_ds_params.resolve_genre_ds_
      params); this is the raw string itself.
    - sonic_mode: raw playlists.sonic_mode string (per-mode corridor width,
      spec section 4, pulled forward from Phase 2 by Dylan's 2026-07-18
      decision). Threaded straight through to build_pier_bridge_playlist's
      sonic_mode kwarg, CORRIDOR-POOLING ONLY (keys the per-segment corridor
      width percentile) -- not consumed by anything else in this function.
      Same "raw string, not a resolved number" seam as genre_mode above.
    """
    # Normalize optional lists to avoid NoneType len issues downstream.
    anchor_seed_ids = list(dict.fromkeys(anchor_seed_ids or []))

    # Optional pier-bridge run audit + infeasible handling configs (default OFF).
    pb_overrides = (overrides or {}).get("pier_bridge", {}) if isinstance(overrides, dict) else {}
    audit_cfg = parse_run_audit_config(pb_overrides.get("audit_run"))
    infeasible_cfg = parse_infeasible_handling_config(pb_overrides.get("infeasible_handling"))
    audit = AuditEmitter(audit_cfg)

    # Parse artist identity config from constraints.artist_identity
    constraints_overrides = (overrides or {}).get("constraints", {}) if isinstance(overrides, dict) else {}
    artist_identity_overrides = constraints_overrides.get("artist_identity", {})
    # Shared parse (artist_identity_resolver.parse_artist_identity_config) so the
    # reporter's seed-share count resolves identity exactly as the beam did.
    artist_identity_cfg = parse_artist_identity_config(artist_identity_overrides)

    if artist_identity_cfg.enabled:
        logger.info("Artist identity resolution enabled for min_gap enforcement")

    bundle = load_artifact_bundle(artifact_path)
    if seed_track_id not in bundle.track_id_to_index:
        raise ValueError(f"Seed track_id not found in artifact: {seed_track_id}")
    seed_idx = bundle.track_id_to_index[seed_track_id]

    # Log requested tuning dials for verification
    if any([sonic_weight, genre_weight, min_genre_similarity, genre_method]):
        logger.info(
            "DS pipeline tuning dials: sonic_weight=%s, genre_weight=%s, min_genre_sim=%s, genre_method=%s",
            sonic_weight, genre_weight, min_genre_similarity, genre_method
        )

    # Restrict bundle to allowed_track_ids / excluded_track_ids when provided.
    bundle, seed_idx, allowed_track_ids_set = restrict_bundle(
        bundle,
        seed_track_id,
        seed_idx,
        anchor_seed_ids=anchor_seed_ids,
        allowed_track_ids=allowed_track_ids,
        excluded_track_ids=excluded_track_ids,
        allowed_track_ids_set=allowed_track_ids_set,
    )

    internal_connector_indices: Optional[Set[int]] = None
    if internal_connector_ids:
        internal_connector_indices = set()
        for tid in internal_connector_ids:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is not None:
                internal_connector_indices.add(idx)

    playlist_len = min(num_tracks, bundle.track_ids.shape[0])
    # Pass config.yaml overrides to default_ds_config for initial config creation
    cfg = default_ds_config(mode, playlist_len=playlist_len, overrides=overrides)
    # pb_overrides (playlists.ds_pipeline.pier_bridge.*) was already extracted
    # above (audit_cfg/infeasible_cfg parsing) -- reused here, not re-derived.
    pace_settings = resolve_pace_mode(
        pace_mode,
        overrides=_resolve_pace_overrides(overrides, pb_overrides) or None,
    )
    # default_ds_config resolves admission caps from `overrides`, which does NOT
    # carry pace_mode (it flows as a separate parameter) — so resolve_thresholds
    # falls back to the "dynamic" caps. Patch the BPM + onset admission caps from
    # the real pace_settings here, mirroring how the bridge caps are patched into
    # pb_cfg downstream. Without this, narrow/strict admission silently ran at the
    # dynamic caps (0.75) instead of their calibrated values (CLAUDE.md: a
    # configured knob that acts at the wrong value is a wiring bug).
    # pace_settings now also carries any yaml overrides (_resolve_pace_overrides
    # above), so an explicit candidate_pool.pace_admission_floor / .pace_bridge_
    # floor / .bpm_admission_max_log_distance / .onset_admission_max_log_distance
    # value survives this replace() instead of being silently discarded.
    cfg = replace(
        cfg,
        candidate=replace(
            cfg.candidate,
            pace_admission_floor=float(pace_settings["admission_floor"]),
            pace_bridge_floor=float(pace_settings["bridge_floor"]),
            bpm_admission_max_log_distance=float(
                pace_settings["bpm_admission_max_log_distance"]
            ),
            onset_admission_max_log_distance=float(
                pace_settings["onset_admission_max_log_distance"]
            ),
        ),
    )
    logger.info(
        "Pace mode: %s (admission_floor=%.2f, bridge_floor=%.2f, "
        "bpm_adm=%.2f, onset_adm=%.2f)",
        pace_mode,
        float(pace_settings["admission_floor"]),
        float(pace_settings["bridge_floor"]),
        float(pace_settings["bpm_admission_max_log_distance"]),
        float(pace_settings["onset_admission_max_log_distance"]),
    )

    # Load BPM + onset arrays from DB when pace gates are active
    perceptual_bpm: Optional[np.ndarray] = None
    tempo_stability_bpm: Optional[np.ndarray] = None
    onset_rate_arr: Optional[np.ndarray] = None
    _bpm_adm = float(pace_settings.get("bpm_admission_max_log_distance", float("inf")))
    _bpm_brd = float(pace_settings.get("bpm_bridge_max_log_distance", float("inf")))
    _onset_adm = float(pace_settings.get("onset_admission_max_log_distance", float("inf")))
    _onset_brd = float(pace_settings.get("onset_bridge_max_log_distance", float("inf")))
    if not (np.isinf(_bpm_adm) and np.isinf(_bpm_brd) and np.isinf(_onset_adm) and np.isinf(_onset_brd)):
        try:
            from src.playlist.bpm_loader import load_bpm_arrays
            _db_path = str(
                (overrides or {}).get("library", {}).get("database_path")
                or "data/metadata.db"
            )
            _bpm_arrays = load_bpm_arrays(bundle.track_ids, db_path=_db_path)
            perceptual_bpm = _bpm_arrays["perceptual_bpm"]
            tempo_stability_bpm = _bpm_arrays["tempo_stability"]
            onset_rate_arr = _bpm_arrays["onset_rate"]
            logger.info(
                "BPM loaded: %d/%d tracks have data",
                int(np.sum(~np.isnan(perceptual_bpm))),
                int(perceptual_bpm.shape[0]),
            )
        except Exception:
            logger.warning("BPM load failed; BPM gates disabled for this run", exc_info=True)

    energy_matrix: Optional[np.ndarray] = None
    _energy_active = any(
        float(pace_settings.get(k, 0.0)) > 0.0
        or float(pb_overrides.get(k, 0.0)) > 0.0
        for k in ("energy_step_strength", "energy_arc_strength")
    ) or int(pace_settings.get("pace_rescue_k_energy", 0)) > 0
    if _energy_active:
        try:
            from src.playlist.energy_loader import load_energy_matrix
            _energy_feats = tuple(
                ((overrides or {}).get("analyze", {}).get("pace", {}) or {}).get(
                    "energy_features", ["arousal_p50"]
                )
            )
            _sidecar = str(Path(artifact_path).parent / "energy" / "energy_sidecar.npz")
            energy_matrix = load_energy_matrix(
                bundle.track_ids, sidecar_path=_sidecar, features=_energy_feats
            )
            logger.info(
                "energy loaded: %d/%d tracks",
                int(np.sum(np.all(np.isfinite(energy_matrix), axis=1))),
                energy_matrix.shape[0],
            )
        except Exception:
            logger.warning("energy load failed; pace energy terms disabled", exc_info=True)
            energy_matrix = None

    # Post-order per-artist cap override for the multi-artist blend's own
    # named/seed/joint artists (xhigh review Finding 6). The relaxed cap used
    # to be applied by scaling cfg.construct.max_artist_fraction_final itself
    # -- but that value feeds the SINGLE `max_per_artist` scalar passed to
    # find_artist_cap_violations below, which applies it to EVERY identity in
    # the playlist, not just the ones the relaxation was meant for. A 3-group
    # 30-track blend raised the cap from 4 to 12 for every identity, so any
    # incidental bridge-fill artist could take up to 12 of 30 slots without
    # tripping this hard diversity guard. Computed here (not by mutating cfg)
    # so `max_per_artist` below stays the UNSCALED cap for everyone else;
    # only `seed_artist_keys_override`'s own keys get the relaxed value, via
    # `max_tracks_per_artist_overrides` at the validation call.
    artist_cap_overrides: Optional[Dict[str, int]] = None
    if single_artist:
        # Disable artist cap for single-artist runs
        cfg = replace(cfg, construct=replace(cfg.construct, max_artist_fraction_final=1.0))
    elif multi_artist_group_count > 1 and seed_artist_keys_override:
        _relaxed_cap = max(
            1,
            math.ceil(
                float(playlist_len) * cfg.construct.max_artist_fraction_final
                * multi_artist_group_count
            ),
        )
        artist_cap_overrides = {
            str(k): _relaxed_cap for k in seed_artist_keys_override if k
        }
        logger.info(
            "Multi-artist: per-artist cap override %d for %d named/seed/joint "
            "artist key(s); every other identity stays on the unscaled cap.",
            _relaxed_cap, len(artist_cap_overrides),
        )
    # Also apply any runtime overrides (for backward compatibility with nested structure)
    cfg = _apply_overrides(cfg, overrides)

    embedding = setup_embedding(
        bundle,
        seed_track_id,
        seed_idx,
        anchor_seed_ids=anchor_seed_ids,
        mode=mode,
        cfg=cfg,
        sonic_weight=sonic_weight,
        genre_weight=genre_weight,
        min_genre_similarity=min_genre_similarity,
        random_seed=random_seed,
    )
    # The orchestrator still needs these by name for the remaining phases.
    min_genre_similarity = embedding.min_genre_similarity
    variant_stats = embedding.variant_stats
    broad_filters = embedding.broad_filters
    genre_vocab = embedding.genre_vocab
    X_genre_raw = embedding.X_genre_raw
    X_genre_smoothed = embedding.X_genre_smoothed
    layered_genre_shadow_available = all(
        getattr(bundle, attr, None) is not None
        for attr in ("X_genre_leaf_idf", "X_genre_family", "X_genre_bridge", "X_facet")
    )
    genre_graph_cfg = (overrides or {}).get("genre_graph", {}) if isinstance(overrides, dict) else {}
    if not isinstance(genre_graph_cfg, dict):
        genre_graph_cfg = {}
    genre_graph_source = str(genre_graph_cfg.get("source") or "legacy").strip().lower()
    if genre_graph_source == "legacy" and str((overrides or {}).get("genre_source") or "").strip().lower() == "layered_shadow":
        genre_graph_source = "layered_shadow"
    if genre_graph_source not in {"legacy", "layered_shadow", "layered"}:
        logger.warning("Invalid genre_graph.source=%r; falling back to legacy", genre_graph_source)
        genre_graph_source = "legacy"

    # Per-seed adaptive admission percentile (Task 4: genre-arc steering).
    # Check both the base key and the mode-specific key (e.g. genre_admission_percentile_narrow)
    # because resolve_pier_bridge_tuning isn't called until after the pool is built.
    _genre_admission_percentile: Optional[float] = None
    _raw_pct = pb_overrides.get(f"genre_admission_percentile_{mode}")
    if _raw_pct is None:
        _raw_pct = pb_overrides.get("genre_admission_percentile")
    if _raw_pct is not None:
        try:
            _genre_admission_percentile = float(_raw_pct)
        except (TypeError, ValueError):
            pass

    # Roam broad-pool (calibrated 2026-06-24): when roam corridors are on, sonic is
    # the North Star and the graph-smoothed genre-neighbor pool keeps it honest, so
    # relax the tag-literal dense PMI-SVD percentile gate — it over-narrows single-
    # artist pools (300-372 -> 554-678) and worsens the worst edge, and is moot for
    # already-broad diverse seeds (~2394). Tunable via roam.genre_gate_percentile
    # (default 0.0 = off; e.g. 0.5 for a light gate); keeps sonic admission + graph.
    _roam = pb_overrides.get("roam") or {}
    if bool(_roam.get("enabled")):
        try:
            _genre_admission_percentile = float(_roam.get("genre_gate_percentile", 0.0))
        except (TypeError, ValueError):
            _genre_admission_percentile = 0.0

    # Fix 2 (2026-07-04): the genre-mode ladder (GENRE_MODE_PRESETS ->
    # genre_ds_params -> explicit param) OWNS the flat-gate percentile when set.
    # The pb_overrides keys above are cohesion-keyed legacy, and the roam clamp
    # was calibrated against the DENSE gate — neither may silently disable the
    # user's genre axis (the 2026-07-04 slider eval found the roam clamp zeroed
    # the percentile in every live run, leaving only the quantization-prone
    # absolute floor acting).
    if genre_admission_percentile is not None:
        _genre_admission_percentile = float(genre_admission_percentile)
        logger.info(
            "Genre admission percentile (genre_mode ladder): p=%.2f",
            _genre_admission_percentile,
        )

    # Per-seed adaptive sonic admission percentile (Task 1).
    # Mode-specific key (e.g. sonic_admission_percentile_narrow) takes priority
    # over the base key — mirrors the _resolve_mode_number_with_source priority.
    _sonic_admission_percentile: Optional[float] = None
    _raw_sonic_pct = pb_overrides.get(f"sonic_admission_percentile_{mode}")
    if _raw_sonic_pct is None:
        _raw_sonic_pct = pb_overrides.get("sonic_admission_percentile")
    if _raw_sonic_pct is not None:
        try:
            _sonic_admission_percentile = float(_raw_sonic_pct)
        except (TypeError, ValueError):
            pass

    # Genre admission aggregate mode: "centroid" (default) | "per_seed".
    _genre_admission_aggregate = str(pb_overrides.get("genre_admission_aggregate", "centroid")).strip().lower()
    if _genre_admission_aggregate not in {"centroid", "per_seed"}:
        _genre_admission_aggregate = "centroid"

    # Tag steering (user-selected genre tags): resolve the dense target once per run.
    _tag_steering_tags = [
        str(t) for t in (pb_overrides.get("tag_steering_tags") or []) if str(t).strip()
    ]
    _tag_steering_target = None
    try:
        _tag_steering_blend = float(pb_overrides.get("tag_steering_pool_blend", 0.5))
    except (TypeError, ValueError):
        _tag_steering_blend = 0.5
    if _tag_steering_tags:
        from src.playlist.tag_steering import resolve_tag_steering_target
        _tag_steering_target, _, _ = resolve_tag_steering_target(
            _tag_steering_tags,
            genre_vocab=[str(v) for v in getattr(bundle, "genre_vocab", [])],
            genre_emb=getattr(bundle, "genre_emb", None),
        )

    # Oops, All Bangers: gate cutoff comes from the explicit pier_bridge_config (artist
    # mode) or from pb_overrides (seed mode injects it via config;
    # pier_bridge_config is None there). Compute the effective cutoff here, then pass
    # it to the helper so _banger_gate_inputs stays a pure function on (bundle, cutoff).
    _cfg_cutoff: Optional[int] = getattr(pier_bridge_config, "popularity_rank_cutoff", None)
    if _cfg_cutoff is None:
        _ovr_cutoff = pb_overrides.get("popularity_rank_cutoff")
        _cfg_cutoff = int(_ovr_cutoff) if _ovr_cutoff is not None else None
    # metadata.db path for album-aware version-preference in popularity resolution
    # (demotes clean-titled live-album tracks). Shared by the gate + beam loaders.
    _meta_db = str((overrides or {}).get("library", {}).get("database_path") or "data/metadata.db")
    _banger_ranks, _banger_cutoff = _banger_gate_inputs(
        bundle, _cfg_cutoff, db_path="", metadata_db_path=_meta_db
    )

    # Tag steering (sonic pool lever): the selected tags learn a *centered*
    # (global-mean-subtracted) sonic prototype = the tag-SPECIFIC direction. Its
    # per-track affinity is blended into the sonic admission similarity so the
    # bridge pool admits on-tag-specific tracks (not generic-centroid neighbors --
    # the uncentered failure mode, retired 2026-07-08 after the centered pool beat
    # it: Real Estate/jangle worst-edge 0.315 -> 0.716). The SAME centered affinity
    # feeds the opt-in beam term (default weight 0.0 -- proven inert-to-harmful).
    _tag_sonic_affinity = None
    # Tag steering pool guarantee: on-tag track ids (authority membership, from
    # _rows below) to force-admit into the pool past the per-artist rank walk.
    # Independent of the cohesion gate below -- the guarantee needs the ids even
    # when the sonic prototype itself is disabled (low support/cohesion).
    _on_tag_guarantee_ids = None
    try:
        _tag_sonic_blend = float(pb_overrides.get("tag_steering_sonic_blend", 0.35))
    except (TypeError, ValueError):
        _tag_sonic_blend = 0.35
    try:
        _beam_tag_weight = float(pb_overrides.get("tag_steering_sonic_beam_weight", 0.0))
    except (TypeError, ValueError):
        _beam_tag_weight = 0.0
    try:
        _beam_edge_band = float(pb_overrides.get("tag_steering_worst_edge_band", 0.0))
    except (TypeError, ValueError):
        _beam_edge_band = 0.0
    try:
        _guar_max = int(pb_overrides.get("tag_steering_pool_guarantee_max", 30))
    except (TypeError, ValueError):
        _guar_max = 30
    try:
        _guar_per_artist = int(pb_overrides.get("tag_steering_pool_guarantee_per_artist", 3))
    except (TypeError, ValueError):
        _guar_per_artist = 3
    # Bridge-side Phase A (stage D, segment_pool_builder.py): relaxed bridge admission
    # (min->max) + on-tag force-include at the per-segment pool, gated the same way as
    # the pool-level guarantee above -- inert unless _on_tag_guarantee_ids resolves
    # (i.e. tag steering is active). Live default ON per #22 (activate fixes; keep a
    # config rollback), never on for non-steered runs regardless of these knobs.
    try:
        _relax_bridge = bool(pb_overrides.get("tag_steering_relax_bridge_admission", True))
    except (TypeError, ValueError):
        _relax_bridge = True
    try:
        _seg_guar_max = int(pb_overrides.get("tag_steering_segment_guarantee_max", 8))
    except (TypeError, ValueError):
        _seg_guar_max = 8
    try:
        _seg_guar_pa = int(pb_overrides.get("tag_steering_segment_guarantee_per_artist", 2))
    except (TypeError, ValueError):
        _seg_guar_pa = 2
    # Anchor gap insertion (spec 2026-07-27). Live default ON per principle #22;
    # false = legacy append + global re-order. Inert unless tag_anchor_track_ids
    # is non-empty, i.e. never on for a non-steered or genre-mode run.
    try:
        _anchor_gap_insertion = bool(
            pb_overrides.get("tag_steering_anchor_gap_insertion", True)
        )
    except (TypeError, ValueError):
        _anchor_gap_insertion = True
    try:
        _anchor_min_bridge = float(
            pb_overrides.get("tag_steering_anchor_min_bridge", 0.35)
        )
    except (TypeError, ValueError):
        _anchor_min_bridge = 0.35
    if _tag_steering_tags:
        _xsonic = embedding.X_sonic_for_embed
        if _xsonic is not None:
            from src.playlist.tag_steering import (
                resolve_tag_sonic_prototype_rows, sonic_prototype_from_rows, sonic_global_mean,
            )
            _min_support = int(pb_overrides.get("tag_steering_prototype_min_support", 25))
            _t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
            # Exclude only the SEED ARTIST (the artist being steered), matching the
            # pier lever. Do NOT use seed_indices_for_floor here -- that is the broad
            # admission-floor neighborhood (dozens of artists), and excluding all of
            # them collapses niche-tag support below the floor and silently disables
            # this lever (2026-07-08: hauntology support 89 -> 21, lever off).
            _seed_artists = None
            _bta = getattr(bundle, "track_artists", None)
            if _bta is not None and 0 <= int(seed_idx) < len(_bta):
                _seed_artists = [str(_bta[int(seed_idx)])]
            _rows, _n, _ = resolve_tag_sonic_prototype_rows(
                _tag_steering_tags, metadata_db_path=_meta_db,
                track_id_to_row=_t2r, exclude_artists=_seed_artists, min_support=_min_support,
            )
            if _rows is not None:
                _on_tag_guarantee_ids = {str(bundle.track_ids[r]) for r in _rows}
                _xs = np.asarray(_xsonic, dtype=np.float64)
                _gm = sonic_global_mean(_xs)
                _cproto, _ccoh, _ = sonic_prototype_from_rows(_xs, _rows, global_mean=_gm)
                _min_coh = float(pb_overrides.get("tag_steering_prototype_min_cohesion", 0.15))
                if _cproto is not None and _ccoh >= _min_coh:
                    _xsn = _xs / (np.linalg.norm(_xs, axis=1, keepdims=True) + 1e-12)
                    _tag_sonic_affinity = (_xsn - _gm) @ _cproto
                    logger.info(
                        "Tag steering sonic prototype: support=%d cohesion=%.3f "
                        "pool_blend=%.2f beam_weight=%.2f",
                        _n, _ccoh, _tag_sonic_blend, _beam_tag_weight,
                    )
                elif _cproto is not None:
                    logger.warning(
                        "Tag steering sonic prototype: cohesion %.3f < %.2f (sonically "
                        "multimodal tag) -- sonic pool/beam levers disabled for this run.",
                        _ccoh, _min_coh,
                    )
                else:
                    logger.warning(
                        "Tag steering sonic prototype: degenerate (near-zero-norm) centroid "
                        "over %d support rows -- sonic pool/beam levers disabled for this run.",
                        _n,
                    )

    # BEAM tag term signal = COMBINED genre-dense (leading) + sonic. The genre-dense
    # affinity is the clean cross-artist discriminator for tag membership (e.g. for a
    # BoC + hauntology steer, Ghost Box score ~0.94 vs Aphex/Autechre ~0.01, while the
    # sonic prototype cannot separate them: ~0.6 vs ~0.66). The sonic term adds
    # within-blended-artist resolution (Real Estate/jangle, where genre-dense is flat
    # and drops out of the ranking as a ~constant). Additive, matching the pier lever.
    _beam_tag_affinity = None
    if _tag_steering_tags:
        _xgd_b = getattr(bundle, "X_genre_dense", None)
        _genre_tag_aff = None
        if _tag_steering_target is not None and _xgd_b is not None:
            _xgdn = np.asarray(_xgd_b, dtype=np.float64)
            _xgdn = _xgdn / (np.linalg.norm(_xgdn, axis=1, keepdims=True) + 1e-12)
            _genre_tag_aff = _xgdn @ np.asarray(_tag_steering_target, dtype=np.float64)
        if _genre_tag_aff is not None and _tag_sonic_affinity is not None:
            _beam_sonic_sub = float(pb_overrides.get("tag_steering_sonic_weight", 0.5))
            _beam_tag_affinity = _genre_tag_aff + _beam_sonic_sub * _tag_sonic_affinity
        elif _genre_tag_aff is not None:
            _beam_tag_affinity = _genre_tag_aff
        else:
            _beam_tag_affinity = _tag_sonic_affinity

    # Instrumental lean: load voice_prob only when enabled; degrade to inert
    # + WARN rather than raising (mirrors the energy-sidecar pattern above).
    # Loaded here (before pool build) rather than after apply_pier_bridge_overrides
    # builds pb_cfg, because build_candidate_pool (Task 8) consumes voice_prob too
    # and the pool is built before pb_cfg exists. cfg.candidate.instrumental_enabled
    # is populated from the same overrides['pier_bridge'] section pb_cfg reads
    # (see default_ds_config), so this single load serves both pool and beam.
    voice_prob: Optional[np.ndarray] = None
    if bool(getattr(cfg.candidate, "instrumental_enabled", False)):
        from src.playlist.instrumental_loader import load_voice_prob
        _instr_sidecar = str(Path(artifact_path).parent / "instrumental" / "instrumental_sidecar.npz")
        voice_prob = load_voice_prob(bundle.track_ids, sidecar_path=_instr_sidecar)
        if voice_prob is None or not np.isfinite(voice_prob).any():
            logger.warning(
                "Instrumental lean ON but voice_prob absent/all-NaN at %s — guard inert this run",
                _instr_sidecar,
            )

    def _build_pool(candidate_cfg: Any, genre_gate: Optional[float],
                    popularity_rank_cutoff: Optional[int] = _banger_cutoff):
        return build_candidate_pool(
            seed_idx=seed_idx,
            seed_indices=embedding.seed_indices_for_floor,
            embedding=embedding.embedding_model.embedding,
            artist_keys=bundle.artist_keys,
            track_ids=bundle.track_ids,
            track_titles=bundle.track_titles,
            track_artists=bundle.track_artists,
            durations_ms=bundle.durations_ms,
            cfg=candidate_cfg,
            random_seed=random_seed,
            X_sonic=embedding.X_sonic_for_embed,
            X_genre_raw=X_genre_raw if genre_gate is not None else None,
            X_genre_smoothed=X_genre_smoothed if genre_gate is not None else None,
            X_genre_dense=getattr(bundle, "X_genre_dense", None) if genre_gate is not None else None,
            min_genre_similarity=genre_gate,
            genre_method=genre_method or "ensemble",
            genre_vocab=genre_vocab,
            broad_filters=broad_filters,
            mode=mode,
            perceptual_bpm=perceptual_bpm,
            tempo_stability=tempo_stability_bpm,
            onset_rate=onset_rate_arr,
            X_energy=(energy_matrix.reshape(-1) if energy_matrix is not None else None),
            genre_admission_percentile=_genre_admission_percentile,
            genre_admission_aggregate=_genre_admission_aggregate,
            layered_genre_diagnostics=layered_genre_shadow_available and genre_graph_source in {"layered_shadow", "layered"},
            X_genre_leaf_idf=getattr(bundle, "X_genre_leaf_idf", None),
            X_genre_family=getattr(bundle, "X_genre_family", None),
            X_genre_bridge=getattr(bundle, "X_genre_bridge", None),
            X_facet=getattr(bundle, "X_facet", None),
            genre_leaf_vocab=getattr(bundle, "genre_leaf_vocab", None),
            genre_family_vocab=getattr(bundle, "genre_family_vocab", None),
            genre_bridge_vocab=getattr(bundle, "genre_bridge_vocab", None),
            facet_vocab=getattr(bundle, "facet_vocab", None),
            genre_graph_source=genre_graph_source,
            popularity_ranks=_banger_ranks,
            popularity_rank_cutoff=popularity_rank_cutoff,
            steering_target=_tag_steering_target,
            steering_blend=_tag_steering_blend,
            sonic_blend=_tag_sonic_blend,
            sonic_pool_affinity=_tag_sonic_affinity,  # centered, tag-specific direction
            on_tag_guarantee_ids=_on_tag_guarantee_ids,
            on_tag_guarantee_max=_guar_max,
            on_tag_guarantee_per_artist=_guar_per_artist,
            voice_prob=voice_prob,
        )

    _candidate_cfg_kwargs: dict = dict(
        pace_rescue_k_energy=int(pace_settings.get("pace_rescue_k_energy", 0)),
    )
    if _sonic_admission_percentile is not None:
        _candidate_cfg_kwargs["sonic_admission_percentile"] = _sonic_admission_percentile
    _candidate_cfg = replace(cfg.candidate, **_candidate_cfg_kwargs)
    pool = _build_pool(_candidate_cfg, min_genre_similarity)
    pool.stats["target_length"] = num_tracks

    # Oops, All Bangers: relax-to-fill cascade. If the banger-gated pool is too
    # small to build a coherent playlist, relax sonic -> pace -> genre -> popularity
    # (popularity LAST — the only purity-breaking rung), rebuilding and stopping the
    # instant the pool fills. Only runs when the gate is active.
    if _banger_cutoff is not None:
        _min_banger_pool = max(2 * int(num_tracks), 40)
        _pool_n = len(getattr(pool, "eligible_indices", pool.pool_indices))
        if _pool_n < _min_banger_pool:
            for _step in _banger_relaxation_steps(_candidate_cfg, min_genre_similarity, _banger_cutoff):
                logger.info(
                    "Bangers relax-to-fill: pool=%d < target=%d -> relaxing [%s]%s",
                    _pool_n, _min_banger_pool, _step.label,
                    "  (ADMITTING NON-BANGERS)" if _step.label.startswith("popularity") else "",
                )
                pool = _build_pool(_step.candidate_cfg, _step.genre_gate,
                                   popularity_rank_cutoff=_step.rank_cutoff)
                pool.stats["target_length"] = num_tracks
                _pool_n = len(getattr(pool, "eligible_indices", pool.pool_indices))
                if _pool_n >= _min_banger_pool:
                    logger.info("Bangers relax-to-fill: filled at [%s] pool=%d", _step.label, _pool_n)
                    break

    max_per_artist = max(1, math.ceil(playlist_len * cfg.construct.max_artist_fraction_final))
    logger.info(
        "PIPELINE STRATEGY: pier-bridge (anchor_seed_ids=%s)",
        len(anchor_seed_ids),
    )

    playlist: PlaylistResult

    # ═══════════════════════════════════════════════════════════════════════════
    # PIER + BRIDGE STRATEGY (always used)
    # - Multiple seeds: seeds as fixed piers, beam-search bridges between
    # - Single seed: seed acts as both start and end pier (arc structure)
    # - No repair pass
    # ═══════════════════════════════════════════════════════════════════════════
    if True:  # Always use pier-bridge
        # ─── PIER BRIDGE PATH: No construct_playlist, no repair ───
        logger.info("ENTERING PIER-BRIDGE PATH with %d anchor seeds", len(anchor_seed_ids))

        pier_seed_indices, seed_track_ids_for_pier = resolve_pier_seeds(
            bundle, seed_idx, anchor_seed_ids,
        )

        # Pier-bridge handles any number of seeds (including 1 seed as arc structure)
        if True:
            # Deduplicate candidate pool by (artist, title), keeping canonical version.
            pool_indices = list(getattr(pool, "eligible_indices", pool.pool_indices))
            pool_indices = dedupe_pool_by_track_key(bundle, pool_indices)

            audit.ensure_context(
                bundle=bundle,
                seed_idx=seed_idx,
                seed_track_id=seed_track_id,
                mode=cfg.mode,
                dry_run=dry_run,
                artifact_path=artifact_path,
                sonic_variant=getattr(bundle, "sonic_variant", None),
                allowed_ids_count=int(len(allowed_track_ids_set or set())),
                pool_source=pool_source,
                artist_style_enabled=artist_style_enabled,
                artist_playlist=artist_playlist,
                audit_context_extra=audit_context_extra,
            )

            pb_cfg, tuning, tuning_sources = apply_pier_bridge_overrides(
                pier_bridge_config=pier_bridge_config,
                cfg=cfg,
                overrides=overrides,
                pb_overrides=pb_overrides,
                artist_playlist=artist_playlist,
                dry_run=dry_run,
                audit_cfg=audit_cfg,
            )
            pb_cfg = replace(
                pb_cfg,
                pace_bridge_floor=float(cfg.candidate.pace_bridge_floor),
                bpm_bridge_max_log_distance=float(pace_settings.get("bpm_bridge_max_log_distance", float("inf"))),
                bpm_stability_min=float(cfg.candidate.bpm_stability_min),
                bpm_trust_min_onset_rate=float(pace_settings.get("bpm_trust_min_onset_rate", 0.0)),
                onset_bridge_max_log_distance=float(pace_settings.get("onset_bridge_max_log_distance", float("inf"))),
                bpm_bridge_soft_penalty_strength=float(pace_settings.get("bpm_bridge_soft_penalty_strength", 0.0)),
                onset_bridge_soft_penalty_strength=float(pace_settings.get("onset_bridge_soft_penalty_strength", 0.0)),
                energy_step_cap=float(pace_settings.get("energy_step_cap", 0.0)),
                energy_step_strength=float(pace_settings.get("energy_step_strength", 0.0)),
                energy_arc_band=float(pace_settings.get("energy_arc_band", 0.0)),
                energy_arc_strength=float(pace_settings.get("energy_arc_strength", 0.0)),
                # C1 duration soft-penalty ON-case + title-hygiene (Task 7 fix):
                # corridor-pooling seam (see PierBridgeConfig's field comments).
                duration_penalty_enabled=bool(cfg.candidate.duration_penalty_enabled),
                duration_penalty_weight=float(cfg.candidate.duration_penalty_weight),
                duration_cutoff_multiplier=float(cfg.candidate.duration_cutoff_multiplier),
                title_hard_exclude_flags=tuple(sorted(cfg.candidate.title_hard_exclude_flags)),
            )
            # (config.yaml pier_bridge.<energy_*> overrides on top of preset defaults
            # are now folded into pace_settings via _resolve_pace_overrides above, so
            # the replace() block just above already reflects them -- no separate
            # post-hoc override pass needed here anymore.)

            # Instrumental lean: voice_prob is loaded ONCE, earlier (before pool
            # build, near _banger_gate_inputs) because build_candidate_pool (Task 8)
            # needs it too. pb_cfg.instrumental_enabled here is sourced from the
            # same pb_overrides dict as cfg.candidate.instrumental_enabled above,
            # so the two stay consistent without a second sidecar load.

            logger.info(
                "Pier-bridge segment policy: artist_playlist=%s strategy=%s pool_max=%d progress=%s disallow_seed_artist_in_interiors=%s disallow_pier_artists_in_interiors=%s",
                bool(artist_playlist),
                str(pb_cfg.segment_pool_strategy),
                int(pb_cfg.segment_pool_max),
                bool(pb_cfg.progress_enabled),
                bool(pb_cfg.disallow_seed_artist_in_interiors),
                bool(pb_cfg.disallow_pier_artists_in_interiors),
            )
            if audit.active and not audit.has_kind("preflight"):
                pool_summary = {
                    "allowed_ids_count": int(len(allowed_track_ids_set or set())),
                    "piers_count": int(len(seed_track_ids_for_pier)),
                    "internal_connectors_count": int(len(internal_connector_indices or set())),
                    "candidate_pool_stats": dict(pool.stats or {}),
                    "candidate_pool_indices_after_dedupe": int(len(pool_indices)),
                }
                style_summary = None
                if isinstance(audit_context_extra, dict):
                    style_summary = audit_context_extra.get("style_summary")

                excluded_ids_set = {str(t) for t in (excluded_track_ids or set())}
                pier_ids_set = {str(t) for t in seed_track_ids_for_pier}
                exempt_piers_count = int(len(excluded_ids_set & pier_ids_set))
                recency_lookback_days = None
                if isinstance(audit_context_extra, dict):
                    recency_raw = audit_context_extra.get("recency")
                    if isinstance(recency_raw, dict) and isinstance(recency_raw.get("lookback_days"), (int, float)):
                        recency_lookback_days = int(recency_raw.get("lookback_days"))
                audit.append(
                    "preflight",
                    {
                            "tuning": {
                                "mode": str(cfg.mode),
                                "transition_floor": float(tuning.transition_floor),
                                "bridge_floor": float(tuning.bridge_floor),
                                "weight_bridge": float(tuning.weight_bridge),
                                "weight_transition": float(tuning.weight_transition),
                                "genre_tiebreak_weight": float(tuning.genre_tiebreak_weight),
                                "genre_penalty_threshold": float(tuning.genre_penalty_threshold),
                                "genre_penalty_strength": float(tuning.genre_penalty_strength),
                                "genre_tie_break_band": (
                                    float(pb_cfg.genre_tie_break_band)
                                    if pb_cfg.genre_tie_break_band is not None
                                    else None
                                ),
                                "segment_pool_strategy": str(pb_cfg.segment_pool_strategy),
                                "segment_pool_max": int(pb_cfg.segment_pool_max),
                                "max_segment_pool_max": int(pb_cfg.max_segment_pool_max),
                                "progress": {
                                    "enabled": bool(pb_cfg.progress_enabled),
                                    "monotonic_epsilon": float(pb_cfg.progress_monotonic_epsilon),
                                    "penalty_weight": float(pb_cfg.progress_penalty_weight),
                                },
                                "experiments": {
                                    "bridge_scoring": {
                                        "enabled": bool(pb_cfg.experiment_bridge_scoring_enabled),
                                        "min_weight": float(pb_cfg.experiment_bridge_min_weight),
                                        "balance_weight": float(pb_cfg.experiment_bridge_balance_weight),
                                    },
                                },
                                "progress_arc": {
                                    "enabled": bool(pb_cfg.progress_arc_enabled),
                                    "weight": float(pb_cfg.progress_arc_weight),
                                    "shape": str(pb_cfg.progress_arc_shape),
                                    "tolerance": float(pb_cfg.progress_arc_tolerance),
                                    "loss": str(pb_cfg.progress_arc_loss),
                                    "huber_delta": float(pb_cfg.progress_arc_huber_delta),
                                    "max_step": (float(pb_cfg.progress_arc_max_step) if pb_cfg.progress_arc_max_step is not None else None),
                                    "max_step_mode": str(pb_cfg.progress_arc_max_step_mode),
                                    "max_step_penalty": float(pb_cfg.progress_arc_max_step_penalty),
                                    "autoscale": {
                                        "enabled": bool(pb_cfg.progress_arc_autoscale_enabled),
                                        "min_distance": float(pb_cfg.progress_arc_autoscale_min_distance),
                                        "distance_scale": float(pb_cfg.progress_arc_autoscale_distance_scale),
                                        "per_step_scale": bool(pb_cfg.progress_arc_autoscale_per_step_scale),
                                    },
                                },
                                "disallow_seed_artist_in_interiors": bool(pb_cfg.disallow_seed_artist_in_interiors),
                                "disallow_pier_artists_in_interiors": bool(pb_cfg.disallow_pier_artists_in_interiors),
                                "infeasible_handling": {
                                    "enabled": bool(infeasible_cfg.enabled),
                                    "strategy": str(infeasible_cfg.strategy),
                                    "min_bridge_floor": float(infeasible_cfg.min_bridge_floor),
                                    "backoff_steps": list(infeasible_cfg.backoff_steps),
                                    "max_attempts_per_segment": int(infeasible_cfg.max_attempts_per_segment),
                                    "widen_search_on_backoff": bool(infeasible_cfg.widen_search_on_backoff),
                                    "extra_neighbors_m": int(infeasible_cfg.extra_neighbors_m),
                                    "extra_bridge_helpers": int(infeasible_cfg.extra_bridge_helpers),
                                    "extra_beam_width": int(infeasible_cfg.extra_beam_width),
                                    "extra_expansion_attempts": int(infeasible_cfg.extra_expansion_attempts),
                                },
                                "audit_run": {
                                    "enabled": bool(audit_cfg.enabled),
                                    "out_dir": str(audit_cfg.out_dir),
                                    "include_top_k": int(audit_cfg.include_top_k),
                                    "max_bytes": int(audit_cfg.max_bytes),
                                    "write_on_success": bool(audit_cfg.write_on_success),
                                    "write_on_failure": bool(audit_cfg.write_on_failure),
                                },
                            },
                            "tuning_sources": dict(tuning_sources),
                            "pool_summary": pool_summary,
                            "ds_inputs": {
                                "allowed_ids_count": int(len(allowed_track_ids_set or set())),
                                "excluded_ids_count": int(len(excluded_ids_set)),
                            },
                            "recency": {
                                "lookback_days": recency_lookback_days,
                                "excluded_count": int(len(excluded_ids_set)),
                                "exempt_piers_count": exempt_piers_count,
                            },
                            "style_summary": style_summary,
                    },
                )
            # Compute a single shared generation deadline so all One-Each retries
            # and all segment loops inside build_pier_bridge_playlist share one
            # budget rather than each call resetting its own start anchor.
            # Read from the typed pb_cfg (already absorbed pb_overrides via
            # apply_pier_bridge_overrides) so a programmatic caller passing a
            # typed config without a raw overrides dict is honored.
            _budget_s = float(pb_cfg.generation_budget_s)
            # generation_budget_s <= 0 disables the wall-clock limit entirely: no soft
            # deadline here, and (because the builder keys its relaxation cap off this
            # being None) no per-build relaxation cap either. Quality-first while we
            # dial in playlists; a positive value re-arms the deadline + fallback later.
            _generation_deadline: Optional[float] = (
                None if _budget_s <= 0 else time.monotonic() + _budget_s
            )

            def _run_pier_bridge(candidate_pool_indices: list[int]) -> PierBridgeResult:
                # Oops, All Bangers: cache-only popularity for the gated pool (no Last.fm
                # fetch here — the eager batch warms the cache; sub-8 artists stay NaN ->
                # ruthlessly demoted). Inert unless a positive strength is configured.
                popularity_values = None
                if float(getattr(pb_cfg, "popularity_penalty_strength", 0.0)) > 0.0:
                    from src.analyze.popularity_runner import (
                        popularity_cache_db_path,
                        load_pool_popularity_values_cached,
                    )
                    popularity_values = load_pool_popularity_values_cached(
                        bundle, candidate_pool_indices, db_path=popularity_cache_db_path(),
                        metadata_db_path=_meta_db)
                return build_pier_bridge_playlist(
                    seed_track_ids=seed_track_ids_for_pier,
                    total_tracks=playlist_len,
                    bundle=bundle,
                    candidate_pool_indices=candidate_pool_indices,
                    cfg=pb_cfg,
                    allowed_track_ids_set=set(allowed_track_ids) if allowed_track_ids else None,
                    internal_connector_indices=internal_connector_indices,
                    internal_connector_max_per_segment=internal_connector_max_per_segment,
                    internal_connector_priority=internal_connector_priority,
                    infeasible_handling=infeasible_cfg,
                    audit_config=audit_cfg,
                    audit_events=audit.events,
                    artist_identity_cfg=artist_identity_cfg,
                    perceptual_bpm=perceptual_bpm,
                    tempo_stability_arr=tempo_stability_bpm,
                    onset_rate=onset_rate_arr,
                    energy_matrix=energy_matrix,
                    voice_prob=voice_prob,
                    popularity_values=popularity_values,
                    min_gap=int(getattr(cfg.construct, "min_gap", 1) or 1),
                    deadline=_generation_deadline,
                    sonic_tag_affinity=_beam_tag_affinity,
                    sonic_tag_beam_weight=_beam_tag_weight,
                    tag_steering_worst_edge_band=_beam_edge_band,
                    tag_steering_relax_bridge_admission=_relax_bridge,
                    on_tag_guarantee_ids=_on_tag_guarantee_ids,
                    on_tag_segment_guarantee_max=_seg_guar_max,
                    on_tag_segment_guarantee_per_artist=_seg_guar_pa,
                    tag_anchor_track_ids=tag_anchor_track_ids,
                    tag_anchor_gap_insertion=_anchor_gap_insertion,
                    tag_anchor_min_bridge=_anchor_min_bridge,
                    # Phase 1 Task 5 req 0: thread the raw genre_mode string
                    # through so the corridor path's genre-mode-keyed
                    # relevance mask (Phase 1 Task 4) gets the real slider
                    # value instead of always landing in the "off" bucket.
                    genre_mode=genre_mode,
                    # Per-mode corridor width task, req 0 mirror: thread the
                    # raw sonic_mode string through so the corridor path's
                    # sonic-mode-keyed width (pier_bridge.corridor.resolve_
                    # corridor_width_percentile) gets the real slider value
                    # instead of always falling back to "dynamic".
                    sonic_mode=sonic_mode,
                    # Phase 1 Task 5 reseat: Oops-All-Bangers on the corridor
                    # universe. Same (ranks, cutoff) pair already resolved
                    # once above via _banger_gate_inputs for the legacy pool
                    # (aligned to the FULL bundle) -- inert on the legacy path
                    # (pier_bridge_builder.py only reads these inside its
                    # pooling=="corridor" branch).
                    popularity_ranks=_banger_ranks,
                    popularity_rank_cutoff=_banger_cutoff,
                    seed_artist_keys_override=seed_artist_keys_override,
                )

            one_each_candidate_relaxation: Optional[Dict[str, Any]] = None
            pb_result: PierBridgeResult = _run_pier_bridge(pool_indices)
            if not pb_result.success and getattr(pb_cfg, "max_non_seed_tracks_per_artist", None) == 1:
                for relaxation in _relaxed_one_each_candidate_attempts(_candidate_cfg, min_genre_similarity):
                    # Reuse the shared generation deadline — do NOT reset it per
                    # retry. Each retry re-runs the entire pier-bridge build, which
                    # was the root cause of the 23-min strict+hyperpop grind.
                    if _generation_deadline is not None and time.monotonic() > _generation_deadline:
                        logger.warning(
                            "One-Each candidate retry %d skipped — generation deadline "
                            "exceeded; using result from previous attempt",
                            int(relaxation.attempt),
                        )
                        break
                    summary = dict(relaxation.summary)
                    logger.info(
                        "One Each candidate fallback attempt %d: similarity_floor %.3f -> %.3f, sonic_floor %s -> %s, genre_gate %s -> %s",
                        int(relaxation.attempt),
                        float(summary["similarity_floor"]["from"]),
                        float(summary["similarity_floor"]["to"]),
                        summary["min_sonic_similarity"]["from"],
                        summary["min_sonic_similarity"]["to"],
                        summary["min_genre_similarity"]["from"],
                        summary["min_genre_similarity"]["to"],
                    )
                    retry_pool = _build_pool(
                        relaxation.candidate_cfg,
                        relaxation.min_genre_similarity,
                    )
                    retry_pool.stats["target_length"] = num_tracks
                    retry_pool_indices = list(getattr(retry_pool, "eligible_indices", retry_pool.pool_indices))
                    retry_pool_indices = dedupe_pool_by_track_key(bundle, retry_pool_indices)
                    retry_result = _run_pier_bridge(retry_pool_indices)
                    summary["candidate_pool_indices_after_dedupe"] = int(len(retry_pool_indices))
                    summary["candidate_pool_stats"] = dict(retry_pool.stats or {})
                    summary["success"] = bool(retry_result.success)
                    summary["failure_reason"] = retry_result.failure_reason
                    if audit.active:
                        audit.append("one_each_candidate_fallback", summary)

                    logger.info(
                        "One Each candidate fallback attempt %d result: success=%s pool_after_dedupe=%d",
                        int(relaxation.attempt),
                        bool(retry_result.success),
                        int(len(retry_pool_indices)),
                    )
                    pool = retry_pool
                    pool_indices = retry_pool_indices
                    pb_result = retry_result
                    min_genre_similarity = relaxation.min_genre_similarity
                    if retry_result.success:
                        one_each_candidate_relaxation = summary
                        break

            if not pb_result.success:
                diag = build_failure_diagnostic(
                    pool.stats or {},
                    pb_result.failure_reason,
                    pool_indices_count=len(pool_indices),
                    seed_track_ids_for_pier_count=len(seed_track_ids_for_pier),
                    cfg_mode=cfg.mode,
                    cfg_genre_gate_min_similarity=min_genre_similarity,
                )
                if audit.can_flush():
                    audit.append(
                        "final_failure",
                        {
                            "failure_reason": diag.failure_reason,
                            "diagnostic_msg": diag.diagnostic_msg,
                            "segment_bridge_floors_used": (pb_result.stats or {}).get("segment_bridge_floors_used"),
                            "segment_backoff_attempts_used": (pb_result.stats or {}).get("segment_backoff_attempts_used"),
                            "pool_diagnostics": {
                                "admitted": diag.pool_diagnostics.admitted,
                                "rejected_sonic": diag.pool_diagnostics.rejected_sonic,
                                "rejected_genre": diag.pool_diagnostics.rejected_genre,
                                "total_considered": diag.pool_diagnostics.total_considered,
                            },
                        },
                    )
                    audit.flush()
                    raise ValueError(
                        f"{diag.diagnostic_msg}\n\n(Detailed audit: {audit.path})"
                    )
                raise ValueError(diag.diagnostic_msg)

            # 6. Log diagnostics
            logger.info(
                "Pier bridge result: %d tracks, %d segments, success=%s",
                len(pb_result.track_indices),
                len(pb_result.segment_diagnostics),
                pb_result.success,
            )
            if not pb_result.success:
                logger.warning("Pier bridge did not fully succeed: %s", pb_result.failure_reason)

            # 7. Build PlaylistResult-compatible output (no repair, no construct_playlist)
            # We create a minimal PlaylistResult to match the expected interface
            pb_track_indices = np.array(pb_result.track_indices, dtype=np.int32)
            artist_counts: Dict[str, int] = {}
            for idx in pb_track_indices.tolist():
                artist_key = str(bundle.artist_keys[int(idx)]) if bundle.artist_keys is not None else ""
                artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1

            # Edge scores (T) produced by pier-bridge builder (matches its scoring)
            edge_scores_list = list((pb_result.stats or {}).get("edge_scores") or [])
            t_vals = [
                float(e.get("T"))
                for e in edge_scores_list
                if isinstance(e, dict) and isinstance(e.get("T"), (int, float))
            ]
            below_floor_count = (
                sum(1 for v in t_vals if v < float(pb_cfg.transition_floor))
                if t_vals
                else 0
            )
            min_transition = float(min(t_vals)) if t_vals else None
            mean_transition = float(sum(t_vals) / len(t_vals)) if t_vals else None
            layered_transition_diagnostics = build_layered_transition_diagnostics(
                bundle=bundle,
                track_indices=pb_track_indices.tolist(),
                edge_scores=edge_scores_list,
                mode=mode,
                enabled=genre_graph_source in {"layered_shadow", "layered"},
            )

            # Receipt (GUI dials, 2026-07-04): admitted/considered/genre-rescued counts
            # from the FINAL settled candidate pool (post bangers relax-to-fill and
            # post one-each-candidate retry — `pool` may have been reassigned above).
            receipt_pool_stats = {
                "admitted": int(len(getattr(pool, "eligible_indices", pool.pool_indices))),
                "considered": int((pool.stats or {}).get("total_candidates_considered", 0) or 0),
                "genre_rescued": int((pool.stats or {}).get("genre_rescue_admitted", 0) or 0),
            }

            # Instrumental lean confession (Task 11): count of vocal-classified
            # tracks (voice_prob > threshold) admitted into the FINAL playlist as
            # a last resort. Seed/pier tracks are excluded — the listener asked
            # for those explicitly, so instrumental lean never second-guesses a
            # seed. Only computed when the guard was actually active this run
            # (mirrors the "missing key -> no notes" contract compose_receipt
            # relies on).
            instrumental_stat: Optional[Dict[str, Any]] = None
            if (
                bool(getattr(cfg.candidate, "instrumental_enabled", False))
                and voice_prob is not None
                and np.isfinite(voice_prob).any()
            ):
                _instr_threshold = 0.5
                _instr_seed_ids = {str(t) for t in seed_track_ids_for_pier}
                _instr_admitted = 0
                for _instr_idx in pb_track_indices.tolist():
                    if str(bundle.track_ids[_instr_idx]) in _instr_seed_ids:
                        continue
                    _vp = voice_prob[_instr_idx]
                    if np.isfinite(_vp) and _vp > _instr_threshold:
                        _instr_admitted += 1
                instrumental_stat = {
                    "enabled": True,
                    "admitted_count": _instr_admitted,
                    "threshold": _instr_threshold,
                }

            # Create minimal PlaylistResult
            playlist = PlaylistResult(
                track_indices=pb_track_indices,
                stats={
                    "strategy": "pier_bridge",
                    "num_segments": len(pb_result.segment_diagnostics),
                    "success": pb_result.success,
                    "failure_reason": pb_result.failure_reason,
                    "segment_diagnostics": [s.__dict__ for s in pb_result.segment_diagnostics],
                    "edge_scores": edge_scores_list,
                    "transition_floor": float(pb_cfg.transition_floor),
                    "transition_gamma": float(cfg.construct.transition_gamma),
                    "transition_centered": bool(pb_cfg.center_transitions),
                    # SP-B: tower transition weights removed from the beam (plain-cosine
                    # sonic space); key kept for downstream consumers that still read it.
                    "transition_weights": None,
                    "below_floor_count": below_floor_count,
                    "min_transition": min_transition,
                    "mean_transition": mean_transition,
                    "artist_counts": artist_counts,
                    "distinct_artists": len(artist_counts),
                    "repair_applied": bool((pb_result.stats or {}).get("edge_repair_applied")),
                    "edge_repair_enabled": bool((pb_result.stats or {}).get("edge_repair_enabled")),
                    "edge_repair_swap_log": (pb_result.stats or {}).get("edge_repair_swap_log") or [],
                    "warnings": (pb_result.stats or {}).get("warnings") or [],
                    "one_each_candidate_relaxation": one_each_candidate_relaxation,
                    "beam_edge_components": (pb_result.stats or {}).get("beam_edge_components") or [],
                    "bpm_summary": (pb_result.stats or {}).get("bpm_summary"),
                    # Corridor pooling diagnostics (Phase 1 Task 8: corridor
                    # is the sole pooling path).
                    "pooling_strategy": (pb_result.stats or {}).get("pooling_strategy"),
                    "corridor_segments": (pb_result.stats or {}).get("corridor_segments") or [],
                    # Task 5 req 0's wiring test caught this missing: without it,
                    # corridor_universe_size never reached a production-path
                    # caller (build_pier_bridge_playlist's raw stats dict has it;
                    # this per-run summarization layer just wasn't copying it).
                    "corridor_universe_size": (pb_result.stats or {}).get("corridor_universe_size"),
                    "layered_transition_diagnostics": layered_transition_diagnostics,
                    "receipt_pool": receipt_pool_stats,
                },
                params_requested={"strategy": "pier_bridge"},
                params_effective={
                    "strategy": "pier_bridge",
                    "pier_config": pb_cfg.__dict__,
                    "one_each_candidate_relaxation": one_each_candidate_relaxation,
                },
            )
            if instrumental_stat is not None:
                playlist.stats["instrumental"] = instrumental_stat

    # Legacy paths (anchor_builder, standard construct_playlist) have been removed.
    # All playlist construction now goes through pier-bridge.

    ordered_track_ids = [str(bundle.track_ids[i]) for i in playlist.track_indices]

    # Post-order validation: DS ordering must be final (no post-filtering).
    # Hard-invariant audit (Pass 1). These use the SAME values the beam enforced —
    # `max_per_artist` is the DERIVED cap from above, not the raw config literal,
    # and `artist_identity_cfg` is the run's own cfg.
    #
    # Matching VALUES is not sufficient; the SCOPE has to match too. This comment
    # used to claim the validator "cannot report a violation the beam never
    # committed", and that was wrong: min_gap governs what the beam places BETWEEN
    # piers, but the validator checked every pair including pier-vs-pier, so artist
    # mode flagged the seed artist against its own piers on every run with
    # min_gap > pier spacing (Cut Worms, 2026-07-25 — positions [1,7,13,19,25,30] at
    # min_gap=9, which needs 46 slots in a 30-track playlist). run_post_order_validation
    # now derives pier positions from seed_track_ids_for_pier and exempts pier-vs-pier
    # only. Note the cap already had this exemption via the single_artist branch above
    # (max_artist_fraction_final=1.0); the gap check simply never got the equivalent.
    validation = run_post_order_validation(
        bundle=bundle,
        ordered_track_ids=ordered_track_ids,
        expected_length=int(playlist_len),
        excluded_track_ids=excluded_track_ids,
        seed_track_ids_for_pier=seed_track_ids_for_pier,
        artist_identity_cfg=artist_identity_cfg,
        min_gap=int(getattr(cfg.construct, "min_gap", 1) or 1),
        max_tracks_per_artist=int(max_per_artist),
        max_tracks_per_artist_overrides=artist_cap_overrides,
        # Pier ratio is RECORDED, not warned on: max_artist_fraction_final is a
        # per-artist cap, not a pier-ratio target, and mini-piers legitimately push
        # the count above the derived base. Pass a real band here once one has been
        # chosen from the recorded `pier_ratio_milli` across the genre corpus.
        pier_ratio_target=0.0,
    )
    post_order_validation = validation.summary

    if validation.errors:
        msg = "post_order_validation_failed: " + " | ".join(validation.errors)
        if audit.can_flush():
            audit.append(
                "final_failure",
                {
                    "failure_reason": msg,
                    "post_order_filters_applied": [],
                    "post_order_validation": dict(post_order_validation),
                },
            )
            audit.flush()
            raise ValueError(f"{msg} (audit: {audit.path})")
        raise ValueError(msg)
    try:
        if allowed_track_ids_set is not None:
            logger.info(
                "Allowed pool enforcement: allowed=%d playlist=%d",
                len(allowed_track_ids_set),
                len(ordered_track_ids),
            )
            enforce_allowed_invariant(ordered_track_ids, allowed_track_ids_set, context="final_playlist")
    except Exception as exc:
        if audit.can_flush():
            audit.append(
                "final_failure",
                {
                    "failure_reason": f"allowed_set_invariant_failed: {exc}",
                },
            )
            audit.flush()
            raise ValueError(f"{exc} (audit: {audit.path})") from exc
        raise

    if audit.can_flush() and bool(audit_cfg.write_on_success):
        playlist_tracks: list[dict[str, Any]] = []
        for idx in playlist.track_indices.tolist():
            artist = ""
            title = ""
            try:
                if bundle.artist_keys is not None:
                    artist = str(bundle.artist_keys[int(idx)])
                if bundle.track_titles is not None:
                    title = str(bundle.track_titles[int(idx)])
            except Exception:
                artist = ""
                title = ""
            playlist_tracks.append(
                {
                    "track_id": str(bundle.track_ids[int(idx)]),
                    "artist": artist,
                    "title": title,
                }
            )

        weakest_edges: list[dict[str, Any]] = []
        try:
            weakest_edges = sorted(
                [e for e in (edge_scores_list or []) if isinstance(e, dict) and isinstance(e.get("T"), (int, float))],
                key=lambda e: float(e.get("T")),
            )[: min(10, len(edge_scores_list or []))]
        except Exception:
            weakest_edges = []

        audit.append(
            "final_success",
            {
                "post_order_filters_applied": [],
                "post_order_validation": dict(post_order_validation),
                "playlist_tracks": playlist_tracks,
                "weakest_edges": weakest_edges,
                "summary_stats": {
                    "final_playlist_size": int(len(ordered_track_ids)),
                    "transition_floor": float(getattr(pb_cfg, "transition_floor", 0.0)),
                    "below_floor_count": int(below_floor_count),
                    "min_transition": min_transition,
                    "mean_transition": mean_transition,
                    "segment_bridge_floors_used": (pb_result.stats or {}).get("segment_bridge_floors_used"),
                    "segment_backoff_attempts_used": (pb_result.stats or {}).get("segment_backoff_attempts_used"),
                    "soft_genre_penalty_hits": (pb_result.stats or {}).get("soft_genre_penalty_hits"),
                    "soft_genre_penalty_edges_scored": (pb_result.stats or {}).get("soft_genre_penalty_edges_scored"),
                    "post_order_filters_applied": [],
                    "post_order_validation": dict(post_order_validation),
                    "receipt_pool": receipt_pool_stats,
                },
            },
        )
        audit.flush()
    params_requested = _params_from_config(cfg)
    if overrides:
        params_requested["overrides"] = overrides
    params_effective: Dict[str, Any] = {
        "sonic_variant": variant_stats,
        "embedding": embedding.embedding_model.params_effective,
        "candidate_pool": pool.params_effective,
        "playlist": playlist.params_effective,
    }
    stats: Dict[str, Any] = {
        "candidate_pool": pool.stats,
        "playlist": playlist.stats,
    }
    if audit.path is not None:
        try:
            stats.setdefault("playlist", {})
            stats["playlist"]["audit_path"] = str(audit.path)
        except Exception:
            pass

    # Hard-invariant warnings ride out on stats so callers can surface a degraded
    # result. Deliberately NOT in `errors` — a breach marks the run degraded, it
    # does not withhold the playlist (spec 2026-07-25-genre-mode-pass1 §4.2).
    try:
        stats.setdefault("playlist", {})
        stats["playlist"]["invariant_warnings"] = list(validation.warnings)
        stats["playlist"]["degraded"] = bool(validation.warnings)
    except Exception:
        pass

    return DSPipelineResult(
        track_ids=ordered_track_ids,
        track_indices=playlist.track_indices.tolist(),
        stats=stats,
        params_requested=params_requested,
        params_effective=params_effective,
    )


