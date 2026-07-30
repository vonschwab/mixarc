#!/usr/bin/env python3
"""
GUI Worker Process - NDJSON Protocol Handler (v1)

This worker runs as a separate process, receiving commands via stdin and
emitting events via stdout using newline-delimited JSON (NDJSON).

Protocol Version: 1
  - All commands include request_id for correlation
  - All events include request_id to match the originating command
  - Cancel command for cooperative cancellation

Commands (GUI -> Worker, one JSON per line):
  {"cmd":"ping","request_id":"<uuid>","protocol_version":1}
  {"cmd":"generate_playlist","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{...},"args":{...}}
  {"cmd":"cancel","request_id":"<uuid-to-cancel>"}
  {"cmd":"scan_library","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"update_genres","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"update_sonic","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"build_artifacts","request_id":"<uuid>","protocol_version":1,"base_config_path":"config.yaml","overrides":{}}
  {"cmd":"find_replacement_suggestions","request_id":"<uuid>","protocol_version":1,"position":3,"mode":"best","top_k":10}

Events (Worker -> GUI, one JSON per line):
  {"type":"log","request_id":"<uuid>","level":"INFO","msg":"..."}
  {"type":"progress","request_id":"<uuid>","stage":"...","current":...,"total":...}
  {"type":"result","request_id":"<uuid>","result_type":"playlist","playlist":{...}}
  {"type":"error","request_id":"<uuid>","message":"...","traceback":"..."}
  {"type":"done","request_id":"<uuid>","cmd":"...","ok":true}
  {"type":"done","request_id":"<uuid>","cmd":"...","ok":false,"cancelled":true}

Security:
  - Never emit secrets (API keys, tokens) in events
  - Redact secret values from error tracebacks
"""
import json
import logging
import subprocess
import sys
import sqlite3
import tempfile
import threading
import traceback
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.playlist.analyze_library_results import (
    parse_analyze_library_paused,
    parse_analyze_library_report,
    parse_analyze_library_stage_progress,
)
from src.config_loader import DEFAULT_MIN_TRACK_DURATION_SECONDS, resolve_database_path
from src.playlist.config import resolve_cohesion_mode
from src.playlist.request_models import GeneratePlaylistRequest, LibraryPipelineRequest
from .utils.redaction import redact_text

try:
    from ..cancellation import CancellationToken, CancelledException
except ImportError:
    try:
        from src.cancellation import CancellationToken, CancelledException
    except ImportError:
        CancellationToken = None

        class CancelledException(Exception):
            """Fallback cancellation type; never aliases broad Exception."""

            pass


# Protocol version
PROTOCOL_VERSION = 1

# Path to the AI genre enrichment sidecar DB (relative to project root / cwd)
SIDECAR_DB_PATH = "data/ai_genre_enrichment.db"


# ─────────────────────────────────────────────────────────────────────────────
# Worker State Management
# ─────────────────────────────────────────────────────────────────────────────

class CancellationError(Exception):
    """Raised when a command is cancelled."""
    pass


@dataclass
class WorkerState:
    """
    Manages worker state for request tracking and cancellation.

    Thread-safe for cancel requests that may arrive during command execution.
    """
    current_request_id: Optional[str] = None
    current_job_id: Optional[str] = None
    current_cmd: Optional[str] = None
    cancel_requested: bool = False
    _lock: threading.Lock = None

    def __post_init__(self):
        self._lock = threading.Lock()

    def start_request(self, request_id: str, cmd: str, job_id: Optional[str]) -> None:
        """Mark a new request as active."""
        with self._lock:
            self.current_request_id = request_id
            self.current_cmd = cmd
            self.current_job_id = job_id
            self.cancel_requested = False

    def end_request(self) -> None:
        """Clear the active request."""
        with self._lock:
            self.current_request_id = None
            self.current_cmd = None
            self.current_job_id = None
            self.cancel_requested = False

    def request_cancel(self, request_id: str) -> bool:
        """
        Request cancellation for a specific request_id.

        Returns True if the request_id matches the active request.
        """
        with self._lock:
            if self.current_request_id == request_id:
                self.cancel_requested = True
                return True
            return False

    def cancel_active(self) -> None:
        """Request cancellation of whatever request is currently active.

        Unlike request_cancel, this does not match a specific request_id — it is
        used on shutdown (stdin EOF) to signal any in-flight tracked command to
        unwind at its next checkpoint so the worker can exit with its parent.
        """
        with self._lock:
            self.cancel_requested = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        with self._lock:
            return self.cancel_requested

    def check_cancelled(self) -> None:
        """
        Check if cancelled and raise CancellationError if so.

        Call this at stage boundaries in command handlers.
        """
        if self.is_cancelled():
            raise CancellationError("Operation cancelled by user")

    def get_request_id(self) -> Optional[str]:
        """Get the current request_id."""
        with self._lock:
            return self.current_request_id

    def get_job_id(self) -> Optional[str]:
        """Get the current job_id, if any."""
        with self._lock:
            return self.current_job_id


# Global worker state
_worker_state = WorkerState()

# Global cancellation token for current operation
_current_cancellation_token: Optional[CancellationToken] = None
_token_lock = threading.Lock()

# Serializes stdout writes: tracked commands run on a worker thread while the
# reader thread dispatches untracked commands (cancel) inline, so two threads can
# emit NDJSON concurrently. Without this lock their lines could interleave.
_stdout_lock = threading.Lock()


@dataclass
class _LastGenerationCache:
    """In-memory snapshot of the most recent generated playlist."""

    playlist_id: Optional[str] = None
    db_path: Optional[str] = None
    track_ids: Optional[np.ndarray] = None
    artist_keys: Optional[np.ndarray] = None
    X_sonic: Optional[np.ndarray] = None
    X_full: Optional[np.ndarray] = None
    X_start: Optional[np.ndarray] = None
    X_mid: Optional[np.ndarray] = None
    X_end: Optional[np.ndarray] = None
    X_genre_smoothed: Optional[np.ndarray] = None
    genre_vocab: Optional[np.ndarray] = None
    perceptual_bpm: Optional[np.ndarray] = None
    tempo_stability: Optional[np.ndarray] = None
    candidate_pool_indices: Optional[np.ndarray] = None
    idf_weights: Optional[np.ndarray] = None
    transition_metric_context: Any = None
    transition_floor: float = 0.20
    playlist_track_ids: Optional[List[str]] = None


_LAST_GENERATION_CACHE = _LastGenerationCache()


def set_cancellation_token(token: Optional[CancellationToken]) -> None:
    """Set the cancellation token for the current operation."""
    global _current_cancellation_token
    with _token_lock:
        _current_cancellation_token = token


def get_cancellation_token() -> Optional[CancellationToken]:
    """Get the current cancellation token."""
    with _token_lock:
        return _current_cancellation_token


def check_cancelled() -> None:
    """Convenience function to check cancellation at stage boundaries."""
    _worker_state.check_cancelled()


# ─────────────────────────────────────────────────────────────────────────────
# Secret Redaction
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Event Emission
# ─────────────────────────────────────────────────────────────────────────────

def emit_event(event: Dict[str, Any]) -> None:
    """
    Emit an event to stdout as NDJSON (one JSON object per line).

    Automatically includes request_id from current worker state.
    """
    # Include request_id if we have an active request
    request_id = _worker_state.get_request_id()
    if request_id and "request_id" not in event:
        event["request_id"] = request_id

    # Include job_id if available
    job_id = _worker_state.get_job_id()
    if job_id and "job_id" not in event:
        event["job_id"] = job_id

    try:
        line = json.dumps(event, default=str)
        with _stdout_lock:
            print(line, flush=True)
    except Exception as e:
        # Fallback for serialization errors
        fallback = {"type": "error", "message": f"Serialization error: {e}"}
        if request_id:
            fallback["request_id"] = request_id
        with _stdout_lock:
            print(json.dumps(fallback), flush=True)


def emit_log(level: str, msg: str, request_id: Optional[str] = None) -> None:
    """Emit a log event."""
    event = {"type": "log", "level": level, "msg": redact_text(msg)}
    if request_id:
        event["request_id"] = request_id
    emit_event(event)


def emit_progress(stage: str, current: int, total: int, detail: Optional[str] = None) -> None:
    """Emit a progress event."""
    event = {"type": "progress", "stage": stage, "current": current, "total": total}
    if detail:
        event["detail"] = detail
    emit_event(event)


def emit_error(message: str, tb: Optional[str] = None) -> None:
    """Emit an error event with redacted traceback."""
    event = {"type": "error", "message": redact_text(message)}
    if tb:
        event["traceback"] = redact_text(tb)
        # Also emit full traceback to the log stream so GUI output isn't truncated
        emit_log("DEBUG", f"Traceback detail:\n{tb}")
    emit_event(event)


def emit_result(result_type: str, data: Dict[str, Any]) -> None:
    """Emit a result event."""
    emit_event({"type": "result", "result_type": result_type, **data})


# Backwards-compatible alias expected by legacy tests
def redact_secrets_in_text(text: str) -> str:
    return redact_text(text)


def emit_done(
    cmd: str,
    ok: bool,
    detail: Optional[str] = None,
    cancelled: bool = False,
    summary: Optional[str] = None,
) -> None:
    """Emit a done event indicating command completion."""
    event = {"type": "done", "cmd": cmd, "ok": ok}
    if detail:
        event["detail"] = detail
    if summary:
        event["summary"] = summary
    if cancelled:
        event["cancelled"] = True
    emit_event(event)


def emit_checkpoint(stage: str, items_completed: int, resumable_state: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit a checkpoint event for resumable operations.

    Args:
        stage: Stage name (e.g., 'scan', 'genres')
        items_completed: Number of items completed so far
        resumable_state: Optional state data for resumption (e.g., last file processed)
    """
    event = {
        "type": "checkpoint",
        "stage": stage,
        "items_completed": items_completed,
    }
    if resumable_state:
        event["resumable_state"] = resumable_state
    emit_event(event)


def emit_verbose_log(level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit a verbose log event (only sent when verbose mode is enabled).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: Log message
        context: Optional context data (file path, item name, etc.)
    """
    import datetime
    event = {
        "type": "verbose_log",
        "level": level,
        "message": redact_text(message),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if context:
        event["context"] = context
    emit_event(event)


def emit_performance(stage: str, elapsed_seconds: float, items_processed: int, throughput: Optional[float] = None) -> None:
    """
    Emit a performance metrics event.

    Args:
        stage: Stage name (e.g., 'scan', 'genres')
        elapsed_seconds: Time elapsed for this stage
        items_processed: Number of items processed
        throughput: Optional throughput (items/sec), calculated if not provided
    """
    if throughput is None and elapsed_seconds > 0:
        throughput = items_processed / elapsed_seconds

    event = {
        "type": "performance",
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "items_processed": items_processed,
        "throughput": round(throughput, 2) if throughput else 0.0,
    }
    emit_event(event)


# ─────────────────────────────────────────────────────────────────────────────
# Logging Integration
# ─────────────────────────────────────────────────────────────────────────────

class WorkerLogHandler(logging.Handler):
    """Logging handler that emits logs as NDJSON events."""

    def emit(self, record):
        try:
            msg = self.format(record)
            emit_log(record.levelname, msg)
        except Exception:
            pass  # Avoid recursion if logging fails


def setup_worker_logging():
    """Configure logging to emit via NDJSON."""
    root = logging.getLogger()
    # Root must allow DEBUG through so a per-playlist log FileHandler
    # (attached later, per-request, by src.logging_utils.playlist_log_file)
    # can actually receive DEBUG records. The NDJSON handler below stays at
    # INFO, so the GUI itself is unaffected -- DEBUG-to-file, INFO-to-GUI.
    root.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add our NDJSON handler
    handler = WorkerLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────────────────────────────────────

# Corridor Phase 0 (2026-07): candidate_pool's per-artist cap walk and
# never-starve backstop were deleted (Task 1: src/playlist/candidate_pool.py),
# and the min_pool_size preset/resolution chains that fed them -- plus the
# reviewer-confirmed-dead PierBridgeTuning.min_pool_size family -- were
# deleted with them (Task 2: src/playlist/mode_presets.py,
# src/playlist/pipeline/core.py). The dataclass fields themselves
# (CandidatePoolConfig.min_pool_size, PierBridgeTuning.min_pool_size) were
# kept as inert placeholders so old config.yaml values don't raise, per the
# "configured knob that can't act is a startup error, not a silent no-op"
# rule -- this is that startup error. Two distinct yaml locations can carry
# retired keys:
#   - playlists.ds_pipeline.candidate_pool.* -- fed default_ds_config's
#     per-artist cap walk (candidates_per_artist/target_artists/
#     seed_artist_bonus were dead outlets even before Phase 0 -- see
#     scripts/corridor_baseline/perturb.py's _CANDIDATE_POOL_FIELD_MAP --
#     max_pool_size/min_pool_size were live until Task 1/2 deleted their
#     consumers)
#   - playlists.ds_pipeline.pier_bridge.min_pool_size (+ cohesion_mode-suffixed
#     variants _strict/_narrow/_dynamic/_discover, resolved via
#     _resolve_mode_number_with_source) -- fed PierBridgeTuning.min_pool_size,
#     which has zero readers in pier_bridge/config.py, pier_bridge/beam.py, or
#     pier_bridge_builder.py (Task 2 reviewer finding)
_RETIRED_CANDIDATE_POOL_KEYS = {
    "candidates_per_artist": "pool artist-cap removed (corridor Phase 0); beam enforces diversity",
    "target_artists": "pool artist-cap removed (corridor Phase 0); beam enforces diversity",
    "max_pool_size": "global pool size cap removed (corridor Phase 0); segment_pool_max bounds beam input",
    "min_pool_size": "never-starve backstop removed (corridor Phase 0); quality-triggered widening replaces it (corridor Phase 1)",
    "seed_artist_bonus": "pool artist-cap removed (corridor Phase 0); only fed the deleted cap allocation",
}

_RETIRED_PIER_BRIDGE_MIN_POOL_SIZE_KEYS = {
    "min_pool_size": "min_pool_size resolution chain removed (corridor Phase 0 Task 2); PierBridgeTuning.min_pool_size has no reader",
    "min_pool_size_strict": "min_pool_size resolution chain removed (corridor Phase 0 Task 2); PierBridgeTuning.min_pool_size has no reader",
    "min_pool_size_narrow": "min_pool_size resolution chain removed (corridor Phase 0 Task 2); PierBridgeTuning.min_pool_size has no reader",
    "min_pool_size_dynamic": "min_pool_size resolution chain removed (corridor Phase 0 Task 2); PierBridgeTuning.min_pool_size has no reader",
    "min_pool_size_discover": "min_pool_size resolution chain removed (corridor Phase 0 Task 2); PierBridgeTuning.min_pool_size has no reader",
}

# Corridor Phase 1 Task 8 (the flip): corridor pooling is now the SOLE
# pier-bridge pooling path -- the "legacy"/"corridor" dev flag and every
# legacy segment-pool builder it selected between (_build_segment_candidate_
# pool_legacy, _build_segment_candidate_pool_scored, SegmentCandidatePool
# Builder) were deleted, along with the legacy per-segment relaxation ladder
# (infeasible_handling's bridge_floor backoff / transition-floor / genre-arc-
# floor tiers -- the corridor widening ladder is the sole segment-level
# recovery mechanism now) and the artist-mode external candidate pool
# (build_balanced_candidate_pool -- corridor's library-wide eligible universe
# replaces it).
_RETIRED_PIER_BRIDGE_TOP_LEVEL_KEYS = {
    "pooling": "the legacy|corridor dev flag was deleted (corridor Phase 1 Task 8); corridor is the sole pooling path",
    "collapse_segment_pool_by_artist": "read only inside the deleted legacy segment-scored pool builder (corridor Phase 1 Task 8); the beam enforces per-segment artist diversity on its own",
}

# infeasible_handling.* (nested under playlists.ds_pipeline.pier_bridge):
# every sub-field EXCEPT guarantee_feasible/greedy_genre_weight was only
# consumed by the deleted bridge_floor-backoff / transition-floor / genre-
# arc-floor relaxation tiers. guarantee_feasible (terminal last-resort
# placement) and greedy_genre_weight (its genre blend weight) remain live
# and are NOT retired.
_RETIRED_PIER_BRIDGE_INFEASIBLE_HANDLING_KEYS = {
    k: "the legacy per-segment relaxation ladder was deleted (corridor Phase 1 Task 8); the corridor widening ladder (corridor_widen_*) is the sole segment-level recovery mechanism"
    for k in (
        "enabled", "strategy", "min_bridge_floor", "backoff_steps",
        "max_attempts_per_segment", "widen_search_on_backoff",
        "extra_neighbors_m", "extra_bridge_helpers", "extra_beam_width",
        "extra_expansion_attempts", "transition_floor_relaxation_enabled",
        "min_transition_floor", "genre_arc_relaxation_enabled",
        "min_genre_arc_percentile",
    )
}

# dj_bridging.pooling.* (nested under playlists.ds_pipeline.pier_bridge):
# the "dj_union" segment-pool strategy's implementation lived entirely
# inside the deleted legacy segment-scored pool builder. `strategy`/
# `cache_enabled` are NOT retired (cache_enabled still backs the live
# dj_connectors cache; `strategy` itself is still read for diagnostics).
_RETIRED_PIER_BRIDGE_DJ_POOLING_KEYS = {
    k: "the dj_union segment-pool strategy's implementation was deleted with the legacy segment-scored pool builder (corridor Phase 1 Task 8)"
    for k in ("step_stride", "debug_compare_baseline")
}
# k_local/k_toward/k_genre/k_union_max (final-review correction,
# corridor-phase1-pooling, 2026-07-18): NOT simply unread. micro_pier.py's
# `_build_dj_relaxation_attempts` ("relax_effort" attempt, :57-61) still
# reads all four into a scaled derived config (`* pool_scale`, e.g.
# `effort_cfg.dj_pooling_k_local`) -- but that derived value is itself never
# consumed by anything: the dj_union pool implementation it would have
# configured was deleted in Task 8, and `_attempt_micro_pier_split` only
# reaches this attempt when `dj_bridging_enabled` is on (off by default,
# superseded by taxonomy steering -- CLAUDE.md Layer 3 item 16). So a value
# set here is still a no-op in config.yaml, just via an inert scale-and-
# discard rather than a plain unread key.
_RETIRED_PIER_BRIDGE_DJ_POOLING_KEYS.update({
    k: "reads into an inert scaled copy in micro_pier.py's relax_effort DJ-bridging relaxation attempt (dj_bridging_enabled is off by default) -- the dj_union segment-pool strategy that copy would have configured was deleted with the legacy segment-scored pool builder (corridor Phase 1 Task 8), so the derived value is never consumed"
    for k in ("k_local", "k_toward", "k_genre", "k_union_max")
})

# playlists.ds_pipeline.artist_style.*: fed build_balanced_candidate_pool,
# the per-cluster sonic-neighbor "external pool" that hard-clamped Artist
# mode's allowed_track_ids -- deleted (corridor Phase 1 Task 8); corridor's
# library-wide eligible universe replaces it.
_RETIRED_ARTIST_STYLE_KEYS = {
    "per_cluster_candidate_pool_size": "build_balanced_candidate_pool deleted (corridor Phase 1 Task 8); corridor's eligible universe replaces the artist-mode external pool",
    "pool_balance_mode": "build_balanced_candidate_pool deleted (corridor Phase 1 Task 8); pool_balance_mode's only non-default value (proportional_capped) was never implemented anyway",
}

# playlists.ds_pipeline.multi_artist.alternation_bonus: forced-interleaving
# rewrite (2026-07-30, docs/superpowers/sdd/2026-07-29-multi-artist-blend/
# forced-interleaving-report.md). order_with_alternation used to score
# candidate orders with `sonic path cost + alternation_bonus * artist
# changes` -- but every candidate came from order_clusters' own greedy
# nearest-neighbor walk, which structurally clumps same-artist piers, so no
# amount of bonus weight could ever produce an interleaved order. Alternation
# is now a hard constraint (the maximum achievable alternation for the actual
# group sizes is computed and forced, then the minimax-best order among those
# is picked) -- there is no bonus weight left to tune.
_RETIRED_MULTI_ARTIST_KEYS = {
    "alternation_bonus": "order_with_alternation forces the maximum achievable alternation now (forced-interleaving rewrite, 2026-07-30); alternation is a hard constraint, not a scored bonus, so there is no weight left to tune",
}

_WARNED_RETIRED_KEYS: set[str] = set()  # per-process dedup (final-review finding, corridor Phase 0)


def _warn_retired_section(
    section: Dict[str, Any], prefix: str, retired: Dict[str, str], found: list[str]
) -> None:
    """Check one flat config dict against one retired-key/rationale map,
    warning loudly (once per process, per qualified key) and appending every
    match to ``found``. Shared by every family _warn_retired_keys checks."""
    for key, rationale in retired.items():
        if key in section:
            qualified = f"{prefix}.{key}"
            found.append(qualified)
            if qualified not in _WARNED_RETIRED_KEYS:
                logging.getLogger(__name__).warning(
                    "Config key 'playlists.ds_pipeline.%s' is RETIRED (%s). "
                    "Remove it from config.yaml.", qualified, rationale)
                _WARNED_RETIRED_KEYS.add(qualified)


def _warn_retired_keys(config: Dict[str, Any]) -> list[str]:
    """
    Warn loudly (once per configured key) for every retired candidate-pool /
    pier-bridge / artist-style knob still present in a merged config, naming
    the replacement rationale. Returns the qualified ("<section>.<key>")
    names found, for tests and callers that want to surface this beyond the
    log.

    "A configured knob that can't act is a startup error, not a silent
    no-op" (CLAUDE.md) -- these families were retired in corridor Phase 0
    (Tasks 1-2) and Phase 1 (Task 8, the flip) but their dataclass fields
    were kept as inert placeholders (or, for `pooling`, simply have no
    effect once absent) so old config.yaml values don't raise. This is that
    startup error, downgraded to a warning since the fields are genuinely
    harmless no-ops now (not silent -- logged).
    """
    ds_pipeline = ((config.get("playlists") or {}).get("ds_pipeline") or {})
    found: list[str] = []

    candidate_pool = ds_pipeline.get("candidate_pool") or {}
    _warn_retired_section(candidate_pool, "candidate_pool", _RETIRED_CANDIDATE_POOL_KEYS, found)

    pier_bridge = ds_pipeline.get("pier_bridge") or {}
    _warn_retired_section(pier_bridge, "pier_bridge", _RETIRED_PIER_BRIDGE_MIN_POOL_SIZE_KEYS, found)
    _warn_retired_section(pier_bridge, "pier_bridge", _RETIRED_PIER_BRIDGE_TOP_LEVEL_KEYS, found)

    infeasible_handling = pier_bridge.get("infeasible_handling") or {}
    _warn_retired_section(
        infeasible_handling, "pier_bridge.infeasible_handling",
        _RETIRED_PIER_BRIDGE_INFEASIBLE_HANDLING_KEYS, found,
    )

    dj_pooling = (pier_bridge.get("dj_bridging") or {}).get("pooling") or {}
    _warn_retired_section(
        dj_pooling, "pier_bridge.dj_bridging.pooling",
        _RETIRED_PIER_BRIDGE_DJ_POOLING_KEYS, found,
    )
    # `strategy` itself is NOT a retired key (its "baseline" value is a
    # harmless label with no corridor-era reader either way) -- but the
    # "dj_union" VALUE specifically is now a silent no-op (its
    # implementation died with the rest of this family), which is exactly
    # the "configured knob that can't act" trap a presence-only check would
    # miss. Value-conditional, so handled separately from
    # _warn_retired_section's key-presence checks.
    if str(dj_pooling.get("strategy", "")).strip().lower() == "dj_union":
        qualified = "pier_bridge.dj_bridging.pooling.strategy=dj_union"
        found.append(qualified)
        if qualified not in _WARNED_RETIRED_KEYS:
            logging.getLogger(__name__).warning(
                "Config value 'playlists.ds_pipeline.pier_bridge.dj_bridging."
                "pooling.strategy: dj_union' is RETIRED (the dj_union segment-"
                "pool strategy's implementation was deleted with the legacy "
                "segment-scored pool builder, corridor Phase 1 Task 8); this "
                "now behaves identically to strategy: baseline. Update "
                "config.yaml.")
            _WARNED_RETIRED_KEYS.add(qualified)

    artist_style = ds_pipeline.get("artist_style") or {}
    _warn_retired_section(artist_style, "artist_style", _RETIRED_ARTIST_STYLE_KEYS, found)

    multi_artist = ds_pipeline.get("multi_artist") or {}
    _warn_retired_section(multi_artist, "multi_artist", _RETIRED_MULTI_ARTIST_KEYS, found)

    return found


def load_config_with_overrides(base_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Load base config and merge with overrides."""
    if not Path(base_path).exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Deep merge overrides
    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(config, overrides)

    # Corridor Phase 0: warn loudly on any retired candidate_pool/pier_bridge
    # min_pool_size knob still configured -- see _warn_retired_keys docstring.
    _warn_retired_keys(merged)

    # Apply mode presets (genre_mode/sonic_mode) to resolve weights and thresholds
    _apply_mode_presets(merged)

    # Publish artifacts.* settings to the artifact loader (process-wide).
    # artifacts.sonic_variant_override wins over the artifact-declared sonic
    # variant; a missing variant key raises at artifact load (configured-knob-
    # must-act rule), never a silent fallback.
    from src.features.artifacts import set_sonic_variant_override

    set_sonic_variant_override((merged.get("artifacts") or {}).get("sonic_variant_override"))

    return merged


def _apply_mode_presets(config: Dict[str, Any]) -> None:
    """
    Apply genre_mode and sonic_mode presets if specified in config.
    Modifies config in-place.
    """
    playlists_cfg = config.get('playlists', {})
    if not playlists_cfg:
        return
    try:
        from src.playlist.mode_presets import apply_mode_presets
    except ImportError:
        return

    apply_mode_presets(playlists_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────────────────────────────────────

def _build_seed_similarity_components(
    *,
    tracks: list[dict[str, Any]],
    ds_report: dict[str, Any],
    edge_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Optional[float]]]]:
    """
    Build T/S1/S2 similarity components for sonic + genre.

    T: similarity to immediately preceding track (transition edge score)
    S1: similarity to the most recent seed track before this track
    S2: similarity to the next seed track after this track
    """
    if not ds_report:
        return {}

    artifact_path = ds_report.get("artifact_path")
    seed_track_id = ds_report.get("seed_track_id")
    anchor_seed_ids = ds_report.get("anchor_seed_ids") or []
    if not artifact_path or not seed_track_id:
        return {}

    track_ids: list[str] = []
    for track in tracks:
        tid = track.get("rating_key") or track.get("id") or track.get("track_id")
        track_ids.append(str(tid) if tid is not None else "")

    if not track_ids:
        return {}

    seed_ids = [str(seed_track_id)] + [str(sid) for sid in anchor_seed_ids if sid]
    seed_set = set(seed_ids)

    try:
        from src.features.artifacts import load_artifact_bundle

        bundle = load_artifact_bundle(artifact_path)
    except Exception:
        return {}

    track_idx_by_id = bundle.track_id_to_index
    track_indices = [track_idx_by_id.get(tid) for tid in track_ids]
    seed_positions = [
        idx for idx, tid in enumerate(track_ids)
        if tid in seed_set and track_indices[idx] is not None
    ]
    if not seed_positions:
        return {}

    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    if X_sonic is None:
        return {}

    X_sonic_norm = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
    X_genre_norm = None
    if X_genre is not None:
        denom = np.linalg.norm(X_genre, axis=1, keepdims=True) + 1e-12
        X_genre_norm = X_genre / denom

    def _dot(mat: Optional[np.ndarray], a: Optional[int], b: Optional[int]) -> Optional[float]:
        if mat is None or a is None or b is None:
            return None
        return float(np.dot(mat[a], mat[b]))

    components: dict[str, dict[str, dict[str, Optional[float]]]] = {}
    for idx, track_id in enumerate(track_ids):
        if not track_id:
            continue
        cur_idx = track_indices[idx]
        if cur_idx is None:
            continue

        prev_seed_pos_idx = bisect_right(seed_positions, idx) - 1
        prev_seed_pos = seed_positions[prev_seed_pos_idx] if prev_seed_pos_idx >= 0 else None
        next_seed_pos_idx = bisect_right(seed_positions, idx)
        next_seed_pos = seed_positions[next_seed_pos_idx] if next_seed_pos_idx < len(seed_positions) else None

        prev_seed_idx = track_indices[prev_seed_pos] if prev_seed_pos is not None else None
        next_seed_idx = track_indices[next_seed_pos] if next_seed_pos is not None else None

        s1_sonic = _dot(X_sonic_norm, cur_idx, prev_seed_idx)
        s2_sonic = _dot(X_sonic_norm, cur_idx, next_seed_idx)
        s1_genre = _dot(X_genre_norm, cur_idx, prev_seed_idx)
        s2_genre = _dot(X_genre_norm, cur_idx, next_seed_idx)

        if prev_seed_pos is not None and prev_seed_pos == idx:
            s1_sonic = 1.0
            if X_genre_norm is not None:
                s1_genre = 1.0

        edge = edge_map.get(track_id, {})
        t_sonic = edge.get("S")
        t_genre = edge.get("G")

        components[track_id] = {
            "sonic": {"t": t_sonic, "s1": s1_sonic, "s2": s2_sonic},
            "genre": {"t": t_genre, "s1": s1_genre, "s2": s2_genre},
        }

    return components


def _resolve_track_genres(
    track: Dict[str, Any],
    *,
    sidecar_db_path: str,
    fallback,
    db_path: Optional[str] = None,
) -> List[str]:
    """Display genres for a track: published authority -> signature -> fallback.

    The published authority (release_effective_genres in metadata.db, maintained
    by the publish stage) is THE genre output and is consulted first when
    db_path is given. The sidecar signature is the older bandcamp-era layer —
    often sparse/stale while the authority is rich — kept only as a fallback
    for releases enriched but not yet published. fallback is a no-arg callable
    returning the raw genres list.
    """
    from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

    track_id = track.get("rating_key") or track.get("id") or track.get("track_id")
    if db_path and track_id:
        from src.genre.authority import display_genre_names_for_track

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                published = display_genre_names_for_track(conn, str(track_id))
            finally:
                conn.close()
        except sqlite3.Error:
            published = []
        if published:
            return published

    artist = track.get("artist") or ""
    album = track.get("album") or ""
    if not artist or not album:
        return list(fallback() or [])
    resolver = EnrichedGenreResolver(sidecar_db_path)
    enriched = resolver.get_enriched_genres(artist=artist, album=album)
    if enriched:
        return enriched
    return list(fallback() or [])


def _resolve_display_genres(
    track: Dict[str, Any],
    *,
    sidecar_db_path: str,
    fallback,
    db_path: Optional[str] = None,
) -> List[str]:
    """Display genres for a track: resolved (authority -> enriched -> fallback),
    then canonicalized through the taxonomy and ordered most-specific first.

    order_genres_for_display applies the raw-tags safety fallback and never
    raises (degrades to raw tags if the taxonomy is unavailable).
    """
    from src.genre.granularity import order_genres_for_display

    return order_genres_for_display(
        _resolve_track_genres(
            track, sidecar_db_path=sidecar_db_path, fallback=fallback, db_path=db_path
        )
    )


def _reset_last_generation_cache() -> None:
    global _LAST_GENERATION_CACHE
    _LAST_GENERATION_CACHE = _LastGenerationCache()


def _replacement_pool_indices_from_report(
    *,
    bundle: Any,
    ds_report: dict[str, Any],
    playlist_indices: set[int],
) -> np.ndarray:
    """Recover the DS-admitted candidate pool from the last generation report."""
    playlist_stats = ds_report.get("playlist_stats") or {}
    pool_stats = playlist_stats.get("candidate_pool") or ds_report.get("candidate_pool") or {}
    admitted_track_ids = pool_stats.get("seed_sonic_sim_track_ids") or {}
    if isinstance(admitted_track_ids, dict) and admitted_track_ids:
        indices = [
            int(bundle.track_id_to_index[str(track_id)])
            for track_id in admitted_track_ids.keys()
            if str(track_id) in bundle.track_id_to_index
        ]
        return np.array(list(dict.fromkeys(indices)), dtype=int)

    return np.array(
        [idx for idx in range(int(bundle.track_ids.shape[0])) if idx not in playlist_indices],
        dtype=int,
    )


def _populate_last_generation_cache(
    *,
    generator: Any,
    playlist_result: dict[str, Any],
    db_path: str,
) -> None:
    """Capture read-only generation artifacts needed for replacement scoring."""
    _reset_last_generation_cache()

    ds_report = getattr(generator, "_last_ds_report", None) or {}
    artifact_path = ds_report.get("artifact_path")
    if not artifact_path:
        emit_log("DEBUG", "Replacement cache skipped: no DS artifact path in last report")
        return

    from src.features.artifacts import load_artifact_bundle
    from src.playlist.bpm_loader import load_bpm_arrays
    from src.playlist.transition_metrics import build_transition_metric_context, resolve_transition_calib

    bundle = load_artifact_bundle(artifact_path)
    playlist_track_ids = [
        str(track.get("rating_key") or track.get("track_id") or track.get("id"))
        for track in playlist_result.get("tracks", [])
        if track.get("rating_key") or track.get("track_id") or track.get("id")
    ]
    if len(playlist_track_ids) < 3:
        emit_log("DEBUG", "Replacement cache skipped: playlist too short")
        return

    try:
        bpm_arrays = load_bpm_arrays(bundle.track_ids, db_path=db_path)
        perceptual_bpm = bpm_arrays.get("perceptual_bpm")
        tempo_stability = bpm_arrays.get("tempo_stability")
    except Exception as exc:
        emit_log("WARNING", f"Replacement cache BPM load failed: {exc}")
        perceptual_bpm = None
        tempo_stability = None

    playlist_indices = {
        int(bundle.track_id_to_index[tid])
        for tid in playlist_track_ids
        if tid in bundle.track_id_to_index
    }
    candidate_pool_indices = _replacement_pool_indices_from_report(
        bundle=bundle,
        ds_report=ds_report,
        playlist_indices=playlist_indices,
    )

    playlist_stats = (ds_report.get("playlist_stats") or {}).get("playlist") or {}
    transition_floor = playlist_stats.get("transition_floor") or ds_report.get("transition_floor") or 0.20

    # Calib must track the ACTIVE variant's cosine band (MuQ hot vs MERT) or the
    # rescale saturates — resolve from bundle.sonic_variant (authoritative mert/muq),
    # exactly as the beam does (pier_bridge_builder.py:490); build_transition_metric_context
    # otherwise defaults to muq's 0.594/0.092 and every replacement-scored edge collapses to ~1.0.
    _cal_c, _cal_s, _cal_g = resolve_transition_calib(getattr(bundle, "sonic_variant", None))
    transition_metric_context = build_transition_metric_context(
        X_sonic=bundle.X_sonic,
        X_start=bundle.X_sonic_start,
        X_mid=bundle.X_sonic_mid,
        X_end=bundle.X_sonic_end,
        X_genre=bundle.X_genre_smoothed,
        center_transitions=bool(
            playlist_stats.get("transition_centered")
            or ds_report.get("transition_centered")
        ),
        transition_gamma=playlist_stats.get("transition_gamma") or ds_report.get("transition_gamma"),
        calib_center=_cal_c,
        calib_scale=_cal_s,
        calib_gain=_cal_g,
    )

    cache = _LAST_GENERATION_CACHE
    cache.playlist_id = str(playlist_result.get("name") or "")
    cache.db_path = str(db_path)
    cache.track_ids = bundle.track_ids
    cache.artist_keys = bundle.artist_keys
    cache.X_sonic = bundle.X_sonic
    cache.X_full = bundle.X_sonic
    cache.X_start = bundle.X_sonic_start
    cache.X_mid = bundle.X_sonic_mid
    cache.X_end = bundle.X_sonic_end
    cache.X_genre_smoothed = bundle.X_genre_smoothed
    cache.genre_vocab = bundle.genre_vocab
    cache.perceptual_bpm = perceptual_bpm
    cache.tempo_stability = tempo_stability
    cache.candidate_pool_indices = candidate_pool_indices
    cache.transition_metric_context = transition_metric_context
    cache.transition_floor = float(transition_floor)
    cache.playlist_track_ids = playlist_track_ids


def _top_genres_for_index(cache: _LastGenerationCache, idx: int, limit: int = 3) -> list[str]:
    if cache.X_genre_smoothed is None or cache.genre_vocab is None:
        return []
    vec = np.asarray(cache.X_genre_smoothed[int(idx)], dtype=float)
    if vec.size == 0:
        return []
    order = np.argsort(vec)[::-1]
    genres: list[str] = []
    for genre_idx in order:
        if len(genres) >= limit:
            break
        if float(vec[int(genre_idx)]) <= 0:
            continue
        genres.append(str(cache.genre_vocab[int(genre_idx)]))
    return genres


def _fetch_replacement_display_rows(track_ids: list[str], db_path: Optional[str]) -> dict[str, dict[str, Any]]:
    if not track_ids or not db_path:
        return {}
    db_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    rows: dict[str, dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in track_ids)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT track_id, title, artist, album, duration_ms, file_path
                FROM tracks
                WHERE track_id IN ({placeholders})
                """,
                tuple(track_ids),
            )
            for row in cursor.fetchall():
                tid = str(row["track_id"])
                rows[tid] = {
                    "title": row["title"] or "",
                    "artist": row["artist"] or "",
                    "album": row["album"] or "",
                    "duration_ms": int(row["duration_ms"] or 0),
                    "file_path": row["file_path"] or "",
                }
        finally:
            conn.close()
    except Exception as exc:
        emit_log("WARNING", f"Replacement metadata lookup failed: {exc}")
    return rows


def _enrich_replacement_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = _LAST_GENERATION_CACHE
    track_ids = [str(candidate.get("track_id")) for candidate in candidates]
    display_rows = _fetch_replacement_display_rows(track_ids, cache.db_path)
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        entry = dict(candidate)
        tid = str(entry.get("track_id"))
        display = display_rows.get(tid, {})
        entry.update(
            {
                "rating_key": tid,
                "title": display.get("title") or tid,
                "artist": display.get("artist") or entry.get("artist_key", ""),
                "album": display.get("album", ""),
                "duration_ms": display.get("duration_ms", 0),
                "file_path": display.get("file_path", ""),
                "genres": _top_genres_for_index(cache, int(entry["index"]), limit=12),  # full curated set (was 3 — truncated enriched genres out of the column)
            }
        )
        enriched.append(entry)
    return enriched


def handle_find_replacement_suggestions(cmd_data: Dict[str, Any]) -> None:
    """Find replacement candidates for a single non-pier playlist position."""
    try:
        position = int(cmd_data["position"])
        mode = str(cmd_data.get("mode", "best"))
        top_k = int(cmd_data.get("top_k", 10))

        cache = _LAST_GENERATION_CACHE
        required = [
            cache.playlist_track_ids,
            cache.track_ids,
            cache.artist_keys,
            cache.X_sonic,
            cache.X_full,
            cache.X_genre_smoothed,
            cache.candidate_pool_indices,
        ]
        if any(value is None for value in required):
            emit_error("No playlist in cache. Generate one first.")
            emit_done("find_replacement_suggestions", False, "No playlist in cache")
            return

        playlist_track_ids = list(cache.playlist_track_ids or [])
        if position < 0 or position >= len(playlist_track_ids):
            emit_error(f"position {position} out of range 0..{len(playlist_track_ids) - 1}")
            emit_done("find_replacement_suggestions", False, "Position out of range")
            return
        if position == 0 or position == len(playlist_track_ids) - 1:
            emit_error("Cannot replace pier (first or last) track.")
            emit_done("find_replacement_suggestions", False, "Cannot replace pier track")
            return

        id_to_index = {str(tid): idx for idx, tid in enumerate(cache.track_ids.tolist())}
        try:
            playlist_indices = [id_to_index[str(tid)] for tid in playlist_track_ids]
        except KeyError as exc:
            emit_error(f"Playlist track missing from cache: {exc}")
            emit_done("find_replacement_suggestions", False, "Playlist/cache mismatch")
            return

        from src.playlist.replacement import ReplacementContext, find_replacement_candidates

        current_idx = playlist_indices[position]
        prev_idx = playlist_indices[position - 1]
        next_idx = playlist_indices[position + 1]
        ctx = ReplacementContext(
            X_sonic=cache.X_sonic,
            X_full=cache.X_full,
            X_start=cache.X_start,
            X_end=cache.X_end,
            X_mid=cache.X_mid,
            X_genre_smoothed=cache.X_genre_smoothed,
            perceptual_bpm=cache.perceptual_bpm,
            tempo_stability=cache.tempo_stability,
            track_ids=cache.track_ids,
            artist_keys=cache.artist_keys,
            candidate_pool_indices=cache.candidate_pool_indices,
            idf_weights=cache.idf_weights,
            transition_metric_context=cache.transition_metric_context,
            transition_floor=cache.transition_floor,
        )
        candidates = find_replacement_candidates(
            prev_idx=prev_idx,
            next_idx=next_idx,
            current_idx=current_idx,
            playlist_indices=playlist_indices,
            ctx=ctx,
            mode=mode,
            top_k=top_k,
        )

        emit_result(
            "replacement_suggestions",
            {
                "position": position,
                "mode": mode,
                "candidates": _enrich_replacement_candidates(candidates),
                "current_track_id": str(cache.track_ids[current_idx]),
                "prev_track_id": str(cache.track_ids[prev_idx]),
                "next_track_id": str(cache.track_ids[next_idx]),
            },
        )
        emit_done("find_replacement_suggestions", True, f"Found {len(candidates)} replacement candidate(s)")
    except Exception as exc:
        tb = traceback.format_exc()
        emit_error(f"{type(exc).__name__}: {exc}", tb)
        emit_done("find_replacement_suggestions", False, str(exc))


def handle_doctor(cmd_data: Dict[str, Any]) -> None:
    """Quick diagnostics."""
    checks = []
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    cfg_path = Path(base_path)

    if cfg_path.exists():
        checks.append({"name": "config_path", "ok": True, "detail": str(cfg_path)})
    else:
        checks.append({"name": "config_path", "ok": False, "detail": f"Missing {cfg_path}"})

    try:
        cfg = load_config_with_overrides(base_path, overrides)
        checks.append({"name": "config_load", "ok": True, "detail": "Loaded"})
    except Exception as e:
        checks.append({"name": "config_load", "ok": False, "detail": str(e)})
        emit_result("doctor", {"checks": checks})
        emit_done("doctor", False, "Config failed")
        return

    db_path = Path(resolve_database_path(cfg))
    if not db_path.is_absolute():
        db_path = cfg_path.parent / db_path
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            conn.close()
            checks.append({"name": "database", "ok": count > 0, "detail": f"{count} tracks"})
        except Exception as e:
            checks.append({"name": "database", "ok": False, "detail": str(e)})
    else:
        checks.append({"name": "database", "ok": False, "detail": f"Missing {db_path}"})

    artifact_path = Path(cfg.get("playlists", {}).get("ds_pipeline", {}).get("artifact_path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz"))
    if not artifact_path.is_absolute():
        artifact_path = cfg_path.parent / artifact_path
    checks.append({"name": "artifacts", "ok": artifact_path.exists(), "detail": str(artifact_path)})

    emit_result("doctor", {"checks": checks})
    emit_done("doctor", True, "Doctor complete")

def handle_ping(cmd_data: Dict[str, Any]) -> None:
    """Handle ping command - simple health check with protocol version."""
    emit_log("INFO", "Pong!")
    # Include protocol version in ping response
    emit_event({
        "type": "result",
        "result_type": "pong",
        "protocol_version": PROTOCOL_VERSION
    })
    emit_done("ping", True)


def handle_cancel(cmd_data: Dict[str, Any]) -> None:
    """
    Handle cancel command - request cancellation of an active command.

    The cancel command itself does not have a request_id context;
    it specifies the request_id to cancel.
    """
    target_request_id = cmd_data.get("request_id")
    if not target_request_id:
        emit_log("WARNING", "Cancel command missing request_id", request_id=None)
        return

    if _worker_state.request_cancel(target_request_id):
        emit_log("WARNING", f"Cancellation requested for {target_request_id}", request_id=target_request_id)

        # Signal the cancellation token if available
        token = get_cancellation_token()
        if token:
            token.cancel(f"User requested cancellation for {target_request_id}")
    else:
        current = _worker_state.get_request_id()
        if current:
            emit_log("INFO", f"Cancel request_id {target_request_id} does not match active {current}")
        else:
            emit_log("INFO", f"Cancel request_id {target_request_id} - no active request")


def handle_generate_playlist(cmd_data: Dict[str, Any]) -> None:
    """Handle playlist generation command with cancellation support."""
    _reset_last_generation_cache()
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    request = GeneratePlaylistRequest.from_worker_args(cmd_data.get("args", {}))

    mode = request.mode
    artist = request.artist
    genre = request.genre
    track_title = request.track
    seed_tracks = request.seed_tracks or None
    seed_track_ids = request.seed_track_ids or None
    track_count = request.tracks
    include_collaborations = request.include_collaborations
    exclude_seed_tracks_from_recency = request.exclude_seed_tracks_from_recency

    # Cooperative cancellation for the generation core (registered before the
    # dispatch below). Imported ahead of the `try` so the `except
    # OperationCancelled` clause can always resolve the name.
    from src.cancellation import OperationCancelled, set_cancellation_hook

    request_id = _worker_state.get_request_id()

    # Per-playlist DEBUG log file (see
    # docs/superpowers/specs/2026-07-02-per-playlist-logging-design.md).
    # Settings are resolved ahead of the try/except below via a
    # best-effort, exception-swallowing config pre-load, so the
    # "Starting playlist generation" line and the authoritative config
    # load (inside the try, with real error handling) both land in the
    # file. A logging failure here must never block a generation.
    from src.logging_utils import cleanup_old_playlist_logs_async, playlist_log_file

    playlist_log_enabled, playlist_log_dir_cfg = True, "logs/playlists"
    playlist_log_retention_days, playlist_log_level_name = 30, "DEBUG"
    try:
        _playlist_logs_cfg = (
            load_config_with_overrides(base_path, overrides)
            .get("logging", {})
            .get("playlist_logs", {})
        ) or {}
        playlist_log_enabled = bool(_playlist_logs_cfg.get("enabled", True))
        playlist_log_dir_cfg = _playlist_logs_cfg.get("dir", "logs/playlists")
        playlist_log_retention_days = int(_playlist_logs_cfg.get("retention_days", 30))
        playlist_log_level_name = _playlist_logs_cfg.get("level", "DEBUG")
    except Exception:
        pass
    playlist_log_level = getattr(logging, str(playlist_log_level_name).upper(), logging.DEBUG)

    with playlist_log_file(
        artist or genre or mode,
        request_id,
        enabled=playlist_log_enabled,
        dir=playlist_log_dir_cfg,
        level=playlist_log_level,
    ):
        emit_log("INFO", f"Starting playlist generation (mode={mode})")
        emit_progress("init", 0, 100, "Loading configuration")

        try:
            # Load and merge config
            config = load_config_with_overrides(base_path, overrides)
            emit_progress("init", 10, 100, "Configuration loaded")

            # Cancellation check after config load
            check_cancelled()

            # Import application modules (heavy imports deferred)
            emit_log("INFO", "Loading application modules...")
            emit_progress("init", 20, 100, "Loading modules")

            # Add project root to path if needed
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from src.local_library_client import LocalLibraryClient
            from src.playlist_generator import PlaylistGenerator
            from src.track_matcher import TrackMatcher
            from src.metadata_client import MetadataClient
            from src.lastfm_client import LastFMClient

            # Cancellation check after imports
            check_cancelled()

            # Apply overrides to config
            # Write merged config to temp or use Config directly
            class MergedConfig:
                """Wrapper that uses merged config dict instead of loading from file."""
                def __init__(self, config_dict: dict, original_path: str):
                    self.config = config_dict
                    self.config_path = original_path
                    self._validate()

                def _validate(self):
                    # Minimal validation
                    if 'library' not in self.config:
                        raise ValueError("Missing 'library' section in config")

                def get(self, section: str, key: str = None, default: Any = None) -> Any:
                    if section not in self.config:
                        return default
                    if key is None:
                        return self.config.get(section, default)
                    return self.config[section].get(key, default)

                @property
                def library_database_path(self) -> str:
                    return self.config['library']['database_path']

                @property
                def library_music_directory(self) -> str:
                    return self.config['library'].get('music_directory', 'E:\\MUSIC')

                @property
                def lastfm_api_key(self) -> str:
                    import os
                    return os.getenv('LASTFM_API_KEY') or self.config.get('lastfm', {}).get('api_key', '')

                @property
                def lastfm_username(self) -> str:
                    import os
                    return os.getenv('LASTFM_USERNAME') or self.config.get('lastfm', {}).get('username', '')

                @property
                def lastfm_history_days(self) -> int:
                    return self.config.get('lastfm', {}).get('history_days', 90)

                @property
                def min_duration_minutes(self) -> int:
                    return self.config.get('playlists', {}).get('min_duration_minutes', 90)

                @property
                def min_track_duration_seconds(self) -> int:
                    # Was hardcoded to 90 (a stray copy of min_duration_minutes's
                    # default just above -- a different key, playlist length in
                    # minutes, not track length in seconds). Reconciled to the
                    # single shared fallback (coordinator review 2026-07-30,
                    # duration-gate-report.md) so this GUI-worker config shim
                    # can't silently disagree with the CLI Config class or the
                    # pier/bridge duration gates on a config missing the key.
                    return self.config.get('playlists', {}).get(
                        'min_track_duration_seconds', DEFAULT_MIN_TRACK_DURATION_SECONDS
                    )

                @property
                def max_track_duration_seconds(self) -> int:
                    return self.config.get('playlists', {}).get('max_track_duration_seconds', 720)

                @property
                def recently_played_filter_enabled(self) -> bool:
                    return self.config.get('playlists', {}).get('recently_played_filter', {}).get('enabled', True)

                @property
                def recently_played_lookback_days(self) -> int:
                    return self.config.get('playlists', {}).get('recently_played_filter', {}).get('lookback_days', 0)

                @property
                def recently_played_min_playcount(self) -> int:
                    return self.config.get('playlists', {}).get('recently_played_filter', {}).get('min_playcount_threshold', 0)

                @property
                def max_tracks_per_artist(self) -> int:
                    return self.config.get('playlists', {}).get('max_tracks_per_artist', 3)

                @property
                def artist_window_size(self) -> int:
                    return self.config.get('playlists', {}).get('artist_window_size', 8)

                @property
                def max_artist_per_window(self) -> int:
                    return self.config.get('playlists', {}).get('max_artist_per_window', 1)

                @property
                def min_seed_artist_ratio(self) -> float:
                    return self.config.get('playlists', {}).get('min_seed_artist_ratio', 0.125)

                @property
                def dynamic_sonic_ratio(self) -> float:
                    return self.config.get('playlists', {}).get('dynamic_mode', {}).get('sonic_ratio', 0.6)

                @property
                def dynamic_genre_ratio(self) -> float:
                    return self.config.get('playlists', {}).get('dynamic_mode', {}).get('genre_ratio', 0.4)

                @property
                def similarity_min_threshold(self) -> float:
                    return self.config.get('playlists', {}).get('similarity', {}).get('min_threshold', 0.5)

                @property
                def limit_similar_tracks(self) -> int:
                    return self.config.get('playlists', {}).get('limits', {}).get('similar_tracks', 50)

                @property
                def title_dedupe_enabled(self) -> bool:
                    return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('enabled', True)

                @property
                def title_dedupe_threshold(self) -> int:
                    return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('threshold', 92)

                @property
                def title_dedupe_mode(self) -> str:
                    return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('mode', 'loose')

                @property
                def title_dedupe_short_title_min_len(self) -> int:
                    return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('short_title_min_len', 6)

            emit_progress("init", 30, 100, "Initializing library client")
            merged_config = MergedConfig(config, base_path)
            library = LocalLibraryClient(db_path=resolve_database_path(config))

            # Cancellation check after library init
            check_cancelled()

            emit_progress("init", 40, 100, "Initializing track matcher")
            matcher = TrackMatcher(library, library_id=None, db_path=resolve_database_path(config))

            # Metadata client
            metadata = None
            try:
                metadata = MetadataClient(resolve_database_path(config))
            except Exception:
                pass

            # Last.FM client (optional)
            lastfm = None
            if merged_config.lastfm_api_key and merged_config.lastfm_username:
                try:
                    lastfm = LastFMClient(
                        api_key=merged_config.lastfm_api_key,
                        username=merged_config.lastfm_username,
                        db_path=resolve_database_path(config)
                    )
                except Exception:
                    pass

            # Cancellation check before playlist generator init
            check_cancelled()

            emit_progress("init", 50, 100, "Initializing playlist generator")
            generator = PlaylistGenerator(
                library,
                merged_config,
                lastfm_client=lastfm,
                track_matcher=matcher,
                metadata_client=metadata
            )

            # Resolve cohesion_mode — drives pier-bridge beam tuning.
            # genre_mode/sonic_mode/pace_mode are independent axes that affect
            # candidate pool composition, not beam scoring.
            cohesion_mode = resolve_cohesion_mode(config.get('playlists', {}))

            # Log which modes are active
            genre_mode = request.genre_mode or config.get('playlists', {}).get(
                'genre_mode'
            )
            sonic_mode = request.sonic_mode or config.get('playlists', {}).get(
                'sonic_mode'
            )
            pace_mode = request.pace_mode or config.get('playlists', {}).get(
                'pace_mode', 'dynamic'
            )
            if genre_mode:
                emit_log("INFO", f"Genre mode: {genre_mode}")
            if sonic_mode:
                emit_log("INFO", f"Sonic mode: {sonic_mode}")
            emit_log("INFO", f"Pace mode: {pace_mode}")
            emit_log("INFO", f"Cohesion mode: {cohesion_mode}")

            # Cancellation check before generation
            check_cancelled()

            # The generation core (build_pier_bridge_playlist -> beam search) runs
            # several layers below this handler and cannot see the worker's cancel
            # flag directly. Register a predicate it polls at its loop checkpoints;
            # on cancel it raises OperationCancelled (caught below). Cleared in the
            # `finally`. This is what makes a long generation actually cancellable.
            set_cancellation_hook(_worker_state.is_cancelled)

            emit_progress("generate", 60, 100, "Generating playlist")
            emit_log("INFO", f"Running playlist generation with cohesion_mode={cohesion_mode}")

            _artists = list(getattr(request, "artists", []) or [])
            if mode == "artist" and (artist or _artists):
                # Single artist mode, or multi-artist blend when 2+ chips are set
                playlist_data = generator.create_playlist_for_artist(
                    artist or _artists[0],
                    track_count,
                    track_title=track_title,
                    track_titles=seed_tracks,
                    dynamic=(cohesion_mode == "dynamic"),
                    cohesion_mode_override=cohesion_mode,
                    include_collaborations=include_collaborations,
                    exclude_seed_tracks_from_recency=exclude_seed_tracks_from_recency,
                    popular_seeds_mode=request.popular_seeds_mode,
                    popularity_mode=request.popularity_mode,
                    seed_epoch=request.seed_epoch,
                    artist_names=_artists,
                )
            elif mode == "seeds" and seed_tracks:
                # Seeds mode (Phase 2 UI)
                playlist_data = generator.create_playlist_from_seed_tracks(
                    seed_tracks,
                    track_count=track_count,
                    dynamic=(cohesion_mode == "dynamic"),
                    cohesion_mode_override=cohesion_mode,
                    seed_track_ids=seed_track_ids,
                    popularity_mode=request.popularity_mode,
                )
            elif mode == "artist" and seed_tracks:
                # Legacy seeds mode (old UI sent mode="artist" with seed_tracks)
                playlist_data = generator.create_playlist_from_seed_tracks(
                    seed_tracks,
                    track_count=track_count,
                    dynamic=(cohesion_mode == "dynamic"),
                    cohesion_mode_override=cohesion_mode,
                    seed_track_ids=seed_track_ids,
                    popularity_mode=request.popularity_mode,
                )
            elif mode == "genre" and genre:
                # Genre mode
                playlist_data = generator.create_playlist_for_genre(
                    genre,
                    track_count,
                    dynamic=(cohesion_mode == "dynamic"),
                    cohesion_mode_override=cohesion_mode,
                    seed_epoch=request.seed_epoch,
                )
            elif mode == "history":
                # Config/history-driven mode, matching the CLI batch-generation flow.
                playlists = generator.create_playlist_batch(
                    1,
                    dynamic=(cohesion_mode == "dynamic"),
                    cohesion_mode_override=cohesion_mode,
                )
                playlist_data = playlists[0] if playlists else None
            else:
                # Fallback for unknown modes or missing required parameters
                raise ValueError(
                    "Invalid mode or missing parameters: "
                    f"mode={mode}, artist={artist}, genre={genre}, seed_tracks={seed_tracks}"
                )

            # Cancellation check after generation, before output
            check_cancelled()

            emit_progress("generate", 90, 100, "Formatting results")

            if playlist_data:
                tracks = playlist_data.get('tracks', [])
                ds_report = playlist_data.get('ds_report', {}) or {}
                edge_scores = (
                    (ds_report.get("playlist_stats") or {}).get("playlist", {}).get("edge_scores")
                    or ds_report.get("edge_scores")
                    or []
                )
                edge_map = {str(edge.get("cur_id")): edge for edge in edge_scores if edge.get("cur_id")}

                similarity_components = _build_seed_similarity_components(
                    tracks=tracks,
                    ds_report=ds_report,
                    edge_map=edge_map,
                )

                formatted_tracks = []

                for i, track in enumerate(tracks):
                    rating_key = track.get('rating_key') or track.get('id') or track.get('track_id')

                    def _raw_genres(_track=track, _rk=rating_key) -> List[str]:
                        raw = _track.get('genres', []) or []
                        if raw:
                            return raw
                        if _rk and getattr(generator, "similarity_calc", None):
                            try:
                                return generator.similarity_calc.get_filtered_combined_genres_for_track(str(_rk)) or []
                            except Exception:
                                return []
                        return []

                    genres = _resolve_display_genres(
                        track,
                        sidecar_db_path=SIDECAR_DB_PATH,
                        fallback=_raw_genres,
                        db_path=resolve_database_path(config),
                    )

                    # Prefer explicit similarity fields; fall back to edge scores from DS report
                    edge = edge_map.get(str(rating_key), {})
                    sonic_sim = (
                        track.get('similarity_score')
                        or track.get('hybrid_score')
                        or track.get('score')
                        or track.get('sonic_similarity')
                        or edge.get('S')
                    )
                    genre_sim = (
                        track.get('genre_sim')
                        or track.get('genre_similarity')
                        or track.get('G')
                        or edge.get('G')
                    )
                    components = similarity_components.get(str(rating_key)) if rating_key else None
                    if components:
                        sonic_comp = components.get("sonic")
                        genre_comp = components.get("genre")
                        if sonic_comp and sonic_comp.get("t") is not None:
                            sonic_sim = sonic_comp.get("t")
                        if genre_comp and genre_comp.get("t") is not None:
                            genre_sim = genre_comp.get("t")
                    else:
                        sonic_comp = None
                        genre_comp = None

                    formatted_tracks.append({
                        "position": i,
                        "rating_key": rating_key,
                        "artist": track.get('artist', 'Unknown'),
                        "title": track.get('title', 'Unknown'),
                        "album": track.get('album', ''),
                        "duration_ms": track.get('duration', 0),
                        "file_path": track.get('file_path', ''),
                        "sonic_similarity": sonic_sim,
                        "genre_similarity": genre_sim,
                        "sonic_similarity_components": sonic_comp,
                        "genre_similarity_components": genre_comp,
                        "genres": genres,
                        "transition_score": edge.get("T"),
                    })

                # Oops, All Bangers: annotate each track with its Last.fm popularity rank
                # (sets track['popularity_rank']) and log it, when Bangers is on.
                if str(getattr(request, "popularity_mode", "off")) in ("on", "oops"):
                    try:
                        from src.analyze.popularity_runner import (
                            annotate_and_log_playlist_popularity,
                            popularity_cache_db_path,
                        )
                        annotate_and_log_playlist_popularity(
                            formatted_tracks, db_path=popularity_cache_db_path())
                    except Exception as _exc:  # diagnostics must never break a generation
                        logging.getLogger(__name__).warning(
                            "Bangers popularity annotation failed: %s", _exc)

                # `warnings`/`degraded` are hard-invariant breaches from post-order
                # validation (artist gap, per-artist cap) — distinct from the
                # `relaxations` list below, which reports guidelines the generator
                # deliberately bent to stay feasible. A relaxation is the system
                # working; a degraded result is the system breaking a guarantee.
                _invariant_warnings = [str(w) for w in (playlist_data.get("warnings") or [])]
                playlist_result = {
                    "name": playlist_data.get('title', 'Generated Playlist'),
                    "tracks": formatted_tracks,
                    "track_count": len(formatted_tracks),
                    "warnings": _invariant_warnings,
                    "degraded": bool(playlist_data.get("degraded") or _invariant_warnings),
                }
                if _invariant_warnings:
                    for _w in _invariant_warnings:
                        emit_log("WARNING", f"Playlist degraded — {_w}")

                # Include DS report metrics if available
                if ds_report:
                    metrics = ds_report.get('metrics', {})
                    playlist_result["metrics"] = {
                        "mean_transition": metrics.get('mean_transition'),
                        "min_transition": metrics.get('min_transition'),
                        "p10_transition": metrics.get('p10_transition'),
                        "p90_transition": metrics.get('p90_transition'),
                        "distinct_artists": metrics.get('distinct_artists'),
                    }
                    # Surface pier-bridge relaxation warnings so the GUI can display
                    # a notice when generation had to bend a guideline to stay feasible.
                    all_warnings = (
                        (ds_report.get("playlist_stats") or {})
                        .get("playlist", {})
                        .get("warnings") or []
                    )
                    playlist_result["relaxations"] = [
                        w for w in all_warnings
                        if isinstance(w, dict) and w.get("type") == "relaxation"
                    ]

                    # Per-generation receipt (GUI dials, 2026-07-04): compact
                    # honor+confess summary composed from this run's own stats.
                    try:
                        from src.playlist_gui.receipt import compose_receipt
                        _pstats = (ds_report.get("playlist_stats") or {}).get("playlist", {}) or {}
                        playlist_result["receipt"] = compose_receipt(
                            _pstats, _pstats.get("receipt_pool") or {})
                    except Exception as exc:
                        emit_log("WARNING", f"Receipt composition failed: {exc}")

                try:
                    _populate_last_generation_cache(
                        generator=generator,
                        playlist_result=playlist_result,
                        db_path=resolve_database_path(config),
                    )
                except Exception as exc:
                    emit_log("WARNING", f"Replacement cache unavailable for this playlist: {exc}")

                emit_result("playlist", {"playlist": playlist_result})
                emit_progress("complete", 100, 100, "Done")
                emit_done("generate_playlist", True, f"Generated {len(formatted_tracks)} tracks")
            else:
                emit_error("No playlist generated")
                emit_done("generate_playlist", False, "No playlist generated")

        except OperationCancelled:
            # Raised by the generation core's cancellation checkpoints (BaseException
            # subclass, so the core's broad `except Exception` blocks don't swallow it).
            emit_log("INFO", "Playlist generation cancelled")
            emit_done("generate_playlist", False, "Cancelled by user", cancelled=True)
        except CancellationError:
            emit_log("INFO", "Playlist generation cancelled")
            emit_done("generate_playlist", False, "Cancelled by user", cancelled=True)
        except Exception as e:
            tb = traceback.format_exc()
            emit_error(str(e), tb)
            emit_done("generate_playlist", False, str(e))
        finally:
            # Never leave the process-global hook set across requests.
            set_cancellation_hook(None)

    cleanup_old_playlist_logs_async(playlist_log_dir_cfg, playlist_log_retention_days)


def handle_scan_library(cmd_data: Dict[str, Any]) -> None:
    """Handle library scan command with cancellation support."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})

    # Check if force flag is set for full scan
    force = overrides.get('force', False)
    resume_from_checkpoint = overrides.get("resume_from_checkpoint") or {}
    scan_type = "full scan" if force else "quick scan (incremental)"
    if resume_from_checkpoint:
        scan_type = f"{scan_type}, resuming"
    emit_log("INFO", f"Starting library scan ({scan_type})")
    emit_progress("scan", 0, 100, "Initializing")

    # Create cancellation token for this operation
    cancellation_token = CancellationToken() if CancellationToken else None
    set_cancellation_token(cancellation_token)

    try:
        config = load_config_with_overrides(base_path, overrides)
        check_cancelled()

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.scan_library import LibraryScanner

        music_dir = config.get('library', {}).get('music_directory', 'E:\\MUSIC')

        check_cancelled()
        emit_progress("scan", 10, 100, f"Scanning {music_dir}")

        # LibraryScanner reads music_dir and db_path from config
        # Default to quick=True (incremental) unless force=True in overrides
        quick = not force
        scanner = LibraryScanner(config_path=base_path, cancellation_token=cancellation_token)
        stats = scanner.run(
            quick=quick,
            cleanup=True,  # Always cleanup missing files during pipeline
            resume_from_checkpoint=resume_from_checkpoint,
        )

        check_cancelled()
        emit_result("scan", {"stats": stats})
        total = stats.get('total', 0) if isinstance(stats, dict) else 0
        new_count = stats.get('new', 0) if isinstance(stats, dict) else 0
        updated = stats.get('updated', 0) if isinstance(stats, dict) else 0
        summary = f"Indexed {total} files (new {new_count}, updated {updated})"
        emit_done("scan_library", True, f"Scanned {total} files", summary=summary)

    except CancellationError:
        emit_log("INFO", "Library scan cancelled by user")
        emit_done("scan_library", False, "Cancelled by user", cancelled=True, summary="Scan cancelled")
    except CancelledException as e:
        # Graceful cancellation with checkpoint
        progress_data = getattr(e, 'progress', {})
        items_completed = progress_data.get('items_completed', 0)
        total_items = progress_data.get('total_items', 0)
        stats = progress_data.get('stats', {})

        emit_log("INFO", f"Library scan cancelled: {e}")

        # Emit checkpoint for potential resumption
        emit_checkpoint(
            stage="scan",
            items_completed=items_completed,
            resumable_state={
                "last_file_index": progress_data.get('last_file_index', 0),
                "stats": stats,
                "base_config_path": base_path,
                "overrides": overrides,
            }
        )

        # Emit partial results
        if stats:
            emit_result("scan", {"stats": stats})

        summary = f"Scan cancelled: {items_completed}/{total_items} files processed"
        emit_done("scan_library", False, str(e), cancelled=True, summary=summary)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("scan_library", False, str(e), summary="Scan failed")
    finally:
        # Clear cancellation token
        set_cancellation_token(None)


def handle_update_genres(cmd_data: Dict[str, Any]) -> None:
    """Handle genre update command with cancellation support."""
    emit_log("INFO", "Starting genre update")
    emit_progress("genres", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import and run genre updater
        from scripts.update_genres_v3_normalized import NormalizedGenreUpdater

        db_path = resolve_database_path(config)
        check_cancelled()
        emit_progress("genres", 20, 100, "Fetching genres")

        # This runs the full genre update (artists + albums)
        updater = NormalizedGenreUpdater(db_path=db_path)
        emit_progress("genres", 30, 100, "Updating artist genres")
        updater.update_artist_genres(progress=False)
        check_cancelled()
        emit_progress("genres", 60, 100, "Updating album genres")
        updater.update_album_genres(progress=False)
        check_cancelled()
        emit_progress("genres", 80, 100, "Updating file tag genres")
        updater.update_track_genres()
        check_cancelled()
        stats = {"artists_updated": 0, "albums_updated": 0}  # Updater doesn't return detailed stats
        updater.close()

        check_cancelled()
        emit_result("genres", {"stats": stats})
        summary = "Updated genres"
        if isinstance(stats, dict):
            artists = stats.get("artists_updated") or stats.get("artists_processed")
            albums = stats.get("albums_updated") or stats.get("albums_processed")
            parts = []
            if artists:
                parts.append(f"{artists} artists")
            if albums:
                parts.append(f"{albums} albums")
            if parts:
                summary = "Updated genres for " + ", ".join(parts)
        emit_done("update_genres", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Genre update cancelled")
        emit_done("update_genres", False, "Cancelled by user", cancelled=True, summary="Genre update cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_genres", False, str(e), summary="Genre update failed")


def handle_update_sonic(cmd_data: Dict[str, Any]) -> None:
    """Handle sonic feature extraction command with cancellation support."""
    emit_log("INFO", "Starting sonic feature extraction")
    emit_progress("sonic", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.update_sonic import SonicFeaturePipeline

        db_path = resolve_database_path(config)
        check_cancelled()
        emit_progress("sonic", 20, 100, "Extracting features")

        # Run sonic analysis with beat3tower
        pipeline = SonicFeaturePipeline(db_path=db_path, use_beat_sync=False, use_beat3tower=True)
        emit_progress("sonic", 30, 100, "Analyzing tracks")
        pipeline.run(limit=None, force=False, progress=False)
        check_cancelled()
        stats = pipeline.get_stats()
        pipeline.close()

        check_cancelled()
        emit_result("sonic", {"stats": stats})
        summary = "Sonic analysis complete"
        if isinstance(stats, dict):
            analyzed = stats.get("analyzed") or stats.get("total") or stats.get("total_tracks")
            if analyzed is not None:
                summary = f"Analyzed {analyzed} tracks"
        emit_done("update_sonic", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Sonic feature extraction cancelled")
        emit_done("update_sonic", False, "Cancelled by user", cancelled=True, summary="Sonic analysis cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("update_sonic", False, str(e), summary="Sonic analysis failed")


def handle_build_artifacts(cmd_data: Dict[str, Any]) -> None:
    """Handle artifact building command with cancellation support."""
    emit_log("INFO", "Starting artifact build")
    emit_progress("artifacts", 0, 100, "Initializing")

    try:
        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        check_cancelled()

        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from scripts.build_beat3tower_artifacts import build_artifacts

        db_path = resolve_database_path(config)
        ds_cfg = config.get('playlists', {}).get('ds_pipeline', {})
        output_path = ds_cfg.get(
            'artifact_path', 'data/artifacts/beat3tower_32k/data_matrices_step1.npz'
        )
        genre_source = _resolve_worker_artifact_genre_source(config)
        genre_sim_path = _resolve_worker_genre_sim_path(config, output_path)

        check_cancelled()
        emit_progress("artifacts", 20, 100, f"Building matrices ({genre_source})")

        # Create argparse-like namespace for build_artifacts
        from argparse import Namespace
        args = Namespace(
            db_path=db_path,
            config=base_path,
            output=output_path,
            genre_sim_path=genre_sim_path,
            max_tracks=0,
            random_seed=42,
            no_genre_normalization=False,
            sidecar_db=SIDECAR_DB_PATH,
            genre_source=genre_source,
            verbose=False
        )
        build_artifacts(args)

        check_cancelled()

        # Rebuild the dense genre embedding sidecar so the new genre system stays in
        # sync with the freshly-built artifact. Without this the sidecar goes stale
        # (its X_genre_dense reflects the OLD genres) and load_artifact_bundle either
        # silently serves stale dense vectors or drops dense steering entirely.
        emit_progress("artifacts", 80, 100, "Building genre embedding sidecar")
        from scripts.build_genre_embedding import build_genre_embedding_sidecar
        sidecar_path = build_genre_embedding_sidecar(output_path, skip_prior=True)
        emit_log("INFO", f"Rebuilt dense genre sidecar: {sidecar_path}")

        # Invalidate the in-process bundle cache so the NEXT generation in this
        # worker picks up the freshly-built artifact + sidecar. load_artifact_bundle
        # is @lru_cache'd, so without this the worker keeps serving the stale bundle
        # it loaded at startup (the rebuild would be invisible until a GUI restart).
        try:
            from src.features.artifacts import load_artifact_bundle
            load_artifact_bundle.cache_clear()
            emit_log("INFO", "Cleared artifact bundle cache; new artifact is live for this session")
        except Exception as exc:  # pragma: no cover - defensive
            emit_log("WARNING", f"Could not clear artifact bundle cache (restart GUI to pick up rebuild): {exc}")

        check_cancelled()
        emit_result("artifacts", {"output_path": output_path, "sidecar_path": str(sidecar_path)})
        summary = f"Built artifacts + genre embedding at {output_path}"
        emit_done("build_artifacts", True, summary, summary=summary)

    except CancellationError:
        emit_log("INFO", "Artifact build cancelled")
        emit_done("build_artifacts", False, "Cancelled by user", cancelled=True, summary="Artifact build cancelled")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("build_artifacts", False, str(e), summary="Artifact build failed")


def _resolve_worker_genre_sim_path(config: Dict[str, Any], artifact_output_path: str) -> Optional[str]:
    """Resolve the genre-similarity smoothing matrix for GUI artifact rebuilds.

    Default (cooccurrence) returns None — the GUI flow has never applied
    smoothing, and that behavior is preserved. With
    playlists.ds_pipeline.genre_similarity.source: graph, the taxonomy-derived
    matrix is (re)built next to the artifact and passed to the builder so GUI
    rebuilds match the CLI analyze flow.
    """
    ds_cfg = config.get("playlists", {}).get("ds_pipeline", {})
    sim_cfg = ds_cfg.get("genre_similarity", {})
    source = ""
    if isinstance(sim_cfg, dict):
        source = str(sim_cfg.get("source") or "").strip().lower()
    if source != "graph":
        return None

    from src.genre.graph_adapter import load_graph_adapter
    from src.genre.graph_similarity import (
        build_graph_similarity,
        npz_similarity_source,
        save_graph_similarity_npz,
    )

    sim_path = Path(artifact_output_path).parent / "genre_similarity_matrix.npz"
    if npz_similarity_source(sim_path) != "graph":
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        save_graph_similarity_npz(build_graph_similarity(load_graph_adapter()), sim_path)
    return str(sim_path)


def _resolve_worker_artifact_genre_source(config: Dict[str, Any]) -> str:
    """Resolve GUI artifact-build source from merged config.

    Runtime layered graph mode requires artifacts with layered matrices. The
    artifact builder emits those under the `layered_shadow` source, while
    `genre_graph.source=layered` controls runtime admission/scoring behavior.
    """
    ds_cfg = config.get("playlists", {}).get("ds_pipeline", {})
    graph_cfg = ds_cfg.get("genre_graph", {})
    graph_source = ""
    if isinstance(graph_cfg, dict):
        graph_source = str(graph_cfg.get("source") or "").strip().lower()
    if graph_source in {"layered", "layered_shadow"}:
        return "layered_shadow"

    raw_source = str(ds_cfg.get("genre_source") or "legacy").strip().lower()
    if raw_source in {"legacy", "enriched", "graph", "hybrid_shadow", "layered_shadow"}:
        return raw_source
    return "legacy"


def _write_merged_temp_config(base_path: str, overrides: Dict[str, Any]) -> Optional[Path]:
    """Write a merged config file for CLI code that only accepts a config path."""
    if not overrides:
        return None
    merged_config = load_config_with_overrides(base_path, overrides)
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="playlist_gui_analyze_",
        delete=False,
    )
    with temp:
        yaml.safe_dump(merged_config, temp, sort_keys=False)
    return Path(temp.name)


class AnalyzeLibraryProgressLogHandler(logging.Handler):
    """Bridge structured Analyze Library log lines into GUI progress events."""

    def __init__(self, stage_order: list[str]) -> None:
        super().__init__(level=logging.INFO)
        self._stage_order = list(stage_order)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            progress = parse_analyze_library_stage_progress(record.getMessage(), self._stage_order)
            if progress:
                emit_progress(
                    progress["stage"],
                    progress["current"],
                    progress["total"],
                    progress["detail"],
                )
        except Exception:
            return


def handle_analyze_library(cmd_data: Dict[str, Any]) -> None:
    """Handle the unified Analyze Library command by invoking the CLI pipeline."""
    request = LibraryPipelineRequest(
        config_path=cmd_data.get("base_config_path", "config.yaml"),
        overrides=cmd_data.get("overrides", {}) or {},
        stages=cmd_data.get("stages") or None,
        force=bool(cmd_data.get("force", False)),
        dry_run=bool(cmd_data.get("dry_run", False)),
    )
    emit_log("INFO", "Starting Analyze Library")
    emit_progress("analyze_library", 0, 100, "Initializing")
    temp_config_path: Optional[Path] = None

    try:
        temp_config_path = _write_merged_temp_config(request.config_path, request.overrides)
        config_path = str(temp_config_path or request.config_path)
        check_cancelled()

        from scripts.analyze_library import DEFAULT_OUT_DIR, parse_args, run_pipeline

        args = parse_args(["--config", config_path])
        args.stages = ",".join(request.stages)
        args.dry_run = request.dry_run
        if request.force or bool(request.overrides.get("force", False)):
            args.force = True
        stage_order = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
        progress_handler = AnalyzeLibraryProgressLogHandler(stage_order)
        analyze_logger = logging.getLogger("analyze_library")

        emit_progress("analyze_library", 5, 100, "Running Analyze Library")
        analyze_logger.addHandler(progress_handler)
        try:
            exit_code = run_pipeline(
                args,
                cancellation_check=check_cancelled,
                console_logging=False,
            )
        finally:
            analyze_logger.removeHandler(progress_handler)
        check_cancelled()
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
        report_path = out_dir / "analyze_run_report.json"

        if exit_code == 0:
            if request.dry_run:
                summary = "Analyze Library dry run complete"
                emit_result(
                    "analyze_library",
                    {
                        "summary": summary,
                        "stages": [{"name": stage, "decision": "planned"} for stage in stage_order],
                        "out_dir": str(out_dir),
                    },
                )
                emit_progress("analyze_library", 100, 100, "Dry run complete")
                emit_done(
                    "analyze_library",
                    True,
                    summary,
                    summary=summary,
                )
                return
            paused = parse_analyze_library_paused(report_path)
            if paused is not None:
                paused_stage, pause_reason = paused
                emit_result(
                    "analyze_library",
                    {
                        "summary": f"Paused at {paused_stage} (resumable)",
                        "paused": True,
                        "paused_stage": paused_stage,
                        "pause_reason": pause_reason,
                        "out_dir": str(out_dir),
                    },
                )
                emit_progress("analyze_library", 100, 100, f"Paused at {paused_stage} (resumable)")
                emit_done(
                    "analyze_library",
                    True,
                    f"Paused at {paused_stage}: {pause_reason} — re-run to resume from cache",
                    summary=f"Analyze Library paused at {paused_stage} (resumable)",
                )
                return
            result_data = parse_analyze_library_report(report_path)
            emit_result("analyze_library", result_data)
            emit_progress("analyze_library", 100, 100, "Complete")
            summary = result_data.get("summary") or "Analyze Library complete"
            emit_done(
                "analyze_library",
                True,
                "Analyze Library complete",
                summary=summary,
            )
        else:
            emit_done(
                "analyze_library",
                False,
                f"Analyze Library exited with code {exit_code}",
                summary="Analyze Library failed",
            )

    except CancellationError:
        emit_log("INFO", "Analyze Library cancelled")
        emit_done(
            "analyze_library",
            False,
            "Cancelled by user",
            cancelled=True,
            summary="Analyze Library cancelled",
        )
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("analyze_library", False, str(e), summary="Analyze Library failed")
    finally:
        if temp_config_path:
            try:
                temp_config_path.unlink(missing_ok=True)
            except Exception as exc:
                emit_log("WARNING", f"Failed to remove temp Analyze Library config: {exc}")


def handle_blacklist_fetch(cmd_data: Dict[str, Any]) -> None:
    """Fetch blacklisted tracks from metadata DB."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = resolve_database_path(config)
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        tracks = metadata.fetch_blacklisted_tracks()
        emit_result("blacklist", {"tracks": tracks, "count": len(tracks)})
        emit_done("blacklist_fetch", True, f"Fetched {len(tracks)} blacklisted tracks")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_fetch", False, str(e))


def handle_blacklist_fetch_scopes(cmd_data: Dict[str, Any]) -> None:
    """Fetch artist/album/track scope blacklists (grouped, for the web UI)."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = resolve_database_path(config)
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        artists = metadata.fetch_artist_blacklist()
        albums = metadata.fetch_album_blacklist()
        tracks = metadata.fetch_track_blacklist()
        emit_result(
            "blacklist_scopes",
            {"artists": artists, "albums": albums, "tracks": tracks},
        )
        total = len(artists) + len(albums) + len(tracks)
        emit_done("blacklist_fetch_scopes", True, f"Fetched {total} blacklist entries")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_fetch_scopes", False, str(e))


def handle_blacklist_set(cmd_data: Dict[str, Any]) -> None:
    """Set blacklisted flag for track ids."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    track_ids = cmd_data.get("track_ids", []) or []
    value = bool(cmd_data.get("value", True))
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = resolve_database_path(config)
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        updated = metadata.set_blacklisted([str(t) for t in track_ids], value)
        emit_result(
            "blacklist_set",
            {"track_ids": track_ids, "value": value, "updated": updated},
        )
        emit_done("blacklist_set", True, f"Updated {updated} track(s)")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_set", False, str(e))


def handle_blacklist_scope_set(cmd_data: Dict[str, Any]) -> None:
    """Set blacklisted flag for an artist or album scope."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    scope = str(cmd_data.get("scope", "") or "").lower()
    value = str(cmd_data.get("value", "") or "")
    artist = str(cmd_data.get("artist", "") or "")
    enabled = bool(cmd_data.get("enabled", True))
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = resolve_database_path(config)
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        if scope == "artist":
            updated = metadata.set_artist_blacklisted(value, enabled)
        elif scope == "album":
            updated = metadata.set_album_blacklisted(artist, value, enabled)
        else:
            raise ValueError(f"Unsupported blacklist scope: {scope}")
        track_ids = sorted(metadata.fetch_blacklisted_track_ids())
        emit_result(
            "blacklist_scope_set",
            {
                "scope": scope,
                "value": value,
                "artist": artist,
                "enabled": enabled,
                "updated": updated,
                "track_ids": track_ids,
            },
        )
        emit_done("blacklist_scope_set", True, f"Updated {updated} track(s)")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_scope_set", False, str(e))


def handle_enrich_artist(cmd_data: Optional[Dict[str, Any]] = None, *, artist: str = "", request_id: str = "") -> Dict[str, Any]:
    """Run the full hybrid enrichment pipeline for an artist as subprocess invocations.

    Steps: ingest-local → extract-lastfm → extract-bandcamp → classify-tags →
    discover releases → hybrid-enrich-one --apply --with-model-prior per release.
    Source-prep failures stop the batch; one per-release hybrid failure is logged
    and skipped so later albums still receive the hybrid signature.

    Can be called directly (unit tests pass artist/request_id as kwargs) or via
    the command dispatch system (cmd_data dict).
    """
    if cmd_data is not None:
        artist = cmd_data.get("artist", artist)
        request_id = cmd_data.get("request_id", request_id)

    steps = _HYBRID_PREP_STEPS
    total_steps = len(steps) + 1
    for i, (command, extra_args) in enumerate(steps, 1):
        argv = _ai_genre_argv(command, artist=artist, extra_args=extra_args)
        emit_progress(f"enrich:{command}", i, total_steps, artist)
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": f"{command} failed: {_subprocess_error_excerpt(completed)}",
                "step": command,
            }
        emit_log("INFO", f"{command} completed for {artist}", request_id=request_id)

    releases = _discover_artist_releases_for_hybrid(artist)
    if not releases:
        return {"ok": False, "error": "discover failed: no releases found", "step": "discover"}

    total_steps = len(steps) + len(releases)
    applied = 0
    succeeded = 0
    failures: list[dict[str, str]] = []
    for offset, release in enumerate(releases, start=1):
        album = release.get("album") or release.get("normalized_album")
        if not album:
            continue
        command = "hybrid-enrich-one"
        argv = _ai_genre_argv(
            command,
            artist=artist,
            album=str(album),
            extra_args=["--with-model-prior", "--include-provisional", "--apply"],
        )
        emit_progress(f"enrich:{command}", len(steps) + offset, total_steps, f"{artist} / {album}")
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            error = f"{command} failed for {artist} / {album}: {_subprocess_error_excerpt(completed)}"
            failures.append({"album": str(album), "error": error})
            emit_log("ERROR", error, request_id=request_id)
            continue
        applied += _applied_count_from_hybrid_output(completed.stdout)
        succeeded += 1
        emit_log("INFO", f"{command} applied for {artist} / {album}", request_id=request_id)

    if succeeded == 0 and failures:
        return {
            "ok": False,
            "error": f"hybrid-enrich-one failed for all {len(failures)} release(s)",
            "step": "hybrid-enrich-one",
            "failed": len(failures),
            "failures": failures[:10],
        }
    return {
        "ok": True,
        "artist": artist,
        "releases": len(releases),
        "applied": applied,
        "failed": len(failures),
        "failures": failures[:10],
    }


_HYBRID_PREP_STEPS = [
    ("ingest-local", ["--no-rebuild-signatures"]),
    ("extract-lastfm", ["--no-rebuild-signatures"]),
    ("extract-bandcamp", ["--no-rebuild-signatures"]),
    ("classify-tags", ["--adjudicate"]),
]


def handle_enrich_genres(
    cmd_data: Optional[Dict[str, Any]] = None,
    *,
    scope: str = "",
    artist: str = "",
    album: str = "",
    request_id: str = "",
) -> Dict[str, Any]:
    """Run hybrid genre enrichment by GUI scope.

    scope:
      - all_unenriched: discover all releases and enrich those missing a sidecar signature.
      - artist: reuse the existing artist batch path.
      - album: run exact-release prep and hybrid apply.
    """
    if cmd_data is not None:
        scope = (cmd_data.get("scope") or scope or "").strip()
        artist = (cmd_data.get("artist") or artist or "").strip()
        album = (cmd_data.get("album") or album or "").strip()
        request_id = cmd_data.get("request_id", request_id)

    if scope == "artist":
        if not artist:
            return {"ok": False, "error": "artist required", "step": "validate"}
        result = handle_enrich_artist(artist=artist, request_id=request_id)
        result["scope"] = scope
        return result

    if scope == "album":
        if not artist or not album:
            return {"ok": False, "error": "artist and album required", "step": "validate"}
        result = _enrich_single_release(artist=artist, album=album, request_id=request_id)
        result["scope"] = scope
        return result

    if scope != "all_unenriched":
        return {"ok": False, "error": f"unsupported enrichment scope: {scope}", "step": "validate"}

    releases = _discover_releases_for_hybrid()
    if not releases:
        return {"ok": False, "error": "discover failed: no releases found", "step": "discover"}

    enriched_keys = _enriched_release_keys()
    pending = [release for release in releases if release.get("release_key") not in enriched_keys]
    if not pending:
        return {
            "ok": True,
            "scope": scope,
            "releases": 0,
            "applied": 0,
            "skipped_enriched": len(releases),
        }

    applied = 0
    completed = 0
    succeeded = 0
    failures: list[dict[str, str]] = []
    total_steps = len(pending) * (len(_HYBRID_PREP_STEPS) + 1)
    for release in pending:
        release_artist = str(release.get("artist") or release.get("normalized_artist") or "")
        release_album = str(release.get("album") or release.get("normalized_album") or "")
        if not release_artist or not release_album:
            continue
        result = _enrich_single_release(
            artist=release_artist,
            album=release_album,
            request_id=request_id,
            progress_start=completed,
            progress_total=total_steps,
        )
        if not result["ok"]:
            error = str(result.get("error") or "enrichment failed")
            failures.append({"artist": release_artist, "album": release_album, "error": error})
            emit_log("ERROR", error, request_id=request_id)
            completed += len(_HYBRID_PREP_STEPS) + 1
            continue
        completed += len(_HYBRID_PREP_STEPS) + 1
        applied += int(result.get("applied") or 0)
        succeeded += 1

    if succeeded == 0 and failures:
        return {
            "ok": False,
            "scope": scope,
            "error": f"enrichment failed for all {len(failures)} pending release(s)",
            "step": "hybrid-enrich-one",
            "failed": len(failures),
            "failures": failures[:10],
        }

    return {
        "ok": True,
        "scope": scope,
        "releases": len(pending),
        "applied": applied,
        "skipped_enriched": len(releases) - len(pending),
        "failed": len(failures),
        "failures": failures[:10],
    }


def _enrich_single_release(
    *,
    artist: str,
    album: str,
    request_id: str = "",
    progress_start: int = 0,
    progress_total: int | None = None,
) -> Dict[str, Any]:
    total_steps = progress_total or (len(_HYBRID_PREP_STEPS) + 1)
    for offset, (command, extra_args) in enumerate(_HYBRID_PREP_STEPS, start=1):
        argv = _ai_genre_argv(command, artist=artist, album=album, extra_args=extra_args)
        emit_progress(f"enrich:{command}", progress_start + offset, total_steps, f"{artist} / {album}")
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": f"{command} failed: {_subprocess_error_excerpt(completed)}",
                "step": command,
                "artist": artist,
                "album": album,
            }
        emit_log("INFO", f"{command} completed for {artist} / {album}", request_id=request_id)

    command = "hybrid-enrich-one"
    argv = _ai_genre_argv(
        command,
        artist=artist,
        album=album,
        extra_args=["--with-model-prior", "--include-provisional", "--apply"],
    )
    emit_progress(
        f"enrich:{command}",
        progress_start + len(_HYBRID_PREP_STEPS) + 1,
        total_steps,
        f"{artist} / {album}",
    )
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": f"{command} failed: {_subprocess_error_excerpt(completed)}",
            "step": command,
            "artist": artist,
            "album": album,
        }
    applied = _applied_count_from_hybrid_output(completed.stdout)
    emit_log("INFO", f"{command} applied for {artist} / {album}", request_id=request_id)
    return {"ok": True, "artist": artist, "album": album, "releases": 1, "applied": applied}


def _ai_genre_argv(
    command: str,
    *,
    artist: str = "",
    album: str = "",
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    argv = [sys.executable, "scripts/ai_genre_enrich.py", command]
    if artist:
        argv.extend(["--artist", artist])
    if album:
        argv.extend(["--album", album])
    argv.extend(extra_args or [])
    return argv


def _discover_artist_releases_for_hybrid(artist: str) -> list[dict[str, Any]]:
    return _discover_releases_for_hybrid(artist=artist)


def _discover_releases_for_hybrid(artist: str = "", album: str = "") -> list[dict[str, Any]]:
    argv = _ai_genre_argv("discover", artist=artist, album=album, extra_args=["--dry-run"])
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return []
    releases: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
            payload = data.get("payload") or {}
        except json.JSONDecodeError:
            continue
        if data.get("release_key") and not payload.get("release_key"):
            payload["release_key"] = data["release_key"]
        if payload.get("album") or payload.get("normalized_album"):
            releases.append(payload)
    return releases


def _enriched_release_keys() -> set[str]:
    try:
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver

        return EnrichedGenreResolver(SIDECAR_DB_PATH).get_all_enriched_release_keys()
    except Exception:
        return set()


def _applied_count_from_hybrid_output(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return int(json.loads(line).get("applied_count") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0
    return 0


def _subprocess_error_excerpt(completed: subprocess.CompletedProcess) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    return text[:200]


def handle_edit_genres(cmd_data: Dict[str, Any]) -> None:
    """Apply a user genre edit: durable override + surgical authority write.

    Resolves typed genres to canonical taxonomy ids (unknowns reported, not
    saved), then writes both the durable ai_genre_user_overrides diff and the
    release_effective_genres rows for the album via the shared publish
    materializer. base for the diff is read server-side from the authority.
    """
    try:
        artist = (cmd_data.get("artist") or "").strip()
        album = (cmd_data.get("album") or "").strip()
        target_names = [
            str(g).strip() for g in (cmd_data.get("genres") or []) if str(g).strip()
        ]
        if not artist or not album:
            raise ValueError("artist and album are required")

        import sqlite3
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.storage import SidecarStore
        from src.genre.genre_edit import apply_user_genre_edit

        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        db_path = resolve_database_path(config)

        meta_conn = sqlite3.connect(db_path)
        meta_conn.row_factory = sqlite3.Row
        try:
            store = SidecarStore(SIDECAR_DB_PATH)
            store.initialize()
            taxonomy = load_default_layered_taxonomy()
            result = apply_user_genre_edit(
                meta_conn, store, taxonomy,
                artist=artist, album=album, target_names=target_names,
            )
        finally:
            meta_conn.close()

        emit_result("edit_genres", {
            "artist": artist, "album": album,
            "resolved": result.resolved, "unknown": result.unknown,
            "added": result.added, "removed": result.removed,
            "no_change": result.no_change,
        })
        emit_done("edit_genres", True, "ok")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("edit_genres", False, str(e))


def handle_refresh_genre_artifact(cmd_data: Dict[str, Any]) -> None:
    """Re-bake only the genre matrices in the artifact NPZ from the authority."""
    emit_log("INFO", "Refreshing genre matrices in artifact")
    emit_progress("refresh_genre", 0, 100, "Loading artifact")
    try:
        import shutil
        from datetime import datetime
        from scripts.build_beat3tower_artifacts import refresh_genre_matrices

        base_path = cmd_data.get("base_config_path", "config.yaml")
        config = load_config_with_overrides(base_path, cmd_data.get("overrides", {}))
        ds = config.get("playlists", {}).get("ds_pipeline", {})
        artifact_path = ds.get(
            "artifact_path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz"
        )
        db_path = resolve_database_path(config)
        # Use the pipeline-maintained matrix next to the artifact — the same file
        # stage_artifacts bakes from (analyze_library.py) and genre-sim rebuilds each
        # run. The old default (data/genre_similarity_graph.npz) is a standalone-script
        # orphan the pipeline never refreshes, so it baked ~month-stale smoothing here
        # while the full analyze used a fresh matrix (divergent-path bug, 2026-07-07).
        genre_sim_path = ds.get("genre_sim_path") or str(
            Path(artifact_path).parent / "genre_similarity_matrix.npz"
        )

        art = Path(artifact_path)
        if not art.exists():
            raise FileNotFoundError(
                f"artifact not found: {artifact_path} — build artifacts first"
            )
        # Timestamped backup before overwrite (artifact is rebuildable; cheap insurance).
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = art.with_suffix(art.suffix + f".genrebak_{ts}")
        shutil.copy2(art, backup)
        emit_progress("refresh_genre", 30, 100, "Re-baking genre matrices")

        result = refresh_genre_matrices(
            str(art), db_path,
            genre_sim_path=genre_sim_path if Path(genre_sim_path).exists() else None,
            sidecar_db=SIDECAR_DB_PATH, config_path=base_path,
        )

        # Rebuild the dense genre embedding sidecar too. refresh_genre_matrices only
        # rewrites X_genre_raw/smoothed; the candidate-pool genre admission gate reads
        # X_genre_dense (PMI-SVD), which stays stale otherwise — so without this the
        # "for generation" promise is only half-kept (smoothed paths update, the dense
        # admission gate does not). Same call "Build Artifacts" uses.
        emit_progress("refresh_genre", 60, 100, "Rebuilding dense genre sidecar")
        from scripts.build_genre_embedding import build_genre_embedding_sidecar
        sidecar_path = build_genre_embedding_sidecar(str(art), skip_prior=True)
        emit_log("INFO", f"Rebuilt dense genre sidecar: {sidecar_path}")

        # Invalidate the @lru_cache'd bundle so the NEXT generation in this worker
        # picks up the fresh genre matrices + dense sidecar (else the worker keeps
        # serving the bundle it loaded at startup until a GUI restart).
        try:
            from src.features.artifacts import load_artifact_bundle
            load_artifact_bundle.cache_clear()
            emit_log("INFO", "Cleared artifact bundle cache; refreshed genres are live for this session")
        except Exception as exc:  # pragma: no cover - defensive
            emit_log("WARNING", f"Could not clear artifact bundle cache (restart GUI to pick up refresh): {exc}")

        emit_progress("refresh_genre", 100, 100, "Done")
        emit_result(
            "refresh_genre_artifact",
            {**result, "sidecar_path": str(sidecar_path), "backup": str(backup)},
        )
        emit_done(
            "refresh_genre_artifact", True,
            f"Re-baked {result['n_genres']} genres across {result['n_tracks']} tracks + dense sidecar",
        )
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("refresh_genre_artifact", False, str(e))


def handle_scan_genre_review(cmd_data: Dict[str, Any]) -> None:
    """Scan all releases for hybrid review terms and persist the queue."""
    try:
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.review_queue import scan_review_queue
        from src.ai_genre_enrichment.storage import SidecarStore

        store = SidecarStore(SIDECAR_DB_PATH)
        store.initialize()
        emit_log("INFO", "Loading layered taxonomy for review scan...")
        taxonomy = load_default_layered_taxonomy()

        def progress(current: int, total: int, detail: str) -> None:
            emit_progress("scan_genre_review", current, total, detail)

        summary = scan_review_queue(
            store, taxonomy=taxonomy, progress_cb=progress, cancel_cb=check_cancelled,
        )
        emit_result("scan_genre_review", summary)
        emit_done(
            "scan_genre_review", True,
            f"Scanned {summary['releases_scanned']} releases",
            summary=f"{summary['pending_terms']} terms pending review",
        )
    except CancellationError:
        emit_done("scan_genre_review", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("scan_genre_review", False, str(e))


def handle_publish_decided(cmd_data: Dict[str, Any]) -> None:
    """Back up metadata.db, then publish() the materialized assignments into the authority.

    Tracked job — the button click is the explicit confirmation; the backup is automatic
    (CLAUDE.md metadata.db discipline). publish() is the only authority writer.
    """
    import datetime

    from src.db_backup import backup_database

    try:
        metadata_db_path = resolve_database_path(None)
        emit_progress("publish_decided", 0, 2, "backing up metadata.db")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{metadata_db_path}.bak.{ts}"
        # Atomic online-backup, NOT a raw file copy of an open DB: copying an open SQLite
        # file can capture a torn WAL state, which is the root cause of the 2026-07
        # idx_tracks_file_path corruption that silently propagated into every .bak. The
        # helper also integrity-checks the snapshot; a not-clean result means metadata.db
        # itself is corrupt, so surface it loudly instead of backing up a bad DB in silence.
        backup_res = backup_database(metadata_db_path, bak)
        if backup_res.integrity_ok is False:
            emit_progress(
                "publish_decided", 0, 2,
                f"WARNING: metadata.db failed integrity_check ({backup_res.integrity_detail}). "
                "Backup saved, but the source DB needs REINDEX/repair.",
            )
        check_cancelled()
        emit_progress("publish_decided", 1, 2, "publishing")
        from src.genre.genre_publish import publish
        stats = publish(metadata_db_path, SIDECAR_DB_PATH, dry_run=False)
        result = {
            "graph_albums": stats.graph_albums, "legacy_albums": stats.legacy_albums,
            "total_albums": stats.total_albums, "collisions": stats.collisions,
            "backup": bak,
        }
        emit_result("publish_decided", result)
        emit_done("publish_decided", True, f"Published {stats.graph_albums} graph albums",
                  summary=f"graph={stats.graph_albums} legacy={stats.legacy_albums}")
    except CancellationError:
        emit_done("publish_decided", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("publish_decided", False, str(e))


def handle_get_genre_review_queue(cmd_data: Dict[str, Any]) -> None:
    """Return the persisted review queue page.

    Registered in UNTRACKED_COMMAND_HANDLERS so the main loop handles it inline
    without joining the worker thread — safe to call while a scan is running.

    These events set request_id (from cmd_data) and job_id (always None)
    explicitly. The bridge routes this command's reply by request_id, and these
    handlers are not jobs. If we let emit_event fill either field from
    _worker_state, an inline run during a scan would inherit the *scan's*
    request_id/job_id and corrupt the scan job (overwrite tool_result, mark it
    done early). Setting job_id=None makes JobRegistry.apply_event a no-op for
    these events.
    """
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.storage import SidecarStore

        # readonly: this runs on the READER thread. It must never write (no
        # initialize/DDL) and never wait behind a scan's write lock — a blocked
        # read here starves every untracked command, including cancel
        # (2026-06-12 review-queue timeout incident).
        store = SidecarStore(SIDECAR_DB_PATH)
        page = store.get_review_queue_page(
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0),
            readonly=True,
        )
        emit_event({"type": "result", "result_type": "genre_review_queue",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_genre_review_queue", "ok": True,
                    "detail": f"{page['pending_terms']} pending",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_genre_review_queue", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_get_genre_review_completed(cmd_data: Dict[str, Any]) -> None:
    """Return the 'Completed' review page: releases with decided terms.

    Untracked + readonly, same reader-thread discipline as
    handle_get_genre_review_queue (see its docstring for the request_id/job_id
    rationale and why this must never write or block behind a scan).
    """
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.storage import SidecarStore

        store = SidecarStore(SIDECAR_DB_PATH)
        page = store.get_completed_review_page(
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0),
            readonly=True,
        )
        emit_event({"type": "result", "result_type": "genre_review_completed",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_genre_review_completed", "ok": True,
                    "detail": f"{page['decided_terms']} decided",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_genre_review_completed", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_apply_genre_review_decision(cmd_data: Dict[str, Any]) -> None:
    """Apply accept/reject/revert for one review-queue row.

    Registered in UNTRACKED_COMMAND_HANDLERS — see handle_get_genre_review_queue
    docstring for rationale (including why request_id/job_id are set explicitly).
    """
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.review_queue import apply_review_decision
        from src.ai_genre_enrichment.storage import SidecarStore

        # No initialize() here: this runs inline on the reader thread, and DDL
        # is a write that can wait behind a scan. If the queue table doesn't
        # exist yet the decision fails fast with a clear done(ok=false).
        store = SidecarStore(SIDECAR_DB_PATH)
        result = apply_review_decision(
            store,
            release_key=str(cmd_data.get("release_key") or ""),
            term=str(cmd_data.get("term") or ""),
            decision=str(cmd_data.get("decision") or ""),
        )
        emit_event({"type": "result", "result_type": "genre_review_decision",
                    "request_id": rid, "job_id": None, **result})
        emit_event({"type": "done", "cmd": "apply_genre_review_decision", "ok": True,
                    "detail": f"{result['term']}: {result['status']}",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "apply_genre_review_decision", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_get_escalation_queue(cmd_data: Dict[str, Any]) -> None:
    """Album-grain escalation queue page. UNTRACKED + read-only (reader thread)."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import list_page
        page = list_page(
            SIDECAR_DB_PATH, status="pending",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0))
        emit_event({"type": "result", "result_type": "escalation_queue",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_escalation_queue", "ok": True,
                    "detail": f"{page['pending_albums']} pending",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_escalation_queue", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_get_escalation_completed(cmd_data: Dict[str, Any]) -> None:
    """Decided escalations page. UNTRACKED + read-only."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import list_page
        page = list_page(
            SIDECAR_DB_PATH, status="decided",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0))
        emit_event({"type": "result", "result_type": "escalation_completed",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_escalation_completed", "ok": True,
                    "detail": f"{page['decided_albums']} decided",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_escalation_completed", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_apply_escalation_decision(cmd_data: Dict[str, Any]) -> None:
    """Apply accept/edit/reject/revert for one escalation. UNTRACKED (quick write)."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.escalation_queue import EscalationQueue
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
        from src.ai_genre_enrichment.storage import SidecarStore

        album_id = str(cmd_data.get("album_id") or "")
        decision = str(cmd_data.get("decision") or "")
        genres = cmd_data.get("genres") or None
        store = SidecarStore(SIDECAR_DB_PATH)
        queue = EscalationQueue(SIDECAR_DB_PATH)
        try:
            if decision == "revert":
                queue.revert(album_id, sidecar_store=store)
                status = "pending"
            else:
                taxonomy = load_default_layered_taxonomy()
                queue.record_decision(album_id, decision, genres=genres,
                                      sidecar_store=store, taxonomy=taxonomy)
                status = {"accept": "accepted", "edit": "edited", "reject": "rejected"}[decision]
        finally:
            queue.close()
        emit_event({"type": "result", "result_type": "escalation_decision",
                    "album_id": album_id, "status": status,
                    "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "apply_escalation_decision", "ok": True,
                    "detail": f"{album_id}: {status}", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "apply_escalation_decision", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_list_artist_links(cmd_data: Dict[str, Any]) -> None:
    """Return the current artist link groups from data/artist_aliases.yaml. UNTRACKED (read)."""
    rid = cmd_data.get("request_id")
    try:
        from src.playlist.artist_aliases import read_artist_link_groups
        groups = read_artist_link_groups()
        emit_event({"type": "result", "result_type": "artist_links",
                    "groups": groups, "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "list_artist_links", "ok": True,
                    "detail": f"{len(groups)} group(s)", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "list_artist_links", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_save_artist_links(cmd_data: Dict[str, Any]) -> None:
    """Validate + write artist link groups, bust the resolver cache. UNTRACKED (quick write).

    On invalid input, emits done ok=False with the validation detail so the route
    surfaces it as a 422 (WorkerCommandError)."""
    rid = cmd_data.get("request_id")
    try:
        from src.playlist.artist_aliases import save_artist_link_groups
        groups = cmd_data.get("groups") or []
        try:
            save_artist_link_groups(groups)
        except ValueError as ve:
            emit_event({"type": "error", "message": str(ve), "request_id": rid, "job_id": None})
            emit_event({"type": "done", "cmd": "save_artist_links", "ok": False,
                        "detail": str(ve), "request_id": rid, "job_id": None})
            return
        emit_event({"type": "result", "result_type": "artist_links_saved",
                    "count": len(groups), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "save_artist_links", "ok": True,
                    "detail": f"saved {len(groups)} group(s)", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "save_artist_links", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy term adjudication (vocabulary-level review; writes the taxonomy YAML)
# ─────────────────────────────────────────────────────────────────────────────

def handle_get_taxonomy_queue(cmd_data: Dict[str, Any]) -> None:
    """Untriaged taxonomy-term candidate page. UNTRACKED + read-only (reader thread)."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
        from src.ai_genre_enrichment.taxonomy_review_queue import list_page, load_artist_names
        page = list_page(
            SIDECAR_DB_PATH, DEFAULT_TAXONOMY_PATH, status="untriaged",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0),
            artist_names=load_artist_names(resolve_database_path(None)))
        emit_event({"type": "result", "result_type": "taxonomy_queue",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_taxonomy_queue", "ok": True,
                    "detail": f"{page['untriaged_terms']} untriaged",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_taxonomy_queue", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_get_taxonomy_completed(cmd_data: Dict[str, Any]) -> None:
    """Decided taxonomy-term page. UNTRACKED + read-only."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
        from src.ai_genre_enrichment.taxonomy_review_queue import list_page, load_artist_names
        page = list_page(
            SIDECAR_DB_PATH, DEFAULT_TAXONOMY_PATH, status="decided",
            search=(cmd_data.get("search") or "").strip() or None,
            limit=int(cmd_data.get("limit") or 50),
            offset=int(cmd_data.get("offset") or 0),
            artist_names=load_artist_names(resolve_database_path(None)))
        emit_event({"type": "result", "result_type": "taxonomy_completed",
                    "request_id": rid, "job_id": None, **page})
        emit_event({"type": "done", "cmd": "get_taxonomy_completed", "ok": True,
                    "detail": f"{page['decided_terms']} decided",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "get_taxonomy_completed", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_adjudicate_taxonomy_term(cmd_data: Dict[str, Any]) -> None:
    """Ask Claude for an add/alias/reject verdict on one term. TRACKED job: the
    Claude (Agent SDK) call is slow — routinely longer than the 60s untracked
    bridge.command cap — and must run on the worker thread, not the reader thread
    (an untracked Claude call there blocks every other untracked command incl.
    cancel; see web-gui skill). Returns the verdict in the job result; does NOT
    persist (the GUI ratifies via record_taxonomy_decision)."""
    try:
        from dataclasses import asdict

        from src.ai_genre_enrichment.layered_taxonomy import (
            load_default_layered_taxonomy, normalize_taxonomy_name,
        )
        from src.ai_genre_enrichment.provider import create_enrichment_client
        from src.ai_genre_enrichment.taxonomy_review_queue import (
            _open_store_readonly, build_candidate_index,
        )
        from src.ai_genre_enrichment.taxonomy_term_adjudicator import (
            RejectVerdict, adjudicate_term,
        )

        term = normalize_taxonomy_name(str(cmd_data.get("term") or ""))
        if not term:
            raise ValueError("term is required")
        emit_progress("adjudicate_taxonomy_term", 0, 1, f"asking Claude about {term}")
        taxonomy = load_default_layered_taxonomy()
        index = build_candidate_index(_open_store_readonly(SIDECAR_DB_PATH), taxonomy)
        candidate = index.get(term)
        if candidate is None:
            raise ValueError(f"term not in candidate queue: {term!r}")
        check_cancelled()
        client = create_enrichment_client()  # web_mode off by default
        verdict = adjudicate_term(candidate, taxonomy, client=client)
        if isinstance(verdict, RejectVerdict):
            out = {"verdict": "reject", "term": term,
                   "proposal": {"reject_reason": verdict.reject_reason,
                                "rationale": verdict.rationale}}
        else:
            out = {"verdict": "alias" if verdict.kind == "alias" else "add",
                   "term": term, "proposal": asdict(verdict)}
        emit_result("adjudicate_taxonomy_term", out)
        emit_done("adjudicate_taxonomy_term", True, term)
    except CancellationError:
        emit_done("adjudicate_taxonomy_term", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("adjudicate_taxonomy_term", False, str(e))


def handle_record_taxonomy_decision(cmd_data: Dict[str, Any]) -> None:
    """Record/revert one term decision to the staging table. UNTRACKED (quick write)."""
    rid = cmd_data.get("request_id")
    try:
        import json as _json

        from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore

        term = str(cmd_data.get("term") or "")
        verdict = str(cmd_data.get("verdict") or "")
        store = TaxonomyDecisionStore(SIDECAR_DB_PATH)
        try:
            if verdict == "revert":
                store.revert(term)
                status = "reverted"
            else:
                store.record_decision(
                    term=term, raw_term=str(cmd_data.get("raw_term") or term),
                    verdict=verdict,
                    proposal_json=_json.dumps(cmd_data.get("proposal") or None),
                    claude_json=_json.dumps(cmd_data.get("claude") or None),
                    human_edited=int(bool(cmd_data.get("human_edited"))))
                status = verdict
        finally:
            store.close()
        emit_event({"type": "result", "result_type": "taxonomy_decision",
                    "term": term, "status": status,
                    "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "record_taxonomy_decision", "ok": True,
                    "detail": f"{term}: {status}", "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "record_taxonomy_decision", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def handle_validate_taxonomy_proposal(cmd_data: Dict[str, Any]) -> None:
    """Structural pre-flight for a manual ADD-wizard proposal. UNTRACKED +
    read-only (reader thread): wraps graph_growth.validate_proposal — the
    single guardrail authority — so the GUI never re-implements placement
    rules. Emits {"errors": [...]}; empty list means safe to stage."""
    rid = cmd_data.get("request_id")
    try:
        from src.ai_genre_enrichment.graph_growth import GrowthProposal, validate_proposal
        from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

        pj = cmd_data.get("proposal")
        pj = pj if isinstance(pj, dict) else {}
        gp_fields = set(GrowthProposal.__dataclass_fields__)  # type: ignore[attr-defined]
        required = {"name", "kind", "status", "specificity_score"}
        proposal = GrowthProposal(
            name=str(pj.get("name") or ""),
            kind=str(pj.get("kind") or ""),
            status=str(pj.get("status") or "active"),
            specificity_score=float(pj.get("specificity_score") or 0.0),
            **{k: v for k, v in pj.items() if k in gp_fields - required},
        )
        errors = validate_proposal(load_default_layered_taxonomy(), proposal)
        emit_event({"type": "result", "result_type": "taxonomy_validate",
                    "errors": errors, "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "validate_taxonomy_proposal", "ok": True,
                    "detail": f"{proposal.name}: {len(errors)} error(s)",
                    "request_id": rid, "job_id": None})
    except Exception as e:
        emit_event({"type": "error", "message": str(e), "request_id": rid, "job_id": None})
        emit_event({"type": "done", "cmd": "validate_taxonomy_proposal", "ok": False,
                    "detail": str(e), "request_id": rid, "job_id": None})


def _next_taxonomy_gui_version(current: str, ts: str) -> str:
    """Bump the minor of a 0.X.Y version and stamp the GUI grow suffix."""
    import re
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(current or ""))
    if m:
        return f"{int(m.group(1))}.{int(m.group(2)) + 1}.0-gui-{ts}-grown"
    return f"0.1.0-gui-{ts}-grown"


def handle_apply_taxonomy_decisions(cmd_data: Dict[str, Any]) -> None:
    """Validate the pending decision batch, back up + write the taxonomy YAML, bump
    the version, reload the graph cache. TRACKED job (the button click is the
    explicit confirmation; the YAML is git-tracked, so the backup + one-diff-per-Apply
    discipline mirror handle_publish_decided). NEVER touches metadata.db."""
    import datetime

    try:
        import yaml as _yaml

        from src.ai_genre_enrichment.graph_growth import GrowthProposal
        from src.ai_genre_enrichment.layered_taxonomy import DEFAULT_TAXONOMY_PATH
        from src.ai_genre_enrichment.taxonomy_apply import Decision, apply_decisions
        from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore

        emit_progress("apply_taxonomy_decisions", 0, 3, "reading pending decisions")
        store = TaxonomyDecisionStore(SIDECAR_DB_PATH)
        try:
            pending = store.list_pending()
            if not pending:
                emit_result("apply_taxonomy_decisions",
                            {"ok": True, "added": 0, "aliased": 0, "rejected": 0,
                             "deferred_edges": [], "applied_terms": []})
                emit_done("apply_taxonomy_decisions", True, "No pending decisions",
                          summary="nothing to apply")
                return

            gp_fields = {f for f in GrowthProposal.__dataclass_fields__}  # type: ignore[attr-defined]
            decisions: List[Decision] = []
            for r in pending:
                pj = r["proposal"] if isinstance(r["proposal"], dict) else {}
                proposal = None
                reject_reason = None
                if r["verdict"] in ("add", "alias"):
                    proposal = GrowthProposal(
                        **{k: v for k, v in pj.items() if k in gp_fields})
                elif r["verdict"] == "reject":
                    reject_reason = pj.get("reject_reason")
                decisions.append(Decision(
                    term=r["term"], verdict=r["verdict"], proposal=proposal,
                    reject_reason=reject_reason,
                    rationale=str(pj.get("rationale") or "")))

            check_cancelled()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            current_version = str(
                (_yaml.safe_load(Path(DEFAULT_TAXONOMY_PATH).read_text(encoding="utf-8"))
                 or {}).get("taxonomy_version") or "")
            version = _next_taxonomy_gui_version(current_version, ts)

            emit_progress("apply_taxonomy_decisions", 1, 3, "validating + writing")
            result = apply_decisions(DEFAULT_TAXONOMY_PATH, decisions, new_version=version)
            if result.validation_failures:
                emit_result("apply_taxonomy_decisions",
                            {"ok": False, "validation_failures": result.validation_failures,
                             "deferred_edges": result.deferred_edges})
                emit_done("apply_taxonomy_decisions", False,
                          "Validation failed — nothing written")
                return
            store.mark_applied([d.term for d in decisions], version)
        finally:
            store.close()

        emit_progress("apply_taxonomy_decisions", 2, 3, "reloading graph cache")
        # §6 cache note: graph_adapter holds the only long-lived LayeredTaxonomy
        # (lru_cache). Bust it so subsequent graph-adapter consumers in this worker
        # session see the new vocabulary (mirrors the load_artifact_bundle bust).
        try:
            from src.genre.graph_adapter import _cached_default_taxonomy
            _cached_default_taxonomy.cache_clear()
        except Exception as exc:  # pragma: no cover - defensive
            emit_log("WARNING", f"Could not clear taxonomy cache (restart GUI): {exc}")

        # "M albums will re-classify on next publish" — distinct albums whose
        # collected tags include a just-added/aliased term (impact = sum reach).
        applied_terms = [d.term for d in decisions if d.verdict in ("add", "alias")]
        result_payload = {
            "ok": True, "added": result.added, "aliased": result.aliased,
            "rejected": result.rejected, "deferred_edges": result.deferred_edges,
            "backup": result.backup_path, "new_version": result.new_version,
            "applied_terms": applied_terms,
        }
        emit_result("apply_taxonomy_decisions", result_payload)
        total = result.added + result.aliased + result.rejected
        emit_done("apply_taxonomy_decisions", True, f"Applied {total} decisions",
                  summary=f"added={result.added} aliased={result.aliased} "
                          f"rejected={result.rejected} v={result.new_version}")
    except CancellationError:
        emit_done("apply_taxonomy_decisions", False, "Cancelled", cancelled=True)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        emit_done("apply_taxonomy_decisions", False, str(e))


def handle_enrich_artist_cmd(cmd_data: Dict[str, Any]) -> None:
    """Command handler wrapper for enrich_artist — called from the dispatch table."""
    artist = cmd_data.get("artist", "")
    request_id = cmd_data.get("request_id", "")
    if not artist:
        emit_error("enrich_artist requires 'artist' field")
        emit_done("enrich_artist", False, "artist required")
        return

    result = handle_enrich_artist(artist=artist, request_id=request_id)
    if result["ok"]:
        emit_result("enrich_artist", result)
        emit_done("enrich_artist", True, artist)
    else:
        emit_error(result.get("error", "enrichment failed"))
        emit_done("enrich_artist", False, artist)


def handle_enrich_genres_cmd(cmd_data: Dict[str, Any]) -> None:
    """Command handler wrapper for scoped hybrid genre enrichment."""
    result = handle_enrich_genres(cmd_data)
    scope = cmd_data.get("scope", "")
    if result["ok"]:
        emit_result("enrich_genres", result)
        detail = scope or result.get("artist") or "enrich_genres"
        summary = f"Enriched {result.get('releases', 0)} release(s); applied {result.get('applied', 0)} genre(s)"
        emit_done("enrich_genres", True, detail, summary=summary)
    else:
        emit_error(result.get("error", "enrichment failed"))
        emit_done("enrich_genres", False, scope or "enrich_genres")


# ─────────────────────────────────────────────────────────────────────────────
# Command Router
# ─────────────────────────────────────────────────────────────────────────────

# Commands that run with request tracking (long-running operations)
TRACKED_COMMAND_HANDLERS = {
    "ping": handle_ping,
    "generate_playlist": handle_generate_playlist,
    "analyze_library": handle_analyze_library,
    "scan_library": handle_scan_library,
    "update_genres": handle_update_genres,
    "update_sonic": handle_update_sonic,
    "build_artifacts": handle_build_artifacts,
    "doctor": handle_doctor,
    "find_replacement_suggestions": handle_find_replacement_suggestions,
    "blacklist_fetch": handle_blacklist_fetch,
    "blacklist_fetch_scopes": handle_blacklist_fetch_scopes,
    "blacklist_set": handle_blacklist_set,
    "blacklist_scope_set": handle_blacklist_scope_set,
    "enrich_artist": handle_enrich_artist_cmd,
    "enrich_genres": handle_enrich_genres_cmd,
    "edit_genres": handle_edit_genres,
    "refresh_genre_artifact": handle_refresh_genre_artifact,
    "scan_genre_review": handle_scan_genre_review,
    "publish_decided": handle_publish_decided,
    "apply_taxonomy_decisions": handle_apply_taxonomy_decisions,
    "adjudicate_taxonomy_term": handle_adjudicate_taxonomy_term,
}

# Commands that don't have their own request context
def handle_set_logging_level(cmd_data: Dict[str, Any]) -> None:
    """
    Handle set_logging_level command - update worker logging level.

    This is an untracked command that doesn't produce tracked events.
    """
    level = cmd_data.get("level", "INFO").upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    if level not in valid_levels:
        emit_log("WARNING", f"Invalid log level: {level}, using INFO")
        level = "INFO"

    try:
        from src import logging_utils
        logging_utils.set_log_level(level)
        emit_log("INFO", f"Logging level changed to {level}")
    except Exception as e:
        emit_log("ERROR", f"Failed to change logging level: {e}")


UNTRACKED_COMMAND_HANDLERS = {
    "cancel": handle_cancel,
    "set_logging_level": handle_set_logging_level,
    # Quick DB reads/writes: run inline on the reader thread so they're
    # not blocked by a long-running tracked command (e.g. scan_genre_review).
    "get_genre_review_queue": handle_get_genre_review_queue,
    "get_genre_review_completed": handle_get_genre_review_completed,
    "apply_genre_review_decision": handle_apply_genre_review_decision,
    "get_escalation_queue": handle_get_escalation_queue,
    "get_escalation_completed": handle_get_escalation_completed,
    "apply_escalation_decision": handle_apply_escalation_decision,
    "list_artist_links": handle_list_artist_links,
    "save_artist_links": handle_save_artist_links,
    "get_taxonomy_queue": handle_get_taxonomy_queue,
    "get_taxonomy_completed": handle_get_taxonomy_completed,
    "record_taxonomy_decision": handle_record_taxonomy_decision,
    "validate_taxonomy_proposal": handle_validate_taxonomy_proposal,
}


def process_command(line: str) -> None:
    """
    Parse and dispatch a command line.

    Handles request_id tracking for correlated events.
    """
    line = line.strip()
    if not line:
        return

    try:
        cmd_data = json.loads(line)
    except json.JSONDecodeError as e:
        emit_error(f"Invalid JSON: {e}")
        return

    cmd = cmd_data.get("cmd")
    if not cmd:
        emit_error("Missing 'cmd' field in command")
        return

    # Handle untracked commands (like cancel) first
    if cmd in UNTRACKED_COMMAND_HANDLERS:
        try:
            UNTRACKED_COMMAND_HANDLERS[cmd](cmd_data)
        except Exception as e:
            emit_log("ERROR", f"Untracked command {cmd} failed: {e}")
            # A missing done leaves the bridge's command() future waiting until
            # TimeoutError — every handler emits its own done, but a crash
            # outside its try must still answer the request.
            emit_event({"type": "done", "cmd": cmd, "ok": False, "detail": str(e),
                        "request_id": cmd_data.get("request_id"), "job_id": None})
        return

    # For tracked commands, get identifiers
    request_id = cmd_data.get("request_id")
    job_id = cmd_data.get("job_id")
    if not request_id:
        # Legacy support: generate a warning but continue
        emit_log("WARNING", f"Command {cmd} missing request_id (legacy mode)")

    handler = TRACKED_COMMAND_HANDLERS.get(cmd)
    if not handler:
        emit_error(f"Unknown command: {cmd}")
        if request_id:
            _worker_state.start_request(request_id, cmd, job_id)
        emit_done(cmd, False, "Unknown command")
        if request_id:
            _worker_state.end_request()
        return

    # Start tracking this request
    if request_id:
        _worker_state.start_request(request_id, cmd, job_id)

    try:
        handler(cmd_data)
    except CancellationError:
        emit_log("INFO", f"Command {cmd} cancelled")
        emit_done(cmd, False, "Cancelled by user", cancelled=True)
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(f"Command {cmd} failed: {e}", tb)
        emit_done(cmd, False, str(e))
    finally:
        # Always clear request state after command completes
        if request_id:
            _worker_state.end_request()


def _peek_cmd(line: str) -> Optional[str]:
    """Return the `cmd` field of an NDJSON line without raising on bad input."""
    try:
        return json.loads(line).get("cmd")
    except (json.JSONDecodeError, AttributeError):
        return None


def main():
    """Main worker loop - read commands from stdin, emit events to stdout.

    Tracked (long-running) commands execute on a single worker thread so the
    reader stays free to dispatch untracked commands — chiefly ``cancel`` — the
    instant they arrive. This is what makes cancellation work: the running
    command polls the cancellation flag at its own safe checkpoints and unwinds.
    The web bridge enforces single-flight, so at most one tracked command runs at
    a time; we still join any in-flight command before starting the next.
    """
    setup_worker_logging()
    emit_log("INFO", "Worker started, ready for commands")

    worker_thread: Optional[threading.Thread] = None
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            if _peek_cmd(line) in UNTRACKED_COMMAND_HANDLERS:
                # cancel / set_logging_level: run inline, immediately, even while
                # a tracked command is mid-flight on the worker thread.
                process_command(line)
            else:
                if worker_thread is not None and worker_thread.is_alive():
                    worker_thread.join()
                worker_thread = threading.Thread(
                    target=process_command, args=(line,),
                    name="worker-command", daemon=True,
                )
                worker_thread.start()
    except KeyboardInterrupt:
        emit_log("INFO", "Worker interrupted")
    except Exception as e:
        emit_error(f"Worker fatal error: {e}", traceback.format_exc())
        sys.exit(1)
    finally:
        # stdin closed → the parent (web bridge) is gone or shutting down. Signal
        # any in-flight tracked command to unwind, then exit promptly. The worker
        # thread is a daemon, so we MUST NOT block the process on a long-running
        # job here — an unbounded join leaves the worker grinding after the parent
        # dies, orphaning it and wedging the bridge (worker-orphan incident
        # 2026-06-12). The bounded join gives a clean unwind a brief window; past
        # that we let the daemon thread be reclaimed when the process exits.
        if worker_thread is not None and worker_thread.is_alive():
            _worker_state.cancel_active()
            worker_thread.join(timeout=2.0)

    emit_log("INFO", "Worker shutdown")


if __name__ == "__main__":
    main()
