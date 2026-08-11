"""
Playlist reporting and diagnostics module.

This module provides functions for generating detailed reports about playlists,
including edge score computation and comprehensive statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence, Tuple
import logging
import os
import numpy as np

from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    resolve_artist_identity_keys,
)
from src.playlist.utils import sanitize_for_logging
from src.features.artifacts import load_artifact_bundle
from src.playlist.transition_metrics import (
    build_transition_metric_context,
    resolve_transition_calib,
    score_transition_edge,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for playlist report generation."""
    verbose: bool = False
    include_edge_scores: bool = True
    include_percentiles: bool = True
    verbose_edges: bool = False


def format_duration(duration_ms: int) -> str:
    """
    Convert milliseconds to human-readable duration format.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Formatted duration string (e.g., "45.2 minutes")
    """
    total_minutes = duration_ms / 1000 / 60
    return f"{total_minutes:.1f} minutes"


def format_track_duration(duration_ms: Optional[int]) -> str:
    """
    Format track duration in milliseconds as M:SS.

    Args:
        duration_ms: Track duration in milliseconds (or None)

    Returns:
        Formatted duration string like "3:45"
    """
    if not duration_ms or duration_ms <= 0:
        return "0:00"
    seconds = int(duration_ms) // 1000
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def log_recency_edge_diff(
    *,
    before_tracks: List[Dict[str, Any]],
    after_tracks: List[Dict[str, Any]]
) -> None:
    """
    Diagnostic: log adjacency changes introduced by recency filtering (no behavior change).

    Args:
        before_tracks: Track list before recency filtering
        after_tracks: Track list after recency filtering
    """
    diag_enabled = bool(os.environ.get("PLAYLIST_DIAG_RECENCY"))
    if not diag_enabled and not logger.isEnabledFor(logging.DEBUG):
        return
    before_ids = [str(t.get("rating_key")) for t in before_tracks]
    after_ids = [str(t.get("rating_key")) for t in after_tracks]
    removed = [tid for tid in before_ids if tid not in after_ids]
    before_edges = list(zip(before_ids, before_ids[1:]))
    after_edges = list(zip(after_ids, after_ids[1:]))
    new_edges = [e for e in after_edges if e not in before_edges]
    logger.info(
        "Recency adjacency diag: before=%d after=%d removed=%d new_edges=%d",
        len(before_ids),
        len(after_ids),
        len(removed),
        len(new_edges),
    )
    if removed:
        logger.info("Recency removed ids: %s", removed[:10])
    if new_edges:
        logger.info("Recency new edges (sample): %s", new_edges[:10])


def diagnose_t_mismatch(
    edges: list[dict],
    *,
    transition_floor: float,
    tolerance: float = 0.05,
) -> list[dict]:
    """Warn when beam-scored T and final-emitted T disagree on a broken edge."""
    issues: list[dict] = []
    for e in edges:
        final_t = e.get("T")
        beam_t = e.get("trans_score_in_beam")
        if final_t is None or beam_t is None:
            continue
        try:
            ft = float(final_t)
            bt = float(beam_t)
        except Exception:
            continue
        if ft < float(transition_floor) and (bt - ft) > float(tolerance):
            logger.warning(
                "T-mismatch edge %s->%s: beam_trans=%.3f final_T=%.3f (floor=%.2f)",
                e.get("from_idx"), e.get("to_idx"),
                bt, ft, float(transition_floor),
            )
            issues.append(dict(e))
    return issues


def emit_selected_edge_audit(edge_rows: list[dict], *, transition_floor: float = 0.20) -> None:
    """Emit one log row per selected edge with full scoring breakdown.

    Diagnostic only; no behavior change. Each row contains the fields
    populated in beam scoring (bridge_score, trans_score_in_beam,
    progress_t/jump, local_sonic_raw_cos, local_sonic_penalty_applied,
    genre_penalty_applied, below_transition_floor) plus the final-emitted
    edge metrics (T, T_centered_cos, S, G). Missing fields render as 'n/a'.
    """
    if not edge_rows:
        return

    def _f(row: dict, key: str, fmt: str = "%.3f") -> str:
        v = row.get(key)
        if v is None:
            return "n/a"
        try:
            return fmt % float(v)
        except Exception:
            return str(v)

    logger.info("=" * 80)
    logger.info("Selected-edge audit (%d edges)", len(edge_rows))
    logger.info("=" * 80)
    for i, row in enumerate(edge_rows):
        from_label = "%s - %s" % (
            row.get("from_artist", "?"),
            row.get("from_title", "?"),
        )
        to_label = "%s - %s" % (
            row.get("to_artist", "?"),
            row.get("to_title", "?"),
        )
        below_floor = bool(row.get("below_transition_floor", False))
        floor_flag = "⚠ " if below_floor else ""
        logger.info(
            "Edge #%02d: %s%s -> %s", i + 1, floor_flag, from_label, to_label,
        )
        flags = row.get("to_title_flags") or set()
        flag_str = ",".join(sorted(flags)) if flags else "-"
        logger.info(
            "  T=%s T_centered_cos=%s S=%s G=%s | bridge=%s trans_beam=%s title_flags=%s",
            _f(row, "T"), _f(row, "T_centered_cos"), _f(row, "S"), _f(row, "G"),
            _f(row, "bridge_score"), _f(row, "trans_score_in_beam"), flag_str,
        )
        logger.info(
            "  progress_t=%s progress_jump=%s local_sonic_cos=%s local_pen=%s genre_pen=%s below_floor=%s",
            _f(row, "progress_t"), _f(row, "progress_jump"),
            _f(row, "local_sonic_raw_cos"),
            _f(row, "local_sonic_penalty_applied"),
            _f(row, "genre_penalty_applied"),
            below_floor,
        )
        bpm_a = row.get("bpm_a")
        bpm_b = row.get("bpm_b")
        bpm_dist = row.get("bpm_log_dist")
        bpm_str = (
            f"bpm={bpm_a:.0f}->{bpm_b:.0f} dist={bpm_dist:.3f}"
            if bpm_a is not None and bpm_b is not None
            else "bpm=n/a"
        )
        logger.info("  %s", bpm_str)

    # Regression check: beam T and final reporter T should share the same metric.
    diagnose_t_mismatch(
        edge_rows,
        transition_floor=float(transition_floor),
        tolerance=0.05,
    )


def emit_edge_repair_log(swap_log: list[dict]) -> None:
    """Emit repair swap diagnostics when the opt-in repair pass ran."""
    if not swap_log:
        return
    logger.info("=" * 80)
    logger.info("Edge repair swap log (%d entries)", len(swap_log))
    logger.info("=" * 80)
    for entry in swap_log:
        if not isinstance(entry, dict):
            continue
        if "new_idx" in entry:
            logger.info(
                "Repair edge=%s pos=%s reason=%s old=%s/%s new=%s/%s old_worst_T=%s new_worst_T=%s",
                entry.get("edge_position"),
                entry.get("position"),
                entry.get("reason"),
                entry.get("old_idx"),
                entry.get("old_id"),
                entry.get("new_idx"),
                entry.get("new_id"),
                entry.get("old_worst_T"),
                entry.get("new_worst_T"),
            )
        else:
            # Per-candidate refusals are DEBUG-only: one line per rejected
            # candidate per edge floods INFO (hundreds of lines/generation).
            # The accepted swaps above + the summary stay at INFO.
            logger.debug(
                "Repair refusal edge=%s pos=%s candidate=%s/%s reason=%s",
                entry.get("edge_position"),
                entry.get("position"),
                entry.get("candidate_idx"),
                entry.get("candidate_id"),
                entry.get("reason"),
            )


def compute_edge_scores_from_artifact(
    *,
    tracks: List[Dict[str, Any]],
    artifact_path: Optional[str],
    transition_floor: Optional[float] = None,
    transition_gamma: Optional[float] = None,
    embedding_random_seed: Optional[int] = None,
    center_transitions: bool = False,
    verbose: bool = False,
    last_ds_report: Optional[Dict[str, Any]] = None,
    transition_weights: Optional[tuple[float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-edge scores (T/S/G) for the final playlist order using artifact matrices.

    Args:
        tracks: List of track dictionaries
        artifact_path: Path to artifact bundle
        transition_floor: Minimum transition score threshold
        transition_gamma: Blend weight for transition scoring
        embedding_random_seed: Random seed for embedding
        center_transitions: Whether to center transition embeddings
        verbose: Enable verbose logging
        last_ds_report: Last DS report dictionary (for storing baseline)

    Returns:
        List of edge score dictionaries
    """
    if not artifact_path or not tracks or len(tracks) < 2:
        return []
    try:
        bundle = load_artifact_bundle(artifact_path)
    except Exception as exc:
        logger.error("Failed to load artifact for edge logging (%s)", exc, exc_info=True)
        return []

    # Map rating_key -> index
    idxs = []
    missing = 0
    missing_ids: List[str] = []
    for t in tracks:
        tid = str(t.get("rating_key"))
        idx = bundle.track_id_to_index.get(tid)
        if idx is None:
            missing += 1
            missing_ids.append(tid)
            idxs.append(None)
        else:
            idxs.append(idx)
    if missing:
        logger.error("Missing %d track ids in artifact; edge scores may be incomplete.", missing)
        if verbose:
            sample = missing_ids[:5]
            logger.info(
                "Missing track id samples: %s",
                [(tid, tid in bundle.track_id_to_index) for tid in sample],
            )

    X_sonic = getattr(bundle, "X_sonic", None)
    X_genre = getattr(bundle, "X_genre_smoothed", None)
    if X_sonic is None:
        return []

    # Transition calibration must track the ACTIVE sonic variant's cosine band
    # (MuQ runs hot vs MERT) or the rescale sigmoid saturates and every reported
    # edge collapses to ~1.0 — the diagnostics-panel bug. Resolve from
    # bundle.sonic_variant (the authoritative mert/muq name), exactly as the beam
    # does (pier_bridge_builder.py:490).
    _cal_c, _cal_s, _cal_g = resolve_transition_calib(getattr(bundle, "sonic_variant", None))
    ctx = build_transition_metric_context(
        X_sonic=X_sonic,
        X_start=getattr(bundle, "X_sonic_start", None),
        X_mid=getattr(bundle, "X_sonic_mid", None),
        X_end=getattr(bundle, "X_sonic_end", None),
        X_genre=X_genre,
        center_transitions=bool(center_transitions),
        transition_gamma=transition_gamma,
        embedding_random_seed=embedding_random_seed,
        calib_center=_cal_c,
        calib_scale=_cal_s,
        calib_gain=_cal_g,
    )

    edge_scores: List[Dict[str, Any]] = []
    missing_edge_pairs: List[Tuple[str, str]] = []

    # Determine valid bounds for indices
    max_idx = 0
    if X_sonic is not None:
        max_idx = X_sonic.shape[0] - 1
    elif X_genre is not None:
        max_idx = X_genre.shape[0] - 1

    for i in range(1, len(tracks)):
        prev_idx = idxs[i - 1]
        cur_idx = idxs[i]
        if prev_idx is None or cur_idx is None:
            if verbose:
                missing_edge_pairs.append(
                    (
                        str(tracks[i - 1].get("rating_key")),
                        str(tracks[i].get("rating_key")),
                    )
                )
            continue
        # Check bounds
        if prev_idx < 0 or prev_idx > max_idx or cur_idx < 0 or cur_idx > max_idx:
            if verbose:
                missing_edge_pairs.append(
                    (
                        str(tracks[i - 1].get("rating_key")),
                        str(tracks[i].get("rating_key")),
                    )
                )
            logger.warning(
                "Track indices out of bounds: prev_idx=%d cur_idx=%d (valid range 0-%d)",
                prev_idx, cur_idx, max_idx
            )
            continue
        edge = score_transition_edge(ctx, prev_idx, cur_idx)
        edge["floor"] = transition_floor
        edge["gamma"] = transition_gamma
        edge_scores.append(
            {
                "prev_id": tracks[i - 1].get("rating_key"),
                "cur_id": tracks[i].get("rating_key"),
                "prev_idx": prev_idx,
                "cur_idx": cur_idx,
                **edge,
            }
        )

    if verbose:
        # Sample from the full artifact candidate space to understand baseline similarity levels.
        cand_indices = list(range(int(getattr(X_sonic, "shape", [0])[0] or 0)))
        if not cand_indices:
            cand_indices = sorted({idx for idx in idxs if idx is not None})
        if len(cand_indices) >= 2:
            rng = np.random.default_rng(0)
            sample_size = min(2000, max(1, len(cand_indices) * 2))
            sample_pairs = rng.choice(cand_indices, size=(sample_size, 2), replace=True)

            def _pct(arr: Optional[np.ndarray]):
                if arr is None or arr.size == 0:
                    return None
                return {
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p99": float(np.percentile(arr, 99)),
                }

            base_s = None
            base_g = None
            base_t_raw = None
            base_t_center = None
            base_t_used = None
            sampled_edges = [score_transition_edge(ctx, int(a), int(b)) for a, b in sample_pairs]
            base_s = np.array([float(e.get("S", float("nan"))) for e in sampled_edges], dtype=float)
            if any(e.get("G") is not None for e in sampled_edges):
                base_g = np.array([float(e.get("G", float("nan"))) for e in sampled_edges], dtype=float)
            base_t_raw = np.array([float(e.get("T_raw", float("nan"))) for e in sampled_edges], dtype=float)
            base_t_center = np.array([float(e.get("T_centered_cos", float("nan"))) for e in sampled_edges], dtype=float)
            base_t_used = np.array([float(e.get("T", float("nan"))) for e in sampled_edges], dtype=float)
            baseline = {
                "S": _pct(base_s),
                "G": _pct(base_g),
                "T": _pct(base_t_used),
                "T_raw": _pct(base_t_raw),
                "T_centered_cos": _pct(base_t_center),
                "T_centered_rescaled": _pct(base_t_used) if center_transitions else None,
            }
            if last_ds_report is not None:
                last_ds_report["baseline"] = baseline

        idx_counts = {}
        for idx in idxs:
            if idx is None:
                continue
            idx_counts[idx] = idx_counts.get(idx, 0) + 1
        if idx_counts:
            unique_indices_used = len(idx_counts)
            max_repeat = max(idx_counts.values())
            repeated = sum(1 for v in idx_counts.values() if v > 1)
            logger.info(
                "Edge mapping summary: unique_indices=%d repeated=%d max_repeat=%d",
                unique_indices_used,
                repeated,
                max_repeat,
            )
            if ctx.X_sonic_norm is not None:
                per_dim_std = np.std(ctx.X_sonic_norm, axis=0)
                mean_std = float(per_dim_std.mean())
                if mean_std < 1e-3:
                    logger.warning("X_sonic appears nearly constant (mean per-dim std=%.6f)", mean_std)
        if missing_edge_pairs and verbose:
            sample_pairs = missing_edge_pairs[:5]
            logger.info(
                "Edges skipped due to missing indices: %d sample ids=%s",
                len(missing_edge_pairs),
                [
                    (
                        p,
                        c,
                        p in bundle.track_id_to_index,
                        c in bundle.track_id_to_index,
                    )
                    for p, c in sample_pairs
                ],
            )

    return edge_scores


def _report_seed_artist_share(
    tracks: List[Dict[str, Any]],
    artist_name: Optional[str],
    seed_artist_names: Optional[Sequence[str]],
    artist_identity_cfg: Optional["ArtistIdentityConfig"],
) -> None:
    """Log how much of the playlist each seed artist actually occupies.

    Counts with ``resolve_artist_identity_keys`` -- the SAME resolution the
    generator used to seat the piers. The previous raw ``.strip().casefold()``
    equality made this number wrong in both directions on real playlists:
    "The Bill Evans Trio" and "Macross 82-99 feat. Desired" were credited to
    nobody, so a Bill Evans blend with six seated piers reported 3.3%, and a
    MACROSS 82-99 playlist holding seven of its tracks reported five.

    For a multi-artist blend every chip gets its own line plus a combined one.
    Reporting only the primary chip is what made a 3/3 split read as 10% when
    the playlist was 20% seed artists, which reads as a shortfall that isn't
    there. The combined figure counts TRACKS, not credits, so a track credited
    to two seed artists at once is not double-counted.
    """
    names = [str(n) for n in (seed_artist_names or []) if str(n or "").strip()]
    if not names and artist_name:
        names = [str(artist_name)]
    if not names or not tracks:
        return

    cfg = artist_identity_cfg if artist_identity_cfg is not None else ArtistIdentityConfig()
    track_keys = [
        resolve_artist_identity_keys(str(t.get("artist") or ""), cfg) for t in tracks
    ]

    total = len(tracks)
    matched_any = [False] * total
    for name in names:
        seed_keys = resolve_artist_identity_keys(name, cfg)
        count = 0
        for i, keys in enumerate(track_keys):
            if keys & seed_keys:
                count += 1
                matched_any[i] = True
        logger.info(
            "  - Seed artist (%s): %d tracks (%.1f%%)",
            sanitize_for_logging(name), count, count / total * 100,
        )
        logger.debug(
            "Seed artist identity keys: %s=%s | first3 track keys: %s",
            sanitize_for_logging(name), sorted(seed_keys),
            [sorted(k) for k in track_keys[:3]],
        )

    if len(names) > 1:
        combined = sum(1 for m in matched_any if m)
        logger.info(
            "  - Seed artists combined: %d tracks (%.1f%%)",
            combined, combined / total * 100,
        )


def print_playlist_report(
    *,
    tracks: List[Dict[str, Any]],
    artist_name: str = None,
    dynamic: bool = False,
    verbose_edges: bool = False,
    last_ds_report: Optional[Dict[str, Any]] = None,
    last_scope: Optional[str] = None,
    last_cohesion_mode: Optional[str] = None,
    seed_artist_names: Optional[Sequence[str]] = None,
    artist_identity_cfg: Optional["ArtistIdentityConfig"] = None,
):
    """
    Print detailed track report showing how each track was selected.

    Args:
        tracks: Final playlist tracks
        artist_name: Name of seed artist (if applicable)
        dynamic: Whether dynamic mode was used
        verbose_edges: Whether to print per-edge scores when available (DS)
        last_ds_report: Last DS report dictionary
        last_scope: Last scope value
        last_cohesion_mode: Last cohesion mode value
        seed_artist_names: Every seed chip for a multi-artist blend. Falls back
            to ``artist_name`` alone when absent.
        artist_identity_cfg: The identity config the generator resolved with.
            Defaults to a DISABLED config, which reproduces plain normalized
            matching -- pass the real one or the share is counted differently
            than the piers were seated.
    """
    logger.info("=" * 80)
    # Pipeline context summary if available
    pipeline_ctx = []
    pipeline_ctx.append("pipeline=ds")
    if last_scope is not None:
        pipeline_ctx.append(f"scope={last_scope}")
    if last_cohesion_mode is not None:
        pipeline_ctx.append(f"mode={last_cohesion_mode}")
    if artist_name:
        pipeline_ctx.append(f'seed="{sanitize_for_logging(artist_name)}"')
    ctx_str = " | ".join(pipeline_ctx)
    logger.info(f"PLAYLIST TRACKLIST  ({ctx_str})")
    logger.info("=" * 80)

    # Group tracks by source for statistics
    source_counts = {'sonic': 0, 'genre': 0, 'unknown': 0}
    edge_scores_for_logging: List[Dict[str, Any]] = []
    edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if last_ds_report:
        report = last_ds_report or {}
        playlist_stats = report.get("playlist_stats") or {}
        playlist_stats_playlist = playlist_stats.get("playlist") or {}
        edge_scores_for_logging = (
            playlist_stats_playlist.get("edge_scores")
            or playlist_stats.get("edge_scores")
            or report.get("edge_scores")
            or []
        )
        last_ds_report["edge_scores"] = edge_scores_for_logging
        edge_map = {(str(e.get("prev_id")), str(e.get("cur_id"))): e for e in edge_scores_for_logging}
        if verbose_edges:
            missing_hydration = [t for t in tracks if not t.get("artist") or not t.get("title")]
            if missing_hydration:
                logger.info(
                    "Hydration gaps: %d tracks missing artist/title; samples=%s",
                    len(missing_hydration),
                    [t.get("rating_key") for t in missing_hydration[:5]],
                )
            if edge_scores_for_logging and len(edge_scores_for_logging) != max(0, len(tracks) - 1):
                logger.error(
                    "Edge score count mismatch (edges=%d, expected=%d); skipping per-edge printing.",
                    len(edge_scores_for_logging),
                    max(0, len(tracks) - 1),
                )
                edge_map = {}

    for i, track in enumerate(tracks, 1):
        raw_artist = str(track.get('artist', 'Unknown') or 'Unknown')
        raw_title = str(track.get('title', 'Unknown') or 'Unknown')
        artist = sanitize_for_logging(raw_artist)
        title = sanitize_for_logging(raw_title)
        source = track.get('source', 'unknown')
        track_id = track.get("rating_key") or track.get("id") or track.get("track_id")
        file_path = track.get("file") or track.get("path") or track.get("file_path")

        sanitized = (artist != raw_artist) or (title != raw_title)
        if sanitized and logger.isEnabledFor(logging.DEBUG):
            # Keep raw values ASCII-safe for console logs.
            raw_artist_dbg = raw_artist.encode("unicode_escape", errors="backslashreplace").decode("ascii")
            raw_title_dbg = raw_title.encode("unicode_escape", errors="backslashreplace").decode("ascii")
            logger.debug(
                "Track metadata sanitized: idx=%d id=%s file=%s raw_artist=%s raw_title=%s safe_artist=%s safe_title=%s",
                i,
                track_id,
                file_path,
                raw_artist_dbg,
                raw_title_dbg,
                artist,
                title,
            )

        # Count by source
        source_counts[source] = source_counts.get(source, 0) + 1

        # Format: Track 01: Artist - Title (dur=3:45, T=0.123)
        marker = f" [id={track_id}]" if sanitized and track_id is not None else ""
        duration_ms = track.get("duration")
        if not duration_ms:
            duration_ms = track.get("duration_ms")
        duration_str = format_track_duration(duration_ms)
        t_score = "n/a"
        if last_ds_report and i > 1:
            prev_id = str(tracks[i - 2].get("rating_key"))
            cur_id = str(tracks[i - 1].get("rating_key"))
            edge = edge_map.get((prev_id, cur_id))
            if edge:
                t_val = edge.get("T", float("nan"))
                if isinstance(t_val, (int, float)) and t_val == t_val:
                    t_score = f"{t_val:.3f}"
        logger.info(
            f"Track {i:02d}: {artist} - {title}{marker} (dur={duration_str}, T={t_score})"
        )
        # Verbose per-edge scores (DS only)
        if verbose_edges and last_ds_report and i > 1:
            prev_id = str(tracks[i - 2].get("rating_key"))
            cur_id = str(tracks[i - 1].get("rating_key"))
            edge = edge_map.get((prev_id, cur_id))
            if edge:
                import math

                t_val = edge.get("T", float("nan"))
                t_raw = edge.get("T_raw", float("nan"))
                t_center_cos = edge.get("T_centered_cos", float("nan"))
                h_val = edge.get("H", float("nan"))
                s_val = edge.get("S", float("nan"))
                g_val = edge.get("G", float("nan"))
                raw_part = ""
                if isinstance(t_raw, (int, float)) and not math.isnan(t_raw):
                    raw_part = f" (raw={t_raw:.3f})"
                center_part = ""
                if isinstance(t_center_cos, (int, float)) and not math.isnan(t_center_cos):
                    center_part = f" center_cos={t_center_cos:.3f}"
                h_part = f"  H={h_val:.3f}" if isinstance(h_val, (int, float)) and not math.isnan(h_val) else "  H=n/a"
                logger.info(
                    "    edge %02d->%02d: T=%.3f%s%s%s  S=%.3f  G=%.3f",
                    i - 1,
                    i,
                    t_val if t_val == t_val else float("nan"),
                    raw_part,
                    center_part,
                    h_part,
                    s_val if s_val == s_val else float("nan"),
                    g_val if g_val == g_val else float("nan"),
                )
            else:
                logger.error("Missing edge score for %s->%s", prev_id, cur_id)

    # Print summary statistics
    logger.info("=" * 80)
    logger.info("PLAYLIST STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total tracks: {len(tracks)}")
    logger.info(f"Unique artists: {len(set(t.get('artist') for t in tracks))}")

    if last_ds_report:
        import math

        report = last_ds_report or {}
        metrics = report.get("metrics") or {}
        playlist_stats = report.get("playlist_stats") or {}
        playlist_stats_playlist = playlist_stats.get("playlist") or {}
        edge_scores = (
            playlist_stats_playlist.get("edge_scores")
            or playlist_stats.get("edge_scores")
            or report.get("edge_scores")
            or []
        )
        report["edge_scores"] = edge_scores
        baseline = report.get("baseline") or {}
        transition_floor_val = (
            metrics.get("transition_floor")
            or playlist_stats_playlist.get("transition_floor")
            or playlist_stats.get("transition_floor")
            or report.get("transition_floor")
        )
        gamma_val = report.get("transition_gamma") or playlist_stats_playlist.get("transition_gamma")
        transition_centered = bool(
            playlist_stats_playlist.get("transition_centered")
            or playlist_stats.get("transition_centered")
            or report.get("transition_centered")
        )
        tracks_by_id = {str(t.get("rating_key")): t for t in tracks}
        if verbose_edges:
            def _nan_count(vals):
                return sum(1 for v in vals if isinstance(v, float) and math.isnan(v))

            logger.info(
                "Edge score arrays: tracks=%d edges=%d missing_T=%d missing_S=%d missing_G=%d",
                len(tracks),
                len(edge_scores),
                _nan_count([e.get("T", float("nan")) for e in edge_scores]),
                _nan_count([e.get("S", float("nan")) for e in edge_scores]),
                _nan_count([e.get("G", float("nan")) for e in edge_scores]),
            )
            if len(edge_scores) != max(0, len(tracks) - 1):
                logger.error(
                    "Edge score count mismatch (edges=%d, expected=%d); skipping edge logging for missing scores.",
                    len(edge_scores),
                    max(0, len(tracks) - 1),
                )
            if edge_scores and all((isinstance(e.get("T", float("nan")), float) and math.isnan(e.get("T"))) for e in edge_scores):
                logger.error("Edge scores contain only NaN values; skipping edge logging.")
                edge_scores = []
            # Index sanity for first 3 and bottom 3
            if edge_scores:
                samples = edge_scores[:3] + sorted(edge_scores, key=lambda e: e.get("T", float("nan")))[:3]
                for e in samples:
                    logger.info(
                        "Edge idx map: prev_idx=%s cur_idx=%s prev=%s - %s -> %s - %s T=%.3f",
                        e.get("prev_idx"),
                        e.get("cur_idx"),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("artist", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("title", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("artist", "")),
                        sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("title", "")),
                        e.get("T", float("nan")),
                    )
            if baseline:
                def _fmt_pct(stats: Optional[dict]) -> str:
                    if not stats:
                        return "n/a"
                    return f"p50={stats.get('p50')} p90={stats.get('p90')} p99={stats.get('p99')}"

                logger.info("Baseline (artifact) percentiles: T=%s | S=%s | G=%s",
                            _fmt_pct(baseline.get("T_centered_rescaled") if transition_centered else baseline.get("T")),
                            _fmt_pct(baseline.get("S")),
                            _fmt_pct(baseline.get("G")))
                if transition_centered:
                    logger.info(
                        "Baseline centered: T_centered_cos=%s | T_centered_rescaled=%s",
                        _fmt_pct(baseline.get("T_centered_cos")),
                        _fmt_pct(baseline.get("T_centered_rescaled")),
                    )
            if edge_scores:
                idxs = [e.get("prev_idx") for e in edge_scores] + [e.get("cur_idx") for e in edge_scores]
                idxs = [i for i in idxs if i is not None]
                if idxs:
                    idx_counts = {}
                    for i in idxs:
                        idx_counts[i] = idx_counts.get(i, 0) + 1
                    unique_indices_used = len(idx_counts)
                    max_repeat = max(idx_counts.values())
                    repeated = sum(1 for v in idx_counts.values() if v > 1)
                    logger.info(
                        "Chain index-degree summary: unique_nodes=%d nodes_with_degree>1=%d max_degree=%d (expected for contiguous chain)",
                        unique_indices_used,
                        repeated,
                        max_repeat,
                    )
        if not edge_scores:
            logger.info("Edge score summaries: none (edges=0)")
            return
        logger.info("Edge score summaries:")

        def _summ(arr):
            import math

            if not arr:
                return None
            arr_np = np.array([v for v in arr if isinstance(v, (int, float)) and not math.isnan(v)], dtype=float)
            if arr_np.size == 0:
                return None
            return {
                "mean": float(np.nanmean(arr_np)),
                "p10": float(np.nanpercentile(arr_np, 10)),
                "p50": float(np.nanpercentile(arr_np, 50)),
                "p90": float(np.nanpercentile(arr_np, 90)),
                "p99": float(np.nanpercentile(arr_np, 99)),
                "min": float(np.nanmin(arr_np)),
            }

        def _fmt(val: Optional[float]) -> str:
            if val is None:
                return "n/a"
            try:
                if math.isnan(val):
                    return "n/a"
            except Exception:
                return "n/a"
            return f"{val:.3f}"

        t_list = [e.get("T", float("nan")) for e in edge_scores]
        t_center_cos_list = [e.get("T_centered_cos", float("nan")) for e in edge_scores]
        s_list = [e.get("S", float("nan")) for e in edge_scores]
        g_list = [e.get("G", float("nan")) for e in edge_scores]
        t_raw_list = [e.get("T_raw", float("nan")) for e in edge_scores]
        h_list = [e.get("H", float("nan")) for e in edge_scores]
        t_stats = _summ(t_list)
        s_stats = _summ(s_list)
        g_stats = _summ(g_list)
        t_raw_stats = _summ(t_raw_list)
        h_stats = _summ(h_list)
        t_center_stats = _summ(t_center_cos_list)
        if t_stats:
            t_clean = [v for v in t_list if isinstance(v, (int, float)) and not np.isnan(v)]
            below_floor_count = None
            if transition_floor_val is not None and t_clean:
                below_floor_count = sum(1 for v in t_clean if v < transition_floor_val)
            logger.info(
                "  T transition: mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s  below_floor=%s (floor=%s gamma=%s centered=%s)",
                _fmt(t_stats.get("mean")),
                _fmt(t_stats.get("p10")),
                _fmt(t_stats.get("p50")),
                _fmt(t_stats.get("p90")),
                _fmt(t_stats.get("p99")),
                _fmt(t_stats.get("min")),
                below_floor_count if below_floor_count is not None else (metrics.get("below_floor") if metrics else None),
                transition_floor_val,
                gamma_val,
                transition_centered,
            )
        if transition_centered and t_center_stats:
            logger.info(
                "  T_centered_cos: mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(t_center_stats.get("mean")),
                _fmt(t_center_stats.get("p10")),
                _fmt(t_center_stats.get("p50")),
                _fmt(t_center_stats.get("p90")),
                _fmt(t_center_stats.get("p99")),
                _fmt(t_center_stats.get("min")),
            )
        if os.environ.get("PLAYLIST_DIAG_SONIC"):
            def _spread(stats: Optional[dict]) -> str:
                if not stats:
                    return "n/a"
                p90 = stats.get("p90")
                p10 = stats.get("p10")
                if p90 is None or p10 is None:
                    return "n/a"
                return _fmt(p90 - p10)
            logger.info(
                "Sonic diag: H(p50=%s p90=%s spread=%s) S(p50=%s p90=%s spread=%s)",
                _fmt(h_stats.get("p50") if h_stats else None),
                _fmt(h_stats.get("p90") if h_stats else None),
                _spread(h_stats),
                _fmt(s_stats.get("p50") if s_stats else None),
                _fmt(s_stats.get("p90") if s_stats else None),
                _spread(s_stats),
            )
        if verbose_edges and t_raw_stats:
            logger.info(
                "  T_raw (end->start): mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(t_raw_stats.get("mean")),
                _fmt(t_raw_stats.get("p10")),
                _fmt(t_raw_stats.get("p50")),
                _fmt(t_raw_stats.get("p90")),
                _fmt(t_raw_stats.get("p99")),
                _fmt(t_raw_stats.get("min")),
            )
        if verbose_edges and h_stats:
            logger.info(
                "  H hybrid:     mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(h_stats.get("mean")),
                _fmt(h_stats.get("p10")),
                _fmt(h_stats.get("p50")),
                _fmt(h_stats.get("p90")),
                _fmt(h_stats.get("p99")),
                _fmt(h_stats.get("min")),
            )
        if s_stats:
            logger.info(
                "  S sonic:      mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(s_stats.get("mean")),
                _fmt(s_stats.get("p10")),
                _fmt(s_stats.get("p50")),
                _fmt(s_stats.get("p90")),
                _fmt(s_stats.get("p99")),
                _fmt(s_stats.get("min")),
            )
        if g_stats:
            logger.info(
                "  G genre:      mean=%s  p10=%s  p50=%s  p90=%s  p99=%s  min=%s",
                _fmt(g_stats.get("mean")),
                _fmt(g_stats.get("p10")),
                _fmt(g_stats.get("p50")),
                _fmt(g_stats.get("p90")),
                _fmt(g_stats.get("p99")),
                _fmt(g_stats.get("min")),
            )
        # BPM per-playlist summary (from pier-bridge bpm_summary stat)
        _bpm_sum = (
            playlist_stats_playlist.get("bpm_summary")
            or playlist_stats.get("bpm_summary")
        )
        if _bpm_sum is not None:
            logger.info(
                "BPM (perceptual): min=%.0f mean=%.0f max=%.0f std=%.0f (%d/%d tracks have data)",
                float(_bpm_sum.get("min", 0)),
                float(_bpm_sum.get("mean", 0)),
                float(_bpm_sum.get("max", 0)),
                float(_bpm_sum.get("std", 0)),
                int(_bpm_sum.get("n", 0)),
                int(_bpm_sum.get("total", 0)),
            )
        if verbose_edges and baseline:
            logger.info("Baseline vs playlist percentiles:")
            def _log_percentiles(label: str, playlist_vals: Optional[dict], base_vals: Optional[dict]) -> None:
                if not playlist_vals and not base_vals:
                    return
                logger.info(
                    "  %s playlist p50=%s p90=%s p99=%s | baseline p50=%s p90=%s p99=%s",
                    label,
                    _fmt(playlist_vals.get("p50") if playlist_vals else None),
                    _fmt(playlist_vals.get("p90") if playlist_vals else None),
                    _fmt(playlist_vals.get("p99") if playlist_vals else None),
                    _fmt((base_vals or {}).get("p50") if base_vals else None),
                    _fmt((base_vals or {}).get("p90") if base_vals else None),
                    _fmt((base_vals or {}).get("p99") if base_vals else None),
                )
            baseline_t = baseline.get("T_centered_rescaled") if transition_centered else baseline.get("T")
            _log_percentiles("T", t_stats, baseline_t)
            _log_percentiles("T_raw", t_raw_stats, baseline.get("T_raw"))
            _log_percentiles("S", s_stats, baseline.get("S"))
            _log_percentiles("G", g_stats, baseline.get("G"))
        # Weakest transitions
        if edge_scores:
            weakest = sorted(
                [e for e in edge_scores if isinstance(e.get("T"), (int, float)) and not np.isnan(e.get("T"))],
                key=lambda e: e.get("T", float("inf")),
            )[:3]
            logger.info("Weakest transitions (bottom 3 by T):")
            for idx, e in enumerate(weakest, 1):
                logger.info(
                    "  #%s  T=%.3f  S=%.3f  G=%.3f  %s - %s (idx=%s) -> %s - %s (idx=%s)",
                    str(idx).zfill(2),
                    e.get("T", float("nan")),
                    e.get("S", float("nan")),
                    e.get("G", float("nan")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("artist", "")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("prev_id")), {}).get("title", "")),
                    e.get("prev_idx"),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("artist", "")),
                    sanitize_for_logging(tracks_by_id.get(str(e.get("cur_id")), {}).get("title", "")),
                    e.get("cur_idx"),
                )
    else:
        if dynamic:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")
            logger.info(f"  - Genre-based: {source_counts.get('genre', 0)} tracks")
        else:
            logger.info(f"  - Sonic similarity: {source_counts.get('sonic', 0)} tracks")

    _report_seed_artist_share(
        tracks, artist_name, seed_artist_names, artist_identity_cfg
    )

    # Duration info
    total_duration_ms = sum((track.get('duration') or 0) for track in tracks)
    total_minutes = total_duration_ms / 1000 / 60
    logger.info(f"Total duration: {total_minutes:.1f} minutes")
    logger.info("=" * 80)
