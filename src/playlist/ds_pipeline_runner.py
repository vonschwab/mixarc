from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.playlist import reporter
from src.playlist.pipeline import generate_playlist_ds as core_generate_playlist_ds
from src.playlist.pier_bridge_builder import PierBridgeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DsRunResult:
    track_ids: List[str]
    requested: Dict[str, Any]
    effective: Dict[str, Any]
    metrics: Dict[str, Any]
    playlist_stats: Dict[str, Any]


def _finite_t_values(edge_scores: List[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for edge in edge_scores:
        if not isinstance(edge, dict):
            continue
        value = edge.get("T")
        if isinstance(value, (int, float)) and value == value:
            values.append(float(value))
    return values


def _refresh_playlist_metrics_from_final_edges(
    *,
    track_ids: List[str],
    playlist_stats: Dict[str, Any],
    artifact_path: str,
    random_seed: int,
) -> None:
    """Make DS summary metrics match final edge-reporting representation."""
    if len(track_ids) < 2:
        return
    transition_floor = playlist_stats.get("transition_floor")
    transition_weights = playlist_stats.get("transition_weights")
    if isinstance(transition_weights, list) and len(transition_weights) == 3:
        transition_weights = tuple(float(v) for v in transition_weights)
    elif isinstance(transition_weights, tuple) and len(transition_weights) == 3:
        transition_weights = tuple(float(v) for v in transition_weights)
    else:
        transition_weights = None
    edge_scores = reporter.compute_edge_scores_from_artifact(
        tracks=[{"rating_key": str(tid)} for tid in track_ids],
        artifact_path=artifact_path,
        transition_floor=transition_floor,
        transition_gamma=playlist_stats.get("transition_gamma"),
        embedding_random_seed=random_seed,
        center_transitions=bool(playlist_stats.get("transition_centered")),
        transition_weights=transition_weights,
    )
    if len(edge_scores) != len(track_ids) - 1:
        return

    t_values = _finite_t_values(edge_scores)
    if not t_values:
        return

    playlist_stats["edge_scores"] = edge_scores
    playlist_stats["min_transition"] = min(t_values)
    playlist_stats["mean_transition"] = sum(t_values) / len(t_values)
    if transition_floor is not None:
        floor = float(transition_floor)
        playlist_stats["below_floor_count"] = sum(1 for value in t_values if value < floor)
    playlist_stats["edge_metric_source"] = "final_emitted_playlist"


def generate_playlist_ds(
    *,
    artifact_path: str,
    seed_track_id: str,
    mode: str,
    length: int,
    random_seed: int,
    pace_mode: str = "dynamic",
    overrides: Optional[Dict[str, Any]] = None,
    enable_logging: bool = False,
    allowed_track_ids: Optional[List[str]] = None,
    excluded_track_ids: Optional[set[str]] = None,
    single_artist: bool = False,
    anchor_seed_ids: Optional[List[str]] = None,
    # Optional pier-bridge audit/backoff context
    dry_run: bool = False,
    pool_source: Optional[str] = None,
    artist_style_enabled: bool = False,
    artist_playlist: bool = False,
    audit_context_extra: Optional[Dict[str, Any]] = None,
    # Genre similarity parameters
    sonic_weight: Optional[float] = None,
    genre_weight: Optional[float] = None,
    min_genre_similarity: Optional[float] = None,
    genre_method: Optional[str] = None,
    genre_admission_percentile: Optional[float] = None,
    genre_mode: Optional[str] = None,
    sonic_mode: Optional[str] = None,
    pier_bridge_config: Optional["PierBridgeConfig"] = None,
    internal_connector_ids: Optional[list[str]] = None,
    internal_connector_max_per_segment: int = 0,
    internal_connector_priority: bool = True,
    tag_anchor_track_ids: Optional[set[str]] = None,
    seed_artist_keys_override: Optional[Sequence[str]] = None,
) -> DsRunResult:
    """Production-facing wrapper around the DS pipeline.

    Always uses pier+bridge strategy:
    - Multiple seeds: seeds as fixed piers, beam-search bridges between them
    - Single seed: seed acts as both start and end pier (arc structure)
    - No repair pass (pier-bridge ordering is final)
    """
    # Env-gated golden capture for the lossless-speedup bit-diff harness.
    # Off unless PLAYLIST_GOLDEN_CAPTURE is set; captures the exact inputs at
    # this deterministic seam so a replay reproduces an identical playlist.
    _golden_dir = os.environ.get("PLAYLIST_GOLDEN_CAPTURE")
    _golden_kwargs = None
    if _golden_dir:
        _golden_kwargs = dict(
            artifact_path=artifact_path, seed_track_id=seed_track_id, mode=mode,
            length=length, random_seed=random_seed, pace_mode=pace_mode,
            overrides=overrides, allowed_track_ids=allowed_track_ids,
            excluded_track_ids=excluded_track_ids, single_artist=single_artist,
            anchor_seed_ids=anchor_seed_ids, pool_source=pool_source, dry_run=dry_run,
            artist_style_enabled=artist_style_enabled, artist_playlist=artist_playlist,
            sonic_weight=sonic_weight, genre_weight=genre_weight,
            min_genre_similarity=min_genre_similarity, genre_method=genre_method,
            genre_admission_percentile=genre_admission_percentile,
            genre_mode=genre_mode,
            sonic_mode=sonic_mode,
            pier_bridge_config=pier_bridge_config,
            internal_connector_ids=internal_connector_ids,
            internal_connector_max_per_segment=internal_connector_max_per_segment,
            internal_connector_priority=internal_connector_priority,
            tag_anchor_track_ids=tag_anchor_track_ids,
            seed_artist_keys_override=seed_artist_keys_override,
        )

    logger.info(
        "DS_PIPELINE_RUNNER: anchor_seed_ids=%d (pier-bridge always enabled)",
        len(anchor_seed_ids or []),
    )

    result = core_generate_playlist_ds(
        artifact_path=artifact_path,
        seed_track_id=seed_track_id,
        anchor_seed_ids=anchor_seed_ids,
        num_tracks=length,
        mode=mode,
        pace_mode=pace_mode,
        random_seed=random_seed,
        overrides=overrides,
        allowed_track_ids=allowed_track_ids,
        excluded_track_ids=excluded_track_ids,
        single_artist=single_artist,
        pier_bridge_config=pier_bridge_config,
        dry_run=dry_run,
        pool_source=pool_source,
        artist_style_enabled=artist_style_enabled,
        artist_playlist=artist_playlist,
        audit_context_extra=audit_context_extra,
        # Pass through genre similarity parameters
        sonic_weight=sonic_weight,
        genre_weight=genre_weight,
        min_genre_similarity=min_genre_similarity,
        genre_method=genre_method,
        genre_admission_percentile=genre_admission_percentile,
        genre_mode=genre_mode,
        sonic_mode=sonic_mode,
        allowed_track_ids_set=set(str(t) for t in allowed_track_ids) if allowed_track_ids else None,
        internal_connector_ids=internal_connector_ids,
        internal_connector_max_per_segment=internal_connector_max_per_segment,
        internal_connector_priority=internal_connector_priority,
        tag_anchor_track_ids=tag_anchor_track_ids,
        seed_artist_keys_override=seed_artist_keys_override,
    )

    playlist_stats = result.stats.get("playlist", {})
    _refresh_playlist_metrics_from_final_edges(
        track_ids=list(result.track_ids),
        playlist_stats=playlist_stats,
        artifact_path=artifact_path,
        random_seed=random_seed,
    )
    metrics: Dict[str, Any] = {
        "below_floor": playlist_stats.get("below_floor_count"),
        "min_transition": playlist_stats.get("min_transition"),
        "mean_transition": playlist_stats.get("mean_transition"),
        "gap_sum": playlist_stats.get("gap_sum"),
        "gap_p90": playlist_stats.get("gap_p90"),
        "seed_sim_min": playlist_stats.get("seed_sim_min"),
        "seed_sim_mean": playlist_stats.get("seed_sim_mean"),
        "artist_counts": playlist_stats.get("artist_counts"),
        "distinct_artists": playlist_stats.get("distinct_artists"),
        # Pier bridge specific
        "strategy": playlist_stats.get("strategy"),
        "repair_applied": playlist_stats.get("repair_applied"),
        "num_segments": playlist_stats.get("num_segments"),
        "edge_metric_source": playlist_stats.get("edge_metric_source"),
    }

    if enable_logging:
        payload = {
            "pipeline": "ds",
            "mode": mode,
            "seed_track_id": seed_track_id,
            "length": length,
            "random_seed": random_seed,
            "metrics": metrics,
            "effective": result.params_effective,
        }
        logger.info(json.dumps(payload))

    if _golden_dir and _golden_kwargs is not None:
        from tests.support.lossless_golden import dump_golden_inputs

        os.makedirs(_golden_dir, exist_ok=True)
        _label = os.environ.get("PLAYLIST_GOLDEN_LABEL", "capture")
        dump_golden_inputs(
            _golden_kwargs, result.track_ids,
            metrics.get("min_transition"), metrics.get("mean_transition"),
            os.path.join(_golden_dir, f"{_label}.json"),
        )

    requested = dict(result.params_requested)
    requested.update({
        "seed_track_id": seed_track_id,
        "mode": mode,
        "pace_mode": pace_mode,
        "length": length,
        "random_seed": random_seed,
    })

    return DsRunResult(
        track_ids=result.track_ids,
        requested=requested,
        effective=result.params_effective,
        metrics=metrics,
        playlist_stats=result.stats,
    )
