"""Micro-pier + relaxation subsystem (Tier-3.1 PR-5).

Self-contained cluster extracted from pier_bridge_builder.py. The entry point
is _attempt_micro_pier_split, called from build_pier_bridge_playlist when a
segment fails. It picks an intermediate "micro-pier" track and runs two
half-beam-searches around it.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.artist_identity_resolver import ArtistIdentityConfig
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge.genre_targets import _build_genre_targets
from src.playlist.transition_metrics import TransitionMetricContext


def _build_dj_relaxation_attempts(cfg: PierBridgeConfig) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    attempts.append({
        "label": "baseline",
        "cfg": cfg,
        "changes": [],
        "force_allow_detours": False,
    })

    relaxed_weight = float(cfg.dj_waypoint_weight) * 0.5
    attempts.append({
        "label": "relax_waypoint",
        "cfg": replace(
            cfg,
            dj_waypoint_weight=float(relaxed_weight),
            dj_waypoint_floor=0.0,
            dj_waypoint_penalty=0.0,
        ),
        "changes": [
            f"waypoint_weight*0.5->{relaxed_weight:.3f}",
            "waypoint_floor->0",
            "waypoint_penalty->0",
        ],
        "force_allow_detours": False,
    })

    pool_scale = 1.25
    effort_cfg = replace(
        cfg,
        segment_pool_max=min(
            int(cfg.max_segment_pool_max),
            int(max(1, round(float(cfg.segment_pool_max) * pool_scale))),
        ),
        dj_pooling_k_local=int(max(1, round(float(cfg.dj_pooling_k_local) * pool_scale))),
        dj_pooling_k_toward=int(max(1, round(float(cfg.dj_pooling_k_toward) * pool_scale))),
        dj_pooling_k_genre=int(max(1, round(float(cfg.dj_pooling_k_genre) * pool_scale))),
        dj_pooling_k_union_max=int(
            max(1, round(float(cfg.dj_pooling_k_union_max) * pool_scale))
        ),
        initial_beam_width=min(
            int(cfg.max_beam_width),
            int(max(1, round(float(cfg.initial_beam_width) * 1.5))),
        ),
    )
    attempts.append({
        "label": "relax_effort",
        "cfg": effort_cfg,
        "changes": [
            "segment_pool_max*1.25",
            "dj_pooling_k_* *1.25",
            "initial_beam_width*1.5",
        ],
        "force_allow_detours": False,
    })

    connector_cfg = replace(
        cfg,
        dj_connector_bias_enabled=True,
        dj_connector_max_per_segment_linear=int(cfg.dj_connector_max_per_segment_linear) + 1,
        dj_connector_max_per_segment_adventurous=int(cfg.dj_connector_max_per_segment_adventurous) + 1,
    )
    attempts.append({
        "label": "relax_connectors",
        "cfg": connector_cfg,
        "changes": [
            "connector_bias_enabled->true",
            "connector_max_per_segment+1",
            "force_allow_detours",
        ],
        "force_allow_detours": True,
    })

    if bool(cfg.dj_relaxation_allow_floor_relaxation):
        relaxed_floor = max(0.0, float(cfg.transition_floor) - 0.02)
        attempts.append({
            "label": "relax_transition_floor",
            "cfg": replace(cfg, transition_floor=float(relaxed_floor)),
            "changes": [f"transition_floor-0.02->{relaxed_floor:.3f}"],
            "force_allow_detours": False,
        })

    max_attempts = max(1, int(cfg.dj_relaxation_max_attempts))
    return attempts[:max_attempts]


def _score_micro_pier_candidates(
    candidates: list[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
) -> list[tuple[int, float]]:
    if not candidates:
        return []
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]
    cand_list = [int(i) for i in candidates]
    sims_a = np.dot(X_full_norm[cand_list], vec_a)
    sims_b = np.dot(X_full_norm[cand_list], vec_b)
    scores = np.minimum(sims_a, sims_b)
    return [(int(cand_list[i]), float(scores[i])) for i in range(len(cand_list))]


def _select_micro_pier_candidates(
    candidates: list[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
    top_k: int,
) -> list[tuple[int, float]]:
    scored = _score_micro_pier_candidates(candidates, X_full_norm, pier_a, pier_b)
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, int(top_k))]


def _micro_pier_candidate_pool(
    source: str,
    last_segment_candidates: list[int],
    pool_cache: Optional[Dict[str, Any]],
) -> list[int]:
    source = str(source or "union_pool").strip().lower()
    connectors: list[int] = []
    if pool_cache is not None:
        cached = pool_cache.get("dj_connectors")
        if cached:
            connectors = [int(i) for i in cached]
    if source == "connectors":
        return connectors
    if source == "both":
        combined = list(dict.fromkeys(connectors + list(last_segment_candidates)))
        return combined
    return list(last_segment_candidates)


def _should_attempt_micro_pier(
    *,
    relaxation_enabled: bool,
    segment_path: Optional[list[int]],
) -> bool:
    return bool(relaxation_enabled) and segment_path is None


def _attempt_micro_pier_split(
    *,
    pier_a: int,
    pier_b: int,
    interior_length: int,
    candidates: list[int],
    X_full: np.ndarray,
    X_full_norm: np.ndarray,
    X_start: Optional[np.ndarray],
    X_mid: Optional[np.ndarray],
    X_end: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray],
    cfg: PierBridgeConfig,
    beam_width: int,
    artist_key_by_idx: Optional[Dict[int, str]],
    seed_artist_keys: frozenset,
    recent_global_artists: Optional[List[str]],
    durations_ms: Optional[np.ndarray],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    bundle: Optional[ArtifactBundle],
    warnings: list[dict[str, Any]],
    X_genre_vocab: Optional[np.ndarray],
    genre_graph: Optional[dict[str, list[tuple[str, float]]]],
    micro_diag: Optional[dict[str, Any]] = None,
    X_genre_norm_idf: Optional[np.ndarray] = None,
    X_genre_raw: Optional[np.ndarray] = None,
    X_genre_smoothed: Optional[np.ndarray] = None,
    genre_idf: Optional[np.ndarray] = None,
    transition_metric_context: Optional[TransitionMetricContext] = None,
    popularity_values: Optional[np.ndarray] = None,
) -> Optional[list[int]]:
    if interior_length < 2 or not candidates:
        return None
    max_micro = max(1, int(cfg.dj_micro_piers_max))
    topk = max(1, int(cfg.dj_micro_piers_topk))

    cand_list = [int(i) for i in candidates]
    metric = str(cfg.dj_micro_piers_selection_metric or "max_min_sim").strip().lower()
    if metric != "max_min_sim":
        metric = "max_min_sim"
    scored = _select_micro_pier_candidates(
        candidates,
        X_full_norm,
        pier_a,
        pier_b,
        top_k=topk,
    )
    micro_candidates = [idx for idx, _ in scored][:topk]

    left_len = interior_length // 2
    right_len = interior_length - left_len - 1
    if right_len < 0:
        return None

    for micro_idx in micro_candidates[:max_micro]:
        if micro_diag is not None:
            micro_diag.update({
                "micro_pier_index": int(micro_idx),
                "micro_pier_metric": "max_min_sim",
                "micro_pier_metric_value": float(
                    next((score for idx, score in scored if idx == micro_idx), 0.0)
                ),
                "left_success": False,
                "right_success": False,
            })
        left_g_targets = None
        right_g_targets = None
        if X_genre_norm is not None and X_genre_vocab is not None and bool(cfg.dj_bridging_enabled):
            left_g_targets = _build_genre_targets(
                pier_a=pier_a,
                pier_b=micro_idx,
                interior_length=left_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=X_genre_vocab,
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                X_genre_raw=None,
                X_genre_smoothed=None,
                genre_idf=None,
            )
            right_g_targets = _build_genre_targets(
                pier_a=micro_idx,
                pier_b=pier_b,
                interior_length=right_len,
                X_full_norm=X_full_norm,
                X_genre_norm=X_genre_norm,
                genre_vocab=X_genre_vocab,
                genre_graph=genre_graph,
                cfg=cfg,
                warnings=warnings,
                X_genre_raw=None,
                X_genre_smoothed=None,
                genre_idf=None,
            )

        keys_map = dict(artist_key_by_idx or {})
        if bundle is not None:
            try:
                keys_map[int(pier_a)] = identity_keys_for_index(bundle, int(pier_a)).artist_key
                keys_map[int(pier_b)] = identity_keys_for_index(bundle, int(pier_b)).artist_key
                keys_map[int(micro_idx)] = identity_keys_for_index(bundle, int(micro_idx)).artist_key
            except Exception:
                pass

        left_path, _, _, _ = _beam_search_segment(
            pier_a,
            micro_idx,
            left_len,
            cand_list,
            X_full,
            X_full_norm,
            X_start,
            X_mid,
            X_end,
            X_genre_norm,
            cfg,
            beam_width,
            X_genre_norm_idf=X_genre_norm_idf,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
            genre_vocab=X_genre_vocab,
            artist_key_by_idx=(keys_map if keys_map else None),
            seed_artist_keys=seed_artist_keys,
            recent_global_artists=recent_global_artists,
            durations_ms=durations_ms,
            artist_identity_cfg=artist_identity_cfg,
            bundle=bundle,
            g_targets_override=left_g_targets,
            transition_metric_context=transition_metric_context,
            popularity_values=popularity_values,
        )
        if left_path is None:
            continue
        if micro_diag is not None:
            micro_diag["left_success"] = True

        used_left = set(int(i) for i in left_path)
        right_candidates = [int(i) for i in cand_list if int(i) not in used_left and int(i) != int(micro_idx)]

        right_path, _, _, _ = _beam_search_segment(
            micro_idx,
            pier_b,
            right_len,
            right_candidates,
            X_full,
            X_full_norm,
            X_start,
            X_mid,
            X_end,
            X_genre_norm,
            cfg,
            beam_width,
            X_genre_norm_idf=X_genre_norm_idf,
            X_genre_raw=X_genre_raw,
            X_genre_smoothed=X_genre_smoothed,
            genre_idf=genre_idf,
            genre_vocab=X_genre_vocab,
            artist_key_by_idx=(keys_map if keys_map else None),
            seed_artist_keys=seed_artist_keys,
            recent_global_artists=recent_global_artists,
            durations_ms=durations_ms,
            artist_identity_cfg=artist_identity_cfg,
            bundle=bundle,
            g_targets_override=right_g_targets,
            transition_metric_context=transition_metric_context,
            popularity_values=popularity_values,
        )
        if right_path is None:
            continue
        if micro_diag is not None:
            micro_diag["right_success"] = True

        warnings.append({
            "type": "micro_pier_used",
            "scope": "segment",
            "message": "Inserted a micro-pier connector to bridge a difficult segment.",
            "micro_pier_index": int(micro_idx),
        })
        return left_path + [int(micro_idx)] + right_path

    return None
