"""Seed ordering + pool prep helpers (Tier-3.1 PR-6).

Four helpers extracted from pier_bridge_builder.py. All operate on candidate
indices + sonic/genre matrices; none touch beam state or PierBridgeConfig
directly (one takes scalar config values).

  * _segment_far_stats — per-segment distance + connector scarcity stats
  * _select_connector_candidates — top-K min-pier-similarity candidates
  * _order_seeds_by_bridgeability — order seeds to maximize pair bridgeability
    (exhaustive for n<=6, greedy for larger)
  * _dedupe_candidate_pool — (artist, title) dedupe with version preference
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.live_albums import LiveAlbumRegistry
from src.playlist.pier_bridge.audit_summary import _compute_bridgeability_score

logger = logging.getLogger(__name__)


def _segment_far_stats(
    *,
    pier_a: int,
    pier_b: int,
    X_full_norm: np.ndarray,
    X_genre_norm: Optional[np.ndarray],
    universe: list[int],
    used_track_ids: Set[int],
    bridge_floor: float,
) -> dict[str, Optional[float]]:
    sim_sonic = float(np.dot(X_full_norm[pier_a], X_full_norm[pier_b]))
    sim_genre = None
    if X_genre_norm is not None:
        sim_genre = float(np.dot(X_genre_norm[pier_a], X_genre_norm[pier_b]))
    available = [int(i) for i in universe if int(i) not in used_track_ids]
    scarcity = None
    if available:
        vec_a = X_full_norm[pier_a]
        vec_b = X_full_norm[pier_b]
        sims_a = np.dot(X_full_norm[available], vec_a)
        sims_b = np.dot(X_full_norm[available], vec_b)
        gate = np.minimum(sims_a, sims_b) >= float(bridge_floor)
        scarcity = float(np.mean(gate)) if gate.size > 0 else None
    return {
        "sonic_sim": sim_sonic,
        "genre_sim": sim_genre,
        "connector_scarcity": scarcity,
    }


def _select_connector_candidates(
    available: List[int],
    X_full_norm: np.ndarray,
    pier_a: int,
    pier_b: int,
    cap: int,
) -> List[int]:
    if cap <= 0 or not available:
        return []
    vec_a = X_full_norm[pier_a]
    vec_b = X_full_norm[pier_b]
    sims_a = np.dot(X_full_norm[available], vec_a)
    sims_b = np.dot(X_full_norm[available], vec_b)
    scores = np.minimum(sims_a, sims_b)
    order = np.argsort(-scores)
    return [int(available[int(i)]) for i in order[:cap]]


def _order_seeds_by_bridgeability(
    seed_indices: List[int],
    X_full_norm: np.ndarray,
    X_start_norm: Optional[np.ndarray],
    X_end_norm: Optional[np.ndarray],
    X_genre_norm: Optional[np.ndarray] = None,
    *,
    weight_sonic: float = 0.0,
    weight_genre: float = 0.0,
    weight_bridge: float = 1.0,
    min_bottleneck: bool = False,
) -> List[int]:
    """
    Order seed indices to maximize bridgeability.
    For <=6 seeds, evaluates all permutations.
    For >6 seeds, uses greedy nearest-neighbor heuristic.

    objective: when ``min_bottleneck`` is False (default, legacy) a permutation is
    scored by the SUM of consecutive pair scores. When True (roam corridors) it is
    scored by the MIN consecutive pair score (the weakest link) — the smoothest
    sequence, so one bad seam can't be averaged away. The greedy path (n>6) is
    nearest-neighbour either way (per-step max approximates both objectives).
    """
    n = len(seed_indices)
    if n <= 1:
        return seed_indices

    weight_sonic = float(weight_sonic) if math.isfinite(float(weight_sonic)) else 0.0
    weight_genre = float(weight_genre) if math.isfinite(float(weight_genre)) else 0.0
    weight_bridge = float(weight_bridge) if math.isfinite(float(weight_bridge)) else 0.0
    weight_sonic = max(0.0, weight_sonic)
    weight_genre = max(0.0, weight_genre)
    weight_bridge = max(0.0, weight_bridge)
    total_weight = weight_sonic + weight_genre + weight_bridge
    if total_weight <= 1e-9:
        weight_bridge = 1.0
        total_weight = 1.0
    weight_sonic /= total_weight
    weight_genre /= total_weight
    weight_bridge /= total_weight

    def _pair_score(a: int, b: int) -> float:
        score = 0.0
        if weight_bridge > 0:
            score += weight_bridge * _compute_bridgeability_score(
                a, b, X_full_norm, X_start_norm, X_end_norm
            )
        if weight_sonic > 0:
            score += weight_sonic * float(np.dot(X_full_norm[a], X_full_norm[b]))
        if weight_genre > 0 and X_genre_norm is not None:
            score += weight_genre * float(np.dot(X_genre_norm[a], X_genre_norm[b]))
        return score

    if n <= 6:
        # Exhaustive search for small seed counts
        best_order = None
        best_score = -float('inf')

        for perm in itertools.permutations(seed_indices):
            pair_scores = [_pair_score(perm[i], perm[i + 1]) for i in range(len(perm) - 1)]
            total_score = min(pair_scores) if min_bottleneck else sum(pair_scores)
            if total_score > best_score:
                best_score = total_score
                best_order = list(perm)

        logger.info("Seed ordering: evaluated %d permutations, best_score=%.4f",
                   math.factorial(n), best_score)
        return best_order or seed_indices
    else:
        # Greedy nearest-neighbor for larger seed counts.
        # min_bottleneck has no separate path here: per-step max-pair locally
        # approximates both the sum and the min-bottleneck objective.
        remaining = set(seed_indices)
        # Start with the first seed
        ordered = [seed_indices[0]]
        remaining.remove(seed_indices[0])

        while remaining:
            current = ordered[-1]
            best_next = None
            best_score = -float('inf')

            for candidate in remaining:
                score = _pair_score(current, candidate)
                if score > best_score:
                    best_score = score
                    best_next = candidate

            if best_next is not None:
                ordered.append(best_next)
                remaining.remove(best_next)

        logger.info("Seed ordering: greedy heuristic for %d seeds", n)
        return ordered


def _dedupe_candidate_pool(
    pool_indices: List[int],
    bundle: ArtifactBundle,
    albums_by_index: Optional[Dict[int, str]] = None,
    live_registry: Optional[LiveAlbumRegistry] = None,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Deduplicate candidate pool by normalized artist+title.
    Returns deduplicated indices and mapping of norm_key -> chosen index.

    Prefers canonical versions based on version preference scoring. This pool is
    multi-artist, so `live_registry` (rather than one flat key set) is narrowed
    PER ROW via that row's own artist_key -- a mark on one seed artist's album
    must never penalize another artist's same-named album.
    """
    from src.title_dedupe import calculate_version_preference_score

    seen: Dict[str, Tuple[int, int]] = {}  # norm_key -> (index, preference_score)

    for idx in pool_indices:
        keys = identity_keys_for_index(bundle, int(idx))
        key = f"{keys.artist_key}|||{keys.title_key}"

        # Compute preference score (higher = more canonical)
        title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""
        album = (albums_by_index or {}).get(int(idx), "")
        live_keys = live_registry.album_keys_for(keys.artist_key) if live_registry else None
        pref_score = calculate_version_preference_score(
            title, album, live_album_keys=live_keys
        )

        if key not in seen or pref_score > seen[key][1]:
            seen[key] = (idx, pref_score)

    deduped = [idx for idx, _ in seen.values()]
    norm_to_idx = {key: idx for key, (idx, _) in seen.items()}

    logger.debug("Deduped candidate pool: %d -> %d tracks", len(pool_indices), len(deduped))
    return deduped, norm_to_idx
