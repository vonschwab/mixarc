from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.string_utils import normalize_artist_key
from src.playlist.history_analyzer import is_collaboration_of
from src.playlist.candidate_pool import _compute_genre_similarity
from src.playlist.genre_compatibility import compute_raw_genre_compatibility
from src.playlist.pier_bridge.vec import _calibrate_transition_cos

logger = logging.getLogger(__name__)


def _artist_indices_in_bundle(
    bundle,
    artist_name: str,
    *,
    include_collaborations: bool = False,
) -> List[int]:
    """
    Return bundle indices whose artist matches the named artist.

    Strict mode (default): match by normalized artist key.
    Inclusive mode: also accept tracks whose raw artist string is a
    collaboration containing the named artist (e.g. "Greg Foat & Art Themen",
    "Maston & Greg Foat"). Uses history_analyzer.is_collaboration_of so the
    collab patterns stay defined in one place.
    """
    if bundle.artist_keys is None:
        return []
    from src.playlist.artist_aliases import resolve_alias
    artist_key = resolve_alias(normalize_artist_key(artist_name))
    raw_artists = getattr(bundle, "track_artists", None)
    indices: List[int] = []
    for i, ak in enumerate(bundle.artist_keys):
        if resolve_alias(normalize_artist_key(str(ak))) == artist_key:
            indices.append(i)
            continue
        if include_collaborations and raw_artists is not None:
            try:
                raw = str(raw_artists[i] or "")
            except Exception:
                raw = ""
            if raw and is_collaboration_of(
                collaboration_name=raw, base_artist=artist_name
            ):
                indices.append(i)
    return indices


@dataclass(frozen=True)
class ArtistStyleConfig:
    enabled: bool = False
    cluster_k_min: int = 3
    cluster_k_max: int = 6
    cluster_k_heuristic_enabled: bool = True
    piers_per_cluster: int = 1
    per_cluster_candidate_pool_size: int = 400
    pool_balance_mode: str = "equal"  # equal | proportional_capped
    internal_connector_priority: bool = True
    internal_connector_max_per_segment: int = 2
    medoid_top_k: int = 5
    bridge_floor_strict: float = 0.10    # Phase 1 (ultra-cohesive)
    bridge_floor_narrow: float = 0.05    # Relaxed from 0.08 (Phase 3A)
    bridge_floor_dynamic: float = 0.02   # Relaxed from 0.03 (Phase 3A)
    bridge_weight: float = 0.7
    transition_weight: float = 0.3
    genre_tiebreak_weight: float = 0.05
    genre_neighbor_pool_enabled: bool = False
    genre_neighbor_pool_size: int = 500
    genre_neighbor_min_similarity: float = 0.25
    genre_neighbor_min_confidence: Optional[float] = 0.50
    genre_neighbor_compatible_threshold: float = 0.35
    genre_neighbor_conflict_threshold: float = 0.15
    # Medoid selection weighting (to avoid interludes/outliers)
    medoid_similarity_weight: float = 0.7  # Weight for sonic similarity to cluster centroid
    medoid_duration_weight: float = 0.3    # Weight for duration typicality (avoid outliers)
    # Energy-aware spread (set-level): pull each cluster's medoid toward an
    # evenly-spaced arousal slot so the pier set tiles the artist's energy range.
    # 0.0 => inert (today's behavior). See spec 2026-06-23-artist-energy-spread.
    medoid_energy_weight: float = 0.0
    energy_feature: str = "arousal_p50"    # which energy sidecar column defines the slots
    energy_slot_lo_pct: float = 10.0       # robust span low percentile of artist z-arousal
    energy_slot_hi_pct: float = 90.0       # robust span high percentile
    # Version de-dup: before clustering, reduce the artist to one canonical version
    # per song (studio/remaster preferred over live/demo/alt) so live takes and
    # duplicate releases don't become seeds. A live cut is kept only if it's the
    # song's sole version. Reuses src/title_dedupe.py.
    dedupe_versions: bool = True
    # Popularity (Last.fm) bias on the within-slot medoid pick. Activated by the
    # "Popular Seeds" checkbox (overrides this to popular_seeds_weight). Keep
    # below medoid_energy_weight so energy-spread keeps the slot structure.
    medoid_popularity_weight: float = 0.0
    # Tag steering (GUI genre-tag chips, artist mode): on-tag bonus in the
    # within-cluster medoid pick. 0.0 => inert (today's behavior).
    medoid_tag_weight: float = 0.0  # tag-steering on-tag bonus in pier scoring
    # Pier bridgeability veto (spec 2026-07-03): a medoid candidate needs >= k
    # non-seed-artist library neighbors at calibrated T >= floor_t to seat as a
    # pier. Vetoes medoid candidacy only; cluster membership is untouched.
    pier_bridgeability_enabled: bool = True
    pier_bridgeability_floor_t: float = 0.30
    pier_bridgeability_k: int = 10
    # Genre-relevance gate on the bridgeability neighbor set (spec Revision
    # 2026-07-04): a library track only counts as a bridge neighbor if its genre
    # cosine to the seed artist's profile is >= this floor. Prevents a
    # degenerate/collapsed sonic embedding (e.g. MuQ mapping near-silent audio to
    # one point) from faking bridgeability via genre-unrelated neighbors.
    pier_bridgeability_genre_floor: float = 0.30
    # Support-aware pier demotion (Corridor Phase 2 Task 3, spec
    # docs/superpowers/plans/2026-07-18-corridor-phase2.md): a candidate whose
    # within-artist-catalog neighborhood is sparser than typical for this artist
    # gets its medoid-selection score demoted (never vetoed -- pier COUNT is
    # never reduced). See compute_within_artist_support for why this is
    # WITHIN-ARTIST, not library-wide (a scratchpad probe found library-wide
    # top-K density does not separate the known outlier cases -- they are
    # perfectly well-supported by OTHER artists elsewhere in the library, just
    # off-character for THIS artist's own catalog).
    pier_support_enabled: bool = True
    pier_support_k: int = 10
    pier_support_demotion_strength: float = 1.0   # 0.0 = inert (today's behavior)
    # Diagnostic-only floor (never filters candidates): below this, a cluster's
    # best-available candidate is logged loudly as "no confidently-supported
    # alternative" rather than silently demoted with no trace.
    pier_support_floor: float = 0.50
    # Arc-aware ordering: prefer NOT seating the lowest-support pier at either
    # terminal position (opening/closing) when an alternate sonic-order start
    # avoids it. Pure tie-break among orderings order_clusters could already
    # produce -- never invents a new topology, never fails (see
    # reorder_avoiding_low_support_terminal).
    pier_support_terminal_avoidance: bool = True


def load_artist_energy_values(bundle, cfg: "ArtistStyleConfig") -> Optional[np.ndarray]:
    """Load z-scored energy aligned to bundle.track_ids for energy-aware spread.

    Returns None (inert) when the energy term is off or the sidecar is unavailable.
    Per the configured-knob-must-act rule, a >0 weight with no usable energy WARNs.
    """
    if cfg.medoid_energy_weight <= 0:
        return None
    artifact_path = getattr(bundle, "artifact_path", None)
    if artifact_path is None:
        logger.warning(
            "artist_style: medoid_energy_weight=%.3f but bundle has no artifact_path; "
            "energy spread inert", cfg.medoid_energy_weight,
        )
        return None
    sidecar = Path(artifact_path).parent / "energy" / "energy_sidecar.npz"
    if not sidecar.exists():
        logger.warning(
            "artist_style: medoid_energy_weight=%.3f but energy sidecar missing at %s; "
            "energy spread inert", cfg.medoid_energy_weight, sidecar,
        )
        return None
    from src.playlist.energy_loader import load_energy_matrix

    matrix = load_energy_matrix(
        bundle.track_ids, sidecar_path=str(sidecar), features=(cfg.energy_feature,)
    )
    vals = np.asarray(matrix[:, 0], dtype=float)
    if not np.any(np.isfinite(vals)):
        logger.warning(
            "artist_style: energy sidecar has no finite %s values; energy spread inert",
            cfg.energy_feature,
        )
        return None
    return vals


def _select_k(track_count: int, cfg: ArtistStyleConfig) -> int:
    """Heuristic for number of clusters."""
    if not cfg.cluster_k_heuristic_enabled:
        return max(cfg.cluster_k_min, min(cfg.cluster_k_max, 3))
    if track_count < 25:
        k = 3
    elif track_count < 60:
        k = 4
    elif track_count < 120:
        k = 5
    else:
        k = 6
    return max(cfg.cluster_k_min, min(cfg.cluster_k_max, k))


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means (cosine via normalized vectors)."""
    n, _ = X.shape
    if n == 0:
        return np.array([], dtype=int), np.empty((0, X.shape[1]))
    # init: random distinct points
    init_idx = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[init_idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(15):
        # assign
        sims = np.dot(X, centroids.T)
        labels = np.argmax(sims, axis=1)
        # update
        new_centroids = []
        for i in range(centroids.shape[0]):
            members = X[labels == i]
            if len(members) == 0:
                new_centroids.append(centroids[i])
            else:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid) + 1e-12
                new_centroids.append(centroid / norm)
        new_centroids = np.stack(new_centroids, axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels, centroids


def _compute_artist_duration_stats(durations_ms: np.ndarray, artist_indices: List[int]) -> Dict[str, float]:
    """
    Compute duration statistics for the artist's catalog for adaptive outlier detection.

    Uses IQR method which is robust to skewed distributions (works for both punk and prog bands).
    """
    if not artist_indices:
        return {}

    artist_durations_sec = durations_ms[artist_indices] / 1000.0

    q1 = float(np.percentile(artist_durations_sec, 25))
    q3 = float(np.percentile(artist_durations_sec, 75))
    iqr = q3 - q1

    return {
        'mean': float(np.mean(artist_durations_sec)),
        'median': float(np.median(artist_durations_sec)),
        'std': float(np.std(artist_durations_sec)),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'min': float(np.min(artist_durations_sec)),
        'max': float(np.max(artist_durations_sec)),
        # Lower fence: Q1 - 1.5*IQR (standard outlier detection)
        'lower_fence': q1 - 1.5 * iqr,
    }


def _duration_outlier_score(duration_sec: float, stats: Dict[str, float]) -> float:
    """
    Calculate how much of an outlier a duration is.

    Returns:
        0.0 = typical duration for this artist
        1.0 = extreme outlier (way too short)

    Uses IQR-based method so it adapts to each artist's typical range.
    For example:
    - Punk band with 1.5 min avg: a 30s track would be an outlier
    - Prog band with 8 min avg: a 30s track would be an outlier
    - But a 1.5 min track would only be an outlier for the prog band
    """
    if not stats:
        return 0.0

    lower_fence = stats['lower_fence']

    # Not an outlier
    if duration_sec >= lower_fence:
        return 0.0

    # Calculate how far below the fence (normalized by IQR)
    iqr = stats['iqr']
    if iqr < 1e-6:  # All tracks same duration (edge case)
        return 0.0

    distance_below = lower_fence - duration_sec
    outlier_magnitude = distance_below / iqr

    # Cap at 1.0 (extreme outlier)
    return min(1.0, outlier_magnitude / 2.0)  # Divided by 2 so 3*IQR below = 1.0


_ENERGY_SPAN_EPS = 1e-6


def _finite_median(values: np.ndarray) -> float:
    """Median of finite entries; NaN if there are none."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def _robust_energy_span(
    values: np.ndarray, lo_pct: float, hi_pct: float
) -> Optional[Tuple[float, float]]:
    """Robust (lo, hi) energy span from finite percentiles, or None if degenerate."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if (hi - lo) < _ENERGY_SPAN_EPS:
        return None
    return (lo, hi)


def _slot_targets_by_quantile(
    cluster_medians: Sequence[float],
    artist_energy: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> List[float]:
    """Energy target per cluster at evenly-spaced *population quantiles* of the
    artist's energy, assigned by the cluster's median-energy rank.

    Spacing by quantile (not by raw value) makes the anchor counts follow the
    band's density: dense intensity regions get more targets, sparse regions
    fewer (e.g. an 81%-aggressive catalog => ~81% of targets land aggressive),
    instead of tiling the value range uniformly (which over-fills sparse ends).

    Clusters with a NaN median get a NaN target (their energy term is inert).
    Single cluster -> the midpoint quantile. Aligned to the input order.
    """
    finite = np.asarray(artist_energy, dtype=float)
    finite = finite[np.isfinite(finite)]
    medians = list(cluster_medians)
    n = len(medians)
    targets = [float("nan")] * n
    if finite.size < 2:
        return targets
    finite_idx = [i for i, m in enumerate(medians) if np.isfinite(m)]
    k = len(finite_idx)
    if k == 0:
        return targets
    if k == 1:
        targets[finite_idx[0]] = float(np.percentile(finite, (lo_pct + hi_pct) / 2.0))
        return targets
    # rank finite clusters by ascending median energy; place each at an evenly
    # spaced quantile so target *counts* follow the band's energy density.
    ordered = sorted(finite_idx, key=lambda i: medians[i])
    for rank, i in enumerate(ordered):
        q = lo_pct + (rank / (k - 1)) * (hi_pct - lo_pct)
        targets[i] = float(np.percentile(finite, q))
    return targets


def _slot_proximity(z: np.ndarray, target: float, span_width: float) -> np.ndarray:
    """Per-member proximity to a slot target in [0,1]; 0 for non-finite members.

    Inert (all zeros) if the target is non-finite or span_width <= 0.
    """
    arr = np.asarray(z, dtype=float)
    if not np.isfinite(target) or span_width <= 0:
        return np.zeros_like(arr)
    dist = np.abs(arr - target) / span_width
    prox = 1.0 - np.clip(dist, 0.0, 1.0)
    prox[~np.isfinite(arr)] = 0.0
    return prox


def _load_albums_for_indices(bundle, indices: List[int], db_path: str) -> Dict[int, str]:
    """track index -> album name for the given bundle indices, from metadata.db.

    The artifact bundle does not carry album names, but version-preference needs
    them to catch album-based live recordings (clean track title, live album).
    Best-effort: returns {} on any error. Reads metadata.db read-only."""
    import sqlite3

    track_ids = getattr(bundle, "track_ids", None)
    if track_ids is None or not indices or not db_path:
        return {}
    idx_by_tid = {str(track_ids[i]): i for i in indices}
    out: Dict[int, str] = {}
    try:
        ph = ",".join("?" * len(idx_by_tid))
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            for tid, album in conn.execute(
                f"SELECT track_id, album FROM tracks WHERE track_id IN ({ph})",
                list(idx_by_tid.keys()),
            ):
                i = idx_by_tid.get(str(tid))
                if i is not None and album:
                    out[i] = str(album)
    except Exception:
        return {}
    return out


def _dedupe_artist_indices(
    indices: List[int],
    track_titles: Optional[Sequence[str]],
    durations_ms: Optional[np.ndarray],
    albums_by_index: Optional[Dict[int, str]] = None,
) -> List[int]:
    """Reduce an artist's track indices to one canonical version per song.

    Groups by loose-normalized title (strips live/remaster/edition suffixes) and
    keeps the highest version-preference version per song (studio/remaster beat
    live/demo/alt; remaster is barely penalized). `albums_by_index` (track index ->
    album name) lets version-preference catch album-based live recordings with
    clean titles ("Polly" on "MTV Unplugged"). Ties break to the longer duration,
    then the lower index, for determinism. A live cut survives only if it is the
    song's sole version. Tracks with no usable title pass through unchanged.
    Returns indices sorted ascending.
    """
    if track_titles is None or len(indices) < 2:
        return list(indices)
    from src.title_dedupe import (
        calculate_version_preference_score,
        normalize_title_for_dedupe,
    )

    groups: Dict[str, List[int]] = {}
    kept: List[int] = []
    for idx in indices:
        title = str(track_titles[idx]) if track_titles[idx] is not None else ""
        norm = normalize_title_for_dedupe(title, mode="loose") if title else ""
        if not norm:
            kept.append(idx)  # untitled -> can't group; keep as-is
            continue
        groups.setdefault(norm, []).append(idx)

    def _rank(i: int) -> tuple:
        title = str(track_titles[i]) if track_titles[i] is not None else ""
        album = albums_by_index.get(i, "") if albums_by_index else ""
        score = calculate_version_preference_score(title, album)
        dur = float(durations_ms[i]) if durations_ms is not None else 0.0
        return (score, dur, -i)  # highest score, then longest, then stable

    for members in groups.values():
        kept.append(members[0] if len(members) == 1 else max(members, key=_rank))
    return sorted(kept)


def compute_within_artist_support(
    X_norm: np.ndarray,
    artist_indices: Sequence[int],
    k: int,
) -> Dict[int, float]:
    """Cheap within-artist-catalog support estimate for pier candidacy.

    For each of the artist's own tracks, computes the mean cosine similarity to
    its ``k`` nearest OTHER same-artist tracks, then normalizes by the artist's
    own median (so 1.0 = typical neighborhood density for THIS artist's
    catalog, < 1.0 = sparser / more off-character than typical, > 1.0 = denser
    than typical). Deliberately scoped to the seed artist's own catalog, not
    the whole library.

    Why within-artist, not library-wide (documented per the design-doc
    requirement to justify the estimator against real data before wiring it
    in; probe script + numbers: docs/corridor_baseline/phase2_task3_probe_findings.md): a
    scratchpad probe against the live MuQ artifact found that library-wide
    top-100-neighbor density (fraction/mean of a candidate's nearest library
    neighbors clearing a threshold, gated to the seed's genre-relevant
    universe like the existing pier-bridgeability veto) does NOT separate the
    two known outlier-EP pier candidates (Parquet Courts "Into the garden",
    Swirlies "His Life of Academic Freedom") from normal candidates -- both
    rank only in the bottom ~25-35th percentile of their artist's candidates,
    not a clear outlier tail, and "Into the garden" specifically is already
    confirmed well-supported library-wide by the EXISTING pier-bridgeability
    signal (kth-of-10 calibrated T = 0.865, comfortably above the 0.30 floor --
    it has plenty of close neighbors, just mostly by OTHER artists). What
    actually distinguishes both known outliers is that they don't resemble the
    REST OF THEIR OWN ARTIST'S catalog: within-artist KNN density puts both in
    the bottom ~10-20th percentile of their artist's candidate pool, robust
    across k=5..20 (the live generation log for Parquet Courts independently
    corroborates this -- "Into the garden"'s raw _medoids_for_cluster score,
    0.774, was the lowest of all 10 medoid candidates across all 5 clusters
    for that artist).

    Returns ``{bundle_index: relative_support}`` for every index in
    ``artist_indices``. An artist with <=1 track returns support=1.0 for it
    (no peers to compare against; inert, never demoted). Pure numpy -- one
    (n_artist_tracks x n_artist_tracks) matmul, a few matvecs total -- cheap
    enough to compute once per generation and cache alongside clustering.
    """
    members = np.asarray(list(artist_indices), dtype=int)
    n = members.size
    if n <= 1:
        return {int(i): 1.0 for i in members}
    sims = X_norm[members] @ X_norm[members].T
    np.fill_diagonal(sims, -np.inf)
    kk = max(1, min(int(k), n - 1))
    part = np.partition(sims, n - kk, axis=1)[:, n - kk:]
    mean_knn = part.mean(axis=1)
    median = float(np.median(mean_knn))
    if median <= 1e-9:
        return {int(i): 1.0 for i in members}
    rel = mean_knn / median
    return {int(idx): float(rel[j]) for j, idx in enumerate(members)}


def _support_penalty(support: float, strength: float) -> float:
    """Continuous rank-demotion penalty from a relative support score.

    0.0 when ``support >= 1.0`` (never promotes above-typical candidates, only
    demotes below-typical ones) or when ``strength <= 0`` (inert). Grows
    linearly as support falls below 1.0, scaled by ``strength``. Pure,
    unit-tested directly (no clustering/artifact dependency).
    """
    return max(0.0, 1.0 - float(support)) * max(0.0, float(strength))


def _medoids_for_cluster(
    X: np.ndarray,
    indices: List[int],
    centroid: np.ndarray,
    bundle_track_ids: Sequence[str],
    per_cluster: int,
    rng: np.random.Generator,
    top_k: int,
    artist_duration_stats: Optional[Dict[str, float]] = None,
    track_durations_ms: Optional[np.ndarray] = None,
    similarity_weight: float = 0.7,
    duration_weight: float = 0.3,
    energy_weight: float = 0.0,
    energy_proximity: Optional[np.ndarray] = None,
    popularity_weight: float = 0.0,
    popularity_values: Optional[np.ndarray] = None,
    tag_weight: float = 0.0,
    tag_affinity: Optional[np.ndarray] = None,
    support_weight: float = 0.0,
    support_values: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Select medoids using weighted scoring that penalizes duration outliers.

    Weighting strategy (configurable):
    - similarity_weight (default 70%): similarity to cluster centroid (sonic cohesion)
    - duration_weight (default 30%): duration typicality (avoid extreme outliers)
    - energy_weight (default 0.0): pull toward this cluster's energy slot (set-level spread)
    - energy_proximity: per-member proximity in [0,1] to the cluster's energy slot, aligned to `indices`

    This ensures we pick representative tracks that are sonically central
    but not weird interludes/outros. The weights adapt to each artist's catalog,
    so it works for both punk bands (short tracks typical) and prog bands (long tracks typical).
    """
    if not indices:
        return []

    # Base similarity to centroid
    sims = np.dot(X[indices], centroid)

    # Initialize duration weights (default to 1.0 = no penalty)
    duration_weights = np.ones(len(indices))

    # Apply duration outlier penalty if we have duration data
    if artist_duration_stats and track_durations_ms is not None:
        for i, idx in enumerate(indices):
            duration_sec = track_durations_ms[idx] / 1000.0
            outlier_score = _duration_outlier_score(duration_sec, artist_duration_stats)

            # Convert outlier score to weight (1.0 = typical, 0.0 = extreme outlier)
            duration_weights[i] = 1.0 - outlier_score

    # Combined weighted score (configurable weights)
    scores = sims * similarity_weight + duration_weights * duration_weight

    # Energy-aware spread: pull the medoid toward this cluster's arousal slot.
    if energy_proximity is not None and energy_weight > 0:
        prox = np.asarray(energy_proximity, dtype=float)
        if prox.shape[0] == len(indices):
            scores = scores + prox * energy_weight
        else:  # defensive: misaligned proximity must never silently corrupt scores
            logger.warning(
                "artist_style: energy_proximity len %d != cluster size %d; skipping energy term",
                prox.shape[0], len(indices),
            )

    # Popularity bias: prefer the recognizable hit WITHIN this cluster's slot.
    if popularity_values is not None and popularity_weight > 0:
        pv = np.asarray(popularity_values, dtype=float)
        if pv.shape[0] == len(indices):
            pv = np.where(np.isfinite(pv), pv, 0.0)   # unknown -> neutral, no bonus
            scores = scores + pv * popularity_weight
        else:
            logger.warning(
                "artist_style: popularity_values len %d != cluster size %d; skipping",
                pv.shape[0], len(indices))

    # Tag steering: prefer the artist's on-tag tracks WITHIN this cluster's slot.
    if tag_affinity is not None and tag_weight > 0:
        ta = np.asarray(tag_affinity, dtype=float)
        if ta.shape[0] == len(indices):
            scores = scores + ta * tag_weight

    # Support-aware demotion (Task 3): candidates sparser than typical for this
    # artist's own catalog lose rank, never get excluded. See
    # compute_within_artist_support / _support_penalty.
    if support_values is not None and support_weight > 0:
        sv = np.asarray(support_values, dtype=float)
        if sv.shape[0] == len(indices):
            penalty = np.array([_support_penalty(s, support_weight) for s in sv])
            scores = scores - penalty
        else:  # defensive: misaligned support array must never silently corrupt scores
            logger.warning(
                "artist_style: support_values len %d != cluster size %d; skipping support term",
                sv.shape[0], len(indices),
            )

    # Select from top-k by combined score
    order = np.argsort(-scores)
    top_k_val = max(1, min(int(top_k), len(indices)))
    medoid_idx = int(order[int(rng.integers(0, top_k_val))])

    medoids: List[int] = [indices[medoid_idx]]

    if per_cluster > 1 and len(indices) > 1:
        # Pick farthest from first medoid to diversify
        # But still respect the quality scores
        first_vec = X[medoids[0]]
        sims2 = np.dot(X[indices], first_vec)

        # Combine diversity (low similarity to first) with quality (high overall score)
        diversity_scores = (1.0 - sims2) * 0.6 + scores * 0.4
        order2 = np.argsort(-diversity_scores)

        for idx in order2:
            cand = indices[int(idx)]
            if cand not in medoids:
                medoids.append(cand)
            if len(medoids) >= per_cluster:
                break

    # Debug logging
    if artist_duration_stats and track_durations_ms is not None:
        medoid_info = []
        for m in medoids:
            cluster_local_idx = indices.index(m)
            dur_sec = track_durations_ms[m] / 1000.0
            score = scores[cluster_local_idx]
            medoid_info.append(f"{bundle_track_ids[m]} ({dur_sec:.0f}s, score={score:.3f})")
        logger.debug("Cluster medoids selected (weighted): %s", medoid_info)
    else:
        logger.debug(
            "Cluster medoids selected: %s",
            [str(bundle_track_ids[i]) for i in medoids],
        )

    return medoids


def compute_pier_bridgeability(
    X_norm: np.ndarray,
    member_indices: Sequence[int],
    excluded_indices: Sequence[int],
    k: int,
    *,
    calib_center: float,
    calib_scale: float,
    calib_gain: float,
    eligible_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calibrated T of each member's k-th best ELIGIBLE library neighbor.

    A neighbor is eligible iff it is not in ``excluded_indices`` (same-artist rows)
    AND, when ``eligible_mask`` is given, ``eligible_mask[col]`` is True. With
    ``eligible_mask=None`` behavior is identical to the ungated library-wide signal.
    See docs/superpowers/specs/2026-07-03-pier-bridgeability-design.md (Revision
    2026-07-04 — genre-relevant neighbor set).
    """
    members = np.asarray(list(member_indices), dtype=int)
    if members.size == 0:
        return np.zeros(0, dtype=float)
    n_cols = int(X_norm.shape[0])
    ineligible = np.zeros(n_cols, dtype=bool)
    excl = np.asarray(sorted({int(i) for i in excluded_indices}), dtype=int)
    if excl.size:
        ineligible[excl] = True
    if eligible_mask is not None:
        ineligible |= ~np.asarray(eligible_mask, dtype=bool)
    n_avail = int(n_cols - int(ineligible.sum()))
    if n_avail <= 0:
        return np.zeros(members.size, dtype=float)
    sims = X_norm[members] @ X_norm.T  # (m, N) cosines; rows are already L2-normalized
    sims[:, ineligible] = -np.inf
    kk = max(1, min(int(k), n_avail))
    kth = np.partition(sims, sims.shape[1] - kk, axis=1)[:, sims.shape[1] - kk]
    return np.asarray(
        [
            _calibrate_transition_cos(
                float(v), center=calib_center, scale=calib_scale, gain=calib_gain
            )
            for v in kth
        ],
        dtype=float,
    )


def seed_genre_relevance_mask(
    X_genre: Optional[np.ndarray],
    artist_indices: Sequence[int],
    genre_floor: float,
) -> Optional[np.ndarray]:
    """Boolean mask (len N) of library rows genre-compatible with the seed artist.

    Eligible iff the row's cosine to the seed artist's aggregate genre profile
    (max over the artist's own tracks) is >= ``genre_floor``. Zero-genre rows are
    never eligible. Returns None when genre data is missing or the seed profile is
    empty, so the caller falls back to the ungated library-wide signal. Non-circular:
    gated on ``artist_indices`` (the seed artist's tracks), not on the piers.

    Public module-level function (no leading underscore, no ``__all__``
    restricting this module's exports) — reused as-is by
    ``pier_bridge_builder.build_pier_bridge_playlist``'s corridor-pooling path
    (Phase 1 Task 4) to build the genre-mode-keyed relevance mask fed into
    ``build_eligible_universe(relevance_mask=...)``. ``artist_indices`` there is
    the run's seed/pier track indices (not a single artist's full catalog) —
    the function itself is agnostic to what the index set represents, it just
    max-pools their genre profile. Behavior here is unchanged by that reuse.
    """
    if X_genre is None:
        return None
    G = np.asarray(X_genre, dtype=float)
    idx = np.asarray(list(artist_indices), dtype=int)
    if idx.size == 0 or G.shape[0] == 0:
        return None
    seed_g = G[idx].max(axis=0)
    sn = float(np.linalg.norm(seed_g))
    if sn < 1e-12:
        return None
    seed_g = seed_g / sn
    row_norms = np.linalg.norm(G, axis=1)
    safe = np.where(row_norms < 1e-12, 1.0, row_norms)
    genre_sim = np.where(row_norms < 1e-12, -np.inf, (G @ seed_g) / safe)
    return genre_sim >= float(genre_floor)


def allocate_piers_by_tag_affinity(
    medoids_by_cluster: list[list[int]],
    cluster_affinities: list[float],
    target_pier_count: int,
    skew: float,
) -> list[int]:
    """Distribute ``target_pier_count`` pier slots across clusters, skewed toward
    high tag-affinity clusters (soft). Each cluster's medoid list is already
    tag-ranked (``_medoids_for_cluster``), so we take its top ``n_c``.

    ``skew=0`` -> uniform across clusters (affinity ignored); ``skew=1`` -> pure
    affinity weighting. A floor of 1 pier per non-empty cluster preserves the sonic
    arc; the cap is each cluster's available medoid count. Returns the selected
    bundle indices (unordered; the caller orders them).
    """
    sizes = [len(m) for m in medoids_by_cluster]
    nonempty = [c for c, s in enumerate(sizes) if s > 0]
    total_available = sum(sizes)
    P = int(target_pier_count)
    if P <= 0 or not nonempty:
        return []
    # Few tracks: everything becomes a pier (matches legacy under-target behavior).
    if total_available <= P:
        return [i for cluster in medoids_by_cluster for i in cluster]

    K = len(nonempty)
    affs = [float(cluster_affinities[c]) for c in nonempty]
    amin, amax = min(affs), max(affs)
    if amax > amin:
        norm = {c: (float(cluster_affinities[c]) - amin) / (amax - amin) for c in nonempty}
    else:
        norm = {c: 0.5 for c in nonempty}
    nsum = sum(norm.values()) or 1.0
    s = float(skew)
    weight = {c: (1.0 - s) * (1.0 / K) + s * (norm[c] / nsum) for c in nonempty}

    alloc = {c: 0 for c in nonempty}
    # Floor: 1 per non-empty cluster, highest-weight first (handles P < K gracefully).
    for c in sorted(nonempty, key=lambda c: (weight[c], -c), reverse=True):
        if sum(alloc.values()) >= P:
            break
        alloc[c] = 1
    # Fill: each remaining slot goes to the cluster furthest below its weighted
    # target (weight[c] * P), respecting each cluster's available-medoid cap.
    while sum(alloc.values()) < P:
        cands = [c for c in nonempty if alloc[c] < sizes[c]]
        if not cands:
            break
        c = max(cands, key=lambda c: (weight[c] * P - alloc[c], weight[c], -c))
        alloc[c] += 1

    selected: list[int] = []
    for c in nonempty:
        selected.extend(medoids_by_cluster[c][: alloc[c]])
    return selected


def select_popular_piers(
    member_indices: list[int],
    popularity_values: np.ndarray,
    target_pier_count: int,
) -> list[int]:
    """🔥 pier selection: the up-to-target_pier_count member indices with the highest
    Last.fm popularity score (1 - rank/n; higher = more popular). Pure top-N — no
    sonic-diversity constraint. Non-finite scores (non-hits) are excluded; ties break
    by index. Returns [] when no member has a finite score (caller falls back to
    medoid piers). The pier-bridge still orders these for cohesion downstream."""
    pv = np.asarray(popularity_values, dtype=float)
    scored = [(int(i), float(pv[int(i)])) for i in member_indices if np.isfinite(pv[int(i)])]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [i for i, _ in scored[: max(0, int(target_pier_count))]]


def cluster_artist_tracks(
    *,
    bundle,
    artist_name: str,
    cfg: ArtistStyleConfig,
    random_seed: int = 0,
    medoid_top_k: int = 1,
    include_collaborations: bool = False,
    excluded_track_ids: Optional[set[str]] = None,
    energy_values: Optional[np.ndarray] = None,
    popularity_values: Optional[np.ndarray] = None,
    metadata_db_path: Optional[str] = None,
    steering_target: Optional[np.ndarray] = None,
    sonic_tag_affinity: Optional[np.ndarray] = None,   # bundle-aligned (N,) sonic prototype affinity
    sonic_tag_weight: float = 0.0,
    target_pier_count: Optional[int] = None,
    restrict_to_track_ids: Optional[set[str]] = None,
    member_indices: Optional[List[int]] = None,
    bridgeability_eligible_mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], List[int], List[List[int]], np.ndarray, Dict[int, float]]:
    """Cluster artist tracks in sonic space and return clusters + medoids.

    The 5th return value, ``support_by_index``, maps every bundle index in the
    (post-filter) artist candidate pool to its within-artist support score
    (see ``compute_within_artist_support``) -- computed once here and reused
    both for the per-cluster medoid demotion below and by callers doing
    arc-aware pier ordering (``reorder_avoiding_low_support_terminal``).
    """
    track_ids = bundle.track_ids
    if bundle.artist_keys is None:
        raise ValueError("Artifact missing artist_keys for clustering.")
    X_raw = getattr(bundle, "X_sonic", None)
    if X_raw is None:
        raise ValueError("Artifact missing X_sonic for clustering.")
    X_norm = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-12)
    if member_indices is not None:
        # Genre mode: the caller supplies a cross-artist index set (spec §3.2).
        # artist_name degrades to a log label. Most of what follows operates on
        # the index list only and is artist-agnostic -- but the pier-bridgeability
        # exclusion set below is routed through member_indices explicitly (not
        # re-derived from artist_name), since a label matches no real artist key.
        artist_indices = [int(i) for i in member_indices]
        logger.info(
            "Clustering scope: caller-supplied member set (label=%s) n=%d",
            artist_name, len(artist_indices),
        )
    else:
        artist_indices = _artist_indices_in_bundle(
            bundle, artist_name, include_collaborations=include_collaborations
        )
    if excluded_track_ids:
        before = len(artist_indices)
        excluded_ids = {str(tid) for tid in excluded_track_ids}
        artist_indices = [
            idx for idx in artist_indices if str(bundle.track_ids[idx]) not in excluded_ids
        ]
        removed = before - len(artist_indices)
        if removed:
            logger.info(
                "Artist style seed freshness: removed %d recent artist tracks before clustering",
                removed,
            )
    if restrict_to_track_ids is not None:
        before = len(artist_indices)
        keep = {str(tid) for tid in restrict_to_track_ids}
        artist_indices = [i for i in artist_indices if str(bundle.track_ids[i]) in keep]
        logger.info(
            "Tag-first piers: restricted clustering to %d/%d on-tag member(s) of %s",
            len(artist_indices), before, artist_name,
        )
    if include_collaborations:
        # Count solo vs collab purely for visibility in the log.
        solo_only = _artist_indices_in_bundle(bundle, artist_name)
        logger.info(
            "Artist style clustering scope: artist=%s solo=%d collab=%d total=%d",
            artist_name, len(solo_only), len(artist_indices) - len(solo_only),
            len(artist_indices),
        )
    if cfg.dedupe_versions:
        before = len(artist_indices)
        albums_by_index = (
            _load_albums_for_indices(bundle, artist_indices, metadata_db_path)
            if metadata_db_path else None
        )
        artist_indices = _dedupe_artist_indices(
            artist_indices, getattr(bundle, "track_titles", None), bundle.durations_ms,
            albums_by_index,
        )
        if before != len(artist_indices):
            logger.info(
                "Artist style version-dedup: %s %d -> %d tracks (one canonical version per song)",
                artist_name, before, len(artist_indices),
            )
    if len(artist_indices) < max(3, cfg.cluster_k_min):
        raise ValueError(f"Not enough tracks to cluster for artist {artist_name}")

    # Support-aware pier demotion (Task 3): within-artist-catalog neighborhood
    # density, computed once over the full (post-filter) candidate pool so it's
    # available both per-cluster below and to the caller's arc-aware ordering
    # step. Cheap (one (n_artist x n_artist) matmul) -- always computed even
    # when demotion/terminal-avoidance are both off, so a later opt-in never
    # needs a new code path (only new weight/flag values).
    support_by_index: Dict[int, float] = compute_within_artist_support(
        X_norm, artist_indices, cfg.pier_support_k
    )

    # Pier bridgeability veto (medoid candidacy only — never mutates clusters).
    bridgeable_set: Optional[set] = None
    if cfg.pier_bridgeability_enabled:
        from src.playlist.transition_metrics import resolve_transition_calib
        _cal_c, _cal_s, _cal_g = resolve_transition_calib(
            getattr(bundle, "sonic_variant", None)
        )
        if member_indices is not None:
            # Genre mode: exclude the caller's own member set. Re-deriving from
            # artist_name would return [] (the label matches no artist key), which
            # silently removes the veto's self-exclusion and lets a track look
            # bridgeable via its own album-mates.
            _excl_cols = artist_indices
        else:
            _excl_cols = _artist_indices_in_bundle(
                bundle, artist_name, include_collaborations=True
            )
        if bridgeability_eligible_mask is not None:
            # Genre mode supplies its own neighbour set (the genre pool). The
            # artist-scoped relevance mask is not merely unnecessary here — it is
            # built by max-pooling the genre profile OF THE MEMBER SET, so a member
            # set contaminated by album-level tags widens the mask to admit its own
            # contaminants' neighbourhoods, which then certifies them as bridgeable.
            # See spec 2026-07-25-genre-mode-pier-admission.md §1.2.
            _genre_mask = bridgeability_eligible_mask
            logger.info(
                "Pier bridgeability: caller-supplied neighbour set, %d/%d rows eligible",
                int(_genre_mask.sum()), int(_genre_mask.size),
            )
        else:
            _xg = getattr(bundle, "X_genre_smoothed", None)
            _genre_mask = seed_genre_relevance_mask(
                _xg, artist_indices, cfg.pier_bridgeability_genre_floor,
            )
            if _genre_mask is not None:
                logger.info(
                    "Pier bridgeability genre gate: %d/%d library rows eligible (genre_floor=%.2f)",
                    int(_genre_mask.sum()), int(_genre_mask.size),
                    float(cfg.pier_bridgeability_genre_floor),
                )
            else:
                logger.warning(
                    "Pier bridgeability: genre-relevance gate inactive (%s) — falling back to "
                    "the ungated library-wide neighbor set.",
                    "X_genre_smoothed absent" if _xg is None
                    else "seed artist has no genre profile",
                )
        _bt = compute_pier_bridgeability(
            X_norm, artist_indices, _excl_cols, cfg.pier_bridgeability_k,
            calib_center=_cal_c, calib_scale=_cal_s, calib_gain=_cal_g,
            eligible_mask=_genre_mask,
        )
        _floor = float(cfg.pier_bridgeability_floor_t)
        bridgeable_set = {
            idx for idx, t in zip(artist_indices, _bt) if float(t) >= _floor
        }
        _vetoed = [
            (idx, float(t)) for idx, t in zip(artist_indices, _bt) if float(t) < _floor
        ]
        if _vetoed:
            logger.info(
                "Pier bridgeability: vetoed %d/%d member(s) (floor_t=%.2f k=%d): %s",
                len(_vetoed), len(artist_indices), _floor,
                int(cfg.pier_bridgeability_k),
                [
                    f"{bundle.track_ids[i]} kth-T={t:.3f}"
                    for i, t in sorted(_vetoed, key=lambda p: p[1])[:10]
                ],
            )
        if not bridgeable_set:
            logger.warning(
                "Pier bridgeability: ALL %d members failed floor_t=%.2f — running "
                "unchecked for this generation (a playlist never fails on a soft axis)",
                len(artist_indices), _floor,
            )
            bridgeable_set = None

    # Energy-aware spread: load energy if not injected, then derive the artist's
    # robust arousal span (slots are spaced across it). None => term stays inert.
    if energy_values is None:
        energy_values = load_artist_energy_values(bundle, cfg)
    energy_span: Optional[Tuple[float, float]] = None
    if energy_values is not None and cfg.medoid_energy_weight > 0:
        artist_energy = np.asarray(energy_values, dtype=float)[artist_indices]
        energy_span = _robust_energy_span(
            artist_energy, cfg.energy_slot_lo_pct, cfg.energy_slot_hi_pct
        )

    k = _select_k(len(artist_indices), cfg)
    rng = np.random.default_rng(random_seed)

    # Compute artist-specific duration statistics for adaptive outlier detection
    artist_duration_stats = None
    if bundle.durations_ms is not None:
        artist_duration_stats = _compute_artist_duration_stats(bundle.durations_ms, artist_indices)
        logger.debug(
            "Artist duration stats for %s: mean=%.1fs, median=%.1fs, Q1=%.1fs, Q3=%.1fs, lower_fence=%.1fs",
            artist_name,
            artist_duration_stats['mean'],
            artist_duration_stats['median'],
            artist_duration_stats['q1'],
            artist_duration_stats['q3'],
            artist_duration_stats['lower_fence'],
        )
    retries = 0
    while True:
        labels, centroids = _kmeans(X_norm[artist_indices], k=k, rng=rng)
        if centroids.shape[0] == 0 or len(set(labels.tolist())) < 2:
            retries += 1
            if k > cfg.cluster_k_min:
                k -= 1
                logger.warning("Cluster retry due to empty/degenerate result; retrying with k=%d", k)
                continue
            raise ValueError(f"Clustering degenerate for artist {artist_name} after retries")
        break
    # First pass: gather non-empty clusters, preserving their centroid index.
    nonempty: List[Tuple[int, List[int]]] = []
    for c in range(centroids.shape[0]):
        members_local = [artist_indices[i] for i, lab in enumerate(labels) if lab == c]
        if members_local:
            nonempty.append((c, members_local))

    # Energy slots: rank clusters by median arousal, then place each target at an
    # evenly-spaced *population quantile* of the artist's energy, so anchor counts
    # follow the band's density (representative) rather than tiling the value range.
    slot_targets: Optional[List[float]] = None
    if energy_span is not None:
        ev = np.asarray(energy_values, dtype=float)
        artist_energy = ev[artist_indices]
        cluster_medians = [_finite_median(ev[members]) for _c, members in nonempty]
        slot_targets = _slot_targets_by_quantile(
            cluster_medians, artist_energy, cfg.energy_slot_lo_pct, cfg.energy_slot_hi_pct
        )
        logger.info(
            "Artist style energy spread: artist=%s span=(%.3f,%.3f) targets=%s",
            artist_name, energy_span[0], energy_span[1],
            [round(t, 3) if np.isfinite(t) else None for t in slot_targets],
        )

    eff_top_k = medoid_top_k
    if bridgeable_set is not None:
        n_eligible_clusters = sum(
            1 for _c, _ml in nonempty if any(m in bridgeable_set for m in _ml)
        )
        if target_pier_count and 0 < n_eligible_clusters < len(nonempty):
            eff_top_k = max(
                medoid_top_k, math.ceil(int(target_pier_count) / n_eligible_clusters)
            )
            logger.warning(
                "Pier bridgeability: %d/%d cluster(s) have no eligible member — "
                "bumping per-cluster medoids to %d to reallocate slots",
                len(nonempty) - n_eligible_clusters, len(nonempty), eff_top_k,
            )

    clusters: List[List[int]] = []
    medoids: List[int] = []
    medoids_by_cluster: List[List[int]] = []
    span_width = (energy_span[1] - energy_span[0]) if energy_span is not None else 0.0
    for ci, (c, members_local) in enumerate(nonempty):
        clusters.append(members_local)
        members_eligible = (
            members_local if bridgeable_set is None
            else [m for m in members_local if m in bridgeable_set]
        )
        if not members_eligible:
            logger.warning(
                "Pier bridgeability: cluster %d has 0/%d eligible members — "
                "contributes no piers; slots reallocate to passing clusters",
                ci, len(members_local),
            )
            medoids_by_cluster.append([])
            continue
        energy_prox: Optional[np.ndarray] = None
        if slot_targets is not None:
            member_energy = np.asarray(energy_values, dtype=float)[members_eligible]
            energy_prox = _slot_proximity(member_energy, slot_targets[ci], span_width)
        pop_slice = None
        if popularity_values is not None and cfg.medoid_popularity_weight > 0:
            pop_slice = np.asarray(popularity_values, dtype=float)[members_eligible]
        tag_slice: Optional[np.ndarray] = None
        _xgd = getattr(bundle, "X_genre_dense", None)
        if steering_target is not None and _xgd is not None and cfg.medoid_tag_weight > 0:
            tag_slice = np.asarray(_xgd, dtype=float)[members_eligible] @ np.asarray(
                steering_target, dtype=float
            )
        # Sonic-prototype term (track-level resolution). Additive: a flat genre
        # term drops out of the ranking, so this decides for genre-blended artists.
        if (
            sonic_tag_affinity is not None
            and float(sonic_tag_weight) > 0.0
            and cfg.medoid_tag_weight > 0
        ):
            _sonic_slice = float(sonic_tag_weight) * np.asarray(
                sonic_tag_affinity, dtype=float
            )[members_eligible]
            tag_slice = _sonic_slice if tag_slice is None else (tag_slice + _sonic_slice)
        support_slice: Optional[np.ndarray] = None
        if cfg.pier_support_enabled and cfg.pier_support_demotion_strength > 0:
            support_slice = np.array(
                [support_by_index.get(m, 1.0) for m in members_eligible], dtype=float
            )
            if float(np.max(support_slice)) < cfg.pier_support_floor:
                logger.warning(
                    "Pier support: cluster %d has NO candidate with typical support "
                    "(best=%.3f, all below diagnostic floor=%.2f) — keeping the "
                    "best-available candidate, no candidates excluded: %s",
                    ci, float(np.max(support_slice)), cfg.pier_support_floor,
                    [str(bundle.track_ids[m]) for m in members_eligible],
                )
        medoid_list = _medoids_for_cluster(
            X_norm,
            members_eligible,
            centroids[c],
            track_ids,
            eff_top_k,
            rng,
            medoid_top_k,
            artist_duration_stats,
            bundle.durations_ms,
            cfg.medoid_similarity_weight,
            cfg.medoid_duration_weight,
            cfg.medoid_energy_weight,
            energy_prox,
            cfg.medoid_popularity_weight,
            pop_slice,
            cfg.medoid_tag_weight,
            tag_slice,
            cfg.pier_support_demotion_strength if cfg.pier_support_enabled else 0.0,
            support_slice,
        )
        medoids_by_cluster.append(medoid_list)
        medoids.extend(medoid_list)
    if bridgeable_set is not None and target_pier_count and len(medoids) < int(target_pier_count):
        logger.warning(
            "Pier bridgeability: after veto+reallocation only %d medoid(s) available vs "
            "target_pier_count=%d — playlist will have fewer seed-artist piers.",
            len(medoids), int(target_pier_count),
        )
    logger.info(
        "Artist style clustering: artist=%s k=%d clusters=%d medoids=%d variant=%s dim=%d",
        artist_name,
        k,
        len(clusters),
        len(medoids),
        getattr(bundle, "sonic_variant", None),
        int(X_norm.shape[1]),
    )
    # Diagnostics: intra/inter stats
    try:
        intra_sims: List[float] = []
        inter_sims: List[float] = []
        for idx_c, members in enumerate(clusters):
            if len(members) > 1:
                sub = X_norm[members]
                sims = np.dot(sub, sub.T)
                mask = ~np.eye(len(members), dtype=bool)
                intra_sims.extend(sims[mask].tolist())
            for idx_d, others in enumerate(clusters):
                if idx_d <= idx_c or not others:
                    continue
                sims_cross = np.dot(X_norm[members], X_norm[others].T).flatten().tolist()
                inter_sims.extend(sims_cross)
        if intra_sims and inter_sims:
            logger.info(
                "Artist style clustering stats: artist=%s k=%d clusters=%d intra_median=%.3f inter_median=%.3f",
                artist_name,
                k,
                len(clusters),
                float(np.median(intra_sims)),
                float(np.median(inter_sims)),
            )
    except Exception:
        logger.debug("Failed to compute intra/inter cluster stats", exc_info=True)
    return clusters, medoids, medoids_by_cluster, X_norm, support_by_index


def order_clusters(
    medoid_indices: List[int],
    X_norm: np.ndarray,
    *,
    start: Optional[int] = None,
) -> List[int]:
    """Order medoids with greedy nearest-neighbor to minimize jumps.

    ``start``: which medoid to begin the walk from; must be a member of
    ``medoid_indices``. Defaults to ``medoid_indices[0]`` (today's behavior,
    byte-identical when omitted). Used by
    ``reorder_avoiding_low_support_terminal`` to explore alternate Hamiltonian
    paths over the SAME pier set that don't terminate on the lowest-support
    pier.
    """
    if not medoid_indices:
        return []
    remaining = set(medoid_indices)
    current = start if start is not None and start in remaining else medoid_indices[0]
    order: List[int] = []
    order.append(current)
    remaining.remove(current)
    while remaining:
        sims = {idx: float(np.dot(X_norm[current], X_norm[idx])) for idx in remaining}
        next_idx = max(sims.items(), key=lambda kv: kv[1])[0]
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
    return order


def reorder_avoiding_low_support_terminal(
    ordered_indices: Sequence[int],
    support_by_index: Dict[int, float],
    X_norm: np.ndarray,
) -> Tuple[List[int], bool]:
    """Arc-aware ordering tie-break (Corridor Phase 2 Task 3).

    Weakest edges cluster at anchor/terminal seats (memory
    ``project_corridor_outlier_anchor_mechanism``; live example: Parquet
    Courts' clustering picked the outlier-EP "Into the garden" as a pier AND
    seated it at the closing position, manufacturing the playlist's globally
    weakest edge right where it's most exposed). Among the Hamiltonian paths
    ``order_clusters`` could already produce over this SAME pier set (same
    greedy-nearest-neighbor algorithm, different start node), prefer one that
    does not seat the lowest-support pier at either terminal position (index 0
    or -1).

    This is a PREFERENCE / tie-break, not an override: it never invents a new
    topology (every candidate order is something ``order_clusters`` itself
    could produce) and it never fails --

    - With < 3 piers there is no interior seat to move a pier to (both
      positions are inescapably terminal); returns the input unchanged.
    - If the lowest-support pier isn't at a terminal seat already, returns the
      input unchanged (nothing to do).
    - If no alternate start avoids the terminal placement either (possible for
      the greedy heuristic in degenerate/near-degenerate sonic layouts), the
      original order is kept -- never raises, never drops a pier.

    Ties in "lowest support" break to the pier appearing earliest in
    ``ordered_indices``, for determinism.

    Returns ``(order, changed)``.
    """
    indices = [int(i) for i in ordered_indices]
    if len(indices) < 3:
        return indices, False
    worst = min(indices, key=lambda i: (support_by_index.get(i, 1.0), indices.index(i)))
    if indices[0] != worst and indices[-1] != worst:
        return indices, False
    for start in indices:
        if start == indices[0]:
            continue  # identical to the current (already-failing) order
        candidate = order_clusters(indices, X_norm, start=start)
        if candidate[0] != worst and candidate[-1] != worst:
            return candidate, True
    return indices, False


# build_balanced_candidate_pool (the per-cluster sonic-neighbor "external pool"
# that hard-clamped Artist mode's allowed_track_ids) was DELETED in Phase 1
# Task 8 -- corridor pooling's library-wide eligible universe (per-segment
# corridors, src/playlist/pier_bridge/corridor.py) replaces it for the
# pier-bridge path; the medoid clustering above (cluster_artist_tracks) is
# unaffected and still selects the artist's pier set. See
# docs/POOL_STARVATION_RESEARCH_2026-07-12.md (M1: this hard clamp was the
# "amputated manifold" pool-starvation mechanism the corridor rework targets).
# Retired config keys: playlists.ds_pipeline.artist_style.per_cluster_candidate_pool_size,
# .pool_balance_mode (warn via src.playlist_gui.worker._warn_retired_keys).


def build_genre_neighbor_candidate_pool(
    *,
    bundle,
    pier_indices: Sequence[int],
    artist_key: str,
    pool_size: int,
    min_similarity: float,
    min_confidence: Optional[float],
    compatible_threshold: float,
    conflict_threshold: float,
    genre_method: str = "ensemble",
    artist_name: Optional[str] = None,
    include_collaborations: bool = False,
) -> List[str]:
    """Build a genre-first artist-style pool to complement sonic cluster neighbors."""
    if not pier_indices:
        return []

    X_raw = getattr(bundle, "X_genre_raw", None)
    X_smooth = getattr(bundle, "X_genre_smoothed", None)
    raw_vocab = getattr(bundle, "genre_vocab", None)
    genre_vocab = [] if raw_vocab is None else [str(g) for g in list(raw_vocab)]
    if X_raw is None and X_smooth is None:
        return []

    genre_matrix = X_smooth if X_smooth is not None else X_raw
    if genre_matrix is None or genre_matrix.ndim != 2:
        return []

    pier_list = [int(i) for i in pier_indices if 0 <= int(i) < genre_matrix.shape[0]]
    if not pier_list:
        return []

    seed_genres = np.max(genre_matrix[pier_list], axis=0)
    genre_sim = _compute_genre_similarity(seed_genres, genre_matrix, method=genre_method)

    confidence = np.ones(genre_matrix.shape[0], dtype=float)
    missing = np.zeros(genre_matrix.shape[0], dtype=bool)
    if X_raw is not None and X_raw.ndim == 2 and X_raw.shape[1] == len(genre_vocab):
        seed_raw = np.max(X_raw[pier_list], axis=0)
        compat = compute_raw_genre_compatibility(
            seed_raw=seed_raw,
            candidate_raw=X_raw,
            genre_vocab=genre_vocab,
            compatible_threshold=compatible_threshold,
            conflict_threshold=conflict_threshold,
            penalty_strength=0.0,
        )
        confidence = np.asarray(compat.confidence, dtype=float)
        missing = np.asarray(compat.missing_or_sparse, dtype=bool)

    total_tracks = genre_matrix.shape[0]
    if include_collaborations and artist_name:
        seed_artist_indices = set(
            _artist_indices_in_bundle(
                bundle, artist_name, include_collaborations=True
            )
        )
        mask_artist = np.array(
            [i in seed_artist_indices for i in range(total_tracks)], dtype=bool
        )
    else:
        mask_artist = np.array([
            normalize_artist_key(str(bundle.artist_keys[i])) == artist_key
            if bundle.artist_keys is not None else False
            for i in range(total_tracks)
        ], dtype=bool)

    pier_set = set(pier_list)
    selected: List[int] = []
    for idx in np.argsort(-genre_sim):
        idx = int(idx)
        if idx in pier_set or mask_artist[idx]:
            continue
        if float(genre_sim[idx]) < float(min_similarity):
            break
        if min_confidence is not None and (not bool(missing[idx])) and float(confidence[idx]) < float(min_confidence):
            continue
        selected.append(idx)
        if len(selected) >= int(pool_size):
            break

    logger.info(
        "Artist style genre-neighbor pool: selected=%d/%d min_similarity=%.3f min_confidence=%s",
        len(selected),
        int(pool_size),
        float(min_similarity),
        min_confidence,
    )
    return [str(bundle.track_ids[i]) for i in selected]


def get_internal_connectors(
    bundle,
    artist_key: str,
    exclude_indices: List[int],
    global_floor: Optional[float],
    pier_indices: List[int],
    X_norm: np.ndarray,
) -> List[str]:
    track_ids = bundle.track_ids
    exclude_set = set(exclude_indices)
    connectors: List[int] = []
    artist_keys = bundle.artist_keys if bundle.artist_keys is not None else []
    for i, ak in enumerate(artist_keys):
        if normalize_artist_key(str(ak)) != artist_key:
            continue
        if i in exclude_set:
            continue
        if global_floor is not None:
            sim = float(np.max(np.dot(X_norm[i], X_norm[pier_indices].T)))
            if sim < global_floor:
                continue
        connectors.append(i)
    return [str(track_ids[i]) for i in connectors]
