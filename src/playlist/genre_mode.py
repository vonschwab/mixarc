"""Genre-mode playlist generation: resolution, membership, pool expansion.

Genre mode picks ONE canonical genre and builds a playlist from it. Seeds come
from exact-genre membership; the bridge pool widens by hub-damped taxonomy
similarity above a threshold (spec 2026-07-24-genre-mode-design.md §3.1).

Everything here is pure over a read-only sqlite connection plus an injected
similarity scorer, so it is testable without an artifact or a live DB.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenreResolution:
    genre_id: str
    name: str


def resolve_genre_query(conn: sqlite3.Connection, query: str) -> Optional[GenreResolution]:
    """Resolve a user-typed genre (canonical name or alias) to a canonical genre.

    Returns None when nothing matches — callers surface that as a user-facing error
    rather than silently generating something unrelated.
    """
    from src.genre.authority import canonical_genre_search

    hits = canonical_genre_search(conn, query, limit=1)
    if not hits:
        return None
    return GenreResolution(genre_id=hits[0][0], name=hits[0][1])


def canonical_names_by_id(conn: sqlite3.Connection) -> dict[str, str]:
    """genre_id -> canonical name for every ACTIVE genre. The similarity scorer is
    canonical-name keyed, so every id must be mapped before scoring."""
    rows = conn.execute(
        "SELECT genre_id, name FROM genre_graph_canonical_genres WHERE status = 'active'"
    ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def seed_member_track_ids(conn: sqlite3.Connection, genre_id: str) -> set[str]:
    """Exact-genre track ids eligible to be SEEDS (observed_leaf/legacy only).

    Inferred layers are excluded: a pier must be unambiguously the genre.
    """
    from src.genre.authority import track_ids_for_genre_ids

    return track_ids_for_genre_ids(conn, {genre_id})


def neighbors_above_threshold(
    steering, genre_name: str, candidate_names: dict[str, str], threshold: float
) -> dict[str, float]:
    """genre_id -> hub-damped similarity for every genre scoring >= threshold.

    Excludes the query genre itself (callers add it at similarity 1.0). Uncovered
    tags score 0.0 from the scorer and drop out naturally.
    """
    out: dict[str, float] = {}
    thr = float(threshold)
    for gid, name in candidate_names.items():
        if name == genre_name:
            continue
        sim = float(steering.similarity(genre_name, name))
        if sim >= thr:
            out[gid] = sim
    return out


def pool_track_ids(
    conn: sqlite3.Connection,
    steering,
    genre_id: str,
    genre_name: str,
    threshold: float,
) -> tuple[set[str], dict[str, float]]:
    """Bridge-pool track ids at ``threshold``, plus the genre_id -> similarity map.

    The exact genre is always included at similarity 1.0, so an impossible threshold
    degrades to exact-only rather than to an empty pool.
    """
    from src.genre.authority import track_ids_for_genre_ids

    sims = {genre_id: 1.0}
    sims.update(
        neighbors_above_threshold(
            steering, genre_name, canonical_names_by_id(conn), threshold
        )
    )
    ids = track_ids_for_genre_ids(conn, set(sims))
    logger.info(
        "stage=genre_pool | genre=%s threshold=%.2f genres=%d tracks=%d",
        genre_id, float(threshold), len(sims), len(ids),
    )
    return ids, sims


@dataclass(frozen=True)
class SeedWeights:
    """Composite seed-score weights (spec §3.2). Tunable via playlists.genre_playlist."""
    prominence: float = 1.0
    canonicity: float = 0.5
    centrality: float = 0.75
    typicality: float = 0.75
    investment: float = 0.15


def sonic_typicality(X_sonic, member_indices) -> dict:
    """Bundle index -> how typical of THIS genre the track sounds, in [0, 1].

    Cosine to the L2-normalized centroid of the genre's member vectors, rescaled
    from [-1, 1]. Second of the two contextual oddball mechanisms in spec §3.4:
    because it is measured against the chosen genre's own centroid, a Boards of
    Canada interstitial is typical of ambient and survives there, while a
    non-shoegaze-shaped intro scores low inside shoegaze. No global rule, so no
    genre gets gutted (§5.1).
    """
    import numpy as np

    idx = [int(i) for i in member_indices]
    if not idx:
        return {}
    X = np.asarray(X_sonic, dtype=np.float64)[idx]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    centroid = Xn.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    cos = Xn @ centroid
    return {i: float((c + 1.0) / 2.0) for i, c in zip(idx, cos)}


# Prominence assumed for an artist absent from the Last.fm cache. Deliberately the
# NEUTRAL midpoint, not 0.0: coverage is lazily populated (29% for funk), so scoring
# uncached artists at the bottom would exclude most of a genre. See spec §3.3.
UNCACHED_PROMINENCE = 0.5


def score_seed_candidates(
    indices,
    *,
    artist_keys: dict,
    prominence_by_artist: dict,
    canonicity_by_index: dict,
    centrality_by_index: dict,
    typicality_by_index: dict,
    investment_by_artist: dict,
    weights: SeedWeights,
) -> dict:
    """Composite seed score per bundle index. Higher is a better pier.

    Missing prominence means UNCACHED, scored at the neutral midpoint. Missing
    canonicity/centrality/typicality/investment mean genuinely absent and score 0.0.
    ``typicality_by_index`` is a required argument (not defaulted): a caller that
    forgets it must fail loudly, not silently zero out the 0.75-weighted
    typicality term for every candidate — a configured signal that can't act is a
    bug, not a no-op (see CLAUDE.md "Activate fixes; never default to legacy").
    """
    out: dict = {}
    for i in indices:
        i = int(i)
        ak = str(artist_keys.get(i, ""))
        prom = prominence_by_artist.get(ak, UNCACHED_PROMINENCE)
        out[i] = (
            weights.prominence * float(prom)
            + weights.canonicity * float(canonicity_by_index.get(i, 0.0))
            + weights.centrality * float(centrality_by_index.get(i, 0.0))
            + weights.typicality * float(typicality_by_index.get(i, 0.0))
            + weights.investment * float(investment_by_artist.get(ak, 0.0))
        )
    return out


def resolve_pool_with_relaxation(
    conn: sqlite3.Connection,
    steering,
    genre_id: str,
    genre_name: str,
    *,
    start_threshold: float,
    steps,
    min_tracks: int,
) -> tuple[set, dict, float]:
    """Widen the similarity threshold until the pool supports ``min_tracks``.

    Relaxation IS the threshold stepping down — there is no separate ladder
    (spec §3.5). Every rung is logged with its resulting pool size so a starved
    generation is diagnosable from the log alone.

    Never raises on a thin genre: if no rung satisfies ``min_tracks``, the widest
    attempt is returned (never-fail on the soft axes).
    """
    attempts = [float(start_threshold)] + [
        float(s) for s in (steps or ()) if float(s) < float(start_threshold)
    ]
    widest: tuple = (set(), {}, attempts[-1])
    for thr in attempts:
        ids, sims = pool_track_ids(conn, steering, genre_id, genre_name, thr)
        widest = (ids, sims, thr)
        if len(ids) >= int(min_tracks):
            logger.info(
                "stage=genre_pool | relaxation settled threshold=%.3f tracks=%d "
                "(needed %d)", thr, len(ids), int(min_tracks),
            )
            return ids, sims, thr
        logger.info(
            "stage=genre_pool | relaxation step threshold=%.3f tracks=%d < %d — widening",
            thr, len(ids), int(min_tracks),
        )
    logger.warning(
        "stage=genre_pool | genre=%s starved: widest threshold=%.3f yields %d tracks "
        "(< %d). Generating from the widest pool.",
        genre_id, widest[2], len(widest[0]), int(min_tracks),
    )
    return widest


def canonicity_by_index(bundle, member_indices, popularity_db_path: str) -> dict:
    """Bundle index -> within-artist Last.fm canonicity in [0, 1] (`1 - rank/N`).

    This is the signal `rank` IS valid for: which track of an artist is the one
    people know. Absent (uncached artist, or track outside the top-N) scores 0.0 —
    unlike prominence, absence here is genuinely "not a known song of theirs".
    """
    from src.analyze.popularity_runner import (
        get_artist_top_tracks_cached, resolve_top_tracks_to_rank,
    )

    by_artist: dict[str, list] = {}
    for i in member_indices:
        by_artist.setdefault(str(bundle.artist_keys[i]), []).append(int(i))
    out: dict = {}
    for artist_key, idxs in by_artist.items():
        top = get_artist_top_tracks_cached(popularity_db_path, artist_key) or []
        if not top:
            continue
        local = [
            {"track_id": str(bundle.track_ids[i]),
             "title": str(bundle.track_titles[i])} for i in idxs
        ]
        ranks = resolve_top_tracks_to_rank(top, local)
        n = max(1, len(top))
        id_to_index = {str(bundle.track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            if tid in id_to_index:
                out[id_to_index[tid]] = 1.0 - float(rank) / n
    return out


def centrality_by_index(
    db_path: str, bundle, member_indices, genre_id: str
) -> dict:
    """Bundle index -> genre purity in [0, 1].

    Authority confidence for the chosen genre, divided down by how many OTHER
    published genres the release carries: a release tagged only `shoegaze` is more
    purely shoegaze than one carrying eight tags (spec §3.2).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT t.track_id, "
            "       MAX(CASE WHEN reg.genre_id = ? THEN reg.confidence END) AS conf, "
            "       COUNT(DISTINCT reg.genre_id) AS n_genres "
            "FROM tracks t JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            "WHERE reg.assignment_layer IN ('observed_leaf', 'legacy') "
            "GROUP BY t.track_id",
            (str(genre_id),),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    by_id = {
        str(r[0]): float(r[1] or 0.0) / (1.0 + 0.15 * max(0, int(r[2]) - 1))
        for r in rows if r[1] is not None
    }
    return {
        int(i): by_id[str(bundle.track_ids[i])]
        for i in member_indices if str(bundle.track_ids[i]) in by_id
    }


def pool_membership_mask(bundle, pool_ids):
    """Boolean mask over bundle rows: True where the track is in the genre pool.

    Used as the pier-bridgeability neighbour set for genre mode, replacing the
    artist-scoped genre relevance mask. That mask is built by max-pooling the genre
    profile OF THE MEMBER SET, so when the member set is contaminated by album-level
    tags it widens to admit the contaminants' own neighbourhoods and then certifies
    them as bridgeable — the self-justifying loop in spec
    2026-07-25-genre-mode-pier-admission.md §1.2.

    With this mask the veto instead asks: "can this candidate reach k good
    neighbours among the genre-adjacent pool, outside the exact-member set?"
    """
    import numpy as np

    n = len(bundle.track_ids)
    mask = np.zeros(n, dtype=bool)
    # Prefer the bundle's own index when present, but don't require it — the map is
    # an ArtifactBundle convenience, not part of the minimal contract this needs.
    idx_of = getattr(bundle, "track_id_to_index", None)
    if idx_of is None:
        idx_of = {str(t): i for i, t in enumerate(bundle.track_ids)}
    for tid in (pool_ids or ()):
        i = idx_of.get(str(tid))
        if i is not None:
            mask[int(i)] = True
    return mask


def pool_genre_breakdown(conn: sqlite3.Connection, genre_ids) -> list:
    """``(genre_id, track_count)`` per contributing genre, most tracks first.

    Answers "which genres actually filled this pool, and how much did each add?"
    — the question the alpha review could not answer from the logs, e.g. why an
    ambient pool is dominated by neighbours rather than ambient itself.

    One grouped query, not one per genre. Counts are per-genre and therefore sum
    to MORE than the pool size when a track carries several pooled genres; the
    pool itself is a distinct union.
    """
    gids = {str(g) for g in (genre_ids or ()) if str(g)}
    if not gids:
        return []
    ph = ",".join("?" for _ in gids)
    try:
        rows = conn.execute(
            f"SELECT reg.genre_id, COUNT(DISTINCT t.track_id) "
            f"FROM tracks t JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            f"WHERE reg.genre_id IN ({ph}) "
            f"  AND reg.assignment_layer IN ('observed_leaf', 'legacy') "
            f"GROUP BY reg.genre_id",
            tuple(gids),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return sorted(((str(r[0]), int(r[1])) for r in rows), key=lambda t: (-t[1], t[0]))


def filter_member_indices_by_duration(
    member_indices, durations_ms, *, min_seconds: int, max_seconds: int
) -> tuple:
    """Drop pier candidates outside the configured duration window.

    Genre-mode piers are ALGORITHM-selected, so unlike a user-supplied seed they
    get no duration exemption — a 77-minute track must never become an anchor
    (spec 2026-07-25-genre-mode-pass1-invariants.md §3).

    A missing/zero duration means "unknown", not "invalid": those are kept, so
    incomplete metadata cannot silently shrink the member set.

    Returns ``(kept_indices, removed_count)``. Pure.
    """
    idx = [int(i) for i in member_indices]
    if not idx:
        return [], 0
    lo_ms = float(min_seconds) * 1000.0
    hi_ms = float(max_seconds) * 1000.0
    kept = []
    for i in idx:
        try:
            d = float(durations_ms[i])
        except (IndexError, TypeError, ValueError):
            kept.append(i)
            continue
        if d <= 0.0 or lo_ms <= d <= hi_ms:
            kept.append(i)
    return kept, len(idx) - len(kept)


def select_piers_from_clusters(
    clusters,
    scores: dict,
    artist_keys: dict,
    *,
    min_cluster_size: int,
    bridgeable: Optional[set],
    epoch: int = 0,
) -> list:
    """One pier per sonic cluster, at most one per artist, best-scoring first.

    Clusters are visited best-candidate-first so a strong cluster claims its artist
    before a weaker one does. A cluster smaller than ``min_cluster_size`` is skipped
    (avoids a cluster of oddballs donating a pier — spec §6). A cluster whose every
    candidate is unbridgeable or artist-blocked is dropped, not crashed.

    ``bridgeable`` is the set of indices passing the pier-bridgeability veto, or None
    to disable the check.

    ``epoch`` rotates each cluster's ranked candidates so "give me another one" yields
    a different playlist while staying reproducible: selection is a pure function of
    (genre, epoch). epoch=0 picks each cluster's top scorer. Rotation only changes
    which candidate a cluster offers first; it does not change cluster VISIT order —
    that always follows each cluster's true best score, computed before rotation, so
    a strong cluster still claims its artist before a weaker one regardless of epoch.
    Spec §3.7 option (C).
    """
    eligible = []
    for cluster in clusters:
        members = [int(i) for i in cluster]
        if len(members) < int(min_cluster_size):
            continue
        members = [i for i in members if bridgeable is None or i in bridgeable]
        if not members:
            continue
        members.sort(key=lambda i: (-float(scores.get(i, 0.0)), i))
        best = float(scores.get(members[0], 0.0))  # true best, BEFORE rotation
        if epoch:
            offset = int(epoch) % len(members)
            members = members[offset:] + members[:offset]
        eligible.append((best, members))
    eligible.sort(key=lambda t: -t[0])

    used_artists: set = set()
    piers: list = []
    for _best, members in eligible:
        for i in members:
            ak = str(artist_keys.get(i, ""))
            if ak and ak in used_artists:
                continue
            piers.append(i)
            used_artists.add(ak)
            break
    piers.sort()
    return piers
