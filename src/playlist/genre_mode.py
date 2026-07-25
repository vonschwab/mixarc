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
