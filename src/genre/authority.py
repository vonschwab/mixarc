"""Read API for the published unified genre store in metadata.db.

Single import point for playlist features (SP2+). All reads come from the
materialized release_effective_genres table; taxonomy-structure helpers delegate
to the loaded LayeredTaxonomy.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GenreRow:
    genre_id: str
    assignment_layer: str
    confidence: float
    source: str


def resolved_genres_for_album(conn: sqlite3.Connection, album_id: str) -> list[GenreRow]:
    rows = conn.execute(
        "SELECT genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres WHERE album_id = ? "
        "ORDER BY assignment_layer, genre_id",
        (album_id,),
    ).fetchall()
    return [GenreRow(r[0], r[1], r[2], r[3]) for r in rows]


def resolved_genres_for_track(conn: sqlite3.Connection, track_id: str) -> list[GenreRow]:
    row = conn.execute(
        "SELECT album_id FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()
    if not row or not row[0]:
        return []
    return resolved_genres_for_album(conn, row[0])


def resolved_genres_by_album(conn: sqlite3.Connection) -> dict[str, list[GenreRow]]:
    """All published genres, batched per album (one query, no N+1)."""
    by_album: dict[str, list[GenreRow]] = {}
    rows = conn.execute(
        "SELECT album_id, genre_id, assignment_layer, confidence, source "
        "FROM release_effective_genres "
        "ORDER BY album_id, assignment_layer, genre_id"
    ).fetchall()
    for album_id, genre_id, layer, confidence, source in rows:
        by_album.setdefault(album_id, []).append(
            GenreRow(genre_id, layer, confidence, source)
        )
    return by_album


def canonical_genre_names(conn: sqlite3.Connection) -> dict[str, str]:
    """genre_id -> display name from the published taxonomy copy."""
    return dict(
        conn.execute(
            "SELECT genre_id, name FROM genre_graph_canonical_genres"
        ).fetchall()
    )


def display_genre_names_for_track(conn: sqlite3.Connection, track_id: str) -> list[str]:
    """Published genres for a track's album as display names, deduped.

    The read for GUI display paths (chips, search results, staged seeds):
    observed + inferred layers, genre_id mapped to the canonical display name
    (unmapped ids pass through as-is). Returns [] when the track's album is
    unpublished or the authority tables are absent — display callers fall back
    to other sources, they don't crash.
    """
    try:
        rows = conn.execute(
            "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) "
            "FROM release_effective_genres reg "
            "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
            "WHERE reg.album_id = (SELECT album_id FROM tracks WHERE track_id = ?) "
            "ORDER BY reg.assignment_layer, reg.genre_id",
            (str(track_id),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for genre_id, name in rows:
        if genre_id not in seen:
            seen.add(genre_id)
            out.append(name)
    return out


def genre_source_for_album(conn: sqlite3.Connection, album_id: str) -> str:
    base = conn.execute(
        "SELECT source FROM release_effective_genres "
        "WHERE album_id = ? AND source != 'user' LIMIT 1",
        (album_id,),
    ).fetchone()
    return base[0] if base else "none"


@lru_cache(maxsize=1)
def _taxonomy():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    return load_default_layered_taxonomy()


def parents_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().parents_for_genre(genre_id)]


def families_for(conn: sqlite3.Connection, genre_id: str) -> list[str]:
    return [g.genre_id for g in _taxonomy().families_for_genre(genre_id)]


def is_facet(conn: sqlite3.Connection, genre_id: str) -> bool:
    return _taxonomy().facet_by_id(genre_id) is not None


def canonical_genre_search(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[tuple[str, str]]:
    """Active canonical genres matching ``query`` by canonical name OR alias.

    Returns ``(genre_id, canonical_name)`` — always the canonical name, even when
    the hit came from an alias, so a pick is always a real taxonomy genre. Ordered
    canonical-name matches first, then most-specific first. Deduped by genre_id.
    """
    q = (query or "").strip()
    if not q:
        return []
    rows = conn.execute(
        "SELECT g.genre_id, g.name, MIN(m.via_alias) AS via_alias "
        "FROM ("
        "  SELECT genre_id, 0 AS via_alias FROM genre_graph_canonical_genres "
        "   WHERE LOWER(name) LIKE '%' || LOWER(?) || '%' "
        "  UNION ALL "
        "  SELECT canonical_genre_id AS genre_id, 1 AS via_alias FROM genre_graph_aliases "
        "   WHERE LOWER(alias) LIKE '%' || LOWER(?) || '%' "
        ") m "
        "JOIN genre_graph_canonical_genres g ON g.genre_id = m.genre_id "
        "WHERE g.status = 'active' "
        "GROUP BY g.genre_id, g.name "
        "ORDER BY via_alias ASC, g.specificity_score DESC, g.name ASC "
        "LIMIT ?",
        (q, q, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def display_genre_names_for_album(conn: sqlite3.Connection, album_id: str) -> list[str]:
    """Published genres for an album_id as deduped display names.

    Mirrors ``display_genre_names_for_track`` but keyed directly by album_id
    (the edit dialog seeds its chips from this).
    """
    rows = conn.execute(
        "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) "
        "FROM release_effective_genres reg "
        "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
        "WHERE reg.album_id = ? ORDER BY reg.assignment_layer, reg.genre_id",
        (str(album_id),),
    ).fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for _gid, name in rows:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


@dataclass(frozen=True)
class ArtistGenreTag:
    genre_id: str
    name: str
    release_count: int
    max_confidence: float


def resolved_genres_for_artist(
    conn: sqlite3.Connection, artist_name: str
) -> list[ArtistGenreTag]:
    """Published observed-leaf/legacy genres across an artist's releases.

    Feeds the tag-steering chips. The input comes from the artist autocomplete,
    which reads ``tracks.artist`` — so an exact case-insensitive match on the
    same column is the correct key (no substring matching). ``inferred_family``
    rows are excluded: hub families carry no steering signal (hub-saturation
    incident 2026-06-12). Ordered strongest-first by (release_count,
    max_confidence). Returns [] for unknown artists or when the authority
    tables are absent — callers render an empty chip row, they don't crash.
    """
    name = (artist_name or "").strip()
    if not name:
        return []
    try:
        rows = conn.execute(
            "SELECT reg.genre_id, COALESCE(g.name, reg.genre_id) AS display_name, "
            "       COUNT(DISTINCT reg.album_id) AS n_releases, "
            "       MAX(reg.confidence) AS max_conf "
            "FROM release_effective_genres reg "
            "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
            "WHERE reg.assignment_layer IN ('observed_leaf', 'legacy') "
            "  AND reg.album_id IN ("
            "      SELECT DISTINCT album_id FROM tracks "
            "      WHERE LOWER(TRIM(artist)) = LOWER(TRIM(?)) "
            "        AND album_id IS NOT NULL"
            "  ) "
            "GROUP BY reg.genre_id "
            "ORDER BY n_releases DESC, max_conf DESC, display_name ASC",
            (name,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [ArtistGenreTag(r[0], r[1], int(r[2]), float(r[3])) for r in rows]


def on_tag_track_ids_for_artist(
    conn: sqlite3.Connection, artist_name: str, genre_ids: set[str]
) -> dict[str, int]:
    """The seed artist's tracks whose album is published (observed_leaf/legacy) with ANY
    of ``genre_ids``, mapped to the count of distinct selected genres that album carries
    (for multi-tag ranking). Union semantics. Same layer + artist-match filter as
    ``resolved_genres_for_artist`` (the chip source), so 'on-tag' == 'would show this chip'.
    Returns {} for empty inputs / unknown artist / absent tables — callers fall back, never crash.
    """
    name = (artist_name or "").strip()
    gids = {str(g) for g in (genre_ids or set()) if str(g)}
    if not name or not gids:
        return {}
    ph = ",".join("?" for _ in gids)
    try:
        rows = conn.execute(
            f"SELECT t.track_id, COUNT(DISTINCT reg.genre_id) AS hits "
            f"FROM tracks t JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            f"WHERE LOWER(TRIM(t.artist)) = LOWER(TRIM(?)) "
            f"  AND reg.genre_id IN ({ph}) "
            f"  AND reg.assignment_layer IN ('observed_leaf', 'legacy') "
            f"GROUP BY t.track_id",
            (name, *gids),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): int(r[1]) for r in rows}
