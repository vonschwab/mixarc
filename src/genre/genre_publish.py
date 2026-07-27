"""Publish authoritative layered genres from the enrichment sidecar into metadata.db.

See docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from src.ai_genre_enrichment.layered_assignment import classify_layered_term
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
from src.ai_genre_enrichment.normalization import make_release_key

try:
    from src.genre.normalize import normalize_and_split_genre
    _NORMALIZE_AVAILABLE = True
except Exception:  # pragma: no cover - normalization optional
    _NORMALIZE_AVAILABLE = False

logger = logging.getLogger(__name__)

_WEIGHT_TRACK = 1.0
_WEIGHT_ALBUM = 0.8
_WEIGHT_ARTIST = 0.5


def _split(raw: str) -> list[str]:
    if not raw or raw == "__EMPTY__":
        return []
    if _NORMALIZE_AVAILABLE:
        return [t for t in normalize_and_split_genre(raw) if t]
    token = raw.strip().casefold()
    return [token] if token else []

# Taxonomy + authority DDL mirrors src/ai_genre_enrichment/storage.py so the
# published tables are schema-faithful copies. Authority tables add `album_id`.
_PUBLISHED_DDL = """
CREATE TABLE IF NOT EXISTS genre_graph_canonical_genres (
    genre_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    specificity_score REAL NOT NULL,
    status TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_aliases (
    alias TEXT PRIMARY KEY,
    canonical_genre_id TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_edges (
    source_genre_id TEXT NOT NULL,
    target_genre_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id, edge_type)
);
CREATE TABLE IF NOT EXISTS genre_graph_canonical_facets (
    facet_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    facet_type TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_bridge_rules (
    source_genre_id TEXT NOT NULL,
    target_genre_id TEXT NOT NULL,
    required_family_min REAL NOT NULL,
    required_facet_overlap REAL NOT NULL,
    required_sonic_similarity REAL NOT NULL,
    required_transition_quality REAL NOT NULL,
    mode_allowed TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (source_genre_id, target_genre_id)
);
CREATE TABLE IF NOT EXISTS genre_graph_rejected_terms (
    term TEXT PRIMARY KEY,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_taxonomy_meta (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
    version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    published_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS genre_graph_release_genre_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
    artist TEXT NOT NULL,
    album TEXT NOT NULL,
    genre_id TEXT NOT NULL,
    assignment_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_reliability REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    rejected_by_user INTEGER NOT NULL DEFAULT 0,
    provenance_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (release_id, genre_id, assignment_layer)
);
CREATE TABLE IF NOT EXISTS genre_graph_release_facet_assignments (
    release_id TEXT NOT NULL,
    album_id TEXT,
    artist TEXT NOT NULL,
    album TEXT NOT NULL,
    facet_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (release_id, facet_id, source)
);
CREATE TABLE IF NOT EXISTS release_effective_genres (
    album_id TEXT NOT NULL,
    release_key TEXT,
    genre_id TEXT NOT NULL,
    assignment_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (album_id, genre_id, assignment_layer)
);
CREATE INDEX IF NOT EXISTS idx_release_effective_genres_album
    ON release_effective_genres (album_id);
"""

# Tables this sub-project owns. Order matters for DROP (children first is N/A
# since FKs are not enforced, but keep a stable list for unpublish()).
# Note: idx_release_effective_genres_album is intentionally absent — SQLite
# drops indexes automatically when their parent table is dropped.
PUBLISHED_TABLES = [
    "release_effective_genres",
    "genre_graph_release_genre_assignments",
    "genre_graph_release_facet_assignments",
    "genre_graph_taxonomy_meta",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
    "genre_graph_canonical_facets",
    "genre_graph_canonical_genres",
]


def create_published_schema(conn: sqlite3.Connection) -> None:
    """Create all published genre tables in metadata.db (idempotent).

    Executes statements individually (NOT executescript, which would implicitly
    COMMIT and break the publish() dry-run rollback). The DDL contains no
    embedded semicolons, so a simple split is safe.
    """
    for stmt in _PUBLISHED_DDL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TAXONOMY_COPY_TABLES = [
    "genre_graph_canonical_genres",
    "genre_graph_canonical_facets",
    "genre_graph_edges",
    "genre_graph_aliases",
    "genre_graph_bridge_rules",
    "genre_graph_rejected_terms",
]


def _taxonomy_fingerprint(conn: sqlite3.Connection) -> str:
    """Stable hash of the published taxonomy (genres + edges)."""
    genres = conn.execute(
        "SELECT genre_id, name, kind, specificity_score, status "
        "FROM genre_graph_canonical_genres ORDER BY genre_id"
    ).fetchall()
    edges = conn.execute(
        "SELECT source_genre_id, target_genre_id, edge_type, weight "
        "FROM genre_graph_edges ORDER BY source_genre_id, target_genre_id, edge_type"
    ).fetchall()
    payload = json.dumps(
        {"genres": [tuple(r) for r in genres], "edges": [tuple(r) for r in edges]},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_taxonomy(conn: sqlite3.Connection) -> None:
    """Copy taxonomy tables from attached `side` DB; rewrite taxonomy_meta.

    Requires the sidecar attached as schema `side`.
    """
    for table in _TAXONOMY_COPY_TABLES:
        conn.execute(f"DELETE FROM {table}")
        conn.execute(f"INSERT INTO {table} SELECT * FROM side.{table}")
    version_row = conn.execute(
        "SELECT taxonomy_version FROM genre_graph_canonical_genres LIMIT 1"
    ).fetchone()
    version = version_row[0] if version_row else "unknown"
    conn.execute("DELETE FROM genre_graph_taxonomy_meta")
    conn.execute(
        "INSERT INTO genre_graph_taxonomy_meta (version, fingerprint, published_at) "
        "VALUES (?, ?, ?)",
        (version, _taxonomy_fingerprint(conn), _now_iso()),
    )


def resolve_release_key_to_album_id(
    conn: sqlite3.Connection,
) -> tuple[dict[str, str], int]:
    """Build release_key -> album_id. Signatures win; albums recompute fills gaps.

    Returns (mapping, collision_count). Requires sidecar attached as `side`.
    """
    mapping: dict[str, str] = {}

    # 1) exact from signatures
    for row in conn.execute(
        "SELECT release_key, album_id FROM side.enriched_genre_signatures "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        mapping[row[0]] = row[1]

    # 2) recompute from albums for keys not already mapped
    collisions = 0
    computed: dict[str, str] = {}
    for album_id, title, artist in conn.execute(
        "SELECT album_id, title, artist FROM albums "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        key = make_release_key(artist, title)
        if not key or key == "::":
            continue
        if key in computed and computed[key] != album_id:
            collisions += 1
            # deterministic: keep the lexicographically smaller album_id
            computed[key] = min(computed[key], album_id)
        else:
            computed[key] = album_id
    for key, album_id in computed.items():
        mapping.setdefault(key, album_id)

    # 3) recompute from tracks for album_ids absent from `albums` (orphans).
    #    Tracks carry the real album_id; an album row may never have been
    #    created (e.g. a double-space artist string). Genre edits on these
    #    must still publish, so derive their release_key -> album_id here.
    #    Best-effort: a reduced `tracks` schema without artist/album columns
    #    (some test fixtures) just skips orphan recovery rather than failing.
    track_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
    if {"album", "artist", "album_id"} <= track_cols:
        for album_id, artist, album in conn.execute(
            "SELECT DISTINCT album_id, artist, album FROM tracks "
            "WHERE album_id IS NOT NULL AND album_id != ''"
        ):
            key = make_release_key(artist, album)
            if not key or key == "::":
                continue
            mapping.setdefault(key, album_id)

    return mapping, collisions


def release_key_to_album_ids(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Build release_key -> ALL album_ids that share it (1:many).

    A logical release is frequently fragmented across multiple album_ids: feat./
    collaboration variants ("Flying Lotus" vs "Flying Lotus feat. X") and duplicate
    imports all normalize to one release_key. Graph genres are computed once per
    release_key, so every fragment must receive them — the single-winner mapping in
    ``resolve_release_key_to_album_id`` reconnects only one album_id and silently drops
    the siblings to legacy (root cause of the fragmented "un-enriched" albums,
    2026-06-25 audit). Requires sidecar attached as `side`.
    """
    acc: dict[str, set[str]] = defaultdict(set)

    for release_key, album_id in conn.execute(
        "SELECT release_key, album_id FROM side.enriched_genre_signatures "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        if release_key:
            acc[release_key].add(album_id)

    for album_id, title, artist in conn.execute(
        "SELECT album_id, title, artist FROM albums "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        key = make_release_key(artist, title)
        if key and key != "::":
            acc[key].add(album_id)

    track_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
    if {"album", "artist", "album_id"} <= track_cols:
        for album_id, artist, album in conn.execute(
            "SELECT DISTINCT album_id, artist, album FROM tracks "
            "WHERE album_id IS NOT NULL AND album_id != ''"
        ):
            key = make_release_key(artist, album)
            if key and key != "::":
                acc[key].add(album_id)

    return {key: sorted(aids) for key, aids in acc.items()}


def populate_authority(conn: sqlite3.Connection, key_to_album: dict[str, str]) -> None:
    """Copy graph genre + facet assignments into metadata.db, stamping album_id.

    The published assignment table is keyed by ``release_id`` (one row per
    release/genre/layer), so it carries a single representative album_id. The 1:many
    fan-out to every fragment of a release happens downstream in ``build_resolved_table``
    via release_key. Requires sidecar attached as `side`. Pure graph (no overrides here).
    """
    conn.execute("DELETE FROM genre_graph_release_genre_assignments")
    conn.execute("DELETE FROM genre_graph_release_facet_assignments")

    for row in conn.execute(
        "SELECT release_id, artist, album, genre_id, assignment_layer, confidence, "
        "source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at "
        "FROM side.genre_graph_release_genre_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_genre_assignments "
            "(release_id, album_id, artist, album, genre_id, assignment_layer, confidence, "
            " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5],
             row[6], row[7], row[8], row[9], row[10]),
        )

    for row in conn.execute(
        "SELECT release_id, artist, album, facet_id, confidence, source, "
        "provenance_json, updated_at FROM side.genre_graph_release_facet_assignments"
    ).fetchall():
        album_id = key_to_album.get(row[0])
        conn.execute(
            "INSERT INTO genre_graph_release_facet_assignments "
            "(release_id, album_id, artist, album, facet_id, confidence, source, "
            " provenance_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (row[0], album_id, row[1], row[2], row[3], row[4], row[5], row[6], row[7]),
        )


def legacy_genres_by_album(
    conn: sqlite3.Connection, album_id: str | None = None, taxonomy=None
) -> dict[str, list[tuple[str, float]]]:
    """Album-grain legacy genres: track(1.0)+album(0.8)+artist(0.5), max weight/token.

    Mirrors the weighting scheme from ``load_genres_for_tracks`` and normalises
    raw genre strings via ``normalize_and_split_genre``.  Returns a mapping of
    ``album_id`` → sorted list of ``(token, weight)`` pairs.  Albums whose only
    genre data is the ``__EMPTY__`` sentinel are excluded from the result.

    When ``album_id`` is given, the scan is restricted to that one album (used
    by the single-release edit path); otherwise the whole library is scanned.

    With ``taxonomy``, tokens are mapped to canonical genre_ids and anything the
    taxonomy rejects is dropped. Without it the raw token is kept, which is what
    every caller did until 2026-07-27 — and why ``release_effective_genres``
    carried ids like ``post-rock`` (canonical: ``post_rock``), ``world``
    (``world_music``) and ``surf`` (``surf_rock``), plus non-genres like
    ``special purpose artist``. Those raw strings became their own dimensions in
    the artifact's genre vocabulary. 46 rows across 25 albums were affected.
    """
    acc: dict[str, dict[str, float]] = defaultdict(dict)

    def add(aid: str, raw: str, base_weight: float) -> None:
        tokens = _split(raw)
        if not tokens:
            return
        per = base_weight / len(tokens)
        for tok in tokens:
            key = tok
            if taxonomy is not None:
                key = _term_to_genre_id(taxonomy, tok)
                if key is None:  # rejected / non-genre / unmappable
                    continue
            if per > acc[aid].get(key, 0.0):
                acc[aid][key] = per

    track_sql = (
        "SELECT t.album_id, tg.genre FROM tracks t "
        "JOIN track_genres tg ON tg.track_id = t.track_id "
        "WHERE t.album_id IS NOT NULL AND t.album_id != ''"
    )
    album_sql = (
        "SELECT album_id, genre FROM album_genres "
        "WHERE album_id IS NOT NULL AND album_id != '' AND genre != '__EMPTY__'"
    )
    artist_sql = (
        "SELECT a.album_id, ag.genre FROM albums a "
        "JOIN artist_genres ag ON ag.artist = a.artist "
        "WHERE a.album_id IS NOT NULL AND a.album_id != '' AND ag.genre != '__EMPTY__'"
    )
    track_params: tuple = ()
    album_params: tuple = ()
    artist_params: tuple = ()
    if album_id is not None:
        track_sql += " AND t.album_id = ?"
        album_sql += " AND album_id = ?"
        artist_sql += " AND a.album_id = ?"
        track_params = album_params = artist_params = (album_id,)

    for aid, genre in conn.execute(track_sql, track_params):
        add(aid, genre, _WEIGHT_TRACK)
    for aid, genre in conn.execute(album_sql, album_params):
        add(aid, genre, _WEIGHT_ALBUM)
    for aid, genre in conn.execute(artist_sql, artist_params):
        add(aid, genre, _WEIGHT_ARTIST)

    return {aid: sorted(toks.items()) for aid, toks in acc.items() if toks}


def _term_to_genre_id(taxonomy, term: str) -> str | None:
    """Return a graph genre_id for a term, or None if unmappable/non-genre."""
    classification = classify_layered_term(taxonomy, term)
    if classification.term_kind in {"reject", "review", "facet", "alias"}:
        if classification.term_kind == "alias" and classification.canonical_id:
            if taxonomy.genre_by_id(classification.canonical_id) is not None:
                return classification.canonical_id
        return None
    return classification.canonical_id


def classify_override_terms(
    taxonomy, add: list[str], remove: list[str]
) -> tuple[list[str], list[str]]:
    """Map override add/remove names to graph genre_ids (unmappable skipped)."""
    add_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in add) if gid]
    remove_ids = [gid for gid in (_term_to_genre_id(taxonomy, t) for t in remove) if gid]
    return list(dict.fromkeys(add_ids)), list(dict.fromkeys(remove_ids))


def _overrides_by_album(conn, key_to_album, taxonomy):
    """album_id -> (add_genre_ids, remove_match) from ai_genre_user_overrides.

    remove_match covers BOTH vocab spaces: graph genre_ids (to drop graph rows)
    AND normalized free-text tokens (to drop legacy rows).
    """
    out: dict[str, tuple[list[str], set[str]]] = {}
    for release_key, add_json, remove_json in conn.execute(
        "SELECT release_key, genres_add_json, genres_remove_json FROM side.ai_genre_user_overrides"
    ):
        album_id = key_to_album.get(release_key)
        if not album_id:
            continue
        add = json.loads(add_json or "[]")
        remove = json.loads(remove_json or "[]")
        add_ids, remove_ids = classify_override_terms(taxonomy, add, remove)
        remove_tokens: set[str] = set()
        for name in remove:
            remove_tokens.update(_split(name))
        out[album_id] = (add_ids, set(remove_ids) | remove_tokens)
    return out


def materialize_album_genres(
    conn,
    album_id: str,
    *,
    graph_album_ids: set[str],
    legacy: dict[str, list[tuple[str, float]]],
    overrides: dict[str, tuple[list[str], set[str]]],
    album_to_key: dict[str, str],
) -> None:
    """Write release_effective_genres rows for one album. Idempotent per album.

    Graph-where-present else legacy, then apply the album's override (drop
    remove_match genre_ids, add user observed_leaf rows). Shared by the full
    publish loop and the single-release edit path so both produce identical
    rows.
    """
    rows: dict[tuple[str, str], tuple[float, str]] = {}
    release_key = album_to_key.get(album_id)
    # Graph genres are stored once per release (keyed by release_id == release_key);
    # read them by release_key so every fragment of a release (feat./collab variants
    # that share the key) resolves to graph — not just the stamped album_id.
    if album_id in graph_album_ids:
        for genre_id, layer, conf in conn.execute(
            "SELECT genre_id, assignment_layer, confidence "
            "FROM genre_graph_release_genre_assignments WHERE release_id = ?",
            (release_key,),
        ):
            rows[(genre_id, layer)] = (conf, "graph")
    elif album_id in legacy:
        for genre_id, weight in legacy[album_id]:
            rows[(genre_id, "legacy")] = (weight, "legacy")

    if album_id in overrides:
        add_ids, remove_match = overrides[album_id]
        rows = {k: v for k, v in rows.items() if k[0] not in remove_match}
        for gid in add_ids:
            rows[(gid, "observed_leaf")] = (1.0, "user")

    conn.execute("DELETE FROM release_effective_genres WHERE album_id = ?", (album_id,))
    for (genre_id, layer), (conf, source) in rows.items():
        conn.execute(
            "INSERT OR REPLACE INTO release_effective_genres "
            "(album_id, release_key, genre_id, assignment_layer, confidence, source) "
            "VALUES (?,?,?,?,?,?)",
            (album_id, release_key, genre_id, layer, conf, source),
        )


def build_resolved_table(conn, key_to_album: dict[str, str], taxonomy) -> None:
    """Build release_effective_genres: graph-where-present else legacy, + overrides."""
    conn.execute("DELETE FROM release_effective_genres")

    # Complete album_id -> release_key for EVERY fragment of a release (feat./collab
    # variants, duplicate imports), not just the single winner in key_to_album, so the
    # graph genres computed once per release_key reach all album_ids that share it.
    album_to_key: dict[str, str] = {}
    for key, aids in release_key_to_album_ids(conn).items():
        for aid in aids:
            album_to_key.setdefault(aid, key)

    assignment_keys = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT release_id FROM genre_graph_release_genre_assignments"
        )
    }
    graph_album_ids = {
        aid for aid, key in album_to_key.items() if key in assignment_keys
    }
    legacy = legacy_genres_by_album(conn, taxonomy=taxonomy)
    overrides = _overrides_by_album(conn, key_to_album, taxonomy)

    all_album_ids = [
        r[0] for r in conn.execute(
            "SELECT album_id FROM albums WHERE album_id IS NOT NULL AND album_id != '' "
            "UNION "
            "SELECT DISTINCT album_id FROM tracks WHERE album_id IS NOT NULL AND album_id != ''"
        )
    ]

    for album_id in all_album_ids:
        materialize_album_genres(
            conn, album_id,
            graph_album_ids=graph_album_ids, legacy=legacy,
            overrides=overrides, album_to_key=album_to_key,
        )


@dataclass
class PublishStats:
    total_albums: int = 0
    graph_albums: int = 0
    legacy_albums: int = 0
    unlinked_releases: int = 0
    collisions: int = 0
    overrides_applied: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def unpublish(conn: sqlite3.Connection) -> None:
    """Drop all published tables. Legacy tables untouched."""
    for table in PUBLISHED_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def _sync_sidecar_taxonomy(sidecar_db: str, taxonomy) -> bool:
    """Re-import the taxonomy into the sidecar graph tables if it's out of date.

    The YAML (`data/layered_genre_taxonomy.yaml`) is the source of truth; the
    sidecar's `genre_graph_*` tables are a materialized copy that ``copy_taxonomy``
    then copies verbatim into metadata.db. Growing the YAML (e.g. via the GUI
    adjudication panel) without re-importing leaves the sidecar stale, so publish
    copies an old canonical table and downstream ids fall back to raw tokens
    (2026-07-02 incident). Syncing here makes every full publish reflect the
    current YAML. Version-gated so it's a no-op when already in sync. Returns
    True iff a re-import was performed.
    """
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(sidecar_db)
    store.initialize()  # idempotent; ensures the graph tables exist
    with store.connect() as sconn:
        row = sconn.execute(
            "SELECT taxonomy_version FROM genre_graph_canonical_genres LIMIT 1"
        ).fetchone()
    if row is not None and row[0] == taxonomy.version:
        return False
    stale = row[0] if row is not None else "(empty)"
    logger.info(
        "publish: sidecar taxonomy %s is stale (YAML is %s); re-importing before copy",
        stale, taxonomy.version,
    )
    store.upsert_layered_taxonomy(taxonomy)
    return True


def publish(metadata_db: str, sidecar_db: str, dry_run: bool = False) -> PublishStats:
    """Publish authoritative genres from sidecar into metadata.db (one transaction)."""
    taxonomy = load_default_layered_taxonomy()
    # Self-heal the sidecar from the YAML before copying it into metadata.db.
    # Skipped on dry_run so a preview mutates nothing.
    resynced = False
    if not dry_run:
        resynced = _sync_sidecar_taxonomy(sidecar_db, taxonomy)
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    # isolation_level=None = autocommit mode. Without this, Python's sqlite3
    # auto-issues BEGIN statements that interfere with explicit ROLLBACK of DDL.
    conn.isolation_level = None
    try:
        # ATTACH must precede BEGIN — SQLite forbids ATTACH inside a transaction.
        conn.execute("ATTACH DATABASE ? AS side", (sidecar_db,))
        conn.execute("BEGIN")
        create_published_schema(conn)
        copy_taxonomy(conn)
        mapping, collisions = resolve_release_key_to_album_id(conn)
        populate_authority(conn, mapping)
        build_resolved_table(conn, mapping, taxonomy)

        graph = {r[0] for r in conn.execute(
            "SELECT DISTINCT album_id FROM release_effective_genres WHERE source='graph'"
        )}
        legacy = {r[0] for r in conn.execute(
            "SELECT DISTINCT album_id FROM release_effective_genres WHERE source='legacy'"
        )}
        total = conn.execute(
            "SELECT COUNT(*) FROM albums WHERE album_id IS NOT NULL AND album_id != ''"
        ).fetchone()[0]
        unlinked = conn.execute(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments WHERE album_id IS NULL"
        ).fetchone()[0]
        overrides_applied = conn.execute(
            "SELECT COUNT(*) FROM release_effective_genres WHERE source='user'"
        ).fetchone()[0]
        stats = PublishStats(
            total_albums=total,
            graph_albums=len(graph),
            legacy_albums=len(legacy),
            unlinked_releases=unlinked,
            collisions=collisions,
            overrides_applied=overrides_applied,
            dry_run=dry_run,
        )
        if dry_run:
            conn.execute("ROLLBACK")
            logger.info(
                "publish: DRY-RUN complete (no changes written) at taxonomy %s; albums: "
                "%d total, %d graph, %d legacy, %d user-override; %d unlinked, %d collisions",
                taxonomy.version, total, len(graph), len(legacy),
                overrides_applied, unlinked, collisions,
            )
        else:
            conn.execute("COMMIT")
            logger.info(
                "publish: SUCCESS — metadata.db committed at taxonomy %s%s; albums: "
                "%d total, %d graph, %d legacy, %d user-override; %d unlinked, %d collisions",
                taxonomy.version,
                " (sidecar re-synced from YAML first)" if resynced else "",
                total, len(graph), len(legacy),
                overrides_applied, unlinked, collisions,
            )
        return stats
    finally:
        try:
            conn.execute("DETACH DATABASE side")
        except sqlite3.OperationalError:
            pass
        conn.close()
