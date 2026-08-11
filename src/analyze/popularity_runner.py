"""Build side of the Last.fm popularity sidecar.

Fetches each artist's top tracks (cached in data/popularity_cache.db so re-runs
skip), resolves each Last.fm track to the *canonical* local track per song
(mbid-first, then loose-title + version-preference), and writes
data/artifacts/beat3tower_32k/popularity/popularity_sidecar.npz aligned to the
artifact's track_ids. Mirrors the energy sidecar. Reads metadata.db read-only.

This cache is deliberately separate from the genre-enrichment DB — it's a
Last.fm artist-popularity cache, not genre data.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

POPULARITY_CACHE_DB_DEFAULT = "data/popularity_cache.db"


def popularity_cache_db_path() -> str:
    """ROOT-anchored absolute path to the popularity cache DB (the per-artist cache)."""
    return str(Path(__file__).resolve().parents[2] / "data" / "popularity_cache.db")


def init_top_tracks_cache(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS artist_top_tracks_cache ("
            "artist_key TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, "
            "track_count INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL)"
        )


def cached_artist_keys(db_path: str) -> set:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT artist_key FROM artist_top_tracks_cache")
        return {r[0] for r in rows}


def upsert_artist_top_tracks(
    db_path: str, artist_key: str, fetched_at: str, top_tracks: List[dict]
) -> None:
    payload = json.dumps(top_tracks)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO artist_top_tracks_cache (artist_key, fetched_at, track_count, payload_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(artist_key) DO UPDATE SET "
            "fetched_at=excluded.fetched_at, track_count=excluded.track_count, "
            "payload_json=excluded.payload_json",
            (artist_key, fetched_at, len(top_tracks), payload),
        )


def get_artist_top_tracks_cached(db_path: str, artist_key: str) -> List[dict]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM artist_top_tracks_cache WHERE artist_key = ?",
            (artist_key,),
        ).fetchone()
    return json.loads(row[0]) if row else []


def resolve_top_tracks_to_rank(
    top_tracks: List[dict], local_tracks: List[dict]
) -> Dict[str, int]:
    """Map one artist's ranked Last.fm top tracks to local track_ids -> 0-based rank.

    mbid-first, else loose-normalized-title grouping with version-preference
    (studio/remaster beat live/demo/alt). On collision keep the LOWER rank (more
    popular). Returns {track_id: 0-based Last.fm rank}.
    """
    if not top_tracks or not local_tracks:
        return {}
    from src.title_dedupe import (
        calculate_version_preference_score,
        normalize_title_for_dedupe,
    )

    def _preference(lt: dict) -> tuple:
        return (
            calculate_version_preference_score(
                str(lt.get("title") or ""), str(lt.get("album") or "")
            ),
            str(lt["track_id"]),
        )

    by_mbid: Dict[str, dict] = {}
    by_norm: Dict[str, List[dict]] = {}
    for lt in local_tracks:
        mbid = str(lt.get("musicbrainz_id") or "")
        if mbid:
            # Sharing an mbid across albums is NOT rare (609 mbids over 2912 tracks
            # in the live library): a live take and the studio cut of the same song
            # routinely carry the same recording id. "First local track wins" let
            # SQLite row order decide, and short-circuited version preference
            # entirely -- the studio version was never scored. Apply the same
            # preference the title branch below uses.
            prev = by_mbid.get(mbid)
            if prev is None or _preference(lt) > _preference(prev):
                by_mbid[mbid] = lt
        norm = normalize_title_for_dedupe(str(lt.get("title") or ""), mode="loose")
        if norm:
            by_norm.setdefault(norm, []).append(lt)

    out: Dict[str, int] = {}
    for t in top_tracks:
        rank = int(t.get("rank", 0))
        tid: Optional[str] = None
        mbid = str(t.get("mbid") or "")
        if mbid and mbid in by_mbid:
            tid = str(by_mbid[mbid]["track_id"])
        else:
            norm = normalize_title_for_dedupe(str(t.get("name") or ""), mode="loose")
            cands = by_norm.get(norm, [])
            if cands:
                tid = str(max(cands, key=_preference)["track_id"])
        if tid is not None and (tid not in out or rank < out[tid]):
            out[tid] = rank
    return out


def resolve_top_tracks_to_popularity(
    top_tracks: List[dict], local_tracks: List[dict]
) -> Dict[str, float]:
    """track_id -> popularity in [0,1] (= 1 - rank/N). See resolve_top_tracks_to_rank."""
    if not top_tracks:
        return {}
    n = len(top_tracks)
    return {
        tid: 1.0 - rank / n
        for tid, rank in resolve_top_tracks_to_rank(top_tracks, local_tracks).items()
    }


def artist_prominence(db_path: str, artist_keys: set) -> Dict[str, float]:
    """Normalized artist_key -> cross-artist prominence in [0, 1].

    Derived from the MAX `listeners` across the artist's cached Last.fm top tracks,
    log-scaled (the raw range spans ~4 orders of magnitude: 1.6M for Slowdive vs
    ~300 for a Bandcamp band) and min-max normalized across the requested set.

    This is the cross-artist signal `1 - rank/N` cannot provide — rank is WITHIN
    artist. See spec 2026-07-24-genre-mode-design.md §3.3.

    An artist absent from the cache is OMITTED from the result, never scored 0.0:
    Last.fm coverage is lazily populated (29% for funk vs 89% for shoegaze), so
    scoring uncached artists at the bottom would silently exclude most of a genre.
    Callers must treat a missing key as "no signal", not "unpopular".
    """
    import math

    keys = {str(k) for k in (artist_keys or set()) if str(k)}
    if not keys:
        return {}
    raw: Dict[str, float] = {}
    for key in keys:
        try:
            top = get_artist_top_tracks_cached(db_path, key)
            listeners = [int(t.get("listeners") or 0) for t in (top or [])]
        except Exception:
            continue
        best = max(listeners) if listeners else 0
        if best > 0:
            raw[key] = math.log10(float(best))
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def log_seed_popularity(
    artist_name: str,
    pier_track_ids: Sequence[str],
    pier_titles: Sequence[str],
    *,
    db_path: str,
    pier_albums: Optional[Sequence[str]] = None,
) -> None:
    """Log each chosen pier's Last.fm popularity rank (diagnostic for Popular Seeds).

    Reads the warm per-artist cache (populated by the lazy fetch during this run)
    and reports where each pier sits on the artist's Last.fm top-N, or that it
    isn't on the list. Never raises.
    """
    from src.string_utils import normalize_artist_key

    try:
        top = get_artist_top_tracks_cached(db_path, normalize_artist_key(artist_name))
    except Exception:  # diagnostics must never break generation
        top = []
    if not top:
        logger.info(
            "Popular Seeds: no cached Last.fm top tracks for %s — piers picked without popularity",
            artist_name,
        )
        return
    n = len(top)
    # Album feeds the live-recording half of version preference; without it a
    # clean-titled live take resolves as if it were the studio cut.
    albums = list(pier_albums or [])
    local = [
        {
            "track_id": str(tid),
            "title": str(title or ""),
            "album": str(albums[i]) if i < len(albums) else "",
            "musicbrainz_id": "",
        }
        for i, (tid, title) in enumerate(zip(pier_track_ids, pier_titles))
    ]
    ranks = resolve_top_tracks_to_rank(top, local)
    logger.info(
        "Popular Seeds: %d/%d piers on %s's Last.fm top-%d (#1 = most popular):",
        len(ranks), len(pier_track_ids), artist_name, n,
    )
    for tid, title in zip(pier_track_ids, pier_titles):
        r = ranks.get(str(tid))
        if r is not None:
            logger.info("    Last.fm #%-3d %s", r + 1, title)
        else:
            logger.info("    (not in top %-4d) %s", n, title)


def _local_tracks_by_artist(metadata_db: str, min_artist_tracks: int) -> Dict[str, List[dict]]:
    by_artist: Dict[str, List[dict]] = {}
    with sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            # `album` is REQUIRED, not decorative: calculate_version_preference_score
            # detects most live recordings from the album, not the (often clean)
            # track title. Selecting only the title left that half of the detector
            # inert on this path, which builds the popularity sidecar.
            "SELECT track_id, title, album, musicbrainz_id, artist_key FROM tracks "
            "WHERE artist_key IS NOT NULL AND artist_key <> ''"
        ):
            by_artist.setdefault(str(r["artist_key"]), []).append({
                "track_id": str(r["track_id"]),
                "title": str(r["title"] or ""),
                "album": str(r["album"] or ""),
                "musicbrainz_id": str(r["musicbrainz_id"] or ""),
            })
    return {k: v for k, v in by_artist.items() if len(v) >= min_artist_tracks}


def build_popularity_sidecar(
    *, artifact_npz: str, metadata_db: str, enrichment_db: str,
    out_path: str, min_artist_tracks: int,
) -> dict:
    """Resolve cached Last.fm top tracks to local track_ids and write the sidecar."""
    tids = [str(t) for t in np.load(artifact_npz, allow_pickle=True)["track_ids"]]
    pos = {t: i for i, t in enumerate(tids)}
    popularity = np.full(len(tids), np.nan, dtype=np.float32)

    by_artist = _local_tracks_by_artist(metadata_db, min_artist_tracks)
    cached = cached_artist_keys(enrichment_db)
    matched = artists_resolved = 0
    for artist_key, local_tracks in by_artist.items():
        if artist_key not in cached:
            continue
        top = get_artist_top_tracks_cached(enrichment_db, artist_key)
        if not top:
            continue
        artists_resolved += 1
        for tid, score in resolve_top_tracks_to_popularity(top, local_tracks).items():
            j = pos.get(tid)
            if j is not None:
                popularity[j] = score
                matched += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, track_ids=np.array(tids, dtype=object), popularity=popularity,
    )
    logger.info("popularity sidecar: %d tracks, %d matched, %d artists resolved -> %s",
                len(tids), matched, artists_resolved, out_path)
    return {"tracks": len(tids), "matched": matched, "artists_resolved": artists_resolved}


def load_artist_popularity_values(
    bundle, artist_name: str, *, client, db_path: str, limit: int,
    max_age_days: int, now_iso: str, include_collaborations: bool = False,
    metadata_db_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Per-track popularity for the seed artist, aligned to bundle.track_ids.

    Lazy cache-first fetch of the seed artist's top tracks + resolve to the
    artist's local tracks (title-based; mbid blank from the bundle). None if no
    client. NaN for non-matched / other-artist tracks (neutral)."""
    if client is None:
        return None
    from src.playlist.artist_style import (
        _artist_indices_in_bundle, _dedupe_artist_indices, _load_albums_for_indices,
    )
    from src.string_utils import normalize_artist_key

    indices = _artist_indices_in_bundle(
        bundle, artist_name, include_collaborations=include_collaborations)
    if not indices:
        return None
    titles = getattr(bundle, "track_titles", None)
    # Album is needed so version-preference can demote a clean-titled track on a live
    # album (e.g. "Corpse Pose" on "Live Leaves") below the studio version.
    _albums = _load_albums_for_indices(bundle, indices, metadata_db_path) if metadata_db_path else {}
    # Resolve popularity against the SAME canonical (deduped) versions the piers come from,
    # using the identical dedup as cluster_artist_tracks. Otherwise, when a song has two
    # equally-preferred versions (e.g. two studio "Corpse Pose" copies on different albums),
    # the resolver's tie-break can land the #1 score on a version the dedup discarded — so the
    # hit carries no score on the surviving version and vanishes from the 🔥 piers.
    canonical_indices = _dedupe_artist_indices(
        indices, titles, getattr(bundle, "durations_ms", None), _albums
    )
    local_tracks = [{
        "track_id": str(bundle.track_ids[i]),
        "title": str(titles[i]) if titles is not None else "",
        "musicbrainz_id": "",
        "album": _albums.get(i, ""),
    } for i in canonical_indices]
    from src.playlist.artist_aliases import resolve_alias, alias_group_member_names
    resolved = resolve_alias(normalize_artist_key(artist_name))
    names = alias_group_member_names(resolved) or [artist_name]
    top: list = []
    for nm in names:
        top.extend(get_artist_top_tracks_cached_or_fetch(
            normalize_artist_key(nm), nm, client=client, db_path=db_path,
            limit=limit, max_age_days=max_age_days, now_iso=now_iso))
    pop = resolve_top_tracks_to_popularity(top, local_tracks)
    if not pop:
        return None
    out = np.full(len(bundle.track_ids), np.nan, dtype=float)
    pos = {str(t): i for i, t in enumerate(bundle.track_ids)}
    for tid, score in pop.items():
        j = pos.get(tid)
        if j is not None:
            out[j] = score
    return out


def load_pool_popularity_values(
    bundle,
    artist_name_by_key,
    *,
    client,
    db_path: str,
    limit: int = 50,
    max_age_days: int = 30,
    now_iso: Optional[str] = None,
    metadata_db_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Per-track popularity (1 - rank/N) for every artist in the candidate pool,
    aligned to bundle.track_ids. NaN where unknown.

    `artist_name_by_key` maps normalized artist_key -> Last.fm display name and
    restricts the scan to the pool's distinct artists. Each artist's top tracks
    are fetched cache-first + TTL and resolved to that artist's bundle rows.
    `metadata_db_path` supplies albums so version preference can demote a
    clean-titled live take; without it that half of the detector is inert, as it
    silently was here until 2026-08-10. Returns None if `client` is None. Never
    raises (a failed fetch leaves that artist's tracks NaN, never gating
    generation).

    NOTE: no production caller today -- `load_pool_popularity_values_cached` is
    what the live path uses. Kept in step with it so the two cannot diverge if
    this one is ever wired up.
    """
    if client is None:
        return None
    track_ids = bundle.track_ids
    titles = getattr(bundle, "track_titles", None)
    keys = getattr(bundle, "artist_keys", None)
    if keys is None:
        return None
    out = np.full(len(track_ids), np.nan, dtype=float)
    rows_by_key: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        k = str(k)
        if k in artist_name_by_key:
            rows_by_key.setdefault(k, []).append(i)
    # Albums so version preference demotes clean-titled live-album tracks.
    from src.playlist.artist_style import _load_albums_for_indices
    _albums = (
        _load_albums_for_indices(
            bundle, [i for idxs in rows_by_key.values() for i in idxs], metadata_db_path
        )
        if metadata_db_path else {}
    )
    for key, idxs in rows_by_key.items():
        try:
            top = get_artist_top_tracks_cached_or_fetch(
                key, artist_name_by_key[key], client=client, db_path=db_path,
                limit=limit, max_age_days=max_age_days, now_iso=now_iso)
        except Exception:  # belt-and-suspenders: never gate generation
            top = []
        if not top:
            continue
        local = [{
            "track_id": str(track_ids[i]),
            "title": str(titles[i]) if titles is not None else "",
            "musicbrainz_id": "",
            "album": _albums.get(i, ""),
        } for i in idxs]
        ranks = resolve_top_tracks_to_rank(top, local)
        n = len(top)
        pos = {str(track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            j = pos.get(tid)
            if j is not None:
                out[j] = 1.0 - rank / n
    return out


def load_pool_popularity_values_cached(
    bundle, pool_indices, *, db_path: str, metadata_db_path: Optional[str] = None
) -> np.ndarray:
    """Cache-ONLY per-track popularity for the given bundle pool indices.

    Reads the warm `artist_top_tracks_cache` (no Last.fm fetch — usable where no
    client is in scope, e.g. deep in the pipeline). Artists not in the cache stay
    NaN (and are thus ruthlessly demoted at any positive strength). Returns a
    vector aligned to bundle.track_ids. Never raises."""
    track_ids = bundle.track_ids
    out = np.full(len(track_ids), np.nan, dtype=float)
    keys = getattr(bundle, "artist_keys", None)
    titles = getattr(bundle, "track_titles", None)
    if keys is None:
        return out
    # Album so version-preference demotes clean-titled live-album tracks (e.g. "Live Leaves").
    from src.playlist.artist_style import _load_albums_for_indices
    _albums = _load_albums_for_indices(bundle, [int(i) for i in pool_indices], metadata_db_path) if metadata_db_path else {}
    rows_by_key: Dict[str, List[int]] = {}
    for i in pool_indices:
        i = int(i)
        rows_by_key.setdefault(str(keys[i]), []).append(i)
    for key, idxs in rows_by_key.items():
        try:
            top = get_artist_top_tracks_cached(db_path, key)
        except Exception:  # never gate generation
            top = []
        if not top:
            continue
        local = [{
            "track_id": str(track_ids[i]),
            "title": str(titles[i]) if titles is not None else "",
            "musicbrainz_id": "",
            "album": _albums.get(i, ""),
        } for i in idxs]
        ranks = resolve_top_tracks_to_rank(top, local)
        n = len(top)
        pos = {str(track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            j = pos.get(tid)
            if j is not None:
                out[j] = 1.0 - rank / n
    return out


def load_pool_popularity_ranks_cached(
    bundle, pool_indices, *, db_path: str, metadata_db_path: Optional[str] = None
) -> np.ndarray:
    """Cache-ONLY per-track Last.fm rank (0-based) for the given bundle pool indices.

    Sibling of load_pool_popularity_values_cached, but stores the rank itself — the
    popularity admission gate compares against a rank cutoff (top-10 / top-50), and
    the score 1 - rank/n is not a fixed-rank threshold (n varies per artist). Artists
    not in the warm cache and tracks not in the artist's top-N stay -1. Aligned to
    bundle.track_ids. Never raises."""
    track_ids = bundle.track_ids
    out = np.full(len(track_ids), -1, dtype=int)
    keys = getattr(bundle, "artist_keys", None)
    titles = getattr(bundle, "track_titles", None)
    if keys is None:
        return out
    # Album so version-preference demotes clean-titled live-album tracks (e.g. "Live Leaves").
    from src.playlist.artist_style import _load_albums_for_indices
    _albums = _load_albums_for_indices(bundle, [int(i) for i in pool_indices], metadata_db_path) if metadata_db_path else {}
    rows_by_key: Dict[str, List[int]] = {}
    for i in pool_indices:
        i = int(i)
        rows_by_key.setdefault(str(keys[i]), []).append(i)
    for key, idxs in rows_by_key.items():
        try:
            top = get_artist_top_tracks_cached(db_path, key)
        except Exception:  # never gate generation
            top = []
        if not top:
            continue
        local = [{
            "track_id": str(track_ids[i]),
            "title": str(titles[i]) if titles is not None else "",
            "musicbrainz_id": "",
            "album": _albums.get(i, ""),
        } for i in idxs]
        ranks = resolve_top_tracks_to_rank(top, local)
        pos = {str(track_ids[i]): i for i in idxs}
        for tid, rank in ranks.items():
            j = pos.get(tid)
            if j is not None:
                out[j] = int(rank)
    return out


def annotate_and_log_playlist_popularity(tracks, *, db_path: str) -> None:
    """Annotate each playlist track dict with its Last.fm popularity rank and log it.

    For each track (uses 'artist' + 'title'), looks up the per-artist rank in the
    warm cache, sets `track['popularity_rank']` (1-based, or None if the track is
    not on its artist's top-50 / the artist is uncached), and logs a per-track
    summary. Used when Bangers (popularity_mode) is on. Never raises."""
    from src.string_utils import normalize_artist_key
    from src.title_dedupe import normalize_title_for_dedupe

    rank_maps: Dict[str, Dict[str, int]] = {}

    def _ranks_for(artist_key: str) -> Dict[str, int]:
        if artist_key not in rank_maps:
            m: Dict[str, int] = {}
            try:
                for row in get_artist_top_tracks_cached(db_path, artist_key) or []:
                    norm = normalize_title_for_dedupe(str(row.get("name") or ""), mode="loose")
                    r = int(row.get("rank", 0))
                    if norm and (norm not in m or r < m[norm]):
                        m[norm] = r
            except Exception:
                m = {}
            rank_maps[artist_key] = m
        return rank_maps[artist_key]

    matched = 0
    for t in tracks:
        artist = str(t.get("artist") or "")
        title = str(t.get("title") or "")
        rank: Optional[int] = None
        if artist and title:
            r = _ranks_for(normalize_artist_key(artist)).get(
                normalize_title_for_dedupe(title, mode="loose"))
            if r is not None:
                rank = r + 1  # 1-based for display
        t["popularity_rank"] = rank
        if rank is not None:
            matched += 1
    logger.info(
        "Bangers: %d/%d playlist tracks on their artist's Last.fm top-50 (#1 = most popular):",
        matched, len(tracks))
    for t in tracks:
        r = t.get("popularity_rank")
        tag = f"Last.fm #{r}" if r is not None else "(not on top-50)"
        logger.info("    %-16s %s - %s", tag, str(t.get("artist") or ""), str(t.get("title") or ""))


def _fetched_at_iso(db_path: str, artist_key: str) -> Optional[str]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at FROM artist_top_tracks_cache WHERE artist_key = ?",
            (artist_key,),
        ).fetchone()
    return row[0] if row else None


def get_artist_top_tracks_cached_or_fetch(
    artist_key: str, artist_name: str, *, client, db_path: str,
    limit: int = 50, max_age_days: int = 30, now_iso: str,
) -> List[dict]:
    """Cache-first per-artist top tracks. Fresh cache -> no network. Miss/stale ->
    one fetch + cache. Fetch failure -> stale cache if any, else []. Never raises."""
    init_top_tracks_cache(db_path)
    cached = get_artist_top_tracks_cached(db_path, artist_key)
    fetched_at = _fetched_at_iso(db_path, artist_key)
    fresh = False
    if fetched_at is not None:
        try:
            age = datetime.fromisoformat(now_iso) - datetime.fromisoformat(fetched_at)
            fresh = age.total_seconds() <= max_age_days * 86400
        except (ValueError, TypeError):
            # Bad/unparseable timestamp OR aware-vs-naive mismatch -> treat as stale
            # and refetch. This block must never raise (never gate generation).
            fresh = False
    # A real artist.getTopTracks response carries MusicBrainz IDs. A cached payload
    # where EVERY entry has a blank mbid is the hallmark of a transient/ambiguous
    # Last.fm response (this exact bug served deep-cut seeds for a month) -> distrust
    # it and re-fetch even within the TTL. Only the seed artist is checked per
    # generation, so the cost is at most one extra Last.fm call.
    cached_all_blank = bool(cached) and not any((t.get("mbid") or "") for t in cached)
    if fetched_at is not None and fresh and not cached_all_blank:
        return cached
    try:
        rows = client.get_artist_top_tracks(artist_name, limit=limit)
        upsert_artist_top_tracks(db_path, artist_key, now_iso, rows)
        return rows
    except Exception as exc:  # network/parse — never gate generation
        logger.warning(
            "popularity lazy fetch failed for %s: %s; using stale/empty", artist_name, exc
        )
        return cached  # stale cache if present, else []
