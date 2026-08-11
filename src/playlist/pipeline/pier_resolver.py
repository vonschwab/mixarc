"""Pier seed resolution + candidate pool dedupe.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

Two responsibilities:
  * resolve_pier_seeds: map anchor seed track_ids to bundle indices, dedupe by
    (artist, title) track key, ensure the primary seed is present.
  * dedupe_pool_by_track_key: collapse the candidate pool to one canonical
    track per (artist, normalized title) group, picking the highest
    version-preference score.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.features.artifacts import ArtifactBundle
from src.playlist.identity_keys import identity_keys_for_index
from src.playlist.live_albums import LiveAlbumRegistry
from src.title_dedupe import calculate_version_preference_score

logger = logging.getLogger(__name__)


def resolve_pier_seeds(
    bundle: ArtifactBundle,
    seed_idx: int,
    anchor_seed_ids: List[str],
) -> Tuple[List[int], List[str]]:
    """Resolve anchor seed track_ids to indices, dedupe, ensure primary seed.

    Returns ``(pier_seed_indices, seed_track_ids_for_pier)`` where
    ``seed_track_ids_for_pier`` is the string-keyed list the pier-bridge
    builder expects.
    """
    pier_seed_indices: List[int] = []
    not_found: List[str] = []
    for sid in anchor_seed_ids:
        idx = bundle.track_id_to_index.get(str(sid))
        if idx is not None:
            pier_seed_indices.append(idx)
        else:
            not_found.append(sid)
    if not_found:
        logger.warning(
            "Pier seeds NOT FOUND in bundle (%d/%d): %s",
            len(not_found),
            len(anchor_seed_ids),
            not_found[:3],
        )
    else:
        logger.info("All %d pier seeds found in bundle", len(anchor_seed_ids))

    # Deduplicate seeds by (artist, title) track key (keep first occurrence).
    pier_seed_indices = list(dict.fromkeys(pier_seed_indices))
    if bundle.track_titles is not None:
        seen_track_keys: set = set()
        dedupe_indices: List[int] = []
        for idx in pier_seed_indices:
            track_key = identity_keys_for_index(bundle, idx).track_key
            if track_key not in seen_track_keys:
                seen_track_keys.add(track_key)
                dedupe_indices.append(idx)
            else:
                title = bundle.track_titles[idx] or ""
                artist = (
                    bundle.track_artists[idx]
                    if bundle.track_artists is not None
                    else ""
                )
                logger.debug("Removing duplicate pier seed: %s - %s", artist, title)
        pier_seed_indices = dedupe_indices

    # Ensure primary seed is included.
    if seed_idx not in pier_seed_indices:
        should_insert = True
        if bundle.track_titles is not None:
            seed_track_key = identity_keys_for_index(bundle, seed_idx).track_key
            for idx in pier_seed_indices:
                if identity_keys_for_index(bundle, idx).track_key == seed_track_key:
                    should_insert = False
                    break
        if should_insert:
            pier_seed_indices.insert(0, seed_idx)

    seed_labels: List[str] = []
    for idx in pier_seed_indices:
        tid = bundle.track_ids[idx]
        title = bundle.track_titles[idx] if bundle.track_titles is not None else ""
        seed_labels.append(f"{tid}:{title}")
    logger.info("Pier seeds (%d): %s", len(pier_seed_indices), seed_labels)

    seed_track_ids_for_pier = [
        str(bundle.track_ids[idx]) for idx in pier_seed_indices
    ]
    return pier_seed_indices, seed_track_ids_for_pier


def dedupe_pool_by_track_key(
    bundle: ArtifactBundle,
    pool_indices: List[int],
    albums_by_index: Optional[Dict[int, str]] = None,
    live_registry: Optional[LiveAlbumRegistry] = None,
) -> List[int]:
    """Group candidate-pool indices by (artist, normalized title) track key,
    keep the canonical version per group (highest version-preference score).

    No-op when ``bundle.track_titles is None``. This pool is multi-artist, so
    `live_registry` is narrowed PER ROW via that row's own artist_key -- a mark
    on one seed artist's album must never penalize another artist's same-named
    album.
    """
    if bundle.track_titles is None:
        return list(pool_indices)

    key_to_indices: Dict[tuple, List[int]] = {}
    for idx in pool_indices:
        key = identity_keys_for_index(bundle, int(idx)).track_key
        key_to_indices.setdefault(key, []).append(idx)

    deduped: List[int] = []
    for key, indices in key_to_indices.items():
        if len(indices) == 1:
            deduped.append(indices[0])
            continue
        # Score each candidate for version preference; higher wins.
        best_idx = indices[0]
        best_score = -1
        # key == (artist_key, title_key) -- key[0] is this group's artist_key already.
        live_keys = live_registry.album_keys_for(key[0]) if live_registry else None
        for idx in indices:
            title = str(bundle.track_titles[idx]) if bundle.track_titles is not None else ""
            album = (albums_by_index or {}).get(int(idx), "")
            score = calculate_version_preference_score(
                title, album, live_album_keys=live_keys
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        deduped.append(best_idx)
        logger.debug(
            "Pool dedupe: kept idx=%d for key=%s (from %d versions)",
            best_idx, key, len(indices),
        )
    logger.info(
        "Pier bridge candidate pool deduped: %d → %d tracks",
        len(pool_indices), len(deduped),
    )
    return deduped
