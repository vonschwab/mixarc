"""Manually-marked live albums (data/live_albums.yaml).

A mark makes a track lose version-preference head-to-heads (-30, same
magnitude as the keyword live detector, non-stacking) and — behind the
exclude_live toggle — bans it from the candidate pool when a non-live
version of the same song exists. Mirrors artist_aliases.py: yaml file,
lru_cache(1) active registry, explicit clear_cache after save.
Design: docs/superpowers/specs/2026-08-10-live-album-marking-design.md.
"""
from __future__ import annotations

import datetime
import logging
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import yaml

from src.metadata_client import _normalize_album_key
from src.string_utils import normalize_artist_key

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "live_albums.yaml"


@dataclass(frozen=True)
class LiveAlbumRegistry:
    entries: Tuple[dict, ...] = ()
    _by_artist: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    def album_keys_for(self, artist_key: str) -> FrozenSet[str]:
        return self._by_artist.get(str(artist_key or "").strip(), frozenset())

    def is_live(self, artist_key: str, album: str) -> bool:
        return _normalize_album_key(album or "") in self.album_keys_for(artist_key)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_artist.values())


EMPTY_REGISTRY = LiveAlbumRegistry()


def build_registry(entries: List[dict]) -> LiveAlbumRegistry:
    by_artist: Dict[str, set] = {}
    for e in entries or []:
        ak = normalize_artist_key(str(e.get("artist") or ""))
        alk = _normalize_album_key(str(e.get("album") or ""))
        if ak and alk:
            by_artist.setdefault(ak, set()).add(alk)
    return LiveAlbumRegistry(
        entries=tuple(entries or []),
        _by_artist={k: frozenset(v) for k, v in by_artist.items()},
    )


def _validate(entries: List[dict]) -> List[str]:
    errors = []
    for i, e in enumerate(entries or []):
        if not str(e.get("artist") or "").strip():
            errors.append(f"entry {i}: missing artist")
        if not str(e.get("album") or "").strip():
            errors.append(f"entry {i}: missing album")
    return errors


def read_live_albums(path: Optional[Path] = None) -> List[dict]:
    p = Path(path) if path is not None else _DEFAULT_PATH
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return list(data.get("albums") or [])
    except Exception as exc:  # malformed yaml must not gate generation
        logger.warning("live_albums: failed to parse %s (%s) — treating as empty", p, exc)
        return []


def save_live_albums(entries: List[dict], path: Optional[Path] = None) -> None:
    errors = _validate(entries)
    if errors:
        raise ValueError("; ".join(errors))
    p = Path(path) if path is not None else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(p, p.with_name(f"{p.name}.bak.{ts}"))
    payload = {"version": 1, "albums": list(entries)}
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    clear_cache()


@lru_cache(maxsize=1)
def _cached_load(path_str: str) -> LiveAlbumRegistry:
    return build_registry(read_live_albums(Path(path_str)))


def get_active_registry() -> LiveAlbumRegistry:
    if _testing_override is not None:
        return _testing_override
    return _cached_load(str(_DEFAULT_PATH))


_testing_override: Optional[LiveAlbumRegistry] = None


def set_live_albums_for_testing(entries_or_none) -> None:
    global _testing_override
    if entries_or_none is None:
        _testing_override = None
    elif isinstance(entries_or_none, LiveAlbumRegistry):
        _testing_override = entries_or_none
    else:
        _testing_override = build_registry(list(entries_or_none))
    clear_cache()


def clear_cache() -> None:
    _cached_load.cache_clear()
