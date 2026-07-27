#!/usr/bin/env python3
"""Dry-run report: sibling editions of one release carrying divergent authority genres.

Motivation
----------
A 2026-07-27 ListenBrainz study (MUSIC_DIGEST_MVP handoff) found that its single
highest-value genre result was not discovery of new labels but *edition
consistency repair*: 7 of 13 conservative nominations were genres the local
authority already carried on a **different local album of the same release**.
That class of defect is entirely local — it needs no external service to detect
and no network call to fix.

Cause: ``make_release_key`` keeps edition decoration in the key
("electric ladyland 50th" vs "electric ladyland reprise 6307 2 2de first
edition without noise reduction"), so publish never links the editions and each
one keeps whatever tags its own evidence happened to produce.

This script is READ-ONLY. It opens metadata.db with ``mode=ro``, writes nothing
back, and emits a deterministic markdown + JSON report under docs/run_audits/.
Nothing here decides an authority change; it produces a review queue.

Two evidence channels
---------------------
``title``      Albums whose (normalized artist, edition-stripped title) match.
               These are repair candidates.
``recording``  Album pairs linked by shared MusicBrainz recording IDs but NOT
               title-matched. High overlap = a retitled edition; low overlap =
               a compilation or appears-on. Compilations are review-only: a
               label sampler's genre set must never propagate to a studio album.

Recording evidence is tiered by trust. ``mbid_status='ok'`` means *we* resolved
the recording; a bare ``musicbrainz_id`` is an unvalidated file tag. Both are
reported, separately, because a raw MBID is not an identity guarantee.

Usage
-----
    python scripts/report_edition_genre_divergence.py
    python scripts/report_edition_genre_divergence.py --metadata-db data/metadata.db
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ai_genre_enrichment.normalization import (  # noqa: E402
    normalize_release_artist,
    normalize_release_name,
)
from src.genre.authority import (  # noqa: E402
    canonical_genre_names,
    descendant_genre_ids,
    resolved_genres_by_album,
)
from src.playlist.artist_aliases import resolve_alias  # noqa: E402
from src.title_dedupe import normalize_title_for_dedupe  # noqa: E402

# Observed layers only. Inferred rows are derived from these by the graph
# resolver, so a divergence in an inferred row is a symptom, not the defect.
OBSERVED_LAYER_PREFIX_EXCLUDED = "inferred"

# Album-level edition decoration that `normalize_title_for_dedupe` does not
# cover: it targets track titles, so it has no notion of "50th", "reissue",
# "expanded", or a trailing disc marker. Applied AFTER the shared normalizer.
_EDITION_WORDS = (
    r"deluxe|expanded|remaster(?:ed)?|reissue|anniversary|edition|"
    r"collector'?s|special|limited|bonus|mono|stereo|"
    r"\d{1,3}(?:st|nd|rd|th)|"
    r"disc\s*\d+|cd\s*\d+|vol(?:ume)?\s*\d+\s*$"
)
_EDITION_SUFFIX_RE = re.compile(rf"\b(?:{_EDITION_WORDS})\b", re.IGNORECASE)

# Pairwise recording overlap at or above this fraction reads as the same
# tracklist under a different title (a retitled edition) rather than a
# compilation that merely shares a track.
EDITION_OVERLAP_THRESHOLD = 0.60

# Below this, two albums share few recordings. That is NOT evidence against
# siblinghood: MusicBrainz mints distinct recording MBIDs per remaster, so
# "Revolver [2009 Mono Remaster]" legitimately shares almost nothing with
# "Revolver". Reported as context only; never used to reject a group.
LOW_OVERLAP_NOTE_CEILING = 0.20

# Sequence markers distinguish *different releases* that the edition stripper
# would otherwise collapse onto one title key: volumes, discs, parts, and
# ordinal album names. `normalize_title_for_dedupe` treats "album", "volume",
# and "disc" as version keywords — correct for track titles, catastrophic here,
# where it merges "Low Level Owl: Volume 1" with "Volume 2" and
# "Suicide (The Second Album)" with "Suicide". Marker mismatch splits the group
# BEFORE any divergence is computed, so a wrong repair is never proposed.
_SEQUENCE_RE = re.compile(
    r"\b(?:vol(?:ume)?|disc|cd|disk|part|pt)\s*\.?\s*(\d+|[ivx]+)\b"
    r"|\b(second|third|fourth|fifth)\s+album\b"
    r"|\b(i{2,3}|iv|v|vi{1,3})\b",
    re.IGNORECASE,
)

# Content type is a second axis the edition stripper flattens. A live album, a
# remix album, and a demos collection are different *records* from the studio
# original, not editions of it — "Stop Making Sense (Live)" and "Hail to the
# Thief (Live Recordings)" both landed beside their studio counterparts before
# this split. Their genre sets legitimately differ and must not be merged.
_CONTENT_TYPE_RE = re.compile(
    r"\b(live|remix(?:es|ed)?|demos?|instrumental|karaoke|acoustic|"
    r"unplugged|soundtrack|score|dub)\b",
    re.IGNORECASE,
)


def edition_stripped_title(album_title: str | None) -> str:
    """Edition-agnostic album title: shared normalizers + album-level decoration.

    Deliberately layered on the repo's existing normalizers rather than a new
    regex pipeline — `normalize_title_for_dedupe` already handles bracketed
    version content and dash suffixes, and `normalize_release_name` already
    defines this project's Unicode punctuation policy.
    """
    if not album_title:
        return ""
    loose = normalize_title_for_dedupe(str(album_title), mode="loose")
    stripped = _EDITION_SUFFIX_RE.sub(" ", loose)
    return normalize_release_name(stripped)


def sequence_marker(album_title: str | None) -> str:
    """Volume/disc/part/ordinal marker read from the RAW title, or "".

    Read before normalization, because `normalize_title_for_dedupe` deletes the
    very tokens that distinguish these releases.
    """
    if not album_title:
        return ""
    markers = {
        " ".join(m.group(0).split()).casefold().replace(".", "").replace("volume", "vol")
        for m in _SEQUENCE_RE.finditer(str(album_title))
    }
    markers |= {m.group(0).casefold().rstrip("s") for m in _CONTENT_TYPE_RE.finditer(str(album_title))}
    return "|".join(sorted(markers))


@dataclass
class Album:
    album_id: str
    artist: str
    title: str
    genres: frozenset[str] = frozenset()
    mbids: frozenset[str] = frozenset()
    validated_mbids: frozenset[str] = frozenset()
    track_count: int = 0

    @property
    def artist_key(self) -> str:
        """Alias-resolved artist identity.

        Grouping on the raw normalized name splits "Jimi Hendrix" from "The Jimi
        Hendrix Experience", which is a *second* fragmentation cause independent
        of edition decoration. `resolve_alias` is the engine's own identity seam
        (data/artist_aliases.yaml); unlinked variants fall through unchanged and
        are surfaced in the alias-candidate section instead.
        """
        return resolve_alias(normalize_release_artist(self.artist))

    @property
    def title_key(self) -> str:
        return edition_stripped_title(self.title)

    @property
    def sequence_marker(self) -> str:
        return sequence_marker(self.title)

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.artist_key, self.title_key, self.sequence_marker)


@dataclass
class Divergence:
    """One album missing one genre that a sibling edition carries."""

    album_id: str
    genre_id: str
    relation: str
    relation_evidence: list[str] = field(default_factory=list)
    carried_by: list[str] = field(default_factory=list)


def load_albums(conn: sqlite3.Connection) -> dict[str, Album]:
    genres_by_album = resolved_genres_by_album(conn)

    mbids: dict[str, set[str]] = defaultdict(set)
    validated: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for album_id, mbid, status in conn.execute(
        "SELECT album_id, musicbrainz_id, mbid_status FROM tracks WHERE album_id IS NOT NULL"
    ):
        counts[album_id] += 1
        if mbid:
            mbids[album_id].add(mbid)
            if status == "ok":
                validated[album_id].add(mbid)

    albums: dict[str, Album] = {}
    for album_id, artist, title in conn.execute(
        "SELECT album_id, artist, title FROM albums ORDER BY album_id"
    ):
        observed = frozenset(
            row.genre_id
            for row in genres_by_album.get(album_id, ())
            if not row.assignment_layer.startswith(OBSERVED_LAYER_PREFIX_EXCLUDED)
        )
        albums[album_id] = Album(
            album_id=album_id,
            artist=artist or "",
            title=title or "",
            genres=observed,
            mbids=frozenset(mbids.get(album_id, ())),
            validated_mbids=frozenset(validated.get(album_id, ())),
            track_count=counts.get(album_id, 0),
        )
    return albums


class RelationClassifier:
    """Taxonomy relation between a candidate genre and an album's existing set.

    ``descendant_genre_ids`` re-reads the cached YAML taxonomy per call, so
    memoize: a run touches a few hundred distinct genres but tens of thousands
    of (album, genre) pairs.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._descendants: dict[str, set[str]] = {}

    def descendants(self, genre_id: str) -> set[str]:
        if genre_id not in self._descendants:
            self._descendants[genre_id] = descendant_genre_ids(self._conn, genre_id)
        return self._descendants[genre_id]

    def classify(self, candidate: str, existing: frozenset[str]) -> tuple[str, list[str]]:
        narrower = sorted((self.descendants(candidate) - {candidate}) & existing)
        if narrower:
            # The album already carries a more specific style under this
            # candidate. Adding it is redundant breadth, not new information.
            return "redundant_broader", narrower
        broader = sorted(g for g in existing if candidate in self.descendants(g) and g != candidate)
        if broader:
            return "specificity_gain", broader
        return "disjoint", []


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def build_title_groups(albums: dict[str, Album]) -> list[list[Album]]:
    groups: dict[tuple[str, str, str], list[Album]] = defaultdict(list)
    for album in albums.values():
        key = album.group_key
        if key[0] and key[1]:
            groups[key].append(album)
    return [
        sorted(members, key=lambda a: a.album_id)
        for _, members in sorted(groups.items())
        if len(members) > 1
    ]


def analyze_group(members: list[Album], classifier: RelationClassifier) -> dict | None:
    """Divergence record for one sibling group, or None if the sets agree."""
    sets = {a.album_id: a.genres for a in members}
    if len({frozenset(s) for s in sets.values()}) <= 1:
        return None

    union = frozenset().union(*sets.values())
    divergences: list[Divergence] = []
    for album in members:
        for genre_id in sorted(union - album.genres):
            relation, evidence = classifier.classify(genre_id, album.genres)
            divergences.append(
                Divergence(
                    album_id=album.album_id,
                    genre_id=genre_id,
                    relation=relation,
                    relation_evidence=evidence,
                    carried_by=sorted(
                        other.album_id for other in members if genre_id in other.genres
                    ),
                )
            )

    # A strict subset is a pure gap: one edition simply has less evidence.
    # Mutual divergence means each side asserts something the other denies,
    # which is a classification question, not a repair.
    non_empty = [s for s in sets.values() if s]
    direction = "mutual"
    if non_empty and any(s == union for s in non_empty):
        direction = "subset"

    pairwise = [
        _overlap(a.mbids, b.mbids)
        for i, a in enumerate(members)
        for b in members[i + 1 :]
    ]
    shared_validated: frozenset[str] = members[0].validated_mbids
    for other in members[1:]:
        shared_validated &= other.validated_mbids
    validated_shared = len(shared_validated)

    # Recording overlap is corroboration, not a gate. A remaster shares no
    # recording MBIDs with its original by MusicBrainz design, so absence of
    # overlap says nothing; presence of it is a positive confirmation.
    measurable = [a for a in members if a.mbids]
    if len(measurable) < 2 or not pairwise:
        grouping = "unverified"
    elif min(pairwise) >= EDITION_OVERLAP_THRESHOLD:
        grouping = "confirmed"
    elif max(pairwise) <= LOW_OVERLAP_NOTE_CEILING:
        grouping = "no_shared_recordings"
    else:
        grouping = "partial"

    return {
        "kind": "edition_sibling",
        "grouping": grouping,
        "artist": members[0].artist,
        "title_key": members[0].group_key[1],
        "direction": direction,
        "albums": [
            {
                "album_id": a.album_id,
                "artist": a.artist,
                "title": a.title,
                "track_count": a.track_count,
                "genres": sorted(a.genres),
                "missing": sorted(union - a.genres),
            }
            for a in members
        ],
        "union": sorted(union),
        "max_missing": max(len(union - a.genres) for a in members),
        "recording_overlap_min": round(min(pairwise), 3) if pairwise else 0.0,
        "recording_overlap_max": round(max(pairwise), 3) if pairwise else 0.0,
        "validated_shared_recordings": validated_shared,
        "divergences": [
            {
                "album_id": d.album_id,
                "genre_id": d.genre_id,
                "relation": d.relation,
                "relation_evidence": d.relation_evidence,
                "carried_by": d.carried_by,
            }
            for d in divergences
        ],
    }


def _name_variant(a_key: str, b_key: str) -> bool:
    """Whether two artist keys plausibly name the same act.

    Token containment only: "jimi hendrix" ⊂ "jimi hendrix experience",
    "sandy alex g" ⊃ "alex g". Two unrelated acts sharing an album title
    (a covers record, a split) fail this and stay out of the alias queue.
    """
    a_tokens, b_tokens = set(a_key.split()), set(b_key.split())
    if not a_tokens or not b_tokens:
        return False
    return a_tokens <= b_tokens or b_tokens <= a_tokens


def find_artist_variant_candidates(albums: dict[str, Album]) -> list[dict]:
    """Same edition-stripped title, different resolved artist key.

    These are candidate `data/artist_aliases.yaml` links, not genre repairs. They
    matter here because an unlinked artist variant hides an edition pair from the
    title channel entirely — the pair then surfaces in the recording-overlap
    section misclassified as a compilation.
    """
    by_title: dict[str, list[Album]] = defaultdict(list)
    for album in albums.values():
        if album.title_key:
            by_title[album.title_key].append(album)

    out: list[dict] = []
    for title_key, members in sorted(by_title.items()):
        keys = {a.artist_key for a in members if a.artist_key}
        if len(keys) < 2:
            continue
        if len(keys) > 2:
            # Three or more artist keys under one title is a various-artist
            # compilation ("Third Man Live", "Eccentric Soul"), not an artist
            # whose name is spelled two ways.
            continue
        members = sorted(members, key=lambda a: a.album_id)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if a.artist_key == b.artist_key or not (a.artist_key and b.artist_key):
                    continue
                if not _name_variant(a.artist_key, b.artist_key):
                    continue
                overlap = _overlap(a.mbids, b.mbids)
                if overlap <= 0.0 and a.genres != b.genres:
                    # No shared recordings: same title by genuinely different
                    # artists (covers, standards). Not an alias candidate.
                    continue
                out.append(
                    {
                        "title_key": title_key,
                        "overlap": round(overlap, 3),
                        "genres_diverge": a.genres != b.genres,
                        "albums": [
                            {
                                "album_id": x.album_id,
                                "artist": x.artist,
                                "artist_key": x.artist_key,
                                "title": x.title,
                                "genres": sorted(x.genres),
                            }
                            for x in (a, b)
                        ],
                    }
                )
    out.sort(key=lambda r: (-r["overlap"], r["title_key"]))
    return out


def taxonomy_coverage(genre_ids: set[str], classifier: RelationClassifier) -> dict:
    """How much is_a structure exists for the genres this report reasons over.

    The relation classifier can only call a candidate redundant or a specificity
    gain when the taxonomy has an is_a edge to lean on. A genre with no children
    and no parents always lands in `disjoint` — so a high disjoint count is
    partly a taxonomy-sparsity artifact, and the report must say so rather than
    let a reader read it as 158 confirmed new-information repairs.
    """
    leaf_only = sorted(g for g in genre_ids if len(classifier.descendants(g)) <= 1)
    return {
        "genres_involved": len(genre_ids),
        "genres_without_descendants": len(leaf_only),
        "examples": leaf_only[:20],
    }


def find_recording_overlap_pairs(
    albums: dict[str, Album], title_grouped: set[str]
) -> list[dict]:
    """Album pairs sharing recordings but not title-matched (comps / appears-on)."""
    by_mbid: dict[str, set[str]] = defaultdict(set)
    for album in albums.values():
        for mbid in album.mbids:
            by_mbid[mbid].add(album.album_id)

    pair_shared: dict[tuple[str, str], int] = defaultdict(int)
    for mbid, album_ids in by_mbid.items():
        if len(album_ids) < 2 or len(album_ids) > 12:
            # A recording on >12 albums is a tagging artifact (silence, hidden
            # track) rather than release identity.
            continue
        ordered = sorted(album_ids)
        for i, left_id in enumerate(ordered):
            for right_id in ordered[i + 1 :]:
                pair_shared[(left_id, right_id)] += 1

    out: list[dict] = []
    for (a_id, b_id), shared in sorted(pair_shared.items()):
        a, b = albums[a_id], albums[b_id]
        if a.group_key == b.group_key and a_id in title_grouped:
            continue
        if a.genres == b.genres:
            continue
        overlap = _overlap(a.mbids, b.mbids)
        out.append(
            {
                "kind": "recording_overlap",
                "classification": (
                    "retitled_edition" if overlap >= EDITION_OVERLAP_THRESHOLD else "compilation_or_appears_on"
                ),
                "shared_recordings": shared,
                "overlap": round(overlap, 3),
                "validated_shared": len(a.validated_mbids & b.validated_mbids),
                "albums": [
                    {
                        "album_id": x.album_id,
                        "artist": x.artist,
                        "title": x.title,
                        "track_count": x.track_count,
                        "genres": sorted(x.genres),
                    }
                    for x in (a, b)
                ],
                "a_missing": sorted(b.genres - a.genres),
                "b_missing": sorted(a.genres - b.genres),
            }
        )
    out.sort(key=lambda r: (-r["overlap"], -r["shared_recordings"], r["albums"][0]["album_id"]))
    return out


def render_markdown(payload: dict, names: dict[str, str]) -> str:
    def gname(genre_id: str) -> str:
        return names.get(genre_id, genre_id)

    s = payload["summary"]
    cov = payload["taxonomy_coverage"]
    lines = [
        "# Edition genre divergence — dry run",
        "",
        f"Source: `{payload['metadata_db']}` (read-only). No authority row was modified.",
        "",
        "## Summary",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| Albums in authority | {s['albums_total']} |",
        f"| Albums with observed genres | {s['albums_with_observed']} |",
        f"| Sibling-edition groups (title channel) | {s['title_groups']} |",
        f"| ...with divergent observed genres | {s['divergent_groups']} |",
        f"| Albums implicated | {s['albums_implicated']} |",
        f"| Missing (album, genre) pairs | {s['divergence_pairs']} |",
        f"| ...disjoint (new information) | {s['relation_disjoint']} |",
        f"| ...specificity gain | {s['relation_specificity_gain']} |",
        f"| ...redundant breadth | {s['relation_redundant_broader']} |",
        f"| Pure-subset groups (clean repair) | {s['direction_subset']} |",
        f"| Mutually divergent groups (review) | {s['direction_mutual']} |",
        f"| Cross-release recording-overlap pairs | {s['recording_overlap_pairs']} |",
        f"| ...classified retitled edition | {s['recording_overlap_editions']} |",
        f"| Artist-string variant candidates | {s['artist_variant_candidates']} |",
        "",
        "| Grouping confidence | Groups |",
        "|---|---:|",
        f"| confirmed (recordings agree) | {s['grouping_confirmed']} |",
        f"| partial | {s['grouping_partial']} |",
        f"| unverified (no recording evidence) | {s['grouping_unverified']} |",
        f"| no shared recordings (expected for remasters) | {s['grouping_no_shared_recordings']} |",
        "",
        f"> **Taxonomy caveat.** {cov['genres_without_descendants']} of "
        f"{cov['genres_involved']} genres in this report have no `is_a` children in the",
        "> live taxonomy. The relation classifier needs an edge to call a candidate",
        "> redundant or a specificity gain, so those genres can only land in `disjoint`.",
        "> Read the disjoint count as an upper bound on new-information repairs, not a",
        "> confirmed one.",
        "",
        "**Relation legend.** `disjoint` — the missing genre is unrelated to anything the",
        "album already carries; real new information. `specificity_gain` — the album carries",
        "only a broader ancestor; the sibling has the narrower style. `redundant_broader` —",
        "the album already carries a narrower descendant, so adding this is breadth without",
        "information (Layer 1 #12: rare > common).",
        "",
        "**Direction.** `subset` means one edition's genre set contains every sibling's —",
        "a pure evidence gap, the safest repair. `mutual` means each edition asserts",
        "something the others lack; that is a classification question and needs review.",
        "",
        "**Grouping corroboration.** Shared recording IDs *confirm* a group; their absence",
        "does not deny it — MusicBrainz mints distinct recording MBIDs per remaster, so",
        "`Revolver [2009 Mono Remaster]` legitimately shares nothing with `Revolver`.",
        "`confirmed` — every pair shares ≥60% of recordings. `partial` — some overlap.",
        "`no_shared_recordings` — ≤20%, expected for remaster pairs. `unverified` — too few",
        "recording IDs to check.",
        "",
        "Different releases that the title normalizer would otherwise merge (`Volume 2`,",
        "`Disc 1`, `The Second Album`) are split by sequence marker *before* divergence is",
        "computed, so they never reach this queue.",
        "",
        "## Sibling-edition groups — repair queue",
        "",
    ]

    for group in payload["edition_groups"]:
        lines.append(
            f"### {group['artist']} — {group['title_key']}  "
            f"`{group['direction']}` · `{group['grouping']}` · max missing {group['max_missing']}"
        )
        lines.append("")
        lines.append("| Album | Tracks | Observed genres | Missing vs siblings |")
        lines.append("|---|---:|---|---|")
        for album in group["albums"]:
            lines.append(
                f"| {album['title']} <br>`{album['album_id']}` | {album['track_count']} | "
                f"{', '.join(gname(g) for g in album['genres']) or '—'} | "
                f"**{', '.join(gname(g) for g in album['missing']) or '—'}** |"
            )
        lines.append("")
        if group["recording_overlap_max"]:
            lines.append(
                f"Recording overlap across the group: "
                f"{group['recording_overlap_min']:.2f}–{group['recording_overlap_max']:.2f} "
                f"(validated shared: {group['validated_shared_recordings']})"
            )
            lines.append("")
        by_relation: dict[str, list[str]] = defaultdict(list)
        for d in group["divergences"]:
            note = f"`{gname(d['genre_id'])}` → {d['album_id']}"
            if d["relation_evidence"]:
                note += f" (already has {', '.join(gname(g) for g in d['relation_evidence'])})"
            by_relation[d["relation"]].append(note)
        for relation in ("disjoint", "specificity_gain", "redundant_broader"):
            if by_relation[relation]:
                lines.append(f"- **{relation}**: " + "; ".join(sorted(by_relation[relation])))
        lines.append("")

    lines += [
        "## Artist-string variant candidates (alias review)",
        "",
        "Same edition-stripped title, different resolved artist key, sharing recordings.",
        "These are candidate `data/artist_aliases.yaml` links — an unlinked variant hides",
        "an edition pair from the title channel, so it resurfaces below misclassified as a",
        "compilation. Fix the alias first, then re-run: some genre divergences will",
        "disappear into their proper sibling group.",
        "",
        "| Overlap | Genres diverge | A | B |",
        "|---:|---|---|---|",
    ]
    for cand in payload["artist_variant_candidates"]:
        a, b = cand["albums"]
        lines.append(
            f"| {cand['overlap']:.2f} | {'yes' if cand['genres_diverge'] else 'no'} | "
            f"{a['artist']} — {a['title']} | {b['artist']} — {b['title']} |"
        )

    lines += [
        "",
        "## Cross-release recording overlap (review only)",
        "",
        "Albums sharing MusicBrainz recording IDs but not title-matched. High overlap is a",
        "retitled edition; low overlap is a compilation or appears-on, where genre",
        "propagation is **not** safe — a label sampler's genre set must not reach a studio",
        "album. `validated` counts recordings where `mbid_status='ok'`; the rest are",
        "unvalidated file tags and carry no identity guarantee.",
        "",
        "| Overlap | Shared | Validated | Classification | A | B |",
        "|---:|---:|---:|---|---|---|",
    ]
    for pair in payload["recording_overlap"]:
        a, b = pair["albums"]
        lines.append(
            f"| {pair['overlap']:.2f} | {pair['shared_recordings']} | {pair['validated_shared']} | "
            f"{pair['classification']} | {a['artist']} — {a['title']} | {b['artist']} — {b['title']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(metadata_db: str, overlap_limit: int) -> tuple[dict, dict[str, str]]:
    conn = sqlite3.connect(f"file:{Path(metadata_db).as_posix()}?mode=ro", uri=True)
    try:
        albums = load_albums(conn)
        names = canonical_genre_names(conn)
        classifier = RelationClassifier(conn)

        title_groups = build_title_groups(albums)
        title_grouped = {a.album_id for group in title_groups for a in group}

        edition_groups = [g for g in (analyze_group(m, classifier) for m in title_groups) if g]
        edition_groups.sort(key=lambda g: (-g["max_missing"], g["artist"], g["title_key"]))

        overlap_pairs = find_recording_overlap_pairs(albums, title_grouped)
        artist_variants = find_artist_variant_candidates(albums)

        divergences = [d for g in edition_groups for d in g["divergences"]]
        relation_counts: dict[str, int] = defaultdict(int)
        for d in divergences:
            relation_counts[d["relation"]] += 1

        involved = {g for group in edition_groups for g in group["union"]}
        coverage = taxonomy_coverage(involved, classifier)

        payload = {
            "metadata_db": str(Path(metadata_db).resolve()),
            "summary": {
                "albums_total": len(albums),
                "albums_with_observed": sum(1 for a in albums.values() if a.genres),
                "title_groups": len(title_groups),
                "divergent_groups": len(edition_groups),
                "albums_implicated": len({a["album_id"] for g in edition_groups for a in g["albums"]}),
                "divergence_pairs": len(divergences),
                "relation_disjoint": relation_counts["disjoint"],
                "relation_specificity_gain": relation_counts["specificity_gain"],
                "relation_redundant_broader": relation_counts["redundant_broader"],
                "direction_subset": sum(1 for g in edition_groups if g["direction"] == "subset"),
                "direction_mutual": sum(1 for g in edition_groups if g["direction"] == "mutual"),
                "recording_overlap_pairs": len(overlap_pairs),
                "recording_overlap_editions": sum(
                    1 for p in overlap_pairs if p["classification"] == "retitled_edition"
                ),
                "artist_variant_candidates": len(artist_variants),
                "grouping_confirmed": sum(1 for g in edition_groups if g["grouping"] == "confirmed"),
                "grouping_partial": sum(1 for g in edition_groups if g["grouping"] == "partial"),
                "grouping_unverified": sum(1 for g in edition_groups if g["grouping"] == "unverified"),
                "grouping_no_shared_recordings": sum(
                    1 for g in edition_groups if g["grouping"] == "no_shared_recordings"
                ),
            },
            "taxonomy_coverage": coverage,
            "edition_groups": edition_groups,
            "artist_variant_candidates": artist_variants,
            "recording_overlap": overlap_pairs[:overlap_limit],
            "recording_overlap_truncated": max(0, len(overlap_pairs) - overlap_limit),
        }
        return payload, names
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--metadata-db", default=str(ROOT / "data" / "metadata.db"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "docs" / "run_audits" / "edition_divergence"),
        help="Directory for the markdown + JSON report",
    )
    parser.add_argument("--stem", default="edition_genre_divergence")
    parser.add_argument(
        "--overlap-limit",
        type=int,
        default=60,
        help="Max cross-release recording-overlap pairs to include",
    )
    args = parser.parse_args(argv)

    payload, names = build_report(args.metadata_db, args.overlap_limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem}.json"
    md_path = out_dir / f"{args.stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload, names), encoding="utf-8")

    s = payload["summary"]
    print(f"divergent sibling groups: {s['divergent_groups']} / {s['title_groups']}")
    print(
        f"missing pairs: {s['divergence_pairs']} "
        f"(disjoint={s['relation_disjoint']} specificity={s['relation_specificity_gain']} "
        f"redundant={s['relation_redundant_broader']})"
    )
    print(f"subset={s['direction_subset']} mutual={s['direction_mutual']}")
    print(f"recording-overlap pairs: {s['recording_overlap_pairs']}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
