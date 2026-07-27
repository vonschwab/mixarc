#!/usr/bin/env python3
"""Apply human-reviewed genre decisions for one album at a time.

Companion to `report_edition_genre_divergence.py`. That report's *subset* groups
are mechanical and handled by `repair_edition_genre_divergence.py`. Its
*mutually divergent* groups are not: each edition asserts something the other
lacks, so merging them blindly spreads a wrong tag as readily as it fills a real
gap (2026-07-27: one copy of the Beatles' 1963 debut was tagged psychedelic
rock). Those need a person to decide, and this script applies what they decided.

Input is a decisions file: a JSON list of
``{album_id, artist, title, target_names, reason}``, where ``target_names`` is
the COMPLETE desired genre set for that album — including the genres it already
has. Anything omitted is removed, so removals are first-class here, unlike the
purely additive repair script.

Writes go through `genre_edit.apply_user_genre_edit`, the same call the GUI's
edit-genres dialog makes: the durable override lands in `ai_genre_user_overrides`
so a later publish reproduces the decision instead of reverting it.

Dry-run by default. Every album is verified against its declared target after
the edit — for reviewed decisions the check is "the result equals what was
asked for", not "nothing was lost".

Usage
-----
    python scripts/apply_reviewed_genre_decisions.py --decisions <file.json> \
        --metadata-db <db> --sidecar-db <db> [--apply]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.repair_edition_genre_divergence import graph_key_is_live  # noqa: E402
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402
from src.genre.authority import display_genre_names_for_album  # noqa: E402
from src.genre.genre_edit import apply_user_genre_edit  # noqa: E402


def asserted_leaf_names(conn: sqlite3.Connection, album_id: str) -> set[str]:
    """Display names this album carries as a user-asserted observed_leaf row."""
    return {
        name
        for (name,) in conn.execute(
            "SELECT COALESCE(g.name, reg.genre_id) FROM release_effective_genres reg "
            "LEFT JOIN genre_graph_canonical_genres g ON g.genre_id = reg.genre_id "
            "WHERE reg.album_id = ? AND reg.source = 'user' "
            "AND reg.assignment_layer = 'observed_leaf'",
            (album_id,),
        )
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--metadata-db", default=str(ROOT / "data" / "metadata.db"))
    parser.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    parser.add_argument("--apply", action="store_true", help="Write the decisions.")
    args = parser.parse_args(argv)

    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    mode = "" if args.apply else "?mode=ro"
    conn = sqlite3.connect(f"file:{Path(args.metadata_db).as_posix()}{mode}", uri=True)
    try:
        print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(decisions)} decisions")
        print(f"  metadata: {args.metadata_db}\n")

        planned, blocked, noop = [], [], []
        for entry in decisions:
            current = set(display_genre_names_for_album(conn, entry["album_id"]))
            target = set(entry["target_names"])
            entry["_add"] = sorted(target - current)
            entry["_remove"] = sorted(current - target)
            # A force_assert entry can be a no-op by NAME while still needing a
            # write: the genre is present, just at an inferred layer. Only the
            # layer check can tell, so consult it before calling this a no-op.
            forced_pending = [
                g for g in (entry.get("force_assert") or [])
                if g not in asserted_leaf_names(conn, entry["album_id"])
            ]
            entry["_forced"] = forced_pending
            if not entry["_add"] and not entry["_remove"] and not forced_pending:
                noop.append(entry)
            elif not graph_key_is_live(conn, entry["artist"], entry["title"]):
                blocked.append(entry)
            else:
                planned.append(entry)

        for entry in planned:
            add = ", ".join(entry["_add"]) or "—"
            rem = ", ".join(entry["_remove"]) or "—"
            print(f"  {entry['artist'][:20]:<22} {entry['title'][:34]:<36}")
            print(f"      + {add}")
            if entry["_remove"]:
                print(f"      - {rem}")
            if entry["_forced"]:
                print(f"      ! assert (currently inferred only): {', '.join(entry['_forced'])}")

        if noop:
            print(f"\n  NO-OP ({len(noop)}) — already matches the target:")
            for entry in noop:
                print(f"    {entry['artist'][:20]:<22} {entry['title'][:34]}")
        if blocked:
            print(f"\n  BLOCKED ({len(blocked)}) — derived release_key finds no graph")
            print("  assignments; editing would delete this album's graph genres:")
            for entry in blocked:
                print(f"    {entry['artist'][:20]:<22} {entry['title'][:34]}")

        if not args.apply:
            print(f"\nNothing written. {len(planned)} edits ready; re-run with --apply.")
            return 0

        taxonomy = load_default_layered_taxonomy()
        store = SidecarStore(args.sidecar_db)
        applied = mismatched = 0
        print()
        for entry in planned:
            result = apply_user_genre_edit(
                conn, store, taxonomy,
                artist=entry["artist"], album=entry["title"],
                target_names=entry["target_names"],
                force_assert=entry.get("force_assert"),
            )
            after = set(display_genre_names_for_album(conn, entry["album_id"]))
            target = set(entry["target_names"])
            applied += 1
            print(f"  applied  {entry['artist'][:20]:<22} {entry['title'][:32]:<34} "
                  f"+{len(entry['_add'])} -{len(entry['_remove'])}")
            # Reviewed decisions declare the whole target, so the authority must
            # equal it afterwards. A genre the graph only INFERS cannot be turned
            # into an assertion by this path (by design — see genre_edit.py), so
            # surface the difference instead of implying the decision landed.
            if after != target:
                mismatched += 1
                print(f"           unmet: missing={sorted(target - after)} "
                      f"extra={sorted(after - target)}")
            if result.unknown:
                print(f"           unresolved terms: {result.unknown}")

        print(f"\napplied={applied} unmet={mismatched} noop={len(noop)} blocked={len(blocked)}")
        return 1 if mismatched else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
