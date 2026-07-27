#!/usr/bin/env python3
"""Apply sibling-edition genre repairs through the existing user-edit path.

Reads the review queue produced by `report_edition_genre_divergence.py` and, for
every album in a **pure-subset** group, adds the genres its sibling editions
carry. Subset groups only: one edition's set strictly contains the others', so
the repair is a pure evidence gap. Mutually divergent groups are a
classification question and are never touched here.

The write goes through `genre_edit.apply_user_genre_edit` — the same call the
GUI's edit-genres dialog makes. That writes the durable override to
`ai_genre_user_overrides` AND surgically materializes
`release_effective_genres`, so a later full publish reproduces the repair
byte-for-byte instead of silently reverting it.

Target-set discipline
---------------------
The target is `display_genre_names_for_album(album_id) + missing`, mirroring
exactly what the dialog would submit if a user opened it and added the missing
chips. Passing only the *observed* names would make every inferred family row
look like a user deletion, and `apply_user_genre_edit` would record a removal
for it. The union with the displayed set is what keeps this additive.

Safety
------
Dry-run by default: prints the planned edits and writes nothing. `--apply`
requires `--metadata-db`/`--sidecar-db` to be passed explicitly, so pointing it
at the live database is always a deliberate act.

Usage
-----
    # dry run against a copy
    python scripts/repair_edition_genre_divergence.py \
        --metadata-db /tmp/edrepair/metadata.db --sidecar-db /tmp/edrepair/ai_genre_enrichment.db

    # apply
    ... same flags ... --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy  # noqa: E402
from src.ai_genre_enrichment.normalization import make_release_key  # noqa: E402
from src.ai_genre_enrichment.storage import SidecarStore  # noqa: E402
from src.genre.authority import canonical_genre_names, display_genre_names_for_album  # noqa: E402
from src.genre.genre_edit import apply_user_genre_edit  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "run_audits" / "edition_divergence" / "edition_genre_divergence.json"


def graph_key_is_live(conn: sqlite3.Connection, artist: str, album: str) -> bool:
    """Whether this release's derived key still finds its graph assignments.

    `materialize_album_genres` rebuilds an album from
    `genre_graph_release_genre_assignments WHERE release_id = <derived key>`,
    then deletes and rewrites the album's authority rows. `make_release_key`
    resolves artist aliases, so the graph rows are keyed with whatever alias map
    was in effect **at graph build time**. Add a new alias to
    `data/artist_aliases.yaml` and that artist's derived key stops matching —
    the rebuild finds nothing, and the edit silently DESTROYS the album's
    graph-derived genres instead of adding to them.

    Verified on a database copy 2026-07-27: adding a Jimi Hendrix alias made an
    edit to "Electric Ladyland 50th" drop acid_rock, blues_rock, and
    psychedelic_rock. Pre-existing aliases (Alex G) were unaffected because the
    graph was built with them already in place.

    This guard fails the album closed rather than trusting the key.
    """
    key = make_release_key(artist, album)
    row = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments WHERE release_id = ?",
        (key,),
    ).fetchone()
    return bool(row and row[0])


def planned_edits(report: dict, id_to_name: dict[str, str]) -> list[dict]:
    """One entry per album needing genres, from subset groups only."""
    plan: list[dict] = []
    for group in report["edition_groups"]:
        if group["direction"] != "subset":
            continue
        for album in group["albums"]:
            if not album["missing"]:
                continue
            plan.append(
                {
                    "album_id": album["album_id"],
                    "artist": album["artist"],
                    "title": album["title"],
                    "missing_ids": album["missing"],
                    "missing_names": [id_to_name.get(g, g) for g in album["missing"]],
                    "grouping": group["grouping"],
                }
            )
    plan.sort(key=lambda e: (e["artist"], e["title"]))
    return plan


def authority_rows(conn: sqlite3.Connection, album_id: str) -> set[tuple[str, str, str]]:
    """(genre_id, layer, source) rows for one album — the before/after check.

    This repair is additive by definition: a sibling edition carries a genre this
    one lacks. Anything it *loses* is a bug, so each album is diffed around its
    edit rather than trusted. That check is what caught two real defects on
    2026-07-27 — an alias change orphaning graph assignments (see
    `graph_key_is_live`) and `apply_user_genre_edit` demoting a user leaf to an
    inferred row (fixed in `genre_edit.py`; regression test
    `test_apply_edit_keeps_user_leaf_that_also_has_an_inferred_row`).
    """
    return {
        (g, layer, src)
        for g, layer, src in conn.execute(
            "SELECT genre_id, assignment_layer, source FROM release_effective_genres "
            "WHERE album_id = ?",
            (album_id,),
        )
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--metadata-db", default=str(ROOT / "data" / "metadata.db"))
    parser.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repairs. Without this the script only prints the plan.",
    )
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))

    mode = "" if args.apply else "?mode=ro"
    conn = sqlite3.connect(f"file:{Path(args.metadata_db).as_posix()}{mode}", uri=True)
    try:
        id_to_name = canonical_genre_names(conn)
        plan = planned_edits(report, id_to_name)

        print(f"{'APPLY' if args.apply else 'DRY RUN'} — {len(plan)} albums to repair")
        print(f"  metadata: {args.metadata_db}")
        print(f"  sidecar : {args.sidecar_db}\n")

        blocked: list[dict] = []
        for entry in plan:
            current = display_genre_names_for_album(conn, entry["album_id"])
            entry["target_names"] = sorted(set(current) | set(entry["missing_names"]))
            entry["graph_key_live"] = graph_key_is_live(conn, entry["artist"], entry["title"])
            if not entry["graph_key_live"]:
                blocked.append(entry)
                continue
            print(
                f"  {entry['artist'][:26]:<28} {entry['title'][:38]:<40} "
                f"+{', '.join(entry['missing_names'])}"
            )

        if blocked:
            print(f"\n  BLOCKED ({len(blocked)}) — repairing these would lose existing genres:")
            for entry in blocked:
                print(
                    f"    {entry['artist'][:24]:<26} {entry['title'][:32]:<34} "
                    "derived release_key finds no graph assignments (rebuild the graph)"
                )

        plan = [e for e in plan if e["graph_key_live"]]

        if not args.apply:
            print("\nNothing written. Re-run with --apply to write these edits.")
            return 0

        taxonomy = load_default_layered_taxonomy()
        store = SidecarStore(args.sidecar_db)
        applied, skipped = 0, 0
        print()
        lost_total = 0
        for entry in plan:
            rows_before = authority_rows(conn, entry["album_id"])
            result = apply_user_genre_edit(
                conn,
                store,
                taxonomy,
                artist=entry["artist"],
                album=entry["title"],
                target_names=entry["target_names"],
            )
            lost = rows_before - authority_rows(conn, entry["album_id"])
            if lost:
                lost_total += len(lost)
                print(f"  LOST ROWS   {entry['artist'][:24]} — {entry['title'][:34]}: {sorted(lost)}")
                print("              ^^ this repair must be additive; investigate before continuing")
            if result.no_change:
                skipped += 1
                print(f"  no-change  {entry['artist'][:24]} — {entry['title'][:34]}")
                continue
            applied += 1
            print(
                f"  applied    {entry['artist'][:24]} — {entry['title'][:34]}  "
                f"+{', '.join(result.added)}"
                + (f"  -{', '.join(result.removed)}" if result.removed else "")
            )
            if result.removed:
                print("             ^^ UNEXPECTED REMOVAL — this repair should be additive only")
            if result.unknown:
                print(f"             unresolved terms: {result.unknown}")

        print(f"\napplied={applied} no_change={skipped} rows_lost={lost_total}")
        return 1 if lost_total else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
