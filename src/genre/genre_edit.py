"""User genre edit orchestration: resolve terms, locate album, apply override.

Writes the durable add/remove override (ai_genre_user_overrides) AND the
surgical release_effective_genres rows via the shared publish materializer, so
the edit is authoritative immediately and reproduced byte-for-byte by a later
full publish.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ai_genre_enrichment.normalization import (
    make_release_key,
    normalize_release_artist,
    normalize_release_name,
)
from src.genre import genre_publish
from src.genre.authority import canonical_genre_names, resolved_genres_for_album


def resolve_terms_to_genre_ids(taxonomy, names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map free-typed names to canonical genre_ids. Unresolved names returned."""
    resolved: dict[str, str] = {}
    unknown: list[str] = []
    for name in names:
        term = (name or "").strip()
        if not term:
            continue
        gid = genre_publish._term_to_genre_id(taxonomy, term)
        if gid:
            resolved[term] = gid
        else:
            unknown.append(term)
    return resolved, unknown


def album_id_for_release(conn, artist: str, album: str) -> str | None:
    """Resolve album_id from the tracks table (orphan-safe).

    Exact (artist, album) first; else normalized release_key grouped over
    tracks, picking the album_id with the most tracks (ties: lexicographically
    smallest) for determinism.
    """
    row = conn.execute(
        "SELECT album_id, COUNT(*) c FROM tracks "
        "WHERE artist = ? AND album = ? AND album_id IS NOT NULL AND album_id != '' "
        "GROUP BY album_id ORDER BY c DESC, album_id ASC LIMIT 1",
        (artist, album),
    ).fetchone()
    if row and row[0]:
        return row[0]

    target_key = make_release_key(artist, album)
    counts: dict[str, int] = {}
    for aid, a, alb in conn.execute(
        "SELECT album_id, artist, album FROM tracks "
        "WHERE album_id IS NOT NULL AND album_id != ''"
    ):
        if make_release_key(a, alb) == target_key:
            counts[aid] = counts.get(aid, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _album_in_graph(conn, album_id: str, release_key: str) -> set[str]:
    """`{album_id}` when this release's graph assignments exist, else empty.

    Membership is decided by RELEASE_KEY, mirroring
    ``genre_publish.build_resolved_table`` — and matching what
    ``materialize_album_genres`` then reads. Gating on a stamped ``album_id``
    instead (as this did until 2026-07-27) disagreed with both: a fragment of a
    release that shares the key but was never stamped — feat./collab variants,
    duplicate imports — looked absent from the graph, so an edit rebuilt it from
    the LEGACY path and destroyed its graph genres. 31 albums were in that state,
    including the unstamped twin of 'Are You Experienced?' (14 rows, among them
    acid_rock/blues_rock/hard_rock/psychedelic_rock).
    """
    row = conn.execute(
        "SELECT 1 FROM genre_graph_release_genre_assignments WHERE release_id = ? LIMIT 1",
        (release_key,),
    ).fetchone()
    return {album_id} if row else set()


@dataclass
class EditResult:
    resolved: list[str]   # canonical names of the saved target set
    unknown: list[str]    # input names that did not resolve to a genre_id
    added: list[str]      # canonical names added vs the non-user authority base
    removed: list[str]    # canonical names removed vs the non-user authority base
    no_change: bool


def apply_user_genre_edit(
    meta_conn,
    sidecar_store,
    taxonomy,
    *,
    artist: str,
    album: str,
    target_names: list[str],
    force_assert: list[str] | None = None,
) -> EditResult:
    """Make ``target_names`` the user-authoritative genres for a release.

    Writes the durable add/remove override (diffed against the NON-user
    authority base, so a later full publish reproduces it) and surgically
    materializes ``release_effective_genres`` for the album via the shared
    publish materializer. ``no_change`` is detected against the FULL current
    authority (incl. user rows): a re-save of the displayed set writes nothing.

    ``force_assert`` names genres that must become user ``observed_leaf`` rows
    even when the graph already carries them as *inferred* rows. Default
    behaviour deliberately leaves an inferred genre inferred (see ``add_ids``
    below), which is right for the edit dialog but blocks a legitimate case:
    sibling-edition repair, where another edition of the same release asserts
    the genre as an observed leaf. Callers must pass only genres they have that
    evidence for, and every name must also appear in ``target_names``.
    """
    resolved_map, unknown = resolve_terms_to_genre_ids(taxonomy, target_names)
    target_ids = set(resolved_map.values())

    force_map, force_unknown = resolve_terms_to_genre_ids(taxonomy, list(force_assert or []))
    force_ids = set(force_map.values())
    if force_unknown:
        raise ValueError(f"force_assert has unresolvable genres: {sorted(force_unknown)}")
    if not force_ids <= target_ids:
        # A knob that cannot act is a bug here, not a silent no-op (CLAUDE.md).
        raise ValueError(
            "force_assert genres must also be in target_names; "
            f"extra: {sorted(force_ids - target_ids)}"
        )

    album_id = album_id_for_release(meta_conn, artist, album)
    if album_id is None:
        raise ValueError(f"no album_id for {artist!r} / {album!r}")

    id_to_name = canonical_genre_names(meta_conn)

    def name_of(gid: str) -> str:
        return id_to_name.get(gid, gid)

    all_rows = resolved_genres_for_album(meta_conn, album_id)
    full_ids = {r.genre_id for r in all_rows}
    non_user_ids = {r.genre_id for r in all_rows if r.source != "user"}
    # Genres the user previously asserted. These must be re-asserted whenever
    # they survive in the target: a prior user observed_leaf row whose genre the
    # graph ALSO carries as an inferred row was suppressed by the plain
    # `target_ids - non_user_ids` rule, so the rewrite dropped the leaf and the
    # genre survived only as inferred — which the artifact excludes from
    # X_genre_raw, silently removing it from generation.
    #
    # Scoped deliberately to prior user rows. The edit dialog seeds its chips
    # from every published layer, so the target routinely contains inferred hub
    # families (pop, r&b/soul); promoting those to observed_leaf would bake hub
    # genres into the artifact at full weight — the 2026-06-12 saturation
    # incident. An inferred genre nobody asserted stays inferred.
    prior_user_ids = {r.genre_id for r in all_rows if r.source == "user"}

    add_ids = (target_ids - non_user_ids) | (prior_user_ids & target_ids) | force_ids
    remove_ids = non_user_ids - target_ids
    resolved_names = sorted(name_of(gid) for gid in target_ids)

    # Nothing visibly changes when the target equals the full current authority.
    # A forced genre is the exception: it may already be PRESENT (so the id sets
    # match) while sitting at the wrong layer, which is exactly the state
    # force_assert exists to correct. Comparing ids alone would short-circuit
    # before writing anything.
    asserted_leaf_ids = {
        r.genre_id
        for r in all_rows
        if r.source == "user" and r.assignment_layer == "observed_leaf"
    }
    if target_ids == full_ids and force_ids <= asserted_leaf_ids:
        return EditResult(resolved=resolved_names, unknown=unknown,
                          added=[], removed=[], no_change=True)

    add_names = sorted(name_of(gid) for gid in add_ids)
    remove_names = sorted(name_of(gid) for gid in remove_ids)

    release_key = make_release_key(artist, album)
    sidecar_store.set_user_override(
        release_key=release_key,
        normalized_artist=normalize_release_artist(artist),
        normalized_album=normalize_release_name(album),
        genres_add=add_names,
        genres_remove=remove_names,
    )

    # Surgical materialize via the SAME path publish uses (parity).
    remove_match: set[str] = set(remove_ids)
    for n in remove_names:
        remove_match |= set(genre_publish._split(n))
    overrides = {album_id: (list(add_ids), remove_match)}
    genre_publish.materialize_album_genres(
        meta_conn, album_id,
        graph_album_ids=_album_in_graph(meta_conn, album_id, release_key),
        legacy=genre_publish.legacy_genres_by_album(meta_conn, album_id, taxonomy=taxonomy),
        overrides=overrides,
        album_to_key={album_id: release_key},
    )
    meta_conn.commit()
    return EditResult(resolved=resolved_names, unknown=unknown,
                      added=add_names, removed=remove_names, no_change=False)
