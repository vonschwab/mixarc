import json
import sqlite3
from src.genre import genre_publish
from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy


def _meta_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_sidecar(tmp_path):
    """A sidecar with the real taxonomy loaded, nothing else."""
    side = tmp_path / "sidecar.db"
    store = SidecarStore(str(side))
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())
    return side


def _attach(meta_conn, side_path):
    # Plain path (no file: URI) to avoid ATTACH URI-mode pitfalls. The sidecar is
    # only ever read (SELECT) by the publish code, so writable-attach is harmless.
    meta_conn.execute("ATTACH DATABASE ? AS side", (str(side_path),))


def test_create_published_schema_creates_all_tables():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {
        "genre_graph_canonical_genres",
        "genre_graph_canonical_facets",
        "genre_graph_edges",
        "genre_graph_aliases",
        "genre_graph_bridge_rules",
        "genre_graph_rejected_terms",
        "genre_graph_taxonomy_meta",
        "genre_graph_release_genre_assignments",
        "genre_graph_release_facet_assignments",
        "release_effective_genres",
    }
    assert expected == names


def test_create_published_schema_is_idempotent():
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    genre_publish.create_published_schema(conn)  # must not raise
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info('release_effective_genres')"
    ).fetchall()}
    assert {"album_id", "release_key", "genre_id", "assignment_layer",
            "confidence", "source"} <= cols


def test_copy_taxonomy_copies_rows_and_writes_meta(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)

    side_genres = conn.execute(
        "SELECT COUNT(*) FROM side.genre_graph_canonical_genres"
    ).fetchone()[0]
    main_genres = conn.execute(
        "SELECT COUNT(*) FROM genre_graph_canonical_genres"
    ).fetchone()[0]
    assert main_genres == side_genres > 0

    meta = conn.execute(
        "SELECT version, fingerprint, published_at FROM genre_graph_taxonomy_meta"
    ).fetchall()
    assert len(meta) == 1
    assert meta[0]["version"]  # non-empty
    assert len(meta[0]["fingerprint"]) == 64


def test_copy_taxonomy_is_idempotent(tmp_path):
    side = _make_sidecar(tmp_path)
    conn = _meta_conn()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    genre_publish.copy_taxonomy(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM genre_graph_taxonomy_meta"
    ).fetchone()[0] == 1


def _make_metadata(tmp_path):
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
                             album_id TEXT, norm_artist TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        """
    )
    conn.commit()
    conn.close()
    return meta


def test_resolve_keys_uses_signature_then_albums(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    # signature row maps a release_key to an album_id
    sconn = sqlite3.connect(side)
    sconn.execute(
        "INSERT INTO enriched_genre_signatures "
        "(release_key, normalized_artist, normalized_album, album_id, signature_json, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("acetone::york blvd", "acetone", "york blvd", "ALB_SIG", "{}", "2026-01-01T00:00:00+00:00"),
    )
    sconn.commit()
    sconn.close()
    # albums table can recompute a different key via normalizers
    mconn = sqlite3.connect(meta)
    mconn.execute("INSERT INTO albums VALUES ('ALB_CALC', 'Rocket', '(Sandy) Alex G')")
    mconn.commit()
    mconn.close()

    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    _attach(conn, side)
    mapping, collisions = genre_publish.resolve_release_key_to_album_id(conn)
    assert mapping["acetone::york blvd"] == "ALB_SIG"      # signature path
    # recomputed path: normalize_release_artist/name of the albums row
    from src.ai_genre_enrichment.normalization import (
        normalize_release_artist, normalize_release_name)
    calc_key = f"{normalize_release_artist('(Sandy) Alex G')}::{normalize_release_name('Rocket')}"
    assert mapping[calc_key] == "ALB_CALC"
    assert collisions == 0


def _insert_graph_assignment(side_path, release_id, artist, album, genre_id, layer):
    sconn = sqlite3.connect(side_path)
    sconn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (release_id, artist, album, genre_id, layer, 0.9, 0.7, 2, 0, "{}", "t"),
    )
    sconn.commit()
    sconn.close()


def test_populate_authority_stamps_album_id(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    _insert_graph_assignment(side, "acetone::york blvd", "acetone", "york blvd",
                             "alternative_rock", "observed_leaf")
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    mapping = {"acetone::york blvd": "ALB1"}
    genre_publish.populate_authority(conn, mapping)
    rows = conn.execute(
        "SELECT album_id, genre_id FROM genre_graph_release_genre_assignments"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["album_id"] == "ALB1"
    assert rows[0]["genre_id"] == "alternative_rock"


def test_legacy_genres_by_album_aggregates_sources(tmp_path):
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'Some Album', 'Some Artist')")
    conn.execute("INSERT INTO tracks VALUES ('T1', 'Some Artist', 'Some Album', 'ALB1', 'some artist')")
    conn.execute("INSERT INTO track_genres VALUES ('T1', 'Slowcore', 'file', 1.0)")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', 'Indie Rock', 'discogs_release')")
    conn.execute("INSERT INTO artist_genres VALUES ('Some Artist', 'Rock', 'musicbrainz_artist')")
    conn.commit()
    result = genre_publish.legacy_genres_by_album(conn)
    assert "ALB1" in result
    genres = {g for g, _w in result["ALB1"]}
    # normalized tokens from the three sources are all present
    assert "slowcore" in genres
    assert "rock" in genres


def test_legacy_genres_are_canonicalized_when_a_taxonomy_is_given(tmp_path):
    """Legacy tokens must land as canonical genre_ids, not raw text.

    The legacy layer wrote its normalized token straight through as the
    genre_id, so `release_effective_genres` carried ids like `post-rock` (hyphen)
    while the canonical id is `post_rock`, and the artifact vocabulary picked up
    the raw string as its own dimension. 46 rows across 25 albums were in this
    state on 2026-07-27, including junk like `special purpose artist` that the
    taxonomy rejects outright.
    """
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'A', 'X')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', 'Post-Rock', 'discogs_release')")
    conn.commit()
    taxonomy = load_default_layered_taxonomy()

    result = genre_publish.legacy_genres_by_album(conn, taxonomy=taxonomy)

    genres = {g for g, _w in result.get("ALB1", [])}
    assert "post_rock" in genres
    assert "post-rock" not in genres


def test_legacy_genres_drop_terms_the_taxonomy_rejects(tmp_path):
    """`special purpose artist` is a MusicBrainz artist type, not a genre."""
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'A', 'X')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', 'Special Purpose Artist', 'musicbrainz_release')")
    conn.commit()
    taxonomy = load_default_layered_taxonomy()

    result = genre_publish.legacy_genres_by_album(conn, taxonomy=taxonomy)

    assert "ALB1" not in result or result["ALB1"] == []


def test_legacy_genres_skip_empty_marker(tmp_path):
    meta = _make_metadata(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'A', 'X')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB1', '__EMPTY__', 'discogs_release')")
    conn.commit()
    result = genre_publish.legacy_genres_by_album(conn)
    assert "ALB1" not in result or result["ALB1"] == []


def test_classify_override_terms_maps_names_to_ids():
    taxonomy = load_default_layered_taxonomy()
    add_ids, remove_ids = genre_publish.classify_override_terms(
        taxonomy, add=["slowcore"], remove=["indie rock"]
    )
    # slowcore is a known leaf -> mapped to its genre_id
    assert any("slowcore" in gid for gid in add_ids)
    # indie rock maps to a known canonical genre id (e.g. indie_rock)
    assert remove_ids  # non-empty


def test_classify_override_terms_skips_unmappable():
    taxonomy = load_default_layered_taxonomy()
    add_ids, remove_ids = genre_publish.classify_override_terms(
        taxonomy, add=["zzzz not a genre zzzz"], remove=[]
    )
    assert add_ids == []


def _insert_override(side_path, release_key, artist, album, add, remove):
    sconn = sqlite3.connect(side_path)
    sconn.execute(
        "INSERT INTO ai_genre_user_overrides "
        "(release_key, normalized_artist, normalized_album, genres_add_json, "
        " genres_remove_json, updated_at) VALUES (?,?,?,?,?,?)",
        (release_key, artist, album, json.dumps(add), json.dumps(remove), "t"),
    )
    sconn.commit()
    sconn.close()


def test_build_resolved_graph_album_uses_graph_rows(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone')")
    conn.commit()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    # graph authority for ALB1
    conn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, album_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES ('acetone::york blvd','ALB1','acetone','york blvd','alternative_rock',"
        " 'observed_leaf',0.9,0.7,2,0,'{}','t')"
    )
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={"acetone::york blvd": "ALB1"},
                                       taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB1'").fetchall()
    assert ("alternative_rock", "graph") in [(r["genre_id"], r["source"]) for r in rows]


def test_build_resolved_legacy_album_uses_legacy_rows(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB2', 'B', 'Y')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release')")
    conn.commit()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={}, taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB2'").fetchall()
    assert rows and all(r["source"] == "legacy" for r in rows)
    assert "slowcore" in {r["genre_id"] for r in rows}


def test_build_resolved_applies_overrides(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    _insert_override(side, "y::b", "y", "b", add=["slowcore"], remove=["jangle pop"])
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO albums VALUES ('ALB3', 'B', 'Y')")
    conn.execute("INSERT INTO album_genres VALUES ('ALB3', 'Jangle Pop', 'discogs_release')")
    conn.commit()
    genre_publish.create_published_schema(conn)
    _attach(conn, side)
    genre_publish.copy_taxonomy(conn)
    taxonomy = load_default_layered_taxonomy()
    genre_publish.build_resolved_table(conn, key_to_album={"y::b": "ALB3"}, taxonomy=taxonomy)
    rows = conn.execute("SELECT genre_id, source FROM release_effective_genres "
                        "WHERE album_id='ALB3'").fetchall()
    ids = {r["genre_id"] for r in rows}
    assert any("slowcore" in gid for gid in ids)
    assert "jangle pop" not in ids
    assert any(r["source"] == "user" for r in rows)


def _full_fixture(tmp_path):
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    mconn = sqlite3.connect(meta)
    mconn.execute("INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone')")
    mconn.execute("INSERT INTO albums VALUES ('ALB2', 'B', 'Y')")
    mconn.execute("INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release')")
    mconn.commit()
    mconn.close()
    _insert_graph_assignment(side, "acetone::york blvd", "acetone", "york blvd",
                             "alternative_rock", "observed_leaf")
    return meta, side


def test_publish_end_to_end_and_stats(tmp_path):
    meta, side = _full_fixture(tmp_path)
    stats = genre_publish.publish(str(meta), str(side), dry_run=False)
    assert stats.graph_albums == 1
    assert stats.legacy_albums == 1
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    src = {r["album_id"]: r["source"] for r in conn.execute(
        "SELECT album_id, source FROM release_effective_genres GROUP BY album_id")}
    assert src["ALB1"] == "graph"
    assert src["ALB2"] == "legacy"


def test_publish_is_idempotent(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    first = conn.execute(
        "SELECT * FROM release_effective_genres ORDER BY album_id, genre_id, assignment_layer"
    ).fetchall()
    conn.close()
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    second = conn.execute(
        "SELECT * FROM release_effective_genres ORDER BY album_id, genre_id, assignment_layer"
    ).fetchall()
    conn.close()
    assert first == second


def test_publish_dry_run_writes_nothing(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=True)
    conn = sqlite3.connect(meta)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "release_effective_genres" not in names


def test_unpublish_drops_published_only(tmp_path):
    meta, side = _full_fixture(tmp_path)
    genre_publish.publish(str(meta), str(side), dry_run=False)
    conn = sqlite3.connect(meta)
    genre_publish.unpublish(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "release_effective_genres" not in names
    assert "albums" in names and "album_genres" in names


def test_cli_main_runs_dry_run(tmp_path, capsys):
    meta, side = _full_fixture(tmp_path)
    from scripts import publish_genres
    rc = publish_genres.main([
        "--metadata-db", str(meta), "--sidecar-db", str(side), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "graph_albums" in out


def test_publish_fans_graph_genres_to_all_album_ids_sharing_release_key(tmp_path):
    """A release fragmented into multiple album_ids (feat./collab variants that share
    one normalized release_key) must ALL receive the release's graph genres — not just
    the single lexicographically-smaller album_id, with the siblings silently falling
    back to legacy. Root cause of ~23 'un-enriched' albums (2026-06-25 audit)."""
    from src.ai_genre_enrichment.normalization import (
        normalize_release_artist, normalize_release_name)
    meta = _make_metadata(tmp_path)
    side = _make_sidecar(tmp_path)
    mconn = sqlite3.connect(meta)
    mconn.execute("INSERT INTO albums VALUES ('ALB_BASE', 'Flamagra', 'Flying Lotus')")
    mconn.execute(
        "INSERT INTO albums VALUES ('ALB_FEAT', 'Flamagra', 'Flying Lotus feat. Anderson Paak')")
    mconn.commit()
    mconn.close()
    key = f"{normalize_release_artist('Flying Lotus')}::{normalize_release_name('Flamagra')}"
    # precondition: the feat variant collapses to the same normalized release key
    assert key == (
        f"{normalize_release_artist('Flying Lotus feat. Anderson Paak')}::"
        f"{normalize_release_name('Flamagra')}"
    )
    _insert_graph_assignment(side, key, "flying lotus", "flamagra",
                             "alternative_rock", "observed_leaf")

    genre_publish.publish(str(meta), str(side), dry_run=False)

    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    src = {r["album_id"]: r["source"] for r in conn.execute(
        "SELECT album_id, source FROM release_effective_genres WHERE genre_id='alternative_rock'")}
    conn.close()
    assert src.get("ALB_BASE") == "graph"
    assert src.get("ALB_FEAT") == "graph"


def test_escalated_album_retains_prior_assignments(tmp_path):
    """An album with prior graph assignments that is later escalated (and therefore NOT
    re-materialized by apply) keeps its prior observed_leaf after publish."""
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.escalation_queue import EscalationQueue
    from src.ai_genre_enrichment.adjudication_apply import apply_adjudications
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    class FakeAdapter:
        def canonicalize_tag(self, t): return t
        def node(self, n): return None

    side = tmp_path / "side.db"
    store = SidecarStore(str(side))
    store.initialize()
    meta = sqlite3.connect(tmp_path / "m.db")
    meta.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT,
            release_year INTEGER, musicbrainz_release_id TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, album_id TEXT, title TEXT);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT);
        INSERT INTO albums VALUES ('a1','Souvlaki','Slowdive',1993,NULL);
        """
    )
    meta.commit()
    taxonomy = load_default_layered_taxonomy()
    queue = EscalationQueue(side)

    # First apply: non-escalated -> materializes shoegaze.
    rows1 = [("a1", "std", {"genres": [{"term": "shoegaze", "confidence": 0.9}],
                            "facets": [], "escalate": False})]
    apply_adjudications(rows=rows1, thorough_pv="tho", std_pv="std", meta_conn=meta,
                        id2name={}, taxonomy=taxonomy, adapter=FakeAdapter(),
                        sidecar_store=store, queue=queue)
    _conn = sqlite3.connect(side)
    before = _conn.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    _conn.close()
    assert before >= 1

    # Second apply: same album now ESCALATED -> must NOT clear prior assignments.
    rows2 = [("a1", "std", {"genres": [{"term": "dream pop", "confidence": 0.9}],
                            "facets": [], "escalate": True, "escalate_reason": "x",
                            "dropped_file_tags": []})]
    apply_adjudications(rows=rows2, thorough_pv="tho", std_pv="std", meta_conn=meta,
                        id2name={}, taxonomy=taxonomy, adapter=FakeAdapter(),
                        sidecar_store=store, queue=queue)
    _conn = sqlite3.connect(side)
    after = _conn.execute(
        "SELECT COUNT(*) FROM genre_graph_release_genre_assignments "
        "WHERE release_id='slowdive::souvlaki' AND assignment_layer='observed_leaf'"
    ).fetchone()[0]
    _conn.close()
    meta.close()
    assert after == before  # prior assignments preserved; nothing cleared


def test_publish_reimports_stale_sidecar_taxonomy(tmp_path):
    """publish() re-syncs a stale sidecar taxonomy from the YAML before copying.

    Regression: growing the taxonomy via the GUI wrote only the YAML; the sidecar
    graph tables stayed at the old version, so publish copied a stale canonical
    table and downstream ids fell back to raw tokens (2026-07-02).
    """
    meta, side = _full_fixture(tmp_path)
    real = load_default_layered_taxonomy()

    # Stale the sidecar: fake old version + drop all but a handful of genres.
    sconn = sqlite3.connect(side)
    keep = [r[0] for r in sconn.execute(
        "SELECT genre_id FROM genre_graph_canonical_genres LIMIT 3")]
    sconn.execute("UPDATE genre_graph_canonical_genres SET taxonomy_version='0.0.1-stale'")
    sconn.execute(
        "DELETE FROM genre_graph_canonical_genres WHERE genre_id NOT IN (?,?,?)", keep)
    sconn.commit()
    assert sconn.execute(
        "SELECT COUNT(*) FROM genre_graph_canonical_genres").fetchone()[0] == 3
    sconn.close()

    genre_publish.publish(str(meta), str(side), dry_run=False)

    sconn = sqlite3.connect(side)
    versions = [r[0] for r in sconn.execute(
        "SELECT DISTINCT taxonomy_version FROM genre_graph_canonical_genres")]
    count = sconn.execute(
        "SELECT COUNT(*) FROM genre_graph_canonical_genres").fetchone()[0]
    sconn.close()
    assert versions == [real.version]      # re-synced to the current YAML version
    assert count == len(real.genres)       # full canonical genre set restored


def test_publish_dry_run_does_not_reimport_sidecar(tmp_path):
    """A dry run previews without mutating the sidecar (re-import is skipped)."""
    meta, side = _full_fixture(tmp_path)
    sconn = sqlite3.connect(side)
    sconn.execute("UPDATE genre_graph_canonical_genres SET taxonomy_version='0.0.1-stale'")
    sconn.commit()
    sconn.close()

    genre_publish.publish(str(meta), str(side), dry_run=True)

    sconn = sqlite3.connect(side)
    versions = [r[0] for r in sconn.execute(
        "SELECT DISTINCT taxonomy_version FROM genre_graph_canonical_genres")]
    sconn.close()
    assert versions == ["0.0.1-stale"]     # untouched by the dry run
