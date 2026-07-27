import sqlite3

from src.genre import authority


def _canon(conn):
    conn.execute(
        "CREATE TABLE genre_graph_canonical_genres ("
        "genre_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        "specificity_score REAL NOT NULL, status TEXT NOT NULL, taxonomy_version TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE genre_graph_aliases ("
        "alias TEXT, canonical_genre_id TEXT, source TEXT, confidence REAL)"
    )
    conn.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        [("dream_pop", "Dream Pop", "genre", 0.8, "active", "v1"),
         ("dreamo", "Dreamo", "genre", 0.7, "active", "v1"),
         ("shoegaze", "Shoegaze", "genre", 0.9, "active", "v1"),
         ("old_thing", "Old Dream", "genre", 0.5, "deprecated", "v1")],
    )


def test_canonical_genre_search_matches_active_by_name():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _canon(conn)
    out = authority.canonical_genre_search(conn, "dream", limit=10)
    names = [n for _, n in out]
    assert "Dream Pop" in names and "Dreamo" in names
    assert "Old Dream" not in names  # deprecated excluded
    assert ("shoegaze", "Shoegaze") not in out


def test_canonical_genre_search_empty_query():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _canon(conn)
    assert authority.canonical_genre_search(conn, "  ", limit=10) == []


def test_album_id_for_release_exact_and_orphan():
    from src.genre import genre_edit
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT)")
    conn.executemany(
        "INSERT INTO tracks VALUES (?,?,?,?)",
        [("t1", "The  Radio Dept.", "Pet Grief", "ORPH1"),
         ("t2", "The  Radio Dept.", "Pet Grief", "ORPH1"),
         ("t3", "Acetone", "York Blvd.", "A1")],
    )
    assert genre_edit.album_id_for_release(conn, "The  Radio Dept.", "Pet Grief") == "ORPH1"
    # normalized fallback: double-space vs single-space artist still resolves
    assert genre_edit.album_id_for_release(conn, "The Radio Dept.", "Pet Grief") == "ORPH1"
    assert genre_edit.album_id_for_release(conn, "Nobody", "Nothing") is None


def _edit_dbs(tmp_path):
    """A metadata.db with the tables the edit path reads/writes."""
    meta = sqlite3.connect(tmp_path / "m.db")
    meta.row_factory = sqlite3.Row
    meta.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, album_id TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE genre_graph_release_genre_assignments "
        "(release_id TEXT, album_id TEXT, genre_id TEXT, assignment_layer TEXT, confidence REAL);"
        "CREATE TABLE genre_graph_canonical_genres "
        "(genre_id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, "
        " specificity_score REAL NOT NULL, status TEXT NOT NULL, taxonomy_version TEXT NOT NULL);"
        "CREATE TABLE release_effective_genres "
        "(album_id TEXT NOT NULL, release_key TEXT, genre_id TEXT NOT NULL, "
        " assignment_layer TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, "
        " PRIMARY KEY (album_id, genre_id, assignment_layer));"
    )
    meta.execute("INSERT INTO tracks VALUES ('t1','The  Radio Dept.','Pet Grief','ORPH1')")
    meta.commit()
    return meta


def test_apply_edit_orphan_zero_to_two(tmp_path):
    from src.genre import genre_edit
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()

    res = genre_edit.apply_user_genre_edit(
        meta, store, taxonomy,
        artist="The  Radio Dept.", album="Pet Grief",
        target_names=["dream pop", "shoegaze"],
    )
    assert res.no_change is False
    user_rows = meta.execute(
        "SELECT genre_id FROM release_effective_genres "
        "WHERE album_id='ORPH1' AND source='user'"
    ).fetchall()
    assert len(user_rows) == len(res.added) == 2
    ov = store.get_user_override("the radio dept::pet grief")
    assert ov is not None and len(ov["genres_add"]) == 2


def test_apply_edit_no_op_when_unchanged(tmp_path):
    from src.genre import genre_edit
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["dream pop"])
    # Re-apply identical target → no change.
    res2 = genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["dream pop"])
    assert res2.no_change is True
    assert res2.added == [] and res2.removed == []


def test_apply_edit_keeps_user_leaf_that_also_has_an_inferred_row(tmp_path):
    """A user genre that ALSO appears as an inferred graph row keeps its leaf row.

    Regression (found 2026-07-27 on Dua Lipa's "Radical Optimism (Extended)"):
    ``add_ids = target_ids - non_user_ids`` and ``non_user_ids`` spanned every
    layer, so a genre the user asserted as ``observed_leaf`` that the graph also
    carried as ``inferred_parent`` got no user row on rewrite. It survived only
    as the inferred row — and the artifact builder excludes inferred layers from
    ``X_genre_raw``, so the genre silently left the generation vector.
    """
    from src.genre import genre_edit, genre_publish
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()

    electro_id = genre_publish._term_to_genre_id(taxonomy, "electropop")
    house_id = genre_publish._term_to_genre_id(taxonomy, "house")
    assert electro_id and house_id and electro_id != house_id

    # Graph carries house as a leaf and electropop only as an inferred parent.
    meta.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "VALUES ('the radio dept::pet grief','ORPH1',?,'observed_leaf',0.9)", (house_id,))
    meta.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "VALUES ('the radio dept::pet grief','ORPH1',?,'inferred_parent',0.5)", (electro_id,))
    # Published state: the user additionally asserted electropop as a leaf.
    meta.execute("INSERT INTO release_effective_genres "
                 "VALUES ('ORPH1','k',?, 'observed_leaf',0.9,'graph')", (house_id,))
    meta.execute("INSERT INTO release_effective_genres "
                 "VALUES ('ORPH1','k',?, 'inferred_parent',0.5,'graph')", (electro_id,))
    meta.execute("INSERT INTO release_effective_genres "
                 "VALUES ('ORPH1','k',?, 'observed_leaf',1.0,'user')", (electro_id,))
    meta.commit()

    # A user opens the dialog (chips = every published name) and adds one genre.
    genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["house", "electropop", "dream pop"])

    layers = {
        (r[0], r[1])
        for r in meta.execute(
            "SELECT genre_id, assignment_layer FROM release_effective_genres "
            "WHERE album_id='ORPH1'")
    }
    assert (electro_id, "observed_leaf") in layers, (
        "electropop lost its observed_leaf row and survives only as inferred"
    )


def test_apply_edit_does_not_promote_an_unasserted_inferred_genre(tmp_path):
    """An inferred genre nobody asserted stays inferred, even though it's in the target.

    The edit dialog seeds its chips from every published layer, so the target set
    routinely contains inferred hub families. Promoting those to ``observed_leaf``
    would bake hub genres into ``X_genre_raw`` at full weight — the 2026-06-12
    genre-vector saturation incident. Guards the fix for
    ``test_apply_edit_keeps_user_leaf_that_also_has_an_inferred_row`` from
    over-correcting.
    """
    from src.genre import genre_edit, genre_publish
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()

    house_id = genre_publish._term_to_genre_id(taxonomy, "house")
    pop_id = genre_publish._term_to_genre_id(taxonomy, "pop")
    assert house_id and pop_id and house_id != pop_id

    meta.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "VALUES ('the radio dept::pet grief','ORPH1',?,'observed_leaf',0.9)", (house_id,))
    meta.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "VALUES ('the radio dept::pet grief','ORPH1',?,'inferred_family',0.4)", (pop_id,))
    meta.execute("INSERT INTO release_effective_genres "
                 "VALUES ('ORPH1','k',?, 'observed_leaf',0.9,'graph')", (house_id,))
    meta.execute("INSERT INTO release_effective_genres "
                 "VALUES ('ORPH1','k',?, 'inferred_family',0.4,'graph')", (pop_id,))
    meta.commit()

    # Chips carry the inferred family; the user only adds dream pop.
    genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["house", "pop", "dream pop"])

    rows = {
        (r[0], r[1], r[2])
        for r in meta.execute(
            "SELECT genre_id, assignment_layer, source FROM release_effective_genres "
            "WHERE album_id='ORPH1'")
    }
    assert (pop_id, "inferred_family", "graph") in rows
    assert (pop_id, "observed_leaf", "user") not in rows, "hub family promoted to a leaf"


def test_apply_edit_removes_graph_genre(tmp_path):
    """Removing a graph-sourced genre drops it from the authority and records a
    remove override (diffed against the non-user base, so publish reproduces it)."""
    from src.genre import genre_edit, genre_publish
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    meta = _edit_dbs(tmp_path)
    store = SidecarStore(str(tmp_path / "s.db"))
    store.initialize()
    taxonomy = load_default_layered_taxonomy()

    slow_id = genre_publish._term_to_genre_id(taxonomy, "slowcore")
    dream_id = genre_publish._term_to_genre_id(taxonomy, "dream pop")
    assert slow_id and dream_id and slow_id != dream_id

    # ORPH1 currently has two GRAPH genres (mirrors a prior publish): graph
    # assignments are the materializer's source, release_effective_genres the base.
    for gid in (slow_id, dream_id):
        meta.execute(
            "INSERT INTO genre_graph_release_genre_assignments "
            "VALUES ('the radio dept::pet grief', 'ORPH1', ?, 'observed_leaf', 0.9)", (gid,))
        meta.execute(
            "INSERT INTO release_effective_genres "
            "VALUES ('ORPH1','k', ?, 'observed_leaf', 0.9, 'graph')", (gid,))
    meta.commit()

    res = genre_edit.apply_user_genre_edit(
        meta, store, taxonomy, artist="The  Radio Dept.", album="Pet Grief",
        target_names=["slowcore"])  # keep slowcore, drop dream pop

    assert res.no_change is False
    remaining = {r[0] for r in meta.execute(
        "SELECT genre_id FROM release_effective_genres WHERE album_id='ORPH1'")}
    assert slow_id in remaining
    assert dream_id not in remaining

    ov = store.get_user_override("the radio dept::pet grief")
    assert ov is not None
    assert ov["genres_add"] == []
    assert len(ov["genres_remove"]) == 1
