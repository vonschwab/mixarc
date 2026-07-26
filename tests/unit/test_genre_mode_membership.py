# tests/unit/test_genre_mode_membership.py
import sqlite3
import pytest
from src.playlist import genre_mode


class FakeSteering:
    """Stands in for TaxonomySteering: canonical-name keyed similarity."""

    def __init__(self, sims):
        self._sims = sims

    def similarity(self, a, b):
        if a == b:
            return 1.0
        return self._sims.get((a, b), self._sims.get((b, a), 0.0))


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute(
        "CREATE TABLE genre_graph_canonical_genres "
        "(genre_id TEXT, name TEXT, kind TEXT, specificity_score REAL, "
        " status TEXT, taxonomy_version TEXT)"
    )
    c.execute(
        "CREATE TABLE genre_graph_aliases "
        "(alias TEXT, canonical_genre_id TEXT, source TEXT, confidence REAL)"
    )
    c.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT)")
    c.execute(
        "CREATE TABLE release_effective_genres "
        "(album_id TEXT, release_key TEXT, genre_id TEXT, assignment_layer TEXT, "
        " confidence REAL, source TEXT)"
    )
    c.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        [
            ("shoegaze", "shoegaze", "genre", 0.9, "active", "1"),
            ("dream_pop", "dream pop", "genre", 0.8, "active", "1"),
            ("noise_pop", "noise pop", "genre", 0.8, "active", "1"),
            ("polka", "polka", "genre", 0.8, "active", "1"),
        ],
    )
    c.executemany(
        "INSERT INTO tracks VALUES (?,?,?)",
        [("t1", "Slowdive", "alb_sg"), ("t2", "Slowdive", "alb_sg"),
         ("t3", "Beach House", "alb_dp"), ("t4", "Polka Band", "alb_pk"),
         ("t5", "Guessed", "alb_inf")],
    )
    c.executemany(
        "INSERT INTO release_effective_genres VALUES (?,?,?,?,?,?)",
        [
            ("alb_sg", "k1", "shoegaze", "observed_leaf", 0.9, "file"),
            ("alb_dp", "k2", "dream_pop", "observed_leaf", 0.8, "file"),
            ("alb_pk", "k3", "polka", "observed_leaf", 0.8, "file"),
            ("alb_inf", "k4", "shoegaze", "inferred_parent", 0.4, "graph"),
        ],
    )
    return c


@pytest.fixture
def steering():
    return FakeSteering({
        ("shoegaze", "dream pop"): 0.69,
        ("shoegaze", "noise pop"): 0.56,
        ("shoegaze", "polka"): 0.01,
    })


def test_resolve_returns_canonical_id_and_name(conn):
    res = genre_mode.resolve_genre_query(conn, "shoegaze")
    assert (res.genre_id, res.name) == ("shoegaze", "shoegaze")


def test_resolve_unknown_returns_none(conn):
    assert genre_mode.resolve_genre_query(conn, "zzzznotagenre") is None


def test_seed_members_exclude_inferred_layers(conn):
    # t5 is on an inferred_parent assignment and must NOT be a seed candidate.
    assert genre_mode.seed_member_track_ids(conn, {"shoegaze"}) == {"t1", "t2"}


def test_seed_members_union_the_whole_genre_family(conn):
    # `dream_pop` standing in for a transitive is_a descendant: its tracks are
    # members of the parent genre, not merely pool neighbours.
    assert genre_mode.seed_member_track_ids(
        conn, {"shoegaze", "dream_pop"}
    ) == {"t1", "t2", "t3"}


def test_neighbors_above_threshold_excludes_self_and_low(conn, steering):
    names = {"shoegaze": "shoegaze", "dream_pop": "dream pop",
             "noise_pop": "noise pop", "polka": "polka"}
    got = genre_mode.neighbors_above_threshold(steering, "shoegaze", names, 0.5)
    assert set(got) == {"dream_pop", "noise_pop"}
    assert got["dream_pop"] == pytest.approx(0.69)


def test_higher_threshold_narrows(conn, steering):
    names = {"shoegaze": "shoegaze", "dream_pop": "dream pop",
             "noise_pop": "noise pop", "polka": "polka"}
    got = genre_mode.neighbors_above_threshold(steering, "shoegaze", names, 0.6)
    assert set(got) == {"dream_pop"}


def test_pool_includes_exact_plus_neighbors(conn, steering):
    ids, sims = genre_mode.pool_track_ids(conn, steering, {"shoegaze"}, "shoegaze", 0.5)
    assert ids == {"t1", "t2", "t3"}      # shoegaze + dream_pop, not polka
    assert sims["shoegaze"] == 1.0


def test_pool_at_impossible_threshold_is_family_only(conn, steering):
    ids, _ = genre_mode.pool_track_ids(conn, steering, {"shoegaze"}, "shoegaze", 0.99)
    assert ids == {"t1", "t2"}


def test_family_members_are_pinned_at_similarity_one(conn, steering):
    # polka scores 0.01 as a neighbour, far below the threshold. As a (contrived)
    # family member it must still enter the pool at 1.0 — a descendant IS the
    # genre, so its name-similarity score is irrelevant. This is also what
    # guarantees seed members are a subset of the pool.
    ids, sims = genre_mode.pool_track_ids(
        conn, steering, {"shoegaze", "polka"}, "shoegaze", 0.5
    )
    assert sims["polka"] == 1.0
    assert "t4" in ids
