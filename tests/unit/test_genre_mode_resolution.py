import sqlite3
import pytest
from src.genre.authority import canonical_genre_search


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
    c.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?,?,?,?,?,?)",
        [
            ("hip_hop", "hip hop", "genre", 0.5, "active", "1"),
            ("r_b_soul", "r b soul", "genre", 0.5, "active", "1"),
            ("shoegaze", "shoegaze", "genre", 0.9, "active", "1"),
            ("dead_genre", "dead genre", "genre", 0.9, "deprecated", "1"),
            # Mirrors the live funk/funk_metal pair (funk=0.58, funk metal=0.80):
            # a more specific genre substring-matches the exact query.
            ("funk", "funk", "genre", 0.58, "active", "1"),
            ("funk_metal", "funk metal", "genre", 0.80, "active", "1"),
        ],
    )
    c.executemany(
        "INSERT INTO genre_graph_aliases VALUES (?,?,?,?)",
        [
            ("hip-hop", "hip_hop", "reviewed_taxonomy", 1.0),
            ("rnb", "r_b_soul", "reviewed_taxonomy", 1.0),
            ("r&b", "r_b_soul", "reviewed_taxonomy", 1.0),
            ("dead alias", "dead_genre", "reviewed_taxonomy", 1.0),
        ],
    )
    return c


def test_alias_resolves_to_canonical(conn):
    assert canonical_genre_search(conn, "hip-hop") == [("hip_hop", "hip hop")]


def test_ampersand_alias_resolves(conn):
    assert canonical_genre_search(conn, "r&b") == [("r_b_soul", "r b soul")]


def test_canonical_name_still_matches(conn):
    assert canonical_genre_search(conn, "shoegaze") == [("shoegaze", "shoegaze")]


def test_no_duplicate_when_both_match(conn):
    # "hip" matches the canonical name AND the alias; the genre must appear once.
    assert canonical_genre_search(conn, "hip") == [("hip_hop", "hip hop")]


def test_alias_to_inactive_genre_is_excluded(conn):
    assert canonical_genre_search(conn, "dead alias") == []


def test_empty_query_returns_empty(conn):
    assert canonical_genre_search(conn, "  ") == []


def test_exact_name_match_beats_more_specific_substring_match(conn):
    # Regression: querying "funk" used to return "funk metal" first (higher
    # specificity_score, same "%funk%" substring) instead of the exact "funk"
    # match — found via genre mode's acceptance run (2026-07-24).
    results = canonical_genre_search(conn, "funk", limit=10)
    assert results[0] == ("funk", "funk")
    ids = [gid for gid, _ in results]
    assert "funk_metal" in ids  # still returned, just not first
