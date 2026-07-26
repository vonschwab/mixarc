import json
import sqlite3
import pytest
from src.analyze.popularity_runner import artist_prominence, init_top_tracks_cache


@pytest.fixture
def cache_db(tmp_path):
    p = str(tmp_path / "pop.db")
    init_top_tracks_cache(p)
    conn = sqlite3.connect(p)
    def add(key, tracks):
        conn.execute(
            "INSERT INTO artist_top_tracks_cache VALUES (?,?,?,?)",
            (key, "2026-07-24T00:00:00", len(tracks), json.dumps(tracks)),
        )
    add("slowdive", [{"name": "When the Sun Hits", "playcount": 27670000,
                      "listeners": 1591135, "mbid": "", "rank": 0}])
    add("helado negro", [{"name": "Running", "playcount": 472000,
                          "listeners": 92816, "mbid": "", "rank": 0}])
    add("tiny band", [{"name": "Song", "playcount": 300, "listeners": 100,
                       "mbid": "", "rank": 0}])
    add("no data", [])
    conn.commit()
    conn.close()
    return p


def test_bigger_artist_scores_higher(cache_db):
    p = artist_prominence(cache_db, {"slowdive", "helado negro", "tiny band"})
    assert p["slowdive"] > p["helado negro"] > p["tiny band"]


def test_scores_are_bounded_zero_to_one(cache_db):
    p = artist_prominence(cache_db, {"slowdive", "helado negro", "tiny band"})
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_uncached_artist_is_absent_not_zero(cache_db):
    p = artist_prominence(cache_db, {"slowdive", "never fetched"})
    assert "never fetched" not in p
    assert "slowdive" in p


def test_artist_with_empty_payload_is_absent(cache_db):
    assert "no data" not in artist_prominence(cache_db, {"no data", "slowdive"})


def test_empty_input_returns_empty(cache_db):
    assert artist_prominence(cache_db, set()) == {}


def test_single_artist_does_not_divide_by_zero(cache_db):
    p = artist_prominence(cache_db, {"slowdive"})
    assert p["slowdive"] == pytest.approx(1.0)
