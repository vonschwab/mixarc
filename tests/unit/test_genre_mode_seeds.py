# tests/unit/test_genre_mode_seeds.py
import pytest
from src.playlist.genre_mode import (
    SeedWeights, score_seed_candidates, select_piers_from_clusters,
)


import numpy as np
from src.playlist.genre_mode import sonic_typicality


def _score(**kw):
    base = dict(
        indices=[0, 1, 2],
        artist_keys={0: "a", 1: "b", 2: "c"},
        prominence_by_artist={"a": 1.0, "b": 0.5, "c": 0.0},
        canonicity_by_index={0: 1.0, 1: 1.0, 2: 1.0},
        centrality_by_index={0: 1.0, 1: 1.0, 2: 1.0},
        typicality_by_index={0: 1.0, 1: 1.0, 2: 1.0},
        investment_by_artist={"a": 0.0, "b": 0.0, "c": 0.0},
        weights=SeedWeights(),
    )
    base.update(kw)
    return score_seed_candidates(**base)


def test_sonic_typicality_ranks_by_closeness_to_genre_centroid():
    # Indices 0,1 are near-identical; 2 is orthogonal — 2 is the atypical one.
    X = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    t = sonic_typicality(X, [0, 1, 2])
    assert t[0] > t[2] and t[1] > t[2]


def test_sonic_typicality_is_bounded_zero_to_one():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    t = sonic_typicality(X, [0, 1, 2])
    assert all(0.0 <= v <= 1.0 for v in t.values())


def test_sonic_typicality_empty_members_returns_empty():
    assert sonic_typicality(np.zeros((3, 2)), []) == {}


def test_prominence_orders_candidates():
    s = _score()
    assert s[0] > s[1] > s[2]


def test_uncached_artist_is_not_penalised_below_a_scored_low_one():
    # "c" is absent from prominence (uncached) and must NOT rank below an artist
    # explicitly scored 0.0 — absence is "no signal", not "unpopular".
    s = _score(prominence_by_artist={"a": 1.0, "b": 0.0})
    assert s[2] > s[1]


def test_centrality_breaks_ties_when_prominence_equal():
    s = _score(prominence_by_artist={"a": 0.5, "b": 0.5, "c": 0.5},
               centrality_by_index={0: 1.0, 1: 0.5, 2: 0.0})
    assert s[0] > s[1] > s[2]


def test_zero_weight_signal_has_no_effect():
    w = SeedWeights(prominence=0.0)
    a = score_seed_candidates(
        indices=[0, 1], artist_keys={0: "a", 1: "b"},
        prominence_by_artist={"a": 1.0, "b": 0.0},
        canonicity_by_index={0: 1.0, 1: 1.0}, centrality_by_index={0: 1.0, 1: 1.0},
        typicality_by_index={0: 1.0, 1: 1.0},
        investment_by_artist={}, weights=w)
    assert a[0] == pytest.approx(a[1])


def test_one_pier_per_artist_across_clusters():
    # Cluster 0's best and cluster 1's best are the same artist; cluster 1 must
    # fall through to its next-best candidate from a different artist.
    piers = select_piers_from_clusters(
        clusters=[[0, 1], [2, 3]],
        scores={0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2},
        artist_keys={0: "a", 1: "b", 2: "a", 3: "c"},
        min_cluster_size=1,
        bridgeable=None,
    )
    assert piers == [0, 3]


def test_unbridgeable_candidates_are_skipped():
    piers = select_piers_from_clusters(
        clusters=[[0, 1]], scores={0: 0.9, 1: 0.1},
        artist_keys={0: "a", 1: "b"}, min_cluster_size=1, bridgeable={1},
    )
    assert piers == [1]


def test_small_clusters_are_skipped():
    piers = select_piers_from_clusters(
        clusters=[[0], [1, 2, 3]], scores={0: 9.0, 1: 0.5, 2: 0.4, 3: 0.3},
        artist_keys={0: "a", 1: "b", 2: "c", 3: "d"},
        min_cluster_size=3, bridgeable=None,
    )
    assert piers == [1]


def test_cluster_with_no_eligible_candidate_is_dropped_not_crashed():
    piers = select_piers_from_clusters(
        clusters=[[0], [1]], scores={0: 0.9, 1: 0.8},
        artist_keys={0: "a", 1: "a"}, min_cluster_size=1, bridgeable=None,
    )
    assert piers == [0]


def test_atypical_track_loses_to_typical_one():
    s = _score(typicality_by_index={0: 0.1, 1: 1.0, 2: 1.0},
               prominence_by_artist={"a": 1.0, "b": 1.0, "c": 1.0})
    assert s[1] > s[0]


def test_epoch_changes_selection_deterministically():
    """(C) deterministic per (genre, epoch): a different epoch yields a different
    pier set, but the same epoch always yields the same one."""
    args = dict(
        clusters=[[0, 1, 2]], scores={0: 0.9, 1: 0.85, 2: 0.8},
        artist_keys={0: "a", 1: "b", 2: "c"}, min_cluster_size=1, bridgeable=None,
    )
    e0a = select_piers_from_clusters(**args, epoch=0)
    e0b = select_piers_from_clusters(**args, epoch=0)
    e1 = select_piers_from_clusters(**args, epoch=1)
    assert e0a == e0b
    assert e1 != e0a


def test_epoch_zero_picks_the_top_scorer():
    piers = select_piers_from_clusters(
        clusters=[[0, 1, 2]], scores={0: 0.9, 1: 0.85, 2: 0.8},
        artist_keys={0: "a", 1: "b", 2: "c"}, min_cluster_size=1,
        bridgeable=None, epoch=0,
    )
    assert piers == [0]


def test_epoch_preserves_cluster_priority_by_true_best_score():
    """A strong cluster must claim its artist first even when epoch rotates each
    cluster's candidate list — priority follows the cluster's true best score,
    not whichever candidate rotation happened to put first."""
    piers = select_piers_from_clusters(
        clusters=[[10, 11], [20, 21]],
        scores={10: 0.9, 11: 0.1, 20: 0.5, 21: 0.45},
        artist_keys={10: "a", 11: "b", 20: "c", 21: "b"},
        min_cluster_size=1, bridgeable=None, epoch=1,
    )
    # Cluster A (true best 0.9) goes first, takes 11 (artist b). Cluster B then
    # finds its rotated head 21 is artist b (taken) and falls through to 20.
    assert piers == [11, 20]
