# tests/unit/test_genre_mode_seeds.py
import sqlite3
from unittest.mock import patch

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


def test_medoid_precap_would_have_hidden_a_high_prominence_candidate():
    """Illustrates the finding at the select_piers_from_clusters boundary: when
    cluster_artist_tracks's medoid pre-filter caps a cluster's candidate list to
    the top-N by sonic centrality (its own ranking criterion, unrelated to the
    composite score), a track that scores highest on the COMPOSITE score
    (prominence-dominated here) but is only #16 of 20 by centrality never even
    reaches select_piers_from_clusters when N=8 (the old hardcoded cap) -- it
    loses to a sonically-central-but-unremarkable candidate. With the cap lifted
    (the fix), the same composite score correctly promotes it.
    """
    n = 20
    # Index order here IS the centrality rank order (0 = most sonically central,
    # matching _medoids_for_cluster's own argsort-by-similarity-to-centroid
    # ranking) -- mirrors how cluster_artist_tracks orders medoids_by_cluster.
    centrality_ranked = list(range(n))
    high_prominence_index = 15  # rank #16 by centrality -- outside an old top-8 cap
    # Composite scores: monotonically decreasing with centrality rank, EXCEPT the
    # prominence-dominated candidate, which wins the whole cluster on the
    # composite (prominence 1.0 * weight 1.0 dwarfs the ~0.1-scale spread here).
    scores = {i: 0.1 * (n - i) for i in centrality_ranked}
    scores[high_prominence_index] = 100.0
    artist_keys = {i: f"artist_{i}" for i in centrality_ranked}

    # OLD: cluster_artist_tracks's medoid_top_k=8 pre-filter has already thrown
    # away everything past centrality rank 8 BEFORE the composite score is ever
    # computed on it -- select_piers_from_clusters never even sees index 15.
    old_precapped_cluster = centrality_ranked[:8]
    old_piers = select_piers_from_clusters(
        clusters=[old_precapped_cluster], scores=scores, artist_keys=artist_keys,
        min_cluster_size=1, bridgeable=None,
    )
    assert high_prominence_index not in old_piers
    assert old_piers == [0]  # best-by-composite among the precapped 8 is rank 0

    # NEW: the pre-filter is uncapped (a pure veto, not a ranking cut), so the
    # composite score sees the whole cluster and correctly promotes the
    # prominent-but-sonically-peripheral candidate.
    new_piers = select_piers_from_clusters(
        clusters=[centrality_ranked], scores=scores, artist_keys=artist_keys,
        min_cluster_size=1, bridgeable=None,
    )
    assert new_piers == [high_prominence_index]


def test_create_playlist_for_genre_passes_configured_pool_size_not_hardcoded_8(
    monkeypatch, tmp_path,
):
    """Regression for the "medoid pre-filter silently caps the composite score's
    reach" finding (audit Important). create_playlist_for_genre must pass
    playlists.genre_playlist.seed_candidate_pool_size through to
    cluster_artist_tracks's medoid_top_k -- not a hardcoded 8. Patches
    cluster_artist_tracks to capture its kwargs and abort immediately; everything
    upstream (genre resolution/pool relaxation/artifact bundle) is faked to the
    minimum needed to reach that call, so this never touches a real DB or
    artifact.
    """
    from src.config_loader import Config
    from src.playlist_generator import PlaylistGenerator
    from src.playlist import genre_mode as genre_mode_module

    db_path = tmp_path / "empty.db"
    sqlite3.connect(str(db_path)).close()  # just needs to exist & be openable ro

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "library:\n"
        f"  database_path: {db_path.as_posix()}\n"
        "playlists:\n"
        "  genre_playlist:\n"
        "    seed_candidate_pool_size: 54321\n"
        "  ds_pipeline:\n"
        f"    artifact_path: {(tmp_path / 'artifact.npz').as_posix()}\n"
    )
    cfg = Config(str(config_path))

    class _StopEarly(Exception):
        pass

    captured: dict = {}

    def _fake_cluster_artist_tracks(**kwargs):
        captured.update(kwargs)
        raise _StopEarly

    class _FakeBundle:
        track_ids = ["t0", "t1", "t2", "t3"]
        artist_keys = ["a0", "a1", "a2", "a3"]

    class _FakeResolution:
        genre_id = "shoegaze"
        name = "shoegaze"

    monkeypatch.setattr(
        genre_mode_module, "resolve_genre_query",
        lambda conn, q: _FakeResolution(),
    )
    monkeypatch.setattr(
        genre_mode_module, "seed_member_track_ids",
        lambda conn, gid: {"t0", "t1", "t2", "t3"},
    )
    monkeypatch.setattr(
        genre_mode_module, "resolve_pool_with_relaxation",
        lambda *a, **kw: ({"t0", "t1", "t2", "t3"}, {}, 0.35),
    )

    generator = PlaylistGenerator(library_client=object(), config=cfg)

    with patch(
        "src.playlist.pier_bridge.taxonomy_steering.get_taxonomy_steering",
        return_value=object(),
    ), patch(
        "src.playlist_generator.load_artifact_bundle", return_value=_FakeBundle(),
    ), patch(
        "src.playlist_generator.cluster_artist_tracks",
        side_effect=_fake_cluster_artist_tracks,
    ):
        with pytest.raises(_StopEarly):
            generator.create_playlist_for_genre(genre_name="shoegaze", track_count=30)

    assert captured.get("medoid_top_k") == 54321
