import numpy as np
import pytest

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def test_narrow_sonic_floor_rejects_negative():
    # Seed + two candidates; floor=0.10 rejects 0.09 but admits 0.10+
    embedding = np.array([
        [1.0, 0.0],  # seed
        [0.09, np.sqrt(1 - 0.09**2)],  # 0.09 cosine to seed
        [0.10, np.sqrt(1 - 0.10**2)],  # 0.10 cosine to seed
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "low", "pass"])
    track_ids = np.array(["seed", "low_id", "pass_id"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.10,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )

    assert result.pool_indices.tolist() == [2]
    assert result.stats["below_sonic_similarity"] == 1
    assert result.stats["seed_sonic_sim_track_ids"]["pass_id"] == result.sonic_sim[0]


def test_dynamic_sonic_floor_rejects_negative():
    embedding = np.array([
        [1.0, 0.0],
        [-0.01, np.sqrt(1 - 0.01**2)],  # -0.01 cosine
        [0.0, 1.0],  # 0.0 cosine
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "neg", "zero"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.0,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="dynamic",
    )
    assert "neg" not in [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "zero" in [artist_keys[i] for i in result.pool_indices.tolist()]


def test_candidate_pool_excludes_configured_title_words():
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    artist_keys = np.array(["seed", "skit_artist", "song_artist"])
    track_titles = np.array(["Seed", "Intro Skit", "Album Track"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        title_hard_exclude_flags=frozenset({"skit", "interlude", "acapella"}),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=track_titles,
        cfg=cfg,
        random_seed=0,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="dynamic",
    )

    assert [artist_keys[i] for i in result.pool_indices.tolist()] == ["song_artist"]
    assert result.stats["title_exclusion_rejected"] == 1


def test_dynamic_floor_config_override():
    # Floor override to 0.20 should reject 0.19 and admit 0.20
    embedding = np.array([
        [1.0, 0.0],
        [0.19, np.sqrt(1 - 0.19**2)],
        [0.20, np.sqrt(1 - 0.20**2)],
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "low", "high"])
    cfg_override = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.20,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg_override,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )
    assert "low" not in [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "high" in [artist_keys[i] for i in result.pool_indices.tolist()]


def test_multi_seed_max_similarity_used_for_floor():
    embedding = np.array([
        [1.0, 0.0],   # seed A
        [0.0, 1.0],   # seed B
        [0.0, 1.0],   # candidate similar only to seed B
        [-1.0, 0.0],  # candidate below floor to both
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed_a", "seed_b", "cand_pass", "cand_drop"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.5,
        max_pool_size=10,
        target_artists=4,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0, 1],
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )
    artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "cand_pass" in artists
    assert "cand_drop" not in artists


def test_title_interlude_and_skit_words_are_excluded_from_candidate_pool():
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    artist_keys = np.array(["seed", "interlude", "skit", "skitter", "interluded"])
    track_titles = np.array([
        "Seed Song",
        "Quiet Interlude",
        "Studio Skit (Live)",
        "Skitter Theme",
        "Interluded Again",
    ])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=track_titles,
        cfg=cfg,
        random_seed=0,
        X_sonic=embedding,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="dynamic",
    )

    admitted_artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "interlude" not in admitted_artists
    assert "skit" not in admitted_artists
    assert admitted_artists == ["interluded", "skitter"]
    assert result.stats["title_exclusion_rejected"] == 2


def test_broad_genre_mask_prevents_inflation():
    # Seed genres: rock, folk; candidate: indie rock
    # Broad filter removes "rock", leaving no overlap => genre sim zero => rejected by gate
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["a", "b"])

    genre_vocab = ["rock", "folk", "indie rock"]
    X_genre_raw = np.array([
        [1, 1, 0],  # seed
        [1, 0, 1],  # candidate
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=("rock",),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=None,
        min_genre_similarity=0.3,
        genre_method="weighted_jaccard",
        genre_vocab=genre_vocab,
        broad_filters=cfg.broad_filters,
        mode="narrow",
    )

    assert result.pool_indices.tolist() == []  # candidate should be rejected by genre gate


def test_dynamic_broad_genre_overlap_guard_blocks_smoothed_inflation():
    # Artist-mode often runs DS mode=dynamic while genre_mode=narrow supplies
    # broad filters. A candidate that only shares a broad raw tag should not be
    # rescued by smoothed secondary tags.
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "cand"])

    genre_vocab = ["rock", "dream pop", "hip hop"]
    X_genre_raw = np.array([
        [1, 1, 0],  # seed: rock + dream pop
        [1, 0, 1],  # candidate: rock + hip hop; only broad raw overlap
    ], dtype=float)
    X_genre_smoothed = np.array([
        [1.0, 1.0, 0.8],
        [1.0, 0.8, 1.0],
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=("rock",),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        min_genre_similarity=0.3,
        genre_method="ensemble",
        genre_vocab=genre_vocab,
        broad_filters=cfg.broad_filters,
        mode="dynamic",
    )

    assert result.pool_indices.tolist() == []
    assert result.stats["below_genre_similarity"] == 1
    assert result.stats["genre_overlap_guard_rejected"] == 1


def test_genre_compatibility_penalty_demotes_high_conflict_candidate():
    # Candidate has 1 compatible tag and 4 conflict tags — conflict ratio ≈ 0.8.
    # penalty = strength * conflict_ratio ≈ 1.0 * 0.8 = 0.8
    # adjusted sim = 1.0 - 0.8 = 0.2, which falls below similarity_floor=0.3.
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    artist_keys = np.array(["seed", "conflict"])
    genre_vocab = ["indie pop", "punk", "rnb", "house", "soul", "funk"]
    X_genre_raw = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1],
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=0.3,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        genre_compatibility_enabled=True,
        genre_compatibility_penalty_strength=1.0,  # full penalty strength
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        min_genre_similarity=None,
        genre_method="weighted_jaccard",
        genre_vocab=genre_vocab,
        broad_filters=(),
        mode="dynamic",
    )

    assert result.pool_indices.tolist() == []
    assert result.stats["genre_compatibility_penalty_applied"] == 1
    assert result.params_effective["genre_compatibility_enabled"] is True


def test_genre_compatibility_penalty_can_demote_without_rejecting():
    embedding = np.array([
        [1.0, 0.0],
        [0.95, np.sqrt(1 - 0.95**2)],
        [0.80, np.sqrt(1 - 0.80**2)],
    ])
    artist_keys = np.array(["seed", "conflict", "clean"])
    genre_vocab = ["indie pop", "punk", "rnb", "house", "soul", "funk"]
    X_genre_raw = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=0.75,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
        genre_compatibility_enabled=True,
        genre_compatibility_penalty_strength=0.30,
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_raw,
        min_genre_similarity=None,
        genre_method="weighted_jaccard",
        genre_vocab=genre_vocab,
        broad_filters=(),
        mode="dynamic",
    )

    artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert artists == ["clean"]
    assert result.stats["genre_compatibility_penalty_applied"] == 1


class _CapturedBundle(Exception):
    """Short-circuit sentinel — carries the restricted bundle that arrived at
    pier-bridge time, so we can assert on the post-restriction state without
    requiring pier-bridge feasibility on a synthetic fixture."""

    def __init__(self, bundle):
        super().__init__("captured")
        self.bundle = bundle


def _capture_pier_bridge_bundle(monkeypatch):
    """Replace ``build_pier_bridge_playlist`` so it raises with the captured bundle.

    Returns nothing; the test should catch ``_CapturedBundle`` from the
    ``generate_playlist_ds`` call and inspect ``exc.bundle``.
    """
    import src.playlist.pipeline.core as pipeline_core

    def _short_circuit(*args, **kwargs):
        raise _CapturedBundle(kwargs["bundle"])

    monkeypatch.setattr(pipeline_core, "build_pier_bridge_playlist", _short_circuit)


def _write_tiny_artifact(tmp_path, name="tiny_artifact.npz"):
    """Minimal 3-track artifact (t0/t1/t2) for bundle-restriction tests."""
    track_ids = np.array(["t0", "t1", "t2"])
    artist_keys = np.array(["a0", "a1", "a2"])
    track_artists = np.array(["Artist 0", "Artist 1", "Artist 2"])
    track_titles = np.array(["Song 0", "Song 1", "Song 2"])
    X_sonic = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    X_genre_raw = np.array([[0.0], [1.0], [0.5]], dtype=float)
    X_genre_smoothed = X_genre_raw.copy()
    genre_vocab = np.array(["g0"])

    artifact_path = tmp_path / name
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )
    return artifact_path


def test_one_each_retries_with_relaxed_candidate_gate_when_pier_bridge_infeasible(tmp_path, monkeypatch):
    from src.playlist.candidate_pool import CandidatePoolResult
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeResult

    import src.playlist.pipeline.core as pipeline_core

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_one_each_retry.npz")
    pool_calls = []

    def _fake_build_candidate_pool(**kwargs):
        pool_calls.append((kwargs["cfg"], kwargs.get("min_genre_similarity")))
        indices = np.array([2], dtype=int)
        return CandidatePoolResult(
            pool_indices=indices,
            eligible_indices=indices,
            seed_sim=np.ones(len(indices), dtype=float),
            sonic_sim=np.ones(len(indices), dtype=float),
            stats={
                "pool_size": len(indices),
                "eligible_count": len(indices),
                "total_candidates_considered": 1,
                "below_similarity_floor": 0,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 0,
            },
            params_effective={
                "similarity_floor": kwargs["cfg"].similarity_floor,
                "min_sonic_similarity": kwargs["cfg"].min_sonic_similarity,
                "min_genre_similarity": kwargs.get("min_genre_similarity"),
            },
        )

    bridge_calls = []

    def _fake_build_pier_bridge_playlist(**kwargs):
        bridge_calls.append(kwargs)
        if len(bridge_calls) == 1:
            return PierBridgeResult(
                track_ids=[],
                track_indices=[],
                seed_positions=[],
                segment_diagnostics=[],
                stats={},
                success=False,
                failure_reason="Segment 0 infeasible under bridge_floor backoff",
            )
        return PierBridgeResult(
            track_ids=["t0", "t2", "t1"],
            track_indices=[0, 2, 1],
            seed_positions=[0, 2],
            segment_diagnostics=[],
            stats={"edge_scores": []},
            success=True,
        )

    monkeypatch.setattr(pipeline_core, "build_candidate_pool", _fake_build_candidate_pool)
    monkeypatch.setattr(pipeline_core, "build_pier_bridge_playlist", _fake_build_pier_bridge_playlist)

    result = generate_playlist_ds(
        artifact_path=str(artifact_path),
        seed_track_id="t0",
        anchor_seed_ids=["t1"],
        num_tracks=3,
        mode="narrow",
        random_seed=0,
        allowed_track_ids=["t0", "t1", "t2"],
        overrides={
            "candidate_pool": {
                "genre_compatibility_enabled": True,
                "genre_compatibility_penalty_strength": 0.3,
                # After the Task 2 dedup refactor, get_min_sonic_similarity() returns None
                # when no override is set (apply_mode_presets() is now the sole writer in
                # the normal path). Provide an explicit value so the retry-relaxation
                # comparison is between real floats (needs to be > 0.08 relaxation target).
                "min_sonic_similarity": 0.20,
            },
            "pier_bridge": {"max_non_seed_tracks_per_artist": 1},
        },
        sonic_weight=0.5,
        genre_weight=0.5,
        min_genre_similarity=0.4,
        genre_method="ensemble",
    )

    assert result.track_ids == ["t0", "t2", "t1"]
    assert len(pool_calls) == 2
    assert len(bridge_calls) == 2
    assert pool_calls[1][0].similarity_floor < pool_calls[0][0].similarity_floor
    assert pool_calls[1][0].min_sonic_similarity < pool_calls[0][0].min_sonic_similarity
    assert pool_calls[1][0].genre_compatibility_enabled is True
    assert pool_calls[1][0].genre_compatibility_penalty_strength == 0.3
    assert pool_calls[1][1] < pool_calls[0][1]
    assert result.stats["playlist"]["one_each_candidate_relaxation"]["attempt"] == 1


def test_one_each_retry_recomputes_album_map_for_new_pool_indices(tmp_path, monkeypatch):
    """FIX 1 regression (final review, 2026-07-27, version-preference-bootleg-
    detection): the One-Each retry rebuilds the candidate pool with relaxed
    floors, which admits indices that were NOT in the original pool. The retry
    must recompute the album map for its OWN pool -- not reuse the first
    attempt's map -- or newly admitted indices score title-only inside the same
    dedupe pass as album-aware ones.
    """
    from src.playlist.candidate_pool import CandidatePoolResult
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeResult
    import src.playlist.artist_style as artist_style_module
    import src.playlist.pipeline.core as pipeline_core

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_one_each_retry_albums.npz")

    pool_calls = []

    def _fake_build_candidate_pool(**kwargs):
        pool_calls.append((kwargs["cfg"], kwargs.get("min_genre_similarity")))
        # First attempt: only idx 2. Retry (relaxed floors): admits idx 1 too --
        # NOT present in the first attempt's pool.
        indices = np.array([2] if len(pool_calls) == 1 else [1, 2], dtype=int)
        return CandidatePoolResult(
            pool_indices=indices,
            eligible_indices=indices,
            seed_sim=np.ones(len(indices), dtype=float),
            sonic_sim=np.ones(len(indices), dtype=float),
            stats={
                "pool_size": len(indices),
                "eligible_count": len(indices),
                "total_candidates_considered": 1,
                "below_similarity_floor": 0,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 0,
            },
            params_effective={
                "similarity_floor": kwargs["cfg"].similarity_floor,
                "min_sonic_similarity": kwargs["cfg"].min_sonic_similarity,
                "min_genre_similarity": kwargs.get("min_genre_similarity"),
            },
        )

    bridge_calls = []

    def _fake_build_pier_bridge_playlist(**kwargs):
        bridge_calls.append(kwargs)
        if len(bridge_calls) == 1:
            return PierBridgeResult(
                track_ids=[],
                track_indices=[],
                seed_positions=[],
                segment_diagnostics=[],
                stats={},
                success=False,
                failure_reason="Segment 0 infeasible under bridge_floor backoff",
            )
        return PierBridgeResult(
            track_ids=["t0", "t2", "t1"],
            track_indices=[0, 2, 1],
            seed_positions=[0, 2],
            segment_diagnostics=[],
            stats={"edge_scores": []},
            success=True,
        )

    def _fake_load_albums_for_indices(bundle, indices, db_path):
        return {int(i): f"Album{int(i)}" for i in indices}

    monkeypatch.setattr(pipeline_core, "build_candidate_pool", _fake_build_candidate_pool)
    monkeypatch.setattr(pipeline_core, "build_pier_bridge_playlist", _fake_build_pier_bridge_playlist)
    monkeypatch.setattr(artist_style_module, "_load_albums_for_indices", _fake_load_albums_for_indices)

    generate_playlist_ds(
        artifact_path=str(artifact_path),
        seed_track_id="t0",
        anchor_seed_ids=["t1"],
        num_tracks=3,
        mode="narrow",
        random_seed=0,
        allowed_track_ids=["t0", "t1", "t2"],
        overrides={
            "candidate_pool": {
                "genre_compatibility_enabled": True,
                "genre_compatibility_penalty_strength": 0.3,
                "min_sonic_similarity": 0.20,
            },
            "pier_bridge": {"max_non_seed_tracks_per_artist": 1},
        },
        sonic_weight=0.5,
        genre_weight=0.5,
        min_genre_similarity=0.4,
        genre_method="ensemble",
    )

    assert len(bridge_calls) == 2
    # First attempt's album map is scoped to its own (smaller) pool.
    assert bridge_calls[0]["albums_by_index"] == {2: "Album2"}
    # The retry admitted idx 1, which was absent from the first attempt's pool.
    # Its album map must be recomputed for the retry's own pool -- the stale
    # first-attempt map would be missing idx 1 entirely.
    assert bridge_calls[1]["albums_by_index"] == {1: "Album1", 2: "Album2"}


def test_pool_dedupe_warns_with_db_path_when_album_lookup_empty(tmp_path, monkeypatch, caplog):
    """FIX 3 regression: the album-lookup warning must name the database it
    queried (the purpose-built detector for "the worker resolved a different
    database_path"), and must fire when the pool is non-empty but the lookup
    still came back empty."""
    import logging as _logging
    from src.playlist.candidate_pool import CandidatePoolResult
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeResult
    import src.playlist.artist_style as artist_style_module
    import src.playlist.pipeline.core as pipeline_core

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_pool_warn_nonempty.npz")

    def _fake_build_candidate_pool(**kwargs):
        indices = np.array([2], dtype=int)
        return CandidatePoolResult(
            pool_indices=indices,
            eligible_indices=indices,
            seed_sim=np.ones(len(indices), dtype=float),
            sonic_sim=np.ones(len(indices), dtype=float),
            stats={
                "pool_size": len(indices),
                "eligible_count": len(indices),
                "total_candidates_considered": 1,
                "below_similarity_floor": 0,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 0,
            },
            params_effective={
                "similarity_floor": kwargs["cfg"].similarity_floor,
                "min_sonic_similarity": kwargs["cfg"].min_sonic_similarity,
                "min_genre_similarity": kwargs.get("min_genre_similarity"),
            },
        )

    def _fake_build_pier_bridge_playlist(**kwargs):
        return PierBridgeResult(
            track_ids=["t0", "t2", "t1"],
            track_indices=[0, 2, 1],
            seed_positions=[0, 2],
            segment_diagnostics=[],
            stats={"edge_scores": []},
            success=True,
        )

    monkeypatch.setattr(pipeline_core, "build_candidate_pool", _fake_build_candidate_pool)
    monkeypatch.setattr(pipeline_core, "build_pier_bridge_playlist", _fake_build_pier_bridge_playlist)
    monkeypatch.setattr(artist_style_module, "_load_albums_for_indices", lambda *a, **k: {})

    # Scoped to tmp_path -- a relative path here would land (and leak a stray
    # .db file) in the repo root, since some downstream readers (e.g. the BPM
    # loader) connect to it eagerly.
    _fake_db_path = str(tmp_path / "custom_meta_test.db")

    with caplog.at_level(_logging.WARNING, logger="src.playlist.pipeline.core"):
        generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            anchor_seed_ids=["t1"],
            num_tracks=3,
            mode="narrow",
            random_seed=0,
            allowed_track_ids=["t0", "t1", "t2"],
            overrides={"library": {"database_path": _fake_db_path}},
            sonic_weight=0.5,
            genre_weight=0.5,
        )

    warnings = [r.message for r in caplog.records if r.levelno == _logging.WARNING]
    assert any("Pool dedupe" in m and _fake_db_path in m for m in warnings)


def test_pool_dedupe_does_not_warn_when_pool_is_starved(tmp_path, monkeypatch, caplog):
    """FIX 3 regression: an empty (starved) pool must NOT fire the album-lookup
    warning -- "album lookup returned nothing" is a false alarm there, since
    there was nothing to look up in the first place."""
    import logging as _logging
    from src.playlist.candidate_pool import CandidatePoolResult
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeResult
    import src.playlist.artist_style as artist_style_module
    import src.playlist.pipeline.core as pipeline_core

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_pool_warn_empty.npz")

    def _fake_build_candidate_pool(**kwargs):
        indices = np.array([], dtype=int)
        return CandidatePoolResult(
            pool_indices=indices,
            eligible_indices=indices,
            seed_sim=np.ones(0, dtype=float),
            sonic_sim=np.ones(0, dtype=float),
            stats={
                "pool_size": 0,
                "eligible_count": 0,
                "total_candidates_considered": 0,
                "below_similarity_floor": 0,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 0,
            },
            params_effective={
                "similarity_floor": kwargs["cfg"].similarity_floor,
                "min_sonic_similarity": kwargs["cfg"].min_sonic_similarity,
                "min_genre_similarity": kwargs.get("min_genre_similarity"),
            },
        )

    def _fake_build_pier_bridge_playlist(**kwargs):
        return PierBridgeResult(
            track_ids=["t0", "t1"],
            track_indices=[0, 1],
            seed_positions=[0, 1],
            segment_diagnostics=[],
            stats={"edge_scores": []},
            success=True,
        )

    monkeypatch.setattr(pipeline_core, "build_candidate_pool", _fake_build_candidate_pool)
    monkeypatch.setattr(pipeline_core, "build_pier_bridge_playlist", _fake_build_pier_bridge_playlist)
    monkeypatch.setattr(artist_style_module, "_load_albums_for_indices", lambda *a, **k: {})

    with caplog.at_level(_logging.WARNING, logger="src.playlist.pipeline.core"):
        generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            anchor_seed_ids=["t1"],
            num_tracks=2,
            mode="narrow",
            random_seed=0,
            allowed_track_ids=["t0", "t1", "t2"],
            sonic_weight=0.5,
            genre_weight=0.5,
        )

    warnings = [r.message for r in caplog.records if r.levelno == _logging.WARNING]
    assert not any("Pool dedupe" in m for m in warnings)


def test_allowed_set_applies_excluded_track_ids_in_pipeline(tmp_path, monkeypatch):
    """
    Regression test: style-aware runs pass both ``allowed_track_ids`` and
    ``excluded_track_ids``. The bundle that reaches pier-bridge must have
    excluded ids stripped — even when also clamped to the allowed set.

    Strategy: short-circuit ``build_pier_bridge_playlist`` to capture the
    restricted bundle and assert on it. Decouples the test from
    pier-bridge feasibility, which is not the contract under test.
    """
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeConfig

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_artifact.npz")
    _capture_pier_bridge_bundle(monkeypatch)

    with pytest.raises(_CapturedBundle) as exc_info:
        generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            anchor_seed_ids=["t1"],
            num_tracks=3,
            mode="narrow",
            random_seed=0,
            allowed_track_ids=["t0", "t1", "t2"],
            excluded_track_ids={"t2"},
            pier_bridge_config=PierBridgeConfig(
                transition_floor=0.0,
                bridge_floor=0.0,
                progress_enabled=False,
            ),
            sonic_weight=1.0,
            genre_weight=0.0,
            min_genre_similarity=None,
            genre_method=None,
        )

    captured_ids = {str(t) for t in exc_info.value.bundle.track_ids}
    assert "t2" not in captured_ids
    assert {"t0", "t1"}.issubset(captured_ids)


def test_excluded_set_does_not_remove_piers_in_pipeline(tmp_path, monkeypatch):
    """
    Piers (seed + anchor seeds) must be exempt from ``excluded_track_ids``
    so DS can still place them as fixed endpoints — even if the caller
    asks for all three tracks to be excluded.
    """
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeConfig

    artifact_path = _write_tiny_artifact(tmp_path, "tiny_artifact_pier_exempt.npz")
    _capture_pier_bridge_bundle(monkeypatch)

    with pytest.raises(_CapturedBundle) as exc_info:
        generate_playlist_ds(
            artifact_path=str(artifact_path),
            seed_track_id="t0",
            anchor_seed_ids=["t1"],
            num_tracks=3,
            mode="narrow",
            random_seed=0,
            allowed_track_ids=["t0", "t1", "t2"],
            excluded_track_ids={"t0", "t1", "t2"},
            pier_bridge_config=PierBridgeConfig(
                transition_floor=0.0,
                bridge_floor=0.0,
                progress_enabled=False,
            ),
            sonic_weight=1.0,
            genre_weight=0.0,
            min_genre_similarity=None,
            genre_method=None,
        )

    captured_ids = {str(t) for t in exc_info.value.bundle.track_ids}
    assert "t0" in captured_ids
    assert "t1" in captured_ids
    assert "t2" not in captured_ids


def test_recency_filters_require_candidate_pool_stage():
    from src.playlist.filtering import filter_by_recently_played, filter_by_scrobbles

    with pytest.raises(ValueError, match="Recency filter must not run after ordering"):
        filter_by_scrobbles(
            tracks=[],
            scrobbles=[],
            lookback_days=14,
            stage="post_order_validation",
        )

    with pytest.raises(ValueError, match="Recency filter must not run after ordering"):
        filter_by_recently_played(
            tracks=[],
            play_history=[],
            lookback_days=14,
            stage="post_order_validation",
        )


def test_post_order_validation_fails_loudly_on_recency_overlap():
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def get_all_tracks(self, library_id=None):
            return []

    class DummyConfig:
        recently_played_filter_enabled = True
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0

        def get(self, section, key=None, default=None):
            return default

    gen = PlaylistGenerator(DummyLibrary(), DummyConfig())
    with pytest.raises(ValueError, match="post_order_validation_failed"):
        gen._post_order_validate_ds_output(
            ordered_tracks=[{"rating_key": "bad_id", "artist": "A", "title": "T"}],
            expected_length=1,
            excluded_track_ids={"bad_id"},
            exempt_pier_track_ids=set(),
            audit_path="docs/run_audits/test.md",
        )


def test_playlist_generator_does_not_post_filter_ds_results(monkeypatch, caplog):
    import logging
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def __init__(self, tracks):
            self._tracks = list(tracks)

        def get_all_tracks(self, library_id=None):
            return list(self._tracks)

    class DummyConfig:
        recently_played_filter_enabled = True
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0
        # Sane defaults so duration filter (47-720s) accepts our fixtures.
        min_track_duration_seconds = 47
        max_track_duration_seconds = 720

        config = {"playlists": {}}

        def get(self, section, key=None, default=None):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)

    artist = "Test Artist"
    artist_key = "test artist"
    # 180000ms = 3min: well within the 47-720s duration filter.
    library_tracks = [
        {
            "rating_key": f"s{i}",
            "artist": artist,
            "artist_key": artist_key,
            "title": f"Seed {i}",
            "duration": 180000,
        }
        for i in range(4)
    ]

    gen = PlaylistGenerator(DummyLibrary(library_tracks), DummyConfig(), lastfm_client=None)

    # If any post-order recency filtering is attempted, fail the test.
    monkeypatch.setattr(
        gen,
        "filter_tracks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("post-order filter_tracks called")),
    )
    monkeypatch.setattr(
        gen,
        "_filter_by_scrobbles",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("post-order _filter_by_scrobbles called")),
    )
    monkeypatch.setattr(gen, "_print_playlist_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(gen, "_compute_edge_scores_from_artifact", lambda *args, **kwargs: [])

    def _stub_ds(*args, **kwargs):
        gen._last_ds_report = {
            "metrics": {"strategy": "pier_bridge"},
            "playlist_stats": {"playlist": {}},
        }
        return [
            {"rating_key": "p0", "artist": artist, "title": "Pier 0"},
            {"rating_key": "x1", "artist": "Other", "title": "Bridge 1"},
            {"rating_key": "p2", "artist": artist, "title": "Pier 2"},
        ]

    monkeypatch.setattr(gen, "_maybe_generate_ds_playlist", _stub_ds)

    caplog.set_level(logging.INFO)
    result = gen.create_playlist_for_artist(artist, track_count=3)
    assert len(result["tracks"]) == 3
    assert any("stage=post_order_validation" in rec.getMessage() for rec in caplog.records)


def test_playlist_generator_excludes_title_words_from_automatic_artist_seeds(monkeypatch):
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def __init__(self, tracks):
            self._tracks = list(tracks)

        def get_all_tracks(self, library_id=None):
            return list(self._tracks)

    class DummyConfig:
        recently_played_filter_enabled = False
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0
        min_track_duration_seconds = 47
        max_track_duration_seconds = 720
        config = {
            "playlists": {
                "ds_pipeline": {
                    "candidate_pool": {
                        "title_exclusion_enabled": True,
                        "title_exclusion_words": ["skit", "interlude", "acapella"],
                    }
                }
            }
        }

        def get(self, section, key=None, default=None):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)

    artist = "Test Artist"
    library_tracks = [
        {
            "rating_key": f"s{i}",
            "artist": artist,
            "artist_key": "test artist",
            "title": title,
            "duration": 180000,
        }
        for i, title in enumerate([
            "Good Seed 0",
            "Good Seed 1",
            "Good Seed 2",
            "Acapella Version",
            "Good Seed 3",
        ])
    ]
    gen = PlaylistGenerator(DummyLibrary(library_tracks), DummyConfig(), lastfm_client=None)
    monkeypatch.setattr(gen, "_print_playlist_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(gen, "_compute_edge_scores_from_artifact", lambda *args, **kwargs: [])

    captured_seed_titles = []

    def _stub_ds(*args, **kwargs):
        captured_seed_titles.extend(track.get("title") for track in kwargs["anchor_seed_tracks"])
        gen._last_ds_report = {
            "metrics": {"strategy": "pier_bridge"},
            "playlist_stats": {"playlist": {}},
        }
        return [
            {"rating_key": "p0", "artist": artist, "title": "Pier 0", "duration": 180000},
            {"rating_key": "x1", "artist": "Other", "title": "Bridge 1", "duration": 180000},
            {"rating_key": "p2", "artist": artist, "title": "Pier 2", "duration": 180000},
        ]

    monkeypatch.setattr(gen, "_maybe_generate_ds_playlist", _stub_ds)

    result = gen.create_playlist_for_artist(artist, track_count=3)

    assert len(result["tracks"]) == 3
    assert "Acapella Version" not in captured_seed_titles


def test_playlist_generator_can_exclude_recent_automatic_artist_seeds(monkeypatch):
    from datetime import datetime

    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def __init__(self, tracks):
            self._tracks = list(tracks)

        def get_all_tracks(self, library_id=None):
            return list(self._tracks)

    class DummyConfig:
        recently_played_filter_enabled = True
        recently_played_lookback_days = 14
        recently_played_min_playcount = 1
        lastfm_history_days = 90
        min_track_duration_seconds = 47
        max_track_duration_seconds = 720
        config = {"playlists": {"ds_pipeline": {"random_seed": 0}}}

        def get(self, section, key=None, default=None):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)

    class DummyLastFM:
        def get_recent_tracks(self, days=90, use_cache=True):
            return [
                {
                    "artist": "The Strokes",
                    "title": "Hard to Explain",
                    "timestamp": int(datetime.now().timestamp()),
                }
            ]

    artist = "The Strokes"
    library_tracks = [
        {
            "rating_key": f"s{i}",
            "artist": artist,
            "artist_key": "the strokes",
            "title": title,
            "duration": 180000,
        }
        for i, title in enumerate([
            "Hard to Explain",
            "The Modern Age",
            "Barely Legal",
            "Someday",
            "Alone, Together",
        ])
    ]
    gen = PlaylistGenerator(DummyLibrary(library_tracks), DummyConfig(), lastfm_client=DummyLastFM())
    monkeypatch.setattr(gen, "_print_playlist_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(gen, "_compute_edge_scores_from_artifact", lambda *args, **kwargs: [])

    captured_seed_titles = []

    def _stub_ds(*args, **kwargs):
        captured_seed_titles.extend(track.get("title") for track in kwargs["anchor_seed_tracks"])
        gen._last_ds_report = {
            "metrics": {"strategy": "pier_bridge"},
            "playlist_stats": {"playlist": {}},
        }
        return [
            {"rating_key": "p0", "artist": artist, "title": "Pier 0", "duration": 180000},
            {"rating_key": "x1", "artist": "Other", "title": "Bridge 1", "duration": 180000},
            {"rating_key": "p2", "artist": artist, "title": "Pier 2", "duration": 180000},
        ]

    monkeypatch.setattr(gen, "_maybe_generate_ds_playlist", _stub_ds)

    result = gen.create_playlist_for_artist(
        artist,
        track_count=3,
        exclude_seed_tracks_from_recency=True,
    )

    assert len(result["tracks"]) == 3
    assert "Hard to Explain" not in captured_seed_titles


def test_playlist_generator_refreshes_ds_metrics_after_final_edge_recompute(monkeypatch):
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def __init__(self, tracks):
            self._tracks = list(tracks)

        def get_all_tracks(self, library_id=None):
            return list(self._tracks)

    class DummyConfig:
        recently_played_filter_enabled = False
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0
        min_track_duration_seconds = 47
        max_track_duration_seconds = 720
        config = {"playlists": {}}

        def get(self, section, key=None, default=None):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)

    artist = "Test Artist"
    library_tracks = [
        {
            "rating_key": f"s{i}",
            "artist": artist,
            "artist_key": "test artist",
            "title": f"Seed {i}",
            "duration": 180000,
        }
        for i in range(4)
    ]
    gen = PlaylistGenerator(DummyLibrary(library_tracks), DummyConfig(), lastfm_client=None)
    monkeypatch.setattr(gen, "_print_playlist_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gen,
        "_compute_edge_scores_from_artifact",
        lambda *args, **kwargs: [
            {"prev_id": "p0", "cur_id": "x1", "T": 0.1},
            {"prev_id": "x1", "cur_id": "p2", "T": 0.3},
        ],
    )

    def _stub_ds(*args, **kwargs):
        gen._last_ds_report = {
            "metrics": {
                "strategy": "pier_bridge",
                "below_floor": 0,
                "min_transition": 0.9,
                "mean_transition": 0.95,
            },
            "playlist_stats": {
                "playlist": {
                    "transition_floor": 0.2,
                    "below_floor_count": 0,
                    "min_transition": 0.9,
                    "mean_transition": 0.95,
                }
            },
            "transition_floor": 0.2,
        }
        return [
            {"rating_key": "p0", "artist": artist, "title": "Pier 0", "duration": 180000},
            {"rating_key": "x1", "artist": "Other", "title": "Bridge 1", "duration": 180000},
            {"rating_key": "p2", "artist": artist, "title": "Pier 2", "duration": 180000},
        ]

    monkeypatch.setattr(gen, "_maybe_generate_ds_playlist", _stub_ds)

    result = gen.create_playlist_for_artist(artist, track_count=3)

    assert result["ds_report"]["metrics"]["below_floor"] == 1
    assert result["ds_report"]["metrics"]["min_transition"] == 0.1
    assert result["ds_report"]["metrics"]["mean_transition"] == 0.2
    assert result["ds_report"]["metrics"]["distinct_artists"] == 2
