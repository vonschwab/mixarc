"""Live: the sonic prototype lifts Eno's ambient-pier affinity above genre-dense alone."""
from pathlib import Path
import inspect
import sys
import numpy as np
import pytest

sys.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path  # noqa: E402

from src.config_loader import DEFAULT_MIN_TRACK_DURATION_SECONDS  # noqa: E402
from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.artist_style import ArtistStyleConfig, cluster_artist_tracks  # noqa: E402
from src.playlist.candidate_pool import build_candidate_pool  # noqa: E402
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist  # noqa: E402
from src.playlist.pier_bridge.beam import _beam_search_segment  # noqa: E402
from src.playlist.tag_steering import (  # noqa: E402
    resolve_tag_steering_target,
    resolve_tag_sonic_prototype_rows,
    sonic_prototype_from_rows,
    sonic_global_mean,
)

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB = "data/metadata.db"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    resolved_artifact_path()
    return load_artifact_bundle(str(ARTIFACT))


def _sonic_tag_affinity(bundle, tags):
    # NOTE: ArtifactBundle exposes the active sonic space as `X_sonic` (resolved
    # from the artifact's `X_sonic_muq` key via config.yaml's
    # `artifacts.sonic_variant_override: muq` at load time) -- there is no
    # `X_sonic_muq` attribute on the bundle itself. Using `X_sonic` here is what
    # keeps this helper space-consistent with `cluster_artist_tracks`, which
    # clusters on the same `bundle.X_sonic` matrix.
    xmq = np.asarray(bundle.X_sonic, dtype=np.float64)
    t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        tags, metadata_db_path=DB, track_id_to_row=t2r,
        exclude_artist="Brian Eno", min_support=25)
    assert rows is not None, "ambient must have enough library support"
    gm = sonic_global_mean(xmq)
    proto, cohesion, _ = sonic_prototype_from_rows(xmq, rows, global_mean=gm)
    xmn = xmq / (np.linalg.norm(xmq, axis=1, keepdims=True) + 1e-12)
    return (xmn - gm) @ proto


def test_sonic_prototype_raises_ambient_pier_affinity(bundle):
    if getattr(bundle, "genre_emb", None) is None or getattr(bundle, "X_sonic", None) is None:
        pytest.skip("dense genre or MuQ sidecar absent")
    tags = ["ambient", "drone", "dark ambient", "space ambient"]
    target, mapped, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    if target is None:
        pytest.skip("ambient tags did not map")
    aff_vec = _sonic_tag_affinity(bundle, tags)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)

    # min_pier_duration_seconds passed explicitly (not relied-on-default,
    # coordinator review 2026-07-30): the real config value, same one
    # production callers pass, identically on both calls so the comparison
    # below stays apples-to-apples.
    _, med_base, _, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        min_pier_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS)
    _, med_sonic, _, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=aff_vec, sonic_tag_weight=0.5,
        min_pier_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS)

    base_sonic = float(np.mean([aff_vec[m] for m in med_base])) if med_base else 0.0
    with_sonic = float(np.mean([aff_vec[m] for m in med_sonic])) if med_sonic else 0.0
    assert with_sonic > base_sonic, (
        f"sonic-term piers ambient affinity {with_sonic:.3f} !> genre-only {base_sonic:.3f}")


def test_build_candidate_pool_accepts_sonic_pool_affinity():
    sig = inspect.signature(build_candidate_pool)
    assert "sonic_pool_affinity" in sig.parameters
    assert "sonic_blend" in sig.parameters
    # the uncentered prototype path was retired 2026-07-08 in favor of the centered
    # affinity blend; guard that the dead param does not creep back.
    assert "sonic_prototype" not in sig.parameters


def test_sonic_pool_affinity_blend_shifts_admitted_affinity(bundle):
    """Blending the centered tag affinity into the sonic admission similarity raises
    the admitted pool's mean affinity to that (centered, tag-specific) direction."""
    if getattr(bundle, "X_sonic", None) is None:
        pytest.skip("MuQ/X_sonic absent")
    # import path may differ; adjust to candidate_pool.py's own import if needed
    from src.playlist.config import CandidatePoolConfig
    xmq = np.asarray(bundle.X_sonic, dtype=np.float64)
    t2r = {str(t): i for i, t in enumerate(bundle.track_ids)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        ["ambient"], metadata_db_path=DB, track_id_to_row=t2r, min_support=25)
    assert rows is not None
    gm = sonic_global_mean(xmq)
    proto, _, _ = sonic_prototype_from_rows(xmq, rows, global_mean=gm)  # CENTERED, tag-specific
    xmn = xmq / (np.linalg.norm(xmq, axis=1, keepdims=True) + 1e-12)
    aff = (xmn - gm) @ proto

    # CandidatePoolConfig has no field defaults for the pool-sizing knobs, so
    # size them generously (uncapped in practice) to keep the whole eligible
    # set in the pool -- the sonic_admission_percentile floor below is the
    # thing under test, not artist/pool capping.
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=100_000,
        target_artists=100_000,
        candidates_per_artist=100_000,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        duration_penalty_enabled=False,
        title_exclusion_enabled=False,
        sonic_admission_percentile=0.5,
    )
    seed = int(np.argmax(aff))
    common = dict(seed_idx=seed, seed_indices=[seed], embedding=xmq,
                  artist_keys=bundle.artist_keys, track_ids=bundle.track_ids,
                  cfg=cfg, random_seed=0, X_sonic=xmq)
    base = build_candidate_pool(**common)
    steered = build_candidate_pool(**common, sonic_pool_affinity=aff, sonic_blend=0.5)

    def mean_aff(res):
        idx = [int(i) for i in res.pool_indices]
        return float(np.mean([aff[i] for i in idx])) if idx else 0.0
    assert mean_aff(steered) >= mean_aff(base)


def test_no_sonic_affinity_is_byte_identical_piers(bundle):
    tags = ["ambient"]
    target, _, _ = resolve_tag_steering_target(
        tags, genre_vocab=[str(v) for v in bundle.genre_vocab], genre_emb=bundle.genre_emb)
    cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)
    _, med_a, _, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        min_pier_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS)
    _, med_b, _, _, _ = cluster_artist_tracks(
        bundle=bundle, artist_name="Brian Eno", cfg=cfg, random_seed=0,
        medoid_top_k=10, steering_target=target, metadata_db_path=DB,
        sonic_tag_affinity=None, sonic_tag_weight=0.5,
        min_pier_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS)
    assert med_a == med_b


def test_beam_accepts_sonic_tag_params():
    for fn in (build_pier_bridge_playlist, _beam_search_segment):
        p = inspect.signature(fn).parameters
        assert "sonic_tag_affinity" in p, f"{fn.__name__} missing sonic_tag_affinity"
        assert "sonic_tag_beam_weight" in p, f"{fn.__name__} missing sonic_tag_beam_weight"


def _flat_space(n: int) -> np.ndarray:
    """n identical unit vectors so sonic gates pass with floors at -1 (reused
    from tests/unit/test_beam_pace_soft_penalty.py's minimal-fixture pattern)."""
    X = np.ones((n, 3), dtype=float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_sonic_tag_beam_term_shifts_candidate_ranking():
    """Two candidates tied on every other beam score component (identical sonic
    vectors, no genre/waypoint machinery); a strong sonic_tag_affinity favoring
    the second candidate flips the beam's pick from the first (default stable
    tie-break) to the second."""
    from src.playlist.pier_bridge.config import PierBridgeConfig

    X = _flat_space(4)  # 0=pier_a, 1=candidate, 2=pier_b, 3=candidate
    cfg = PierBridgeConfig(bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False)

    path_base, _, _, err_base = _beam_search_segment(
        0, 2, 1, [1, 3],
        X, X, None, None, None, None,
        cfg,
        5,
    )
    assert err_base is None
    assert path_base == [1], "sanity: without the term, ties resolve to the first candidate"

    affinity = np.array([0.0, 0.0, 0.0, 10.0], dtype=float)
    path_steered, _, _, err_steered = _beam_search_segment(
        0, 2, 1, [1, 3],
        X, X, None, None, None, None,
        cfg,
        5,
        sonic_tag_affinity=affinity,
        sonic_tag_beam_weight=1.0,
    )
    assert err_steered is None
    assert path_steered == [3], "sonic_tag_beam_weight did not shift the beam's ranking"
