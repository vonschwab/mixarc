"""Behavioral: tag-weighted pier allocation raises the anchors' tag affinity.

Clusters a real artist via the production cluster_artist_tracks, then compares
the allocator at skew=0.6 vs skew=0.0 on the real medoids/affinities.
"""
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, "tests")
from support.gui_fidelity import resolved_artifact_path  # noqa: E402

from src.config_loader import DEFAULT_MIN_TRACK_DURATION_SECONDS  # noqa: E402
from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.artist_style import (  # noqa: E402
    ArtistStyleConfig,
    allocate_piers_by_tag_affinity,
    cluster_artist_tracks,
)
from src.playlist.tag_steering import resolve_tag_steering_target  # noqa: E402

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
DB = "data/metadata.db"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    resolved_artifact_path()  # publishes sonic_variant_override (muq) as a side effect
    return load_artifact_bundle(str(ARTIFACT))


def test_skew_raises_pier_affinity(bundle):
    if getattr(bundle, "genre_emb", None) is None:
        pytest.skip("dense genre sidecar absent")
    target, mapped, _ = resolve_tag_steering_target(
        ["jangle pop", "indie rock"],
        genre_vocab=[str(v) for v in bundle.genre_vocab],
        genre_emb=bundle.genre_emb,
    )
    if target is None or len(mapped) == 0:
        pytest.skip("steering tags did not map in this artifact vocab")

    # Minimal-but-real ArtistStyleConfig: enable clustering + the tag lever. Read the
    # dataclass definition in artist_style.py and set only the fields without defaults
    # plus medoid_tag_weight=0.3; leave the rest at their dataclass defaults.
    style_cfg = ArtistStyleConfig(enabled=True, medoid_tag_weight=0.3)

    clusters, medoids, medoids_by_cluster, X_norm, _support = cluster_artist_tracks(
        bundle=bundle,
        artist_name="Real Estate",
        cfg=style_cfg,
        random_seed=0,
        medoid_top_k=10,                 # over-produce, mirrors the steering path
        steering_target=target,
        metadata_db_path=DB,
        # Explicit, not relied-on-default (coordinator review 2026-07-30):
        # the real config value, same one production callers pass.
        min_pier_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS,
    )
    xgd = np.asarray(bundle.X_genre_dense, dtype=float)
    tgt = np.asarray(target, dtype=float)
    affs = [float(np.mean(xgd[m] @ tgt)) if len(m) else 0.0 for m in clusters]

    sel_skew = allocate_piers_by_tag_affinity(medoids_by_cluster, affs, 10, 0.6)
    sel_flat = allocate_piers_by_tag_affinity(medoids_by_cluster, affs, 10, 0.0)

    def mean_aff(ids):
        return float(np.mean([xgd[i] @ tgt for i in ids])) if ids else 0.0

    assert mean_aff(sel_skew) >= mean_aff(sel_flat), (
        f"skew mean affinity {mean_aff(sel_skew):.3f} < flat {mean_aff(sel_flat):.3f} — "
        "read the per-cluster affinities before concluding the skew is inert"
    )
    # The lowest-affinity cluster should give up slots under skew vs flat.
    def counts(sel):
        sets = [set(m) for m in medoids_by_cluster]
        return [sum(1 for i in sel if i in s) for s in sets]
    low = int(np.argmin(affs))
    assert counts(sel_skew)[low] <= counts(sel_flat)[low]
