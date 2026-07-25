"""Focused unit tests for the modules extracted in Tier-1.5 pipeline split.

Each module gets a small test class:
  * TestBundleRestrict — restrict_bundle (4 paths + pier exemption)
  * TestPierResolver   — resolve_pier_seeds + dedupe_pool_by_track_key
  * TestAuditEmitter   — disabled vs enabled, lazy context, flush semantics
  * TestPostValidation — build_failure_diagnostic (3 buckets) +
                          run_post_order_validation (length + recency)
  * TestEmbeddingSetupPreScaled — pre_scaled path is exposed in variant_stats

The goldens in test_pipeline_smoke_golden.py cover end-to-end behavior;
these tests cover unit-level seams now that the modules are independently
importable.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pytest

from src.features.artifacts import ArtifactBundle
from src.playlist.pipeline.audit_emitter import AuditEmitter
from src.playlist.pipeline.bundle_restrict import restrict_bundle
from src.playlist.pipeline.pier_resolver import (
    dedupe_pool_by_track_key,
    resolve_pier_seeds,
)
from src.playlist.pipeline.post_validation import (
    build_failure_diagnostic,
    run_post_order_validation,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _make_bundle(
    track_ids: List[str],
    *,
    artist_keys: Optional[List[str]] = None,
    track_titles: Optional[List[str]] = None,
    sonic_dim: int = 2,
    dense_dim: Optional[int] = None,
    layered_dim: Optional[int] = None,
) -> ArtifactBundle:
    """Build a minimal in-memory ArtifactBundle for unit tests.

    When ``dense_dim`` is given, populate ``X_genre_dense`` with a distinct,
    per-row recoverable value (row i = i) so slicing/alignment can be asserted.
    """
    n = len(track_ids)
    artist_keys = artist_keys or [f"a{i}" for i in range(n)]
    track_titles = track_titles or [f"Song {i}" for i in range(n)]
    X_genre_dense = None
    if dense_dim is not None:
        # Row i is filled with the value i so kept rows are identifiable post-slice.
        X_genre_dense = np.tile(
            np.arange(n, dtype=float).reshape(n, 1), (1, dense_dim)
        )
    X_genre_leaf_idf = None
    X_genre_family = None
    X_genre_bridge = None
    X_facet = None
    if layered_dim is not None:
        X_genre_leaf_idf = np.tile(
            np.arange(n, dtype=float).reshape(n, 1), (1, layered_dim)
        )
        X_genre_family = np.arange(n, dtype=float).reshape(n, 1)
        X_genre_bridge = np.tile(
            np.arange(n, dtype=float).reshape(n, 1), (1, layered_dim)
        )
        X_facet = np.arange(n, dtype=float).reshape(n, 1)
    return ArtifactBundle(
        artifact_path=Path("test://bundle"),
        track_ids=np.array(track_ids),
        artist_keys=np.array(artist_keys),
        track_artists=np.array(artist_keys),  # display = key for tests
        track_titles=np.array(track_titles),
        X_sonic=np.zeros((n, sonic_dim), dtype=float),
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.zeros((n, 1), dtype=float),
        X_genre_smoothed=np.zeros((n, 1), dtype=float),
        X_genre_dense=X_genre_dense,
        X_genre_leaf_idf=X_genre_leaf_idf,
        X_genre_family=X_genre_family,
        X_genre_bridge=X_genre_bridge,
        X_facet=X_facet,
        genre_leaf_vocab=np.array([f"leaf{i}" for i in range(layered_dim)], dtype=object) if layered_dim else None,
        genre_family_vocab=np.array(["family"], dtype=object) if layered_dim else None,
        genre_bridge_vocab=np.array([f"leaf{i}" for i in range(layered_dim)], dtype=object) if layered_dim else None,
        facet_vocab=np.array(["facet"], dtype=object) if layered_dim else None,
        genre_graph_taxonomy_version=np.array("test-taxonomy", dtype=object) if layered_dim else None,
        genre_graph_sidecar_fingerprint=np.array("b" * 64, dtype=object) if layered_dim else None,
        genre_vocab=np.array(["g0"]),
        track_id_to_index={tid: i for i, tid in enumerate(track_ids)},
    )


# --------------------------------------------------------------------------- #
# TestBundleRestrict                                                          #
# --------------------------------------------------------------------------- #

class TestBundleRestrict:
    def test_no_restriction_passthrough(self):
        """Both args None → bundle returned unchanged, seed_idx unchanged."""
        bundle = _make_bundle(["t0", "t1", "t2"])
        out_bundle, out_seed, out_allowed = restrict_bundle(
            bundle, "t1", seed_idx=1,
            anchor_seed_ids=[],
            allowed_track_ids=None,
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )
        assert out_bundle is bundle
        assert out_seed == 1
        assert out_allowed is None

    def test_allowed_only_clamps_and_reindexes_seed(self):
        bundle = _make_bundle(["t0", "t1", "t2", "t3"])
        out_bundle, out_seed, out_allowed = restrict_bundle(
            bundle, "t2", seed_idx=2,
            anchor_seed_ids=[],
            allowed_track_ids=["t0", "t2"],
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )
        assert set(out_bundle.track_ids) == {"t0", "t2"}
        # Seed must remap to its position in the restricted bundle.
        assert out_bundle.track_ids[out_seed] == "t2"
        # allowed_track_ids_set is augmented with the seed.
        assert "t2" in out_allowed

    def test_allowed_plus_excluded_applies_intersection(self):
        bundle = _make_bundle(["t0", "t1", "t2", "t3"])
        out_bundle, _, out_allowed = restrict_bundle(
            bundle, "t0", seed_idx=0,
            anchor_seed_ids=[],
            allowed_track_ids=["t0", "t1", "t2"],
            excluded_track_ids={"t1"},
            allowed_track_ids_set=None,
        )
        ids = set(out_bundle.track_ids)
        assert "t1" not in ids
        assert {"t0", "t2"}.issubset(ids)
        assert "t0" in out_allowed

    def test_allowed_plus_excluded_exempts_piers(self):
        """Seed + anchors survive excluded_track_ids inside the allowed branch."""
        bundle = _make_bundle(["t0", "t1", "t2", "t3"])
        out_bundle, _, _ = restrict_bundle(
            bundle, "t0", seed_idx=0,
            anchor_seed_ids=["t1"],
            allowed_track_ids=["t0", "t1", "t2"],
            excluded_track_ids={"t0", "t1", "t2"},  # exclude everything
            allowed_track_ids_set=None,
        )
        ids = set(out_bundle.track_ids)
        assert {"t0", "t1"}.issubset(ids)  # piers preserved
        assert "t2" not in ids               # non-pier excluded

    def test_excluded_only_masks_full_bundle(self):
        bundle = _make_bundle(["t0", "t1", "t2", "t3"])
        out_bundle, out_seed, out_allowed = restrict_bundle(
            bundle, "t0", seed_idx=0,
            anchor_seed_ids=[],
            allowed_track_ids=None,
            excluded_track_ids={"t2"},
            allowed_track_ids_set=None,
        )
        ids = set(out_bundle.track_ids)
        assert "t2" not in ids
        assert out_bundle.track_ids[out_seed] == "t0"
        assert out_allowed is None  # not constructed in this branch

    def test_excluded_only_exempts_seed(self):
        """Excluding the seed itself: pier exemption keeps it in the bundle."""
        bundle = _make_bundle(["t0", "t1", "t2"])
        out_bundle, out_seed, _ = restrict_bundle(
            bundle, "t1", seed_idx=1,
            anchor_seed_ids=[],
            allowed_track_ids=None,
            excluded_track_ids={"t1"},  # try to exclude the seed
            allowed_track_ids_set=None,
        )
        assert "t1" in out_bundle.track_ids
        assert out_bundle.track_ids[out_seed] == "t1"

    def test_restrict_carries_dense_and_realigns_rows(self):
        """Dense PMI-SVD matrix must survive restriction and stay row-aligned.

        Regression guard: _slice_bundle previously dropped X_genre_dense, which
        silently routed restricted (artist-style) playlists onto the legacy
        ensemble genre path instead of the redesigned dense embedding.
        """
        bundle = _make_bundle(["t0", "t1", "t2", "t3"], dense_dim=4)
        out_bundle, out_seed, _ = restrict_bundle(
            bundle, "t2", seed_idx=2,
            anchor_seed_ids=[],
            allowed_track_ids=["t1", "t3"],  # plus seed t2 via exemption
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )
        assert out_bundle.X_genre_dense is not None
        # Kept ids are {t1, t2, t3}; dense dim preserved.
        assert out_bundle.X_genre_dense.shape == (3, 4)
        # Each surviving row must still equal its ORIGINAL row value (i), proving
        # rows were sliced with the same index order as track_ids (alignment).
        for i, tid in enumerate(out_bundle.track_ids):
            orig_idx = int(tid[1:])  # "t3" -> 3
            assert np.allclose(out_bundle.X_genre_dense[i], float(orig_idx))
        # Seed row must align too.
        seed_orig = int(out_bundle.track_ids[out_seed][1:])
        assert np.allclose(out_bundle.X_genre_dense[out_seed], float(seed_orig))

    def test_restrict_none_dense_passes_through_as_none(self):
        """Artifacts without a dense sidecar restrict cleanly (no crash, stays None)."""
        bundle = _make_bundle(["t0", "t1", "t2"])  # dense_dim omitted -> None
        out_bundle, _, _ = restrict_bundle(
            bundle, "t0", seed_idx=0,
            anchor_seed_ids=[],
            allowed_track_ids=["t0", "t1"],
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )
        assert out_bundle.X_genre_dense is None

    def test_restrict_carries_layered_shadow_matrices_and_realigns_rows(self):
        bundle = _make_bundle(["t0", "t1", "t2", "t3"], layered_dim=2)
        out_bundle, out_seed, _ = restrict_bundle(
            bundle,
            "t2",
            seed_idx=2,
            anchor_seed_ids=[],
            allowed_track_ids=["t1", "t3"],
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )

        assert out_bundle.X_genre_leaf_idf is not None
        assert out_bundle.X_genre_family is not None
        assert out_bundle.X_genre_bridge is not None
        assert out_bundle.X_facet is not None
        assert out_bundle.X_genre_leaf_idf.shape == (3, 2)
        for i, tid in enumerate(out_bundle.track_ids):
            orig_idx = int(tid[1:])
            assert np.allclose(out_bundle.X_genre_leaf_idf[i], float(orig_idx))
            assert np.allclose(out_bundle.X_genre_family[i], float(orig_idx))
            assert np.allclose(out_bundle.X_genre_bridge[i], float(orig_idx))
            assert np.allclose(out_bundle.X_facet[i], float(orig_idx))
        seed_orig = int(out_bundle.track_ids[out_seed][1:])
        assert np.allclose(out_bundle.X_genre_leaf_idf[out_seed], float(seed_orig))
        assert out_bundle.genre_leaf_vocab is not None
        assert out_bundle.genre_leaf_vocab.tolist() == ["leaf0", "leaf1"]
        assert out_bundle.genre_graph_sidecar_fingerprint is not None
        assert out_bundle.genre_graph_sidecar_fingerprint.item() == "b" * 64

    def test_restrict_preserves_sonic_pre_scaled(self):
        """sonic_pre_scaled must survive bundle restriction.

        Regression guard: _slice_bundle previously omitted this scalar/metadata
        field, causing sonic_pre_scaled to revert to False on the restricted
        bundle. Downstream, embedding_setup.py then took the non-pre_scaled path
        and applied a StandardScaler to the already-weighted blend before PCA,
        undermining the baked-in variant weighting.
        """
        n = 4
        bundle = ArtifactBundle(
            artifact_path=Path("test://bundle"),
            track_ids=np.array([f"t{i}" for i in range(n)]),
            artist_keys=np.array([f"a{i}" for i in range(n)]),
            track_artists=np.array([f"Artist {i}" for i in range(n)]),
            track_titles=np.array([f"Track {i}" for i in range(n)]),
            X_sonic=np.zeros((n, 162), dtype=float),
            X_sonic_start=None,
            X_sonic_mid=None,
            X_sonic_end=None,
            X_genre_raw=np.zeros((n, 1), dtype=float),
            X_genre_smoothed=np.zeros((n, 1), dtype=float),
            X_genre_dense=None,
            genre_vocab=np.array(["g0"]),
            track_id_to_index={f"t{i}": i for i in range(n)},
            sonic_variant="muq",
            sonic_pre_scaled=True,
        )
        out_bundle, _, _ = restrict_bundle(
            bundle, "t0", seed_idx=0,
            anchor_seed_ids=[],
            allowed_track_ids=["t0", "t1"],
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )
        assert out_bundle.sonic_pre_scaled is True
        assert out_bundle.sonic_variant == "muq"

    def test_empty_allowed_raises(self):
        bundle = _make_bundle(["t0", "t1"])
        with pytest.raises(ValueError, match="No allowed track_ids"):
            restrict_bundle(
                bundle, "ghost", seed_idx=0,
                anchor_seed_ids=[],
                allowed_track_ids=["does_not_exist"],
                excluded_track_ids=None,
                allowed_track_ids_set=None,
            )

    def test_config_sized_artist_style_allowed_pool_is_accepted(self):
        """Artist-style defaults can legitimately build 13.5k allowed ids."""
        n = 13_500
        track_ids = [f"t{i}" for i in range(n)]
        bundle = _make_bundle(track_ids)
        out_bundle, out_seed, out_allowed = restrict_bundle(
            bundle,
            "t0",
            seed_idx=0,
            anchor_seed_ids=[],
            allowed_track_ids=track_ids,
            excluded_track_ids=None,
            allowed_track_ids_set=None,
        )

        assert out_bundle.track_ids.shape[0] == n
        assert out_bundle.track_ids[out_seed] == "t0"
        assert out_allowed is not None
        assert len(out_allowed) == n


# --------------------------------------------------------------------------- #
# TestPierResolver                                                            #
# --------------------------------------------------------------------------- #

class TestPierResolver:
    def test_resolve_anchors_dedupes_and_includes_primary(self):
        bundle = _make_bundle(
            ["t0", "t1", "t2"],
            artist_keys=["A", "B", "C"],
            track_titles=["Song A", "Song B", "Song C"],
        )
        indices, ids = resolve_pier_seeds(bundle, seed_idx=0, anchor_seed_ids=["t1", "t2"])
        # Primary seed (t0) is prepended; anchors follow.
        assert ids == ["t0", "t1", "t2"]
        assert indices[0] == 0

    def test_resolve_missing_anchors_are_skipped_not_fatal(self, caplog):
        bundle = _make_bundle(["t0", "t1"])
        indices, ids = resolve_pier_seeds(
            bundle, seed_idx=0, anchor_seed_ids=["t1", "ghost"]
        )
        # Missing anchors filtered out; t0 still prepended as primary.
        assert ids == ["t0", "t1"]
        assert any("NOT FOUND" in rec.getMessage() for rec in caplog.records)

    def test_resolve_dedupes_by_track_key_drops_primary_insertion(self):
        """If an anchor has the same (artist, title) as the primary seed, the
        anchor stays (added first) and the primary insertion is suppressed.

        Order: anchors are added first, then dedupe runs, then the primary
        seed is inserted at position 0 *only if* no existing index shares
        its track_key. So with anchors=[t1, t2] and primary=t0 where t0
        shares a track_key with t1, t0 is *not* inserted.
        """
        bundle = _make_bundle(
            ["t0", "t1", "t2"],
            artist_keys=["A", "A", "C"],
            track_titles=["Song A", "Song A", "Song C"],
        )
        _, ids = resolve_pier_seeds(
            bundle, seed_idx=0, anchor_seed_ids=["t1", "t2"]
        )
        assert "t0" not in ids
        assert set(ids) == {"t1", "t2"}

    def test_resolve_dedupes_duplicate_anchors(self):
        """Anchors that share (artist, title) with another *anchor* are dropped."""
        bundle = _make_bundle(
            ["t0", "t1", "t2", "t3"],
            artist_keys=["Z", "A", "A", "C"],
            track_titles=["Seed", "Song A", "Song A", "Song C"],
        )
        _, ids = resolve_pier_seeds(
            bundle, seed_idx=0, anchor_seed_ids=["t1", "t2", "t3"]
        )
        # t2 deduped against t1 (same track_key). Primary t0 prepended (unique).
        assert "t2" not in ids
        assert ids == ["t0", "t1", "t3"]

    def test_resolve_primary_skipped_if_duplicate_already_in_anchors(self):
        """When an anchor has the same track_key as the primary seed, the
        primary is *not* re-inserted (no duplicate insertion)."""
        bundle = _make_bundle(
            ["t0", "t1"],
            artist_keys=["A", "A"],
            track_titles=["Song A", "Song A"],  # same track_key
        )
        _, ids = resolve_pier_seeds(bundle, seed_idx=0, anchor_seed_ids=["t1"])
        # t0 was primary; t1 has same track_key; insertion is suppressed.
        assert ids == ["t1"]

    def test_dedupe_pool_keeps_canonical_version(self):
        """Same (artist, title) → keep the higher version-preference score."""
        bundle = _make_bundle(
            ["t0", "t1", "t2"],
            artist_keys=["A", "A", "B"],
            track_titles=["Song A", "Song A (Live)", "Song B"],
        )
        # t0 (no penalty, score=100) > t1 (Live, score=70). t2 distinct.
        deduped = dedupe_pool_by_track_key(bundle, [0, 1, 2])
        assert 0 in deduped       # canonical kept
        assert 1 not in deduped   # Live version dropped
        assert 2 in deduped       # different track preserved

    def test_dedupe_pool_no_titles_is_passthrough(self):
        bundle = _make_bundle(["t0", "t1"])
        # Construct a bundle with track_titles=None by replacing the field.
        bundle_no_titles = ArtifactBundle(
            **{**bundle.__dict__, "track_titles": None}
        )
        result = dedupe_pool_by_track_key(bundle_no_titles, [0, 1])
        assert result == [0, 1]


# --------------------------------------------------------------------------- #
# TestAuditEmitter                                                            #
# --------------------------------------------------------------------------- #

class TestAuditEmitter:
    def _cfg(self, enabled, tmp_path: Optional[Path] = None):
        return SimpleNamespace(
            enabled=enabled,
            out_dir=str(tmp_path) if tmp_path else "docs/run_audits",
            max_bytes=10_000,
        )

    def test_disabled_emitter_is_all_noop(self, tmp_path):
        em = AuditEmitter(self._cfg(False))
        assert not em.active
        assert em.events is None
        em.append("preflight", {"x": 1})
        em.flush()  # no path/context; must not raise
        assert not em.has_kind("preflight")
        assert not em.can_flush()

    def test_enabled_emitter_appends_and_returns_events_by_reference(self, tmp_path):
        em = AuditEmitter(self._cfg(True, tmp_path))
        assert em.active
        assert em.events == []
        events_ref = em.events
        em.append("preflight", {"x": 1})
        # The list returned earlier must reflect the new append: critical
        # contract because pier_bridge_builder mutates this list directly.
        assert len(events_ref) == 1
        assert events_ref[0].kind == "preflight"
        assert em.has_kind("preflight")
        assert not em.has_kind("final_success")

    def test_ensure_context_is_idempotent(self, tmp_path):
        em = AuditEmitter(self._cfg(True, tmp_path))
        bundle = _make_bundle(["t0"], artist_keys=["A"], track_titles=["Song A"])
        em.ensure_context(
            bundle=bundle, seed_idx=0, seed_track_id="t0",
            mode="dynamic", dry_run=False,
            artifact_path=Path("test://bundle"),
            sonic_variant=None, allowed_ids_count=1,
            pool_source=None, artist_style_enabled=False,
            artist_playlist=False, audit_context_extra=None,
        )
        first_ctx = em.context
        first_path = em.path
        # Calling again must not rebuild.
        em.ensure_context(
            bundle=bundle, seed_idx=0, seed_track_id="t0",
            mode="dynamic", dry_run=False,
            artifact_path=Path("test://bundle"),
            sonic_variant=None, allowed_ids_count=1,
            pool_source=None, artist_style_enabled=False,
            artist_playlist=False, audit_context_extra=None,
        )
        assert em.context is first_ctx
        assert em.path == first_path
        assert em.can_flush()

    def test_flush_writes_markdown(self, tmp_path):
        em = AuditEmitter(self._cfg(True, tmp_path))
        bundle = _make_bundle(["t0"], artist_keys=["A"], track_titles=["Song A"])
        em.ensure_context(
            bundle=bundle, seed_idx=0, seed_track_id="t0",
            mode="dynamic", dry_run=False,
            artifact_path=Path("test://bundle"),
            sonic_variant=None, allowed_ids_count=1,
            pool_source=None, artist_style_enabled=False,
            artist_playlist=False, audit_context_extra=None,
        )
        em.append("preflight", {"x": 1})
        em.flush()
        assert em.path is not None
        assert em.path.exists()
        # The report format is section-structured (not a raw event dump);
        # we only assert the file was written with the run id header.
        content = em.path.read_text(encoding="utf-8")
        assert em.context is not None
        assert em.context.run_id in content

    def test_flush_swallows_write_errors(self, tmp_path, monkeypatch, caplog):
        """flush() must not propagate write errors — its caller may need to
        raise its own ValueError downstream."""
        from src.playlist.pipeline import audit_emitter as mod
        em = AuditEmitter(self._cfg(True, tmp_path))
        bundle = _make_bundle(["t0"])
        em.ensure_context(
            bundle=bundle, seed_idx=0, seed_track_id="t0",
            mode="dynamic", dry_run=False,
            artifact_path=Path("test://bundle"),
            sonic_variant=None, allowed_ids_count=1,
            pool_source=None, artist_style_enabled=False,
            artist_playlist=False, audit_context_extra=None,
        )
        def _boom(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "write_markdown_report", _boom)
        em.flush()  # must not raise
        assert any("Failed to write run audit" in rec.getMessage() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# TestPostValidation                                                          #
# --------------------------------------------------------------------------- #

class TestBuildFailureDiagnostic:
    def test_genre_isolation_bucket(self):
        result = build_failure_diagnostic(
            pool_stats={
                "pool_size": 0,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 5,
                "total_candidates_considered": 5,
            },
            pb_failure_reason="bridge segment infeasible",
            pool_indices_count=0,
            seed_track_ids_for_pier_count=1,
            cfg_mode="narrow",
            cfg_genre_gate_min_similarity=0.3,
        )
        assert "GENRE ISOLATION" in result.diagnostic_msg
        assert result.pool_diagnostics.admitted == 0
        assert result.pool_diagnostics.rejected_genre == 5

    def test_sonic_isolation_bucket(self):
        result = build_failure_diagnostic(
            pool_stats={
                "pool_size": 0,
                "below_sonic_similarity": 8,
                "below_genre_similarity": 0,
                "total_candidates_considered": 8,
            },
            pb_failure_reason="bridge segment infeasible",
            pool_indices_count=0,
            seed_track_ids_for_pier_count=1,
            cfg_mode="strict",
            cfg_genre_gate_min_similarity=0.3,
        )
        assert "SONIC ISOLATION" in result.diagnostic_msg
        assert result.pool_diagnostics.rejected_sonic == 8

    def test_insufficient_pool_bucket(self):
        result = build_failure_diagnostic(
            pool_stats={
                "pool_size": 5,
                "below_sonic_similarity": 2,
                "below_genre_similarity": 1,
                "total_candidates_considered": 8,
            },
            pb_failure_reason="bridge segment infeasible",
            pool_indices_count=5,
            seed_track_ids_for_pier_count=2,
            cfg_mode="narrow",
            cfg_genre_gate_min_similarity=0.3,
        )
        assert "INSUFFICIENT CANDIDATE POOL" in result.diagnostic_msg
        assert result.pool_diagnostics.admitted == 5

    def test_passthrough_when_no_bucket_applies(self):
        """admitted >= 10 → no diagnostic decoration, raw reason returned."""
        result = build_failure_diagnostic(
            pool_stats={
                "pool_size": 50,
                "below_sonic_similarity": 0,
                "below_genre_similarity": 0,
                "total_candidates_considered": 50,
            },
            pb_failure_reason="some other failure",
            pool_indices_count=50,
            seed_track_ids_for_pier_count=1,
            cfg_mode="narrow",
            cfg_genre_gate_min_similarity=0.3,
        )
        assert result.diagnostic_msg == "some other failure"
        assert "ISOLATION" not in result.diagnostic_msg
        assert "INSUFFICIENT" not in result.diagnostic_msg


class TestRunPostOrderValidation:
    def test_happy_path(self):
        bundle = _make_bundle(["t0", "t1", "t2"])
        result = run_post_order_validation(
            bundle=bundle,
            ordered_track_ids=["t0", "t1", "t2"],
            expected_length=3,
            excluded_track_ids=None,
            seed_track_ids_for_pier=["t0"],
        )
        assert result.errors == []
        assert result.recency_overlap_ids == []
        assert result.summary["final_size"] == 3
        assert result.summary["expected_size"] == 3

    def test_length_in_band_is_informational_not_a_warning(self, caplog):
        """A band length is EXPECTED behaviour, so it logs at INFO, not WARNING.

        History: exact-N was retired when variable-bridge mode landed (it flexes
        interior length to lift the worst edge), and the check was demoted from
        error to WARNING. Pass 1 (2026-07-25) demotes it again to INFO, on Dylan's
        explicit ruling that exact playlist length is not a goal at all — track
        count is arbitrary given tracks vary in duration, and the meaningful
        invariant is the pier-to-bridge ratio, checked separately.

        The demotion matters beyond taste: Pass 1 introduces genuine hard-invariant
        breaches (artist gap, per-artist cap) that log at WARNING. Leaving expected
        behaviour at the same level would dilute that signal and train readers to
        ignore it.
        """
        import logging
        bundle = _make_bundle(["t0", "t1", "t2"])
        with caplog.at_level(logging.INFO, logger="src.playlist.pipeline.post_validation"):
            result = run_post_order_validation(
                bundle=bundle,
                ordered_track_ids=["t0", "t1"],
                expected_length=3,
                excluded_track_ids=None,
                seed_track_ids_for_pier=["t0"],
            )
        # Still no error — band lengths must not block generation.
        assert result.errors == []
        # And no warning either: a band length does not degrade the run.
        assert result.warnings == []
        # Visible at INFO so the band remains auditable.
        records = [r for r in caplog.records if "length_in_band" in r.message]
        assert records, "band length should still be logged"
        assert all(r.levelno == logging.INFO for r in records)

    def test_recency_overlap_excludes_piers(self):
        """Pier ids in the excluded set don't count as overlap violations."""
        bundle = _make_bundle(["t0", "t1", "t2"])
        result = run_post_order_validation(
            bundle=bundle,
            ordered_track_ids=["t0", "t1", "t2"],
            expected_length=3,
            excluded_track_ids={"t0", "t2"},  # t0 is a pier
            seed_track_ids_for_pier=["t0"],
        )
        # Only t2 is a real overlap (t0 is exempt as a pier).
        assert result.recency_overlap_ids == ["t2"]
        assert any("recency_overlap=1" in e for e in result.errors)

    def test_recency_offenders_carry_artist_and_title(self):
        bundle = _make_bundle(
            ["t0", "t1"],
            artist_keys=["Artist X", "Artist Y"],
            track_titles=["Title X", "Title Y"],
        )
        result = run_post_order_validation(
            bundle=bundle,
            ordered_track_ids=["t0", "t1"],
            expected_length=2,
            excluded_track_ids={"t1"},
            seed_track_ids_for_pier=["t0"],
        )
        joined = " ".join(result.errors)
        # Offender string includes artist + title for debugging.
        assert "Artist Y" in joined
        assert "Title Y" in joined


# --------------------------------------------------------------------------- #
# TestEmbeddingSetupPreScaled                                                  #
# --------------------------------------------------------------------------- #

class TestEmbeddingSetupPreScaled:
    """The pre_scaled path in setup_embedding must not re-standardize an
    already-weighted sonic space.

    Regression guard: before the fix, sonic_pre_scaled was dropped by
    _slice_bundle, causing the pipeline to apply a StandardScaler to the
    already-weighted blend before PCA, undermining the baked-in variant
    weighting. SP-B removed the tower_dims/tower_pca_dims plumbing this test
    used to also cover (rhythm-axis gating is BPM-based now, not tower-axis
    based) — only the pre_scaled propagation survives here.
    """

    def _make_pre_scaled_bundle(self) -> ArtifactBundle:
        n = 10
        tids = [f"t{i}" for i in range(n)]
        np.random.seed(0)
        return ArtifactBundle(
            artifact_path=Path("test://pre_scaled"),
            track_ids=np.array(tids),
            artist_keys=np.array(tids),
            track_artists=np.array([f"Artist {i}" for i in range(n)]),
            track_titles=np.array([f"Track {i}" for i in range(n)]),
            X_sonic=np.random.randn(n, 162).astype(np.float32),
            X_sonic_start=None,
            X_sonic_mid=None,
            X_sonic_end=None,
            X_genre_raw=np.zeros((n, 4), dtype=np.float32),
            X_genre_smoothed=np.zeros((n, 4), dtype=np.float32),
            X_genre_dense=None,
            genre_vocab=np.array(["g0", "g1", "g2", "g3"]),
            track_id_to_index={tid: i for i, tid in enumerate(tids)},
            sonic_variant="muq",
            sonic_pre_scaled=True,
        )

    @staticmethod
    def _minimal_cfg():
        """Build the bare minimum DSPipelineConfig that setup_embedding needs."""
        from types import SimpleNamespace
        return SimpleNamespace(candidate=SimpleNamespace(broad_filters=()))

    def test_pre_scaled_path_is_exposed_in_variant_stats(self):
        """When bundle.sonic_pre_scaled=True, variant_stats must say so (so
        downstream skips the StandardScaler that would erase the weighting)."""
        from src.playlist.pipeline.embedding_setup import setup_embedding

        bundle = self._make_pre_scaled_bundle()
        result = setup_embedding(
            bundle,
            seed_track_id="t0",
            seed_idx=0,
            anchor_seed_ids=[],
            mode="narrow",
            cfg=self._minimal_cfg(),
            sonic_weight=None,
            genre_weight=None,
            min_genre_similarity=None,
            random_seed=0,
        )
        assert result.variant_stats.get("pre_scaled") is True, "should take pre_scaled path"
