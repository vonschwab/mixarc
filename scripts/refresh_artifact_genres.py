#!/usr/bin/env python3
"""
Surgical genre-matrix refresh
=============================

Rebuilds ONLY the genre matrices (X_genre_raw, X_genre_smoothed, genre_vocab) of
an existing beat3tower artifact, propagating metadata.db genre edits (e.g. a
removed bad album tag) WITHOUT touching the sonic matrices or track ordering.

Reuses the official builder functions (load_genres_for_tracks, build_genre_matrices)
aligned to the artifact's stored track_ids, so the genre output is identical to a
full rebuild — only the (deterministic, already-stored) sonic side is skipped.

Dry-run by default; pass --apply to write (backs up the artifact first).

    python scripts/refresh_artifact_genres.py                 # dry-run report
    python scripts/refresh_artifact_genres.py --apply         # write (with .bak)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_beat3tower_artifacts import build_genre_matrices, load_genres_for_tracks

ARTIFACT = ROOT / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
DB = ROOT / "data" / "metadata.db"
SIDECAR_DB = ROOT / "data" / "ai_genre_enrichment.db"
YUJI = "dcfd2c974b15618546c8326a8ce9844d"  # canary: bad 'rock' tag removed from DB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--sidecar-db", type=Path, default=SIDECAR_DB)
    ap.add_argument("--genre-sim-path", default=None)
    ap.add_argument("--apply", action="store_true", help="write the refreshed artifact")
    args = ap.parse_args()

    data = dict(np.load(args.artifact, allow_pickle=True))
    track_ids = [str(t) for t in data["track_ids"]]
    old_vocab = [str(g) for g in data["genre_vocab"]]
    old_raw = data["X_genre_raw"]
    old_smoothed = data["X_genre_smoothed"]
    smoothing_was_applied = not np.array_equal(old_raw, old_smoothed)
    print(f"Artifact: {len(track_ids)} tracks, old vocab={len(old_vocab)}, "
          f"smoothing_applied={smoothing_was_applied}")
    if smoothing_was_applied and not args.genre_sim_path:
        print("  WARNING: original had genre smoothing but no --genre-sim-path given; "
              "refreshed X_genre_smoothed will equal X_genre_raw. Pass the sim path to match.")

    # Refresh with the SAME genre source the artifact was built with. Reading it
    # from the artifact's own build_config rather than defaulting: this script
    # previously always attached an EnrichedGenreResolver and never passed
    # use_graph_genres, so on a graph-sourced artifact it rebuilt from the
    # deprecated enriched_genre_signatures layer instead of the published
    # authority. On 2026-07-27 that dry run showed vocab 442 -> 1013, 628 tracks
    # LOSING genres, and the Yuji canary going from chamber music/modern
    # classical to 'rock'. Same class as the 2026-06-11 incident where the
    # builder hardcoded genre_source='legacy' and enrichment never reached
    # generation (genre-data-authority skill, trap catalog).
    build_config = data.get("build_config")
    if build_config is not None and getattr(build_config, "shape", None) == ():
        build_config = build_config.item()
    genre_source = (build_config or {}).get("genre_source") if isinstance(build_config, dict) else None
    if genre_source is None:
        print("  ERROR: artifact has no build_config.genre_source; refusing to guess "
              "which genre layer to rebuild from.", file=sys.stderr)
        return 2
    print(f"  genre_source (from artifact build_config): {genre_source}")

    resolver = None
    if genre_source == "graph":
        # The authority IS the source; the enriched sidecar must not override it.
        print("  enriched resolver: disabled (graph authority is the source)")
    elif args.sidecar_db.exists():
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        resolver = EnrichedGenreResolver(str(args.sidecar_db))
        print(f"  enriched resolver: {args.sidecar_db.name}")

    # Rebuild genres aligned to the artifact's EXISTING track order.
    genre_lists, vocab, stats = load_genres_for_tracks(
        str(args.db), track_ids, normalize_genres=True,
        tracks_metadata=None, enriched_resolver=resolver,
        use_graph_genres=(genre_source == "graph"),
    )
    new_raw, new_smoothed = build_genre_matrices(genre_lists, vocab, args.genre_sim_path)

    # ---- Report what changes ----
    print(f"\n  new vocab = {len(vocab)} (was {len(old_vocab)})")
    old_nnz = (old_raw > 0).sum(axis=1)
    new_nnz = (new_raw > 0).sum(axis=1)
    changed = int((old_raw.shape[1] != new_raw.shape[1]) or
                  np.any(old_nnz != new_nnz))
    n_tracks_changed_count = int((old_nnz != new_nnz).sum()) if old_raw.shape[1] == new_raw.shape[1] or True else -1
    print(f"  tracks with >=1 genre: {int((new_nnz>0).sum())} (was {int((old_nnz>0).sum())})")
    print(f"  tracks whose genre-count changed: {int((old_nnz != new_nnz).sum())}")
    # Yuji canary
    if YUJI in track_ids:
        yi = track_ids.index(YUJI)
        old_g = [old_vocab[j] for j in np.where(old_raw[yi] > 0)[0]]
        new_g = [vocab[j] for j in np.where(new_raw[yi] > 0)[0]]
        print(f"  CANARY Yuji 'Angel's Room': old genres={old_g} -> new genres={new_g}")

    if not args.apply:
        print("\n  DRY RUN — no files written. Re-run with --apply to write.")
        return 0

    # ---- Apply: back up, replace genre arrays only, re-save ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = args.artifact.with_suffix(f".npz.bak_{ts}")
    shutil.copy2(args.artifact, bak)
    print(f"\n  backed up artifact -> {bak.name}")

    data["X_genre_raw"] = new_raw
    data["X_genre_smoothed"] = new_smoothed
    data["genre_vocab"] = np.array(vocab, dtype=object)
    # refresh genre_stats inside build_config if present
    if "build_config" in data:
        bc = data["build_config"].item() if hasattr(data["build_config"], "item") else data["build_config"]
        try:
            bc["genre_stats"] = stats
            bc["genre_refreshed_at"] = ts
            data["build_config"] = bc
        except Exception:
            pass

    np.savez(args.artifact, **data)
    print(f"  wrote refreshed artifact: {args.artifact.name}")
    print("  NEXT: rebuild dense sidecar -> python scripts/build_genre_embedding.py --skip-prior")
    return 0


if __name__ == "__main__":
    sys.exit(main())
