"""Post-order validation + failure diagnostic messages.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

Two responsibilities:
  * build_failure_diagnostic: classifies a pier-bridge failure into one of
    three user-facing root-cause buckets (genre isolation, sonic isolation,
    insufficient pool) and returns the diagnostic message + raw counters
    for audit emission.
  * run_post_order_validation: enforces the post-order DS contract — length
    must match, no recency-excluded ids may appear (except piers).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.features.artifacts import ArtifactBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolDiagnostics:
    """Bottleneck classification numbers from the candidate-pool stats."""
    admitted: int
    rejected_sonic: int
    rejected_genre: int
    total_considered: int


@dataclass(frozen=True)
class FailureDiagnostic:
    """User-facing failure message + structured counters for the audit."""
    failure_reason: str
    diagnostic_msg: str
    pool_diagnostics: PoolDiagnostics


def build_failure_diagnostic(
    pool_stats: Dict[str, Any],
    pb_failure_reason: Optional[str],
    *,
    pool_indices_count: int,
    seed_track_ids_for_pier_count: int,
    cfg_mode: str,
    cfg_genre_gate_min_similarity: float,
) -> FailureDiagnostic:
    """Classify a pier-bridge failure and produce the user-facing message.

    Three root-cause buckets, in priority order:
      1. GENRE ISOLATION — admitted=0 and any tracks were rejected by genre
      2. SONIC ISOLATION — admitted=0 and any tracks were rejected by sonic
      3. INSUFFICIENT CANDIDATE POOL — 0 < admitted < 10
    Otherwise the raw failure reason is returned unmodified.
    """
    failure_reason = str(pb_failure_reason or "pier-bridge failed")

    admitted = int(pool_stats.get("pool_size", pool_indices_count))
    rejected_sonic = int(pool_stats.get("below_sonic_similarity", 0))
    rejected_genre = int(pool_stats.get("below_genre_similarity", 0))
    total_considered = int(pool_stats.get("total_candidates_considered", 0))

    diagnostic_msg = failure_reason
    if admitted == 0 and rejected_genre > 0:
        # Genre gating eliminated all candidates
        passed_sonic = total_considered - rejected_sonic if total_considered > rejected_sonic else 0
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: GENRE ISOLATION\n"
            f"   - {passed_sonic} tracks passed sonic similarity\n"
            f"   - ALL {rejected_genre} were rejected by genre filter (min_similarity={cfg_genre_gate_min_similarity:.2f})\n"
            f"   - Zero candidates remained for playlist generation\n\n"
            f"💡 This artist's genres are too isolated from your library.\n"
            f"   Suggestions:\n"
            f"   - Reduce genre mode from {cfg_mode} to 'narrow' or 'dynamic'\n"
            f"   - Add more music in similar genres to your library\n"
            f"   - Use genre mode 'discover' to disable hard genre filtering"
        )
    elif admitted == 0 and rejected_sonic > 0:
        # Sonic similarity eliminated all candidates
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: SONIC ISOLATION\n"
            f"   - {rejected_sonic} of {total_considered} tracks rejected by sonic floor\n"
            f"   - Zero candidates remained for playlist generation\n\n"
            f"💡 This artist's sound is too sonically isolated from your library.\n"
            f"   Suggestions:\n"
            f"   - Reduce sonic mode from {cfg_mode} to 'narrow' or 'dynamic'\n"
            f"   - Add more sonically similar music to your library"
        )
    elif admitted > 0 and admitted < 10:
        # Small pool passed filters but not enough for bridging
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: INSUFFICIENT CANDIDATE POOL\n"
            f"   - Only {admitted} candidates passed all filters\n"
            f"   - Rejected: {rejected_sonic} sonic, {rejected_genre} genre\n"
            f"   - Need more candidates to bridge between {seed_track_ids_for_pier_count} seeds\n\n"
            f"💡 The candidate pool is too small for multi-seed bridging.\n"
            f"   Suggestions:\n"
            f"   - Use fewer seeds (1-2 instead of {seed_track_ids_for_pier_count})\n"
            f"   - Reduce filtering: set genre/sonic modes to 'discover'\n"
            f"   - Choose seeds that are more sonically/genre similar to each other"
        )

    return FailureDiagnostic(
        failure_reason=failure_reason,
        diagnostic_msg=diagnostic_msg,
        pool_diagnostics=PoolDiagnostics(
            admitted=admitted,
            rejected_sonic=rejected_sonic,
            rejected_genre=rejected_genre,
            total_considered=total_considered,
        ),
    )


@dataclass(frozen=True)
class PostOrderValidation:
    """Result of post-order validation — summary stats, errors, and warnings.

    ``errors`` make the caller RAISE (``pipeline/core.py``). ``warnings`` do not:
    they mark the run degraded so a hard-invariant breach is visible without
    turning a flawed playlist into no playlist. See
    ``docs/superpowers/specs/2026-07-25-genre-mode-pass1-invariants.md`` §4.2 —
    artist-gap violations move to ``errors`` once Pass 3 can reseed piers.
    """
    summary: Dict[str, int]
    errors: List[str]
    recency_overlap_ids: List[str]
    warnings: List[str] = field(default_factory=list)


def find_artist_gap_violations(identity_key_sets, min_gap: int) -> List[tuple]:
    """Pairs of tracks closer than ``min_gap`` whose identity key sets INTERSECT.

    Returns ``(pos_i, pos_j, shared_key, gap)`` with 1-indexed positions.

    Each element of ``identity_key_sets`` is one track's set of identity keys from
    ``resolve_artist_identity_keys`` — a collaboration contributes one key per
    participant. Conflict is set intersection, matching how the beam enforces
    min_gap: "Bill Evans" and "Bob Brookmeyer & Bill Evans" ARE the same artist
    for this purpose, and collapsing each track to a single string would silently
    miss that.

    ``shared_key`` is the lexicographically smallest intersecting key so output is
    deterministic. ``min_gap <= 0`` disables the check.
    """
    sets = [set(s) for s in identity_key_sets]
    gap = int(min_gap)
    if gap <= 0:
        return []
    out: List[tuple] = []
    for i in range(len(sets)):
        for j in range(i + 1, min(i + gap, len(sets))):
            shared = sets[i] & sets[j]
            if shared:
                out.append((i + 1, j + 1, min(shared), j - i))
    return out


def find_artist_cap_violations(identity_key_sets, max_per_artist: int) -> List[tuple]:
    """Identity keys appearing in more than ``max_per_artist`` tracks, as
    ``(key, count)`` ordered by count desc then key.

    Counts per PARTICIPANT: a collaboration counts toward every member's total,
    matching the beam's per-artist accounting. ``max_per_artist <= 0`` disables
    the check.
    """
    from collections import Counter

    cap = int(max_per_artist)
    if cap <= 0:
        return []
    counts: Counter = Counter()
    for s in identity_key_sets:
        for key in set(s):
            if key:
                counts[key] += 1
    return sorted(
        ((a, n) for a, n in counts.items() if n > cap),
        key=lambda t: (-t[1], t[0]),
    )


def _resolve_artist_identities(
    bundle: ArtifactBundle, ordered_track_ids: List[str], cfg
) -> List[set]:
    """One identity key SET per ordered track, matching how the beam compares artists.

    Mirrors the beam's gate (``artist_identity_cfg is not None and cfg.enabled``):
    when identity resolution is on, use ``resolve_artist_identity_keys`` so a
    collaboration yields one key per participant; when off, fall back to the
    bundle's own artist key as a single-element set. Unresolvable tracks get an
    empty set, which by construction never conflicts with anything.
    """
    use_identity = cfg is not None and getattr(cfg, "enabled", False)
    resolver = None
    if use_identity:
        from src.playlist.artist_identity_resolver import resolve_artist_identity_keys

        resolver = resolve_artist_identity_keys

    out: List[set] = []
    for tid in ordered_track_ids:
        idx = bundle.track_id_to_index.get(str(tid))
        if idx is None:
            out.append(set())
            continue
        try:
            artist = ""
            if bundle.track_artists is not None:
                artist = str(bundle.track_artists[int(idx)])
            if resolver is not None and artist:
                out.append(set(resolver(artist, cfg)))
                continue
            if bundle.artist_keys is not None:
                out.append({str(bundle.artist_keys[int(idx)]).strip().lower()})
            else:
                out.append({artist.strip().lower()} if artist else set())
        except Exception:
            out.append(set())
    return out


def run_post_order_validation(
    *,
    bundle: ArtifactBundle,
    ordered_track_ids: List[str],
    expected_length: int,
    excluded_track_ids: Optional[set],
    seed_track_ids_for_pier: List[str],
    artist_identity_cfg: Any = None,
    min_gap: int = 0,
    max_tracks_per_artist: int = 0,
    pier_ratio_target: float = 0.0,
    pier_ratio_tolerance: float = 0.0,
) -> PostOrderValidation:
    """Run the post-order DS contract checks.

    Hard checks (appended to ``errors``; the caller raises):
      * No track in ``excluded_track_ids`` may appear (piers are exempt).

    Soft checks (appended to ``warnings``; the run is marked degraded but still
    returns a playlist — see spec 2026-07-25-genre-mode-pass1-invariants.md §4.2):
      * Artist gap: same-identity tracks closer than ``min_gap``.
      * Per-artist cap: an identity appearing more than ``max_tracks_per_artist``
        times. Pass the beam's DERIVED cap, not the raw config literal.
      * Pier ratio: piers/total must stay within ``pier_ratio_tolerance`` of
        ``pier_ratio_target``.

    Length is deliberately NOT checked. Variable-bridge mode flexes interior
    length to lift the worst edge (approved 2026-06-28), so totals intentionally
    land in a band — exact-N is not a goal. Track count is arbitrary anyway given
    tracks vary in duration; the meaningful invariant is the pier-to-bridge ratio,
    checked above.

    All new checks default to disabled so existing callers are unaffected.

    Returns a ``PostOrderValidation`` carrying the summary dict (for audit
    emission), error strings, and warning strings.
    """
    excluded_ids_set = {str(t) for t in (excluded_track_ids or set())}
    pier_ids_set = {str(t) for t in seed_track_ids_for_pier}
    recency_overlap_ids = [
        tid for tid in ordered_track_ids if (tid in excluded_ids_set and tid not in pier_ids_set)
    ]
    summary = {
        "recency_overlap_count": int(len(recency_overlap_ids)),
        "final_size": int(len(ordered_track_ids)),
        "expected_size": int(expected_length),
    }

    errors: List[str] = []
    warnings: List[str] = []

    # Length is informational only — the band is deliberate (see docstring).
    if expected_length > 0 and len(ordered_track_ids) != expected_length:
        logger.info(
            "post_order_validation: length_in_band final=%d requested=%d "
            "(variable-bridge flex; exact-N is not a goal)",
            len(ordered_track_ids),
            expected_length,
        )

    identities = _resolve_artist_identities(bundle, ordered_track_ids, artist_identity_cfg)

    gap_violations = find_artist_gap_violations(identities, min_gap)
    if gap_violations:
        by_artist: Dict[str, List[int]] = {}
        for i, j, artist, _g in gap_violations:
            positions = by_artist.setdefault(artist, [])
            for p in (i, j):
                if p not in positions:
                    positions.append(p)
        for artist, positions in sorted(by_artist.items()):
            msg = (
                f"artist_gap_violation: artist={artist} positions={sorted(positions)} "
                f"configured_min_gap={int(min_gap)}"
            )
            warnings.append(msg)
            logger.warning("post_order_validation: %s", msg)

    cap_violations = find_artist_cap_violations(identities, max_tracks_per_artist)
    for artist, count in cap_violations:
        msg = (
            f"artist_cap_violation: artist={artist} count={count} "
            f"configured_max={int(max_tracks_per_artist)}"
        )
        warnings.append(msg)
        logger.warning("post_order_validation: %s", msg)

    if pier_ratio_target > 0.0 and ordered_track_ids:
        pier_hits = sum(1 for tid in ordered_track_ids if str(tid) in pier_ids_set)
        actual_ratio = pier_hits / float(len(ordered_track_ids))
        summary["pier_ratio_milli"] = int(round(actual_ratio * 1000))
        if abs(actual_ratio - pier_ratio_target) > pier_ratio_tolerance:
            msg = (
                f"pier_ratio_out_of_band: actual={actual_ratio:.3f} "
                f"target={pier_ratio_target:.3f} tolerance={pier_ratio_tolerance:.3f}"
            )
            warnings.append(msg)
            logger.warning("post_order_validation: %s", msg)

    summary["artist_gap_violations"] = int(len(gap_violations))
    summary["artist_cap_violations"] = int(len(cap_violations))

    if recency_overlap_ids:
        offenders: List[str] = []
        for tid in recency_overlap_ids[:10]:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is None:
                offenders.append(str(tid))
                continue
            artist = ""
            title = ""
            try:
                if bundle.track_artists is not None:
                    artist = str(bundle.track_artists[int(idx)])
                elif bundle.artist_keys is not None:
                    artist = str(bundle.artist_keys[int(idx)])
                if bundle.track_titles is not None:
                    title = str(bundle.track_titles[int(idx)])
            except Exception:
                artist = ""
                title = ""
            offenders.append(f"{tid} ({artist} - {title})")
        errors.append(f"recency_overlap={len(recency_overlap_ids)} offenders={offenders}")

    return PostOrderValidation(
        summary=summary,
        errors=errors,
        recency_overlap_ids=recency_overlap_ids,
        warnings=warnings,
    )
