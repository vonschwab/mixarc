# src/playlist/pier_bridge/anchor_placement.py
"""Gap insertion for tag-steering on-tag ANCHOR piers (pure, unit-testable).

Phase B injects on-tag tracks as piers so a sonically-peripheral on-tag clique is
guaranteed to appear. Appending them to the pier list and letting
``_order_seeds_by_bridgeability`` re-sequence everything as co-equals permits every
bad arrangement -- anchors clustering, or one taking the closing seat (observed
2026-07-27: a Sonic Youth playlist ending on Built To Spill).

This module makes Dylan's two rules structural rather than checked:

  1. an anchor is never terminal  -> gap insertion never touches position 0 or -1;
  2. no two anchors are adjacent  -> at most one anchor per gap.

Same shape as ``mini_pier_select.plan_pier_sequence``: order the real piers, then
insert non-seed waypoints into the gaps between them.

Spec: docs/superpowers/specs/2026-07-27-tag-steering-anchor-placement.md
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AnchorPlacement:
    """Result of placing anchors into the gaps between ordered artist piers.

    ``sequence``            -- artist piers with placed anchors inserted.
    ``placed``              -- (anchor, gap_index) pairs, gap g sitting between
                               artist piers g and g+1. Ordered by gap index.
    ``dropped_clamped``     -- anchors dropped because K > P-1 (lowest rank first).
    ``dropped_unbridgeable``-- anchors dropped because no gap met ``min_bridge``.
    """

    sequence: List[int]
    placed: List[Tuple[int, int]]
    dropped_clamped: List[int]
    dropped_unbridgeable: List[int]


def place_anchors_in_gaps(
    artist_piers: Sequence[int],
    anchors: Sequence[int],
    pair_score: Callable[[int, int], float],
    *,
    min_bridge: float = 0.35,
) -> AnchorPlacement:
    """Insert ``anchors`` into the interior gaps of ``artist_piers``.

    ``anchors`` must be in SELECTION-RANK order (best first): rank decides who
    survives the ``K <= P-1`` clamp.

    ``pair_score(a, b)`` is the directed bridgeability of playing ``a`` then ``b``
    (the builder passes ``_compute_bridgeability_score``). A gap's score for an
    anchor is the MINIMAX over its two flanking edges --
    ``min(pair_score(p_i, a), pair_score(a, p_{i+1}))`` -- because the weaker
    flank is what the listener hears, consistent with roam corridors' switch from
    sum to min in seed ordering.

    Assignment is exhaustive over injective anchor->gap maps (K <= 3, gaps <= ~9),
    preferring, in order: more anchors placed, higher worst score, higher total
    score, then earlier gaps -- the last purely for determinism.
    """
    piers = [int(p) for p in artist_piers]
    ranked = [int(a) for a in anchors]

    num_gaps = len(piers) - 1
    if num_gaps < 1 or not ranked:
        return AnchorPlacement(
            sequence=list(piers), placed=[], dropped_clamped=list(ranked),
            dropped_unbridgeable=[],
        )

    kept = ranked[:num_gaps]
    dropped_clamped = ranked[num_gaps:]

    # score[i][g] for kept anchor i in gap g.
    scores: List[List[float]] = [
        [
            min(
                float(pair_score(piers[g], a)),
                float(pair_score(a, piers[g + 1])),
            )
            for g in range(num_gaps)
        ]
        for a in kept
    ]

    # An anchor with no gap at/above the floor is an island: drop it rather than
    # force-place it (spec section C).
    eligible = [i for i in range(len(kept)) if max(scores[i]) >= float(min_bridge)]
    dropped_unbridgeable = [kept[i] for i in range(len(kept)) if i not in eligible]

    best_key: Optional[tuple] = None
    best_assignment: List[Tuple[int, int]] = []
    # Subsets of the eligible anchors, largest first; within a subset, every
    # injective map into the gaps. Bounded: K <= 3 so at most 8 subsets x 9*8*7 maps.
    for size in range(len(eligible), -1, -1):
        for subset in itertools.combinations(eligible, size):
            for gaps in itertools.permutations(range(num_gaps), size):
                pairs = list(zip(subset, gaps))
                if any(scores[i][g] < float(min_bridge) for i, g in pairs):
                    continue
                values = [scores[i][g] for i, g in pairs]
                key = (
                    size,
                    min(values) if values else 0.0,
                    sum(values),
                    # Prefer earlier-ranked anchors and earlier gaps on ties.
                    tuple(-i for i in subset),
                    tuple(-g for g in gaps),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_assignment = pairs
        if best_key is not None:
            break  # a larger subset always beats a smaller one

    placed_by_gap = {g: kept[i] for i, g in best_assignment}
    placed_anchor_ids = set(placed_by_gap.values())
    dropped_unbridgeable += [
        a for i, a in enumerate(kept) if i in eligible and a not in placed_anchor_ids
    ]

    sequence: List[int] = []
    for g, pier in enumerate(piers):
        sequence.append(pier)
        if g in placed_by_gap:
            sequence.append(placed_by_gap[g])

    return AnchorPlacement(
        sequence=sequence,
        placed=sorted(((a, g) for g, a in placed_by_gap.items()), key=lambda t: t[1]),
        dropped_clamped=dropped_clamped,
        dropped_unbridgeable=dropped_unbridgeable,
    )
