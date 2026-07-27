"""Pure placement rules for tag-steering on-tag anchors (spec
docs/superpowers/specs/2026-07-27-tag-steering-anchor-placement.md).

The two rules are STRUCTURAL: gap insertion can't touch position 0/-1, and at
most one anchor lands per gap. These tests pin that, plus the clamp (K <= P-1),
the min_bridge drop, the byte-identical K=0 path, and determinism.

`pair_score` is injected so the tests can express bridgeability as a plain dict
lookup -- no matrices, no bundle.
"""
from src.playlist.pier_bridge.anchor_placement import place_anchors_in_gaps


def _uniform(score=0.9):
    return lambda a, b: score


def test_anchors_are_never_terminal_and_never_adjacent():
    # 6 artist piers (0..5) -> 5 gaps; 3 anchors (100, 101, 102).
    seq = place_anchors_in_gaps([0, 1, 2, 3, 4, 5], [100, 101, 102], _uniform()).sequence
    anchors = {100, 101, 102}
    assert seq[0] not in anchors and seq[-1] not in anchors
    assert not any(a in anchors and b in anchors for a, b in zip(seq, seq[1:]))
    assert [x for x in seq if x not in anchors] == [0, 1, 2, 3, 4, 5]
    assert sorted(x for x in seq if x in anchors) == [100, 101, 102]


def test_clamps_to_p_minus_one_and_reports_dropped_by_rank():
    # 3 artist piers -> 2 gaps, 3 anchors: the lowest-ranked (last) is dropped.
    res = place_anchors_in_gaps([0, 1, 2], [100, 101, 102], _uniform())
    assert res.dropped_clamped == [102]
    assert res.dropped_unbridgeable == []
    assert res.dropped_displaced == []
    assert len([x for x in res.sequence if x in {100, 101, 102}]) == 2
    assert res.sequence[0] == 0 and res.sequence[-1] == 2


def test_zero_anchors_is_the_identity():
    res = place_anchors_in_gaps([7, 8, 9], [], _uniform())
    assert res.sequence == [7, 8, 9]
    assert res.placed == [] and res.dropped_clamped == [] and res.dropped_unbridgeable == [] and res.dropped_displaced == []


def test_anchor_below_min_bridge_in_every_gap_is_dropped_not_placed():
    def score(a, b):
        # anchor 100 bridges well; anchor 101 is an island everywhere.
        return 0.10 if 101 in (a, b) else 0.90

    res = place_anchors_in_gaps([0, 1, 2], [100, 101], score, min_bridge=0.35)
    assert res.dropped_unbridgeable == [101]
    assert res.dropped_displaced == []
    assert 100 in res.sequence and 101 not in res.sequence


def test_assignment_maximizes_the_worst_flanking_edge():
    # Minimax test discriminates min(leading, trailing) from leading-edge-only.
    # Anchor 100: weak leading to pier 0 (0.1), strong trailing (0.9) in gap 0;
    #             strong leading to pier 1 (0.9), weak trailing (0.1) in gap 1.
    # Anchor 101: balanced (0.55) on both edges in both gaps.
    # Minimax scores (taking min of both flanks):
    #   - 100 in gap 0: min(0.1, 0.9) = 0.1
    #   - 100 in gap 1: min(0.9, 0.1) = 0.1
    #   - 101 in gap 0: min(0.55, 0.55) = 0.55
    #   - 101 in gap 1: min(0.55, 0.55) = 0.55
    # Assignment {100->gap0, 101->gap1}: worst edge = min(0.1, 0.55) = 0.1
    # Assignment {101->gap0, 100->gap1}: worst edge = min(0.55, 0.1) = 0.1
    # Both tie on minimax (0.1), so tie-break prefers earlier-ranked anchors:
    # {100->gap0, 101->gap1} wins.
    # Leading-edge-only would see: {100->gap0}: 0.1, {100->gap1}: 0.9, and pick
    # {101->gap0, 100->gap1} (0.55 + 0.9 = 1.45 > 0.1 + 0.55 = 0.65), proving
    # the algorithm uses min(), not leading-only or sum.

    def score(a, b):
        # Directed scores: pair_score(a, b) = score of a->b transition
        if a == 0 and b == 100:
            return 0.1  # pier 0 -> anchor 100 (weak leading)
        if a == 100 and b == 1:
            return 0.9  # anchor 100 -> pier 1 (strong trailing)
        if a == 1 and b == 100:
            return 0.9  # pier 1 -> anchor 100 (strong leading)
        if a == 100 and b == 2:
            return 0.1  # anchor 100 -> pier 2 (weak trailing)
        if a == 0 and b == 101:
            return 0.55
        if a == 101 and b == 1:
            return 0.55
        if a == 1 and b == 101:
            return 0.55
        if a == 101 and b == 2:
            return 0.55
        return 0.5  # default for pier-pier transitions

    res = place_anchors_in_gaps([0, 1, 2], [100, 101], score, min_bridge=0.0)
    # Minimax picks {100->gap0, 101->gap1} on tie-break (earlier-ranked anchors)
    assert res.sequence == [0, 100, 1, 101, 2]
    assert res.dropped_displaced == []


def test_deterministic_across_repeated_calls():
    def score(a, b):
        return 0.5  # everything ties: determinism must come from the tie-break

    runs = [place_anchors_in_gaps([0, 1, 2, 3], [100, 101], score) for _ in range(5)]
    sequences = {tuple(r.sequence) for r in runs}
    assert len(sequences) == 1
    # All runs should have identical dropped buckets
    assert all(r.dropped_displaced == [] for r in runs)


def test_fewer_than_two_artist_piers_drops_every_anchor():
    res = place_anchors_in_gaps([0], [100], _uniform())
    assert res.sequence == [0]
    assert res.dropped_clamped == [100]
    assert res.dropped_unbridgeable == []
    assert res.dropped_displaced == []


def test_displaced_anchors_vs_unbridgeable():
    # Three anchors, three gaps. All three anchors clear min_bridge (0.35) ONLY in
    # gap 0, and fail the floor (0.2) everywhere else. So only gap 0 is viable,
    # and the algorithm places only anchor 100 (highest ranked), while anchors 101
    # and 102 are displaced (cleared the floor somewhere but lost to a competitor).
    # Accounting invariant: placed (1) + dropped_clamped (0) + dropped_unbridgeable (0)
    # + dropped_displaced (2) = 3 total.

    def score(a, b):
        # Gap 0 (0->1): anchors clear the floor (min values 0.5, 0.4, 0.36)
        if b in (100, 101, 102) and a == 0:
            return {100: 0.5, 101: 0.4, 102: 0.36}[b]
        if a in (100, 101, 102) and b == 1:
            return {100: 0.5, 101: 0.4, 102: 0.36}[a]
        # Gaps 1 and 2: anchors fail the floor (0.2 < 0.35)
        return 0.2

    res = place_anchors_in_gaps([0, 1, 2, 3], [100, 101, 102], score, min_bridge=0.35)
    assert res.dropped_clamped == []
    assert res.dropped_unbridgeable == []
    assert res.dropped_displaced == [101, 102]
    assert len([x for x in res.sequence if x in {100, 101, 102}]) == 1
    assert 100 in res.sequence
    # Verify accounting invariant
    placed_count = len(res.placed)
    total_dropped = len(res.dropped_clamped) + len(res.dropped_unbridgeable) + len(res.dropped_displaced)
    assert placed_count + total_dropped == 3
