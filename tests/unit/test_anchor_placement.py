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
    assert len([x for x in res.sequence if x in {100, 101, 102}]) == 2
    assert res.sequence[0] == 0 and res.sequence[-1] == 2


def test_zero_anchors_is_the_identity():
    res = place_anchors_in_gaps([7, 8, 9], [], _uniform())
    assert res.sequence == [7, 8, 9]
    assert res.placed == [] and res.dropped_clamped == [] and res.dropped_unbridgeable == []


def test_anchor_below_min_bridge_in_every_gap_is_dropped_not_placed():
    def score(a, b):
        # anchor 100 bridges well; anchor 101 is an island everywhere.
        return 0.10 if 101 in (a, b) else 0.90

    res = place_anchors_in_gaps([0, 1, 2], [100, 101], score, min_bridge=0.35)
    assert res.dropped_unbridgeable == [101]
    assert 100 in res.sequence and 101 not in res.sequence


def test_assignment_maximizes_the_worst_flanking_edge():
    # Gap 0 = (0,1) and gap 1 = (1,2). Anchor 100 is an outlier next to pier 0
    # (0.2) but strong next to piers 1 and 2; anchor 101 is uniform. Minimax over
    # BOTH flanks: 100 in gap 0 scores min(0.2, 0.8) = 0.2, in gap 1 min(0.8, 0.9)
    # = 0.8. The only worst-edge-maximizing assignment is 101 -> gap 0, 100 -> gap 1.
    _AFF = {100: {0: 0.2, 1: 0.8, 2: 0.9}, 101: {0: 0.7, 1: 0.7, 2: 0.7}}

    def score(a, b):
        if a in _AFF:
            return _AFF[a][b]
        if b in _AFF:
            return _AFF[b][a]
        return 0.5

    res = place_anchors_in_gaps([0, 1, 2], [100, 101], score, min_bridge=0.0)
    assert res.sequence == [0, 101, 1, 100, 2]


def test_deterministic_across_repeated_calls():
    def score(a, b):
        return 0.5  # everything ties: determinism must come from the tie-break

    runs = {
        tuple(place_anchors_in_gaps([0, 1, 2, 3], [100, 101], score).sequence)
        for _ in range(5)
    }
    assert len(runs) == 1


def test_fewer_than_two_artist_piers_drops_every_anchor():
    res = place_anchors_in_gaps([0], [100], _uniform())
    assert res.sequence == [0]
    assert res.dropped_clamped == [100]
