"""Pass 1 invariants: artist gap, per-artist cap, pier ratio.

These land in ``warnings``, never ``errors`` — a violation degrades the run, it
does not raise (spec 2026-07-25-genre-mode-pass1-invariants.md §4.2).

Identity is a SET per track, not a string: ``resolve_artist_identity_keys``
returns one key per participant, so "Bob Brookmeyer & Bill Evans" shares an
identity with "Bill Evans" and the beam treats them as the same artist for
min_gap. Comparing collapsed strings would silently UNDER-report.
"""

from src.playlist.pipeline.post_validation import (
    find_artist_cap_violations,
    find_artist_gap_violations,
)


def _s(*names):
    """One track's identity key SET."""
    return set(names)


# ── gap ──────────────────────────────────────────────────────────────────────

def test_gap_violation_detected():
    # 4 consecutive same-identity entries, like the reggae Augustus Pablo run
    tracks = [_s("a"), _s("b"), _s("pablo"), _s("pablo"), _s("pablo"), _s("pablo")]
    v = find_artist_gap_violations(tracks, min_gap=9)
    pairs = {(i, j) for i, j, _a, _g in v}
    assert (3, 4) in pairs and (4, 5) in pairs and (5, 6) in pairs


def test_collaboration_sharing_a_member_is_a_violation():
    """The beam treats these as the same artist for min_gap; string equality
    would miss it, so the validator must use set intersection."""
    tracks = [_s("bill evans"), _s("bob brookmeyer", "bill evans")]
    v = find_artist_gap_violations(tracks, min_gap=9)
    assert len(v) == 1
    assert v[0][2] == "bill evans"


def test_disjoint_collaborators_are_not_a_violation():
    tracks = [_s("bill evans"), _s("duke ellington", "john coltrane")]
    assert find_artist_gap_violations(tracks, min_gap=9) == []


def test_no_violation_when_spaced_beyond_gap():
    # Filler must be DISTINCT artists — nine identical fillers would violate the
    # gap among themselves and mask what this test is actually pinning.
    tracks = [_s("a")] + [_s(f"x{i}") for i in range(9)] + [_s("a")]
    assert find_artist_gap_violations(tracks, min_gap=9) == []


def test_violation_when_exactly_one_short_of_the_gap():
    """Boundary: gap of 8 with min_gap=9 still conflicts; gap of 9 does not."""
    just_inside = [_s("a")] + [_s(f"x{i}") for i in range(7)] + [_s("a")]
    assert len(find_artist_gap_violations(just_inside, min_gap=9)) == 1
    just_outside = [_s("a")] + [_s(f"x{i}") for i in range(8)] + [_s("a")]
    assert find_artist_gap_violations(just_outside, min_gap=9) == []


def test_gap_zero_disables_the_check():
    assert find_artist_gap_violations([_s("a"), _s("a")], min_gap=0) == []


def test_violation_positions_are_one_indexed():
    v = find_artist_gap_violations([_s("a"), _s("a")], min_gap=2)
    assert v[0][0] == 1 and v[0][1] == 2


def test_shared_key_is_deterministic_when_several_intersect():
    tracks = [_s("m", "z"), _s("m", "z")]
    v = find_artist_gap_violations(tracks, min_gap=2)
    assert v[0][2] == "m"  # lexicographically smallest intersecting key


def test_empty_identity_set_never_conflicts():
    """A track whose artist could not be resolved must not collide with another
    unresolved track — empty ∩ empty is empty."""
    assert find_artist_gap_violations([set(), set()], min_gap=9) == []


# ── cap ──────────────────────────────────────────────────────────────────────

def test_cap_violation_detected():
    tracks = [_s("pablo")] * 5 + [_s("b")]
    assert find_artist_cap_violations(tracks, max_per_artist=4) == [("pablo", 5)]


def test_cap_counts_each_participant_of_a_collaboration():
    tracks = [_s("pablo")] * 4 + [_s("pablo", "tubby")]
    assert find_artist_cap_violations(tracks, max_per_artist=4) == [("pablo", 5)]


def test_cap_exactly_at_limit_is_clean():
    assert find_artist_cap_violations([_s("p")] * 4, max_per_artist=4) == []


def test_cap_zero_disables_the_check():
    assert find_artist_cap_violations([_s("p")] * 9, max_per_artist=0) == []


def test_cap_orders_by_count_desc_then_key():
    tracks = [_s("b")] * 6 + [_s("a")] * 6 + [_s("c")] * 7
    assert find_artist_cap_violations(tracks, max_per_artist=4) == [
        ("c", 7), ("a", 6), ("b", 6),
    ]
