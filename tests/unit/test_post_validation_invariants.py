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



# ── piers are structural anchors, not gap violations ─────────────────────────

def test_artist_mode_piers_are_not_a_gap_violation():
    """REGRESSION (Cut Worms, 2026-07-25): artist mode banner-flagged the seed
    artist against its own piers on every run.

    Positions [1, 7, 13, 19, 25, 30] with min_gap=9 — the real report. In artist
    mode every pier IS the seed artist by construction, and min_gap governs what
    the beam places BETWEEN piers, never pier placement itself.
    """
    tracks = [_s(f"bridge{i}") for i in range(30)]
    pier_pos = [0, 6, 12, 18, 24, 29]          # 0-indexed
    for p in pier_pos:
        tracks[p] = _s("cut worms")

    assert find_artist_gap_violations(tracks, min_gap=9) != [], "precondition: fires without the exemption"
    assert find_artist_gap_violations(tracks, min_gap=9, pier_positions=pier_pos) == []


def test_the_guarantee_was_arithmetically_unsatisfiable():
    """Why this could never have been a real breach: P piers at min_gap G need
    (P-1)*G+1 slots. Cut Worms needed 46 in a 30-track playlist."""
    n_piers, gap, length = 6, 9, 30
    assert (n_piers - 1) * gap + 1 > length


def test_bridge_track_sharing_a_pier_artist_is_still_a_violation():
    """The exemption must stay narrow: only pier-vs-pier is structural. A bridge
    track by the pier artist inside an interior is a real
    disallow_pier_artists_in_interiors breach."""
    tracks = [_s(f"bridge{i}") for i in range(10)]
    tracks[0] = _s("cut worms")      # pier
    tracks[6] = _s("cut worms")      # pier
    tracks[2] = _s("cut worms")      # BRIDGE by the pier artist -- must be caught
    v = find_artist_gap_violations(tracks, min_gap=9, pier_positions=[0, 6])
    pairs = {(i, j) for i, j, _a, _g in v}
    assert (1, 3) in pairs and (3, 7) in pairs
    assert (1, 7) not in pairs       # the pier-vs-pier pair stays exempt


def test_bridge_vs_bridge_unaffected_by_pier_exemption():
    tracks = [_s("a"), _s("b"), _s("a")]
    assert len(find_artist_gap_violations(tracks, min_gap=9, pier_positions=[1])) == 1


def test_no_pier_positions_preserves_prior_behaviour():
    tracks = [_s("a"), _s("a")]
    assert len(find_artist_gap_violations(tracks, min_gap=9)) == 1

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


# ── xhigh review Finding 6: per-artist cap overrides (multi-artist blend) ──


def test_per_artist_override_relaxes_only_the_named_key():
    """A key present in per_artist_overrides uses ITS OWN cap; every other
    key stays on the global max_per_artist -- the multi-artist blend's
    relaxed cap must never apply to an incidental non-seed identity."""
    tracks = [_s("eno")] * 12 + [_s("some_bridge_artist")] * 6
    violations = find_artist_cap_violations(
        tracks, max_per_artist=4, per_artist_overrides={"eno": 12},
    )
    # "eno" is exactly at its own relaxed cap (12) -> clean. The incidental
    # bridge-fill artist at 6 > the UNSCALED cap (4) -> still a violation.
    assert violations == [("some_bridge_artist", 6)]


def test_per_artist_override_still_catches_a_breach_above_the_relaxed_cap():
    tracks = [_s("eno")] * 15
    violations = find_artist_cap_violations(
        tracks, max_per_artist=4, per_artist_overrides={"eno": 12},
    )
    assert violations == [("eno", 15)]


def test_no_overrides_is_byte_identical_to_the_plain_cap():
    tracks = [_s("pablo")] * 5 + [_s("b")]
    assert (
        find_artist_cap_violations(tracks, max_per_artist=4, per_artist_overrides=None)
        == find_artist_cap_violations(tracks, max_per_artist=4)
        == [("pablo", 5)]
    )


def test_zero_global_cap_still_enforces_an_override():
    """max_per_artist<=0 normally disables the whole check -- but a caller
    that supplies overrides for specific keys (multi-artist blend with the
    global cap otherwise off) must still enforce those."""
    tracks = [_s("eno")] * 3 + [_s("other")] * 100
    violations = find_artist_cap_violations(
        tracks, max_per_artist=0, per_artist_overrides={"eno": 2},
    )
    assert violations == [("eno", 3)]
