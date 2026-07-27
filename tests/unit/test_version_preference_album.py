import numpy as np
from src.title_dedupe import calculate_version_preference_score
from src.playlist.artist_style import _dedupe_artist_indices


def test_version_preference_penalizes_live_album():
    # Clean track title, but the ALBUM is a live recording -> penalized.
    studio = calculate_version_preference_score("Polly", "Nevermind")
    unplugged = calculate_version_preference_score("Polly", "MTV Unplugged in New York")
    reading = calculate_version_preference_score("Lithium", "Live at Reading")
    assert studio > unplugged
    assert studio > reading
    # Title-only call (no album) is unchanged / backwards-compatible.
    assert calculate_version_preference_score("Polly") == 100
    # A standalone "live" word IS penalized so live LPs named "Live <X>" (e.g. Unwound's
    # "Live Leaves") are caught. This also penalizes the rare studio LP named "Live <X>"
    # (e.g. "Live Through This") — an accepted tradeoff: the -30 only changes a tie-break
    # between two versions of the SAME song, which those studio LPs' tracks almost never have.
    assert calculate_version_preference_score("Doll Parts", "Live Through This") == 70
    # ...but a substring like "Alive"/"Olive" is NOT a false positive (word boundary).
    assert calculate_version_preference_score("Doll Parts", "Still Alive") == 100


def test_dedupe_album_aware_beats_duration_tiebreak():
    # Two "Polly": idx 0 = MTV Unplugged (LIVE, and LONGER as live takes often are),
    # idx 1 = Nevermind (studio, shorter). Clean titles, so score ties on title alone.
    titles = ["Polly", "Polly"]
    durations = np.array([240000, 196000], dtype=float)  # idx0 live longer, idx1 studio shorter
    albums = {0: "MTV Unplugged in New York", 1: "Nevermind"}
    # Without album info: score ties -> duration tiebreak keeps the LONGER (live) one — the bug.
    assert _dedupe_artist_indices([0, 1], titles, durations, None) == [0]
    # With album info: the studio version wins despite being shorter — the fix.
    assert _dedupe_artist_indices([0, 1], titles, durations, albums) == [1]


def test_version_preference_penalizes_date_named_bootlegs():
    # Date/venue bootlegs carry NO live keyword — they tied studio at 100 before this fix.
    assert calculate_version_preference_score("Kool Thing", "2004/08/20 Asheville, NC") == 70
    assert calculate_version_preference_score("Schizophrenia", "1987/05/22 Trenton, NJ") == 70
    assert calculate_version_preference_score("Pretend", "Manchester 20.09.2019") == 70
    assert calculate_version_preference_score("Jonsi", "6/30/1999: Reykjavik, Iceland (Limited Edition)") == 70
    # Month-name dates.
    assert calculate_version_preference_score("100%", "Battery Park, NYC: July 4th 2008") == 70
    assert calculate_version_preference_score("Foxey Lady", "Hollywood Bowl | August 18, 1967") == 70
    # The studio cut is unaffected.
    assert calculate_version_preference_score("Kool Thing", "Goo [24bit]") == 100


def test_bootleg_detection_does_not_false_positive_on_year_only_titles():
    # A bare year, or a place with no date, must NOT flag — that is why the pattern
    # requires a full date. These are real album-title shapes.
    assert calculate_version_preference_score("Song", "1999") == 100
    assert calculate_version_preference_score("Song", "Berlin") == 100
    assert calculate_version_preference_score("Song", "Chicago, IL") == 100
    assert calculate_version_preference_score("Song", "Summer 2008") == 100
    assert calculate_version_preference_score("Song", "Version 2.0") == 100


def test_bootleg_penalty_does_not_stack_with_live_keyword():
    # An album that is BOTH date-named and keyword-live gets one -30, not two.
    assert calculate_version_preference_score("Song", "Live In Dallas 12/06/2006") == 70
