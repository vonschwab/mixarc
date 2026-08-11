"""A manual live mark lands exactly where keyword detection lands: -30, non-stacking."""
from src.playlist.live_albums import build_registry
from src.title_dedupe import calculate_version_preference_score as S

REG = build_registry([{"artist": "The Smiths", "album": "“Rank”"}])
KEYS = REG.album_keys_for("the smiths")


def test_marked_album_scores_minus_30():
    assert S("The Queen Is Dead", "“Rank”", live_album_keys=KEYS) == 70
    assert S("The Queen Is Dead", "The Queen Is Dead", live_album_keys=KEYS) == 100


def test_penalty_does_not_stack_with_keyword_detection():
    reg = build_registry([{"artist": "Nirvana", "album": "MTV Unplugged in New York"}])
    keys = reg.album_keys_for("nirvana")
    # Album already keyword-detected (\blive\b misses it, but "unplugged" is a marker):
    # marked AND detected must equal detected alone.
    detected_only = S("About a Girl", "MTV Unplugged in New York")
    marked_too = S("About a Girl", "MTV Unplugged in New York", live_album_keys=keys)
    assert marked_too == detected_only


def test_no_registry_is_todays_behavior():
    assert S("The Queen Is Dead", "“Rank”") == 100          # the original tie
    assert S("The Queen Is Dead", "“Rank”", live_album_keys=None) == 100
    assert S("The Queen Is Dead", "“Rank”", live_album_keys=frozenset()) == 100


def test_smiths_regression_popularity_resolution():
    """Marking Rank makes the studio cut win popularity resolution outright."""
    from src.analyze.popularity_runner import resolve_top_tracks_to_rank
    top = [{"name": "The Queen Is Dead", "mbid": "", "rank": 0}]
    live = {"track_id": "live", "title": "The Queen Is Dead", "album": "“Rank”"}
    studio = {"track_id": "studio", "title": "The Queen Is Dead", "album": "The Queen Is Dead"}
    assert resolve_top_tracks_to_rank(top, [live, studio], live_album_keys=KEYS) == {"studio": 0}
    assert resolve_top_tracks_to_rank(top, [studio, live], live_album_keys=KEYS) == {"studio": 0}


def test_call_site_coverage():
    """Every production call to calculate_version_preference_score must thread
    live keys. Exists because three call sites once forgot to pass `album` and
    left half the detector inert for months."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r"calculate_version_preference_score\((?![^)]*live_album_keys)", text):
            if "def calculate_version_preference_score" in text[max(0, m.start() - 60):m.start()]:
                continue
            offenders.append(f"{py}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, f"call sites missing live_album_keys=: {offenders}"
