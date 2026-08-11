"""Seed-artist share reporting.

The share the reporter prints must be counted with the SAME artist-identity
resolution the generator used to seat the piers. Every case below is taken from
a real run in logs/playlists/ where the old raw-casefold count disagreed with
what the playlist actually contained.
"""
import logging

import pytest

from src.playlist.artist_identity_resolver import (
    ArtistIdentityConfig,
    parse_artist_identity_config,
)
from src.playlist.reporter import _report_seed_artist_share


ON = ArtistIdentityConfig(enabled=True)


def _tracks(*artists):
    return [{"artist": a} for a in artists]


def _shares(caplog):
    """Parse the emitted lines into {label: (count, pct)}."""
    out = {}
    for rec in caplog.records:
        msg = rec.getMessage()
        if "Seed artist" not in msg:
            continue
        label = msg.split("(")[1].split(")")[0] if "Seed artist (" in msg else "combined"
        nums = msg.rsplit(":", 1)[1]
        count = int(nums.strip().split()[0])
        pct = float(nums.split("(")[-1].rstrip("%)\n "))
        out[label] = (count, pct)
    return out


def _run(caplog, tracks, artist_name, names=None, cfg=ON):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.playlist.reporter"):
        _report_seed_artist_share(tracks, artist_name, names, cfg)
    return _shares(caplog)


def test_ensemble_and_group_credits_count_toward_their_members(caplog):
    """From 2026-08-01_205737_Bill_Evans: six piers were seated, and the report
    said "Bill Evans: 1 tracks (3.3%)". Raw casefold matched only the bare
    "Bill Evans" credit, missing "Bill Evans Trio", "The Bill Evans Trio", and
    the Miles Davis quintet credit that Red Garland's piers were seated from."""
    tracks = _tracks(
        "Bill Evans", "Thelonious Monk", "Red Garland", "The Bill Evans Trio",
        "Bill Evans Trio", "Herbie Hancock",
        "Miles Davis, John Coltrane, Red Garland, Paul Chambers, Philly Joe Jones",
        "Miles Davis, John Coltrane, Red Garland, Paul Chambers, Philly Joe Jones",
        "Bud Powell", "Wayne Shorter",
    )
    shares = _run(caplog, tracks, "Bill Evans", ["Bill Evans", "Red Garland"])
    assert shares["Bill Evans"][0] == 3      # bare + Trio + The ... Trio
    assert shares["Red Garland"][0] == 3     # bare + both quintet credits


def test_multi_artist_reports_every_chip_and_a_combined_total(caplog):
    """From 2026-08-06_183130_Green-House: a 3/3 split reported "10.0%", which
    reads as half the dial's promise. Only the primary chip was ever counted."""
    tracks = _tracks(
        "Green-House", "Dream Dolphin", "Jonny Nash & Suzanne Kraft", "Unknown Me",
        "Green-House", "Yoshio Ojima", "Jonny Nash & Suzanne Kraft", "Ana Roxanne",
        "Green-House", "Jonny Nash & Suzanne Kraft",
    )
    shares = _run(
        caplog, tracks, "Green-House", ["Green-House", "Jonny Nash & Suzanne Kraft"]
    )
    assert shares["Green-House"] == (3, pytest.approx(30.0))
    assert shares["Jonny Nash & Suzanne Kraft"] == (3, pytest.approx(30.0))
    assert shares["combined"] == (6, pytest.approx(60.0))


def test_combined_counts_tracks_not_credits(caplog):
    """A single track crediting BOTH seed artists is one track of the playlist,
    so the combined share must not double-count it -- otherwise a blend heavy
    in collaborations reports a share above what it occupies."""
    tracks = _tracks("Brian Eno & David Byrne", "Radiohead", "Brian Eno", "Tom Waits")
    shares = _run(caplog, tracks, "Brian Eno", ["Brian Eno", "David Byrne"])
    assert shares["Brian Eno"][0] == 2       # solo + the collaboration
    assert shares["David Byrne"][0] == 1     # the collaboration
    assert shares["combined"] == (2, pytest.approx(50.0)), (
        "the collaboration is ONE track of the playlist, not two"
    )


def test_single_artist_emits_no_combined_line(caplog):
    tracks = _tracks("Duster", "Teethe", "Duster", "Yuck")
    shares = _run(caplog, tracks, "Duster")
    assert shares["Duster"] == (2, pytest.approx(50.0))
    assert "combined" not in shares, "a lone seed artist has nothing to combine"


def test_feat_credits_count_toward_the_seed(caplog):
    """From 2026-08-08_151209_MACROSS_82-99: the report said 5 tracks where the
    playlist held more, because "Macross 82-99 feat. Desired" matched nothing."""
    tracks = _tracks(
        "Macross 82-99", "Toro Y Moi", "Macross 82-99 feat. Desired", "Justice",
        "MACROSS 82-99", "Daft Punk",
    )
    shares = _run(caplog, tracks, "MACROSS 82-99")
    assert shares["MACROSS 82-99"][0] == 3


def test_disabled_identity_config_falls_back_to_plain_normalization(caplog):
    """Identity resolution is opt-in. With it off the count must not silently
    adopt identity semantics -- ensembles stay separate artists."""
    tracks = _tracks("Bill Evans", "Bill Evans Trio", "Thelonious Monk")
    shares = _run(caplog, tracks, "Bill Evans", None, ArtistIdentityConfig())
    assert shares["Bill Evans"][0] == 1


def test_no_seed_artist_emits_nothing(caplog):
    assert _run(caplog, _tracks("A", "B"), None, None) == {}


def test_empty_playlist_does_not_divide_by_zero(caplog):
    assert _run(caplog, [], "Duster", ["Duster"]) == {}


def test_parse_artist_identity_config_matches_the_shipped_config():
    """The reporter and the beam must resolve identity from the same parse."""
    parsed = parse_artist_identity_config(
        {"enabled": True, "split_delimiters": [" & "], "trailing_ensemble_terms": ["trio"]}
    )
    assert parsed.enabled is True
    assert parsed.split_delimiters == [" & "]
    assert parsed.trailing_ensemble_terms == ["trio"]


@pytest.mark.parametrize("raw", [None, "not-a-dict", 42, []])
def test_parse_artist_identity_config_defaults_to_disabled(raw):
    """A missing or malformed mapping must not silently enable identity
    resolution -- it stays opt-in, as it was when the parse lived in core.py."""
    assert parse_artist_identity_config(raw).enabled is False


def test_parse_artist_identity_config_keeps_defaults_for_empty_lists():
    """An empty delimiter list means "not configured", not "split on nothing" --
    the behaviour core.py's original `or ArtistIdentityConfig().x` expressed."""
    defaults = ArtistIdentityConfig()
    parsed = parse_artist_identity_config({"enabled": True, "split_delimiters": []})
    assert parsed.split_delimiters == defaults.split_delimiters
