"""The seed-artist interior block is a SET (multi-artist blend), but its default
derivation is unchanged: exactly the first seed's artist key.

Byte-identity guard. Tag-steering anchors inject other artists into seed_indices;
deriving the set from ALL seeds would newly exclude those artists from bridge
interiors and silently change existing single-artist output.
"""
from __future__ import annotations

import inspect


def test_beam_accepts_a_set_of_seed_artist_keys():
    from src.playlist.pier_bridge import beam

    # NOTE: the brief's guessed name `build_bridge_segment_beam` is not the
    # actual public name at beam.py:193 -- the real function is
    # `_beam_search_segment` (verified by reading the module).
    sig = inspect.signature(beam._beam_search_segment)
    assert "seed_artist_keys" in sig.parameters, \
        "beam must take the widened seed_artist_keys parameter"
    assert "seed_artist_key" not in sig.parameters, \
        "the old singular parameter must be gone, not shadowed"


def test_micro_pier_accepts_a_set_of_seed_artist_keys():
    from src.playlist.pier_bridge import micro_pier

    src = inspect.getsource(micro_pier)
    assert "seed_artist_keys" in src
    assert "seed_artist_key:" not in src, "no singular parameter should remain"


def test_builder_derives_exactly_one_key_by_default():
    """Cheap tripwire: the default derivation still reads seed_indices[0] only.

    Backed by the behavioral test below -- this one just fails fast on a sloppy
    edit. It is deliberately a source-text check because it guards a property of
    the derivation SITE, not of a return value.
    """
    import src.playlist.pier_bridge_builder as pbb

    src = inspect.getsource(pbb)
    assert "seed_indices[0]" in src, \
        "the default derivation must still read only the FIRST seed"
    assert "seed_artist_keys" in src


def test_only_the_first_seeds_artist_is_blocked_by_default():
    """THE REAL GATE for the derivation.

    Given seeds from two different artists (which is exactly what tag-steering
    anchor injection produces), the default derivation must block ONLY the first
    seed's artist. Deriving from all seeds would newly exclude anchor artists from
    bridge interiors and silently change existing single-artist output.
    """
    import numpy as np

    from src.playlist.pier_bridge_builder import _derive_seed_artist_keys

    class _B:
        track_ids = ["t0", "t1"]
        track_artists = ["Brian Eno", "David Bowie"]
        artist_keys = ["brian eno", "david bowie"]
        durations_ms = np.array([240000.0, 240000.0])

    keys = _derive_seed_artist_keys(_B(), [0, 1])
    assert keys == frozenset({"brian eno"}), (
        f"expected only the first seed's artist, got {keys} — deriving from all "
        "seeds changes tag-anchor runs"
    )

    # An explicit override (the multi-artist path) replaces the derivation wholesale.
    override = _derive_seed_artist_keys(_B(), [0, 1], override=["brian eno", "harold budd"])
    assert override == frozenset({"brian eno", "harold budd"})
