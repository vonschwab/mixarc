"""Genre-mode pier admission: pool-membership mask and topology scoring.

Spec: docs/superpowers/specs/2026-07-25-genre-mode-pier-admission.md

The mask replaces the artist-scoped genre relevance mask for genre mode. That mask
is built by max-pooling the genre profile OF THE MEMBER SET, so a member set
contaminated by album-level tags widens the mask to admit the contaminants' own
neighbourhoods — which then certifies them as bridgeable (spec §1.2).
"""

import numpy as np

from src.playlist.genre_mode import pool_membership_mask


class _Bundle:
    def __init__(self, track_ids):
        self.track_ids = list(track_ids)
        self.track_id_to_index = {str(t): i for i, t in enumerate(track_ids)}


# ── pool membership mask ─────────────────────────────────────────────────────

def test_mask_marks_only_pool_members():
    b = _Bundle(["a", "b", "c", "d"])
    m = pool_membership_mask(b, {"b", "d"})
    assert m.dtype == bool
    assert m.tolist() == [False, True, False, True]


def test_ids_absent_from_the_bundle_are_ignored():
    """A pool id with no artifact row must not raise or shift the mask."""
    b = _Bundle(["a", "b"])
    m = pool_membership_mask(b, {"b", "ghost"})
    assert m.tolist() == [False, True]


def test_empty_pool_returns_all_false():
    b = _Bundle(["a", "b"])
    assert pool_membership_mask(b, set()).tolist() == [False, False]


def test_mask_length_matches_the_bundle_not_the_pool():
    b = _Bundle([str(i) for i in range(10)])
    assert pool_membership_mask(b, {"1", "2"}).shape == (10,)
