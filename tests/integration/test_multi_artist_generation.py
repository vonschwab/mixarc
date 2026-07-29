"""Acceptance tests for the multi-artist blend
(spec docs/superpowers/specs/2026-07-29-multi-artist-blend-design.md).

Artist-mode pier discovery is DB-clustering, which tests/support/gui_fidelity.py's
generate_like_gui explicitly does NOT cover (seeds-mode only). Following the
precedent in test_gui_fidelity_regressions.py and test_genre_mode_generation.py,
these tests drive the real production entry point -- create_playlist_for_artist on
a PlaylistGenerator built from a real Config('config.yaml') -- never a hand-built
overrides dict.

Determinism: PlaylistGenerator is constructed WITHOUT a lastfm_client, so
self.lastfm is falsy and no network call happens. random_seed=0 pins clustering
and tie-breaks.

Library facts these tests rely on (verified 2026-07-29 against data/metadata.db):
  Brian Eno                  73 tracks
  Harold Budd                34 tracks
  Harold Budd and Brian Eno  21 tracks  <- the joint group
  David Bowie               162 tracks  <- no joint credit with Eno
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.test_gui_fidelity_regressions import _select_piers
from tests.support.gui_fidelity import resolved_artifact_path

ART = Path(resolved_artifact_path())
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

ENO = "Brian Eno"
BUDD = "Harold Budd"
BOWIE = "David Bowie"

# The pier IDs Brian Eno produced on master @ b4f9c89, before any multi-artist
# code existed. Any later task that changes this list has broken the
# single-artist path -- that is the whole point of this test.
#
# A legitimate artifact rebuild will also change it. That is an informative
# failure, not a false one: re-derive the list, confirm the change is explained
# by the rebuild and not by multi-artist code, and update the constant.
ENO_BASELINE_PIERS = [
    "b9348304f3408cb2c523af9a4f754607",
    "6fd170a37a8d57c0d4d2e504e18e54f1",
    "f44460624925361b21c6c6ac6039afd4",
    "0c283af35fbe6e8200ad44e682d678a9",
]


@_requires_artifact
def test_single_artist_piers_unchanged():
    """THE REGRESSION GATE. Multi-artist must be completely inert for one artist."""
    assert _select_piers(ENO, []) == ENO_BASELINE_PIERS
