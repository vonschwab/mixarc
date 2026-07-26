# tests/unit/test_taxonomy_descendants.py
"""`is_a` descent must be TRANSITIVE.

The taxonomy records only the nearest parent (`melodic death metal is_a death
metal is_a metal`, with no direct metal edge), so a one-hop child query silently
misses grandchildren — which is exactly how umbrella genres ended up hollow
before the 2026-07-25 backfill.
"""
import pytest

from src.ai_genre_enrichment.layered_taxonomy import (
    CanonicalGenre, GenreEdge, LayeredTaxonomy,
)


def _genre(gid, *, status="active", kind="genre"):
    return CanonicalGenre(
        genre_id=gid, name=gid.replace("_", " "), kind=kind,
        specificity_score=0.5, status=status, taxonomy_version="test",
    )


def _isa(child, parent, edge_type="is_a"):
    return GenreEdge(
        source_genre_id=child, target_genre_id=parent, edge_type=edge_type,
        weight=0.75, confidence=0.85, source="test",
    )


@pytest.fixture
def tax():
    return LayeredTaxonomy(
        version="test",
        genres=(
            _genre("metal"), _genre("death_metal"), _genre("melodic_death_metal"),
            _genre("doom_metal"), _genre("shoegaze"),
            _genre("blackwave", status="review"),
            _genre("nu_metal"),
        ),
        aliases=(),
        edges=(
            _isa("death_metal", "metal"),
            _isa("melodic_death_metal", "death_metal"),   # grandchild
            _isa("doom_metal", "metal"),
            _isa("blackwave", "metal"),                   # status: review
            # family_context is NOT is_a: nu metal sits near the family but is not
            # claimed as membership.
            _isa("nu_metal", "metal", edge_type="family_context"),
        ),
        facets=(),
        bridge_rules=(),
    )


def test_walks_is_a_transitively_to_grandchildren(tax):
    assert set(tax.descendant_genre_ids("metal")) == {
        "metal", "death_metal", "melodic_death_metal", "doom_metal",
    }


def test_includes_self(tax):
    assert tax.descendant_genre_ids("shoegaze") == ("shoegaze",)


def test_leaf_genre_expansion_is_inert(tax):
    # The property that makes this change safe to ship: a genre with no children
    # gains nothing, so genres that already generated well cannot regress.
    assert tax.descendant_genre_ids("melodic_death_metal") == ("melodic_death_metal",)


def test_family_context_is_not_membership(tax):
    assert "nu_metal" not in tax.descendant_genre_ids("metal")


def test_non_active_descendants_excluded_by_default(tax):
    assert "blackwave" not in tax.descendant_genre_ids("metal")
    assert "blackwave" in tax.descendant_genre_ids("metal", active_only=False)


def test_queried_genre_is_returned_whatever_its_status(tax):
    # The caller asked for it explicitly; only DESCENDANTS are status-filtered.
    assert tax.descendant_genre_ids("blackwave") == ("blackwave",)


def test_unknown_genre_returns_itself_only(tax):
    assert tax.descendant_genre_ids("xyzzy_not_a_genre") == ("xyzzy_not_a_genre",)


def test_cycle_does_not_hang():
    cyclic = LayeredTaxonomy(
        version="test",
        genres=(_genre("a"), _genre("b")),
        edges=(_isa("a", "b"), _isa("b", "a")),
        aliases=(), facets=(), bridge_rules=(),
    )
    assert set(cyclic.descendant_genre_ids("a")) == {"a", "b"}


def test_live_taxonomy_soul_reaches_classic_soul():
    """Guards the actual motivating case against the shipped YAML: soul's exact
    core was 214 tracks from 13 artists while the classic-soul catalogue sat one
    unrecorded edge away."""
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    descendants = set(load_default_layered_taxonomy().descendant_genre_ids("soul"))
    assert {"classic_soul", "deep_soul", "gospel_soul"} <= descendants
    assert "soul_jazz" not in descendants        # head noun last: that is jazz
