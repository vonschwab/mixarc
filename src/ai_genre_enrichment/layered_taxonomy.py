from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import yaml


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "layered_genre_taxonomy.yaml"
FAMILY_KIND = "family"


def normalize_taxonomy_name(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True)
class CanonicalGenre:
    genre_id: str
    name: str
    kind: str
    specificity_score: float
    status: str
    taxonomy_version: str
    role: str = "leaf"


@dataclass(frozen=True)
class GenreAlias:
    alias: str
    canonical_genre_id: str
    source: str
    confidence: float
    alias_policy: dict[str, Any] | None = None
    target_kind: str = "genre"


@dataclass(frozen=True)
class GenreEdge:
    source_genre_id: str
    target_genre_id: str
    edge_type: str
    weight: float
    confidence: float
    source: str
    notes: str | None = None


@dataclass(frozen=True)
class CanonicalFacet:
    facet_id: str
    name: str
    facet_type: str
    status: str


@dataclass(frozen=True)
class BridgeRule:
    source_genre_id: str
    target_genre_id: str
    required_family_min: float
    required_facet_overlap: float
    required_sonic_similarity: float
    required_transition_quality: float
    mode_allowed: tuple[str, ...]
    notes: str | None = None
    required_facets_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class RejectedTerm:
    term: str
    reason: str
    notes: str | None = None


@dataclass(frozen=True)
class LayeredTaxonomy:
    version: str
    genres: tuple[CanonicalGenre, ...]
    aliases: tuple[GenreAlias, ...]
    edges: tuple[GenreEdge, ...]
    facets: tuple[CanonicalFacet, ...]
    bridge_rules: tuple[BridgeRule, ...]
    rejected_terms: tuple[RejectedTerm, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_genres_by_id", {genre.genre_id: genre for genre in self.genres})
        object.__setattr__(self, "_genres_by_name", {
            normalize_taxonomy_name(genre.name): genre
            for genre in self.genres
        })
        object.__setattr__(self, "_alias_targets_by_name", {
            normalize_taxonomy_name(alias.alias): alias.canonical_genre_id
            for alias in self.aliases
            if _is_plain_alias(alias)
        })
        object.__setattr__(self, "_alias_targets_by_exact_name", {
            alias.alias.casefold().strip(): alias
            for alias in self.aliases
        })
        object.__setattr__(self, "_facets_by_name", {
            normalize_taxonomy_name(facet.name): facet
            for facet in self.facets
        })
        object.__setattr__(self, "_facets_by_id", {facet.facet_id: facet for facet in self.facets})
        object.__setattr__(self, "_rejected_terms_by_name", {
            normalize_taxonomy_name(term.term): term
            for term in self.rejected_terms
        })
        object.__setattr__(self, "_edges_by_source", _group_edges(self.edges))
        object.__setattr__(self, "_parent_edges_by_source", _group_parent_edges(self.edges))
        object.__setattr__(self, "_is_a_children_by_parent", _group_is_a_children(self.edges))
        object.__setattr__(self, "_bridge_rules_by_pair", {
            (rule.source_genre_id, rule.target_genre_id): rule
            for rule in self.bridge_rules
        })

    def genre_by_id(self, genre_id: str) -> CanonicalGenre | None:
        return self._genres_by_id.get(genre_id)

    def genre_by_name(self, name: str, context_terms: list[str] | tuple[str, ...] | None = None) -> CanonicalGenre | None:
        normalized = normalize_taxonomy_name(name)
        genre = self._genres_by_name.get(normalized)
        if genre is not None:
            return genre
        alias = self._alias_for_name(name, context_terms=context_terms)
        if alias is None or alias.target_kind != "genre":
            return None
        return self.genre_by_id(alias.canonical_genre_id)

    def facet_by_name(self, name: str) -> CanonicalFacet | None:
        return self._facets_by_name.get(normalize_taxonomy_name(name))

    def alias_target_for_name(
        self,
        name: str,
        context_terms: list[str] | tuple[str, ...] | None = None,
    ) -> CanonicalGenre | CanonicalFacet | None:
        alias = self._alias_for_name(name, context_terms=context_terms)
        if alias is None:
            return None
        if alias.target_kind == "facet":
            return self.facet_by_id(alias.canonical_genre_id)
        return self.genre_by_id(alias.canonical_genre_id)

    def exact_alias_target_for_name(
        self,
        name: str,
        context_terms: list[str] | tuple[str, ...] | None = None,
    ) -> CanonicalGenre | CanonicalFacet | None:
        return self.alias_target_for_name(name, context_terms=context_terms)

    def facet_by_id(self, facet_id: str) -> CanonicalFacet | None:
        return self._facets_by_id.get(facet_id)

    def rejected_term_by_name(self, name: str) -> RejectedTerm | None:
        return self._rejected_terms_by_name.get(normalize_taxonomy_name(name))

    def describe_recorded_name(self, name: str) -> str | None:
        """Human description of an existing record with this name, or None if the
        name is genuinely unrecorded.

        Context-free by design: a *conditional* alias counts as recorded even
        though it does not resolve without context. The taxonomy editor uses this
        so it neither re-offers an already-recorded term for adjudication nor
        appends a duplicate that would shadow it. Covers genres, facets, aliases
        (plain AND conditional), and rejects — the registries the context-sensitive
        `genre_by_name`/`_alias_for_name` lookups can miss.
        """
        norm = normalize_taxonomy_name(name)
        genre = self._genres_by_name.get(norm)
        if genre is not None:
            return f"{genre.kind} '{genre.name}'"
        facet = self._facets_by_name.get(norm)
        if facet is not None:
            return f"facet '{facet.name}'"
        alias = self._alias_targets_by_exact_name.get(str(name or "").casefold().strip())
        if alias is None:
            alias = next((a for a in self.aliases
                          if normalize_taxonomy_name(a.alias) == norm), None)
        if alias is not None:
            target = (self._genres_by_id.get(alias.canonical_genre_id)
                      or self._facets_by_id.get(alias.canonical_genre_id))
            target_name = target.name if target is not None else alias.canonical_genre_id
            kind = "alias" if _is_plain_alias(alias) else "conditional alias"
            return f"{kind} → {target_name}"
        rejected = self._rejected_terms_by_name.get(norm)
        if rejected is not None:
            return f"reject ({rejected.reason})"
        return None

    def name_is_recorded(self, name: str) -> bool:
        """True if any record (genre/facet/alias — plain or conditional — /reject)
        already uses this name. Context-free; see `describe_recorded_name`."""
        return self.describe_recorded_name(name) is not None

    def alias_for_name(
        self,
        name: str,
        context_terms: list[str] | tuple[str, ...] | None = None,
    ) -> GenreAlias | None:
        return self._alias_for_name(name, context_terms=context_terms)

    def _alias_for_name(
        self,
        name: str,
        context_terms: list[str] | tuple[str, ...] | None = None,
    ) -> GenreAlias | None:
        alias = self._alias_targets_by_exact_name.get(str(name or "").casefold().strip())
        if alias is None:
            return None
        if _is_plain_alias(alias):
            return alias
        context = {normalize_taxonomy_name(term) for term in context_terms or ()}
        policy = alias.alias_policy or {}
        required_any = {normalize_taxonomy_name(term) for term in policy.get("requires_any_context") or ()}
        required_all = {normalize_taxonomy_name(term) for term in policy.get("requires_all_context") or ()}
        if required_all and not required_all <= context:
            return None
        if required_any and not (required_any & context):
            return None
        return alias

    def parents_for_genre(self, genre_id: str) -> tuple[CanonicalGenre, ...]:
        return tuple(
            genre
            for genre in (
                self.genre_by_id(edge.target_genre_id)
                for edge in self._parent_edges_by_source.get(genre_id, ())
            )
            if genre is not None and genre.kind != FAMILY_KIND
        )

    def descendant_genre_ids(self, genre_id: str, *, active_only: bool = True) -> tuple[str, ...]:
        """``genre_id`` plus every genre that ``is_a``-descends from it, TRANSITIVELY.

        The taxonomy records only the NEAREST parent — `melodic death metal is_a
        death metal is_a metal` carries no direct metal edge — so a one-hop child
        query silently misses grandchildren. Any consumer asking "what is in this
        genre" must walk the chain; that is what this does.

        ``is_a`` only. `family_context` is the weaker "sits near this family" edge
        (1.6k of them vs ~340 is_a) and is deliberately NOT traversed: it would drag
        adjacent scenes into a genre's own membership.

        ``genre_id`` itself is always included, whatever its status — the caller
        asked for it. Descendants default to ACTIVE only, so a genre still under
        review does not silently enlarge a published genre's membership. Cycle-safe.
        """
        seen = {genre_id}
        stack = [genre_id]
        while stack:
            for child in self._is_a_children_by_parent.get(stack.pop(), ()):
                if child in seen:
                    continue
                record = self._genres_by_id.get(child)
                if active_only and record is not None and record.status != "active":
                    continue
                seen.add(child)
                stack.append(child)
        return tuple(sorted(seen))

    def edge_for_genre(self, source_genre_id: str, target_genre_id: str, edge_type: str | None = None) -> GenreEdge | None:
        for edge in self._edges_by_source.get(source_genre_id, ()):
            if edge.target_genre_id == target_genre_id and (edge_type is None or edge.edge_type == edge_type):
                return edge
        return None

    def families_for_genre(self, genre_id: str) -> tuple[CanonicalGenre, ...]:
        families: dict[str, CanonicalGenre] = {}
        visited: set[str] = set()
        stack = [genre_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for edge in self._parent_edges_by_source.get(current, ()):
                parent = self.genre_by_id(edge.target_genre_id)
                if parent is None:
                    continue
                if parent.kind == FAMILY_KIND or parent.role == "family":
                    families[parent.genre_id] = parent
                else:
                    stack.append(parent.genre_id)
        return tuple(sorted(families.values(), key=lambda genre: genre.name))

    def bridge_rule_for(self, source_genre_id: str, target_genre_id: str) -> BridgeRule | None:
        direct = self._bridge_rules_by_pair.get((source_genre_id, target_genre_id))
        if direct is not None:
            return direct
        reverse = self._bridge_rules_by_pair.get((target_genre_id, source_genre_id))
        if reverse is None:
            return None
        return BridgeRule(
            source_genre_id=source_genre_id,
            target_genre_id=target_genre_id,
            required_family_min=reverse.required_family_min,
            required_facet_overlap=reverse.required_facet_overlap,
            required_sonic_similarity=reverse.required_sonic_similarity,
            required_transition_quality=reverse.required_transition_quality,
            mode_allowed=reverse.mode_allowed,
            notes=reverse.notes,
            required_facets_any=reverse.required_facets_any,
        )


def load_default_layered_taxonomy() -> LayeredTaxonomy:
    return load_layered_taxonomy(DEFAULT_TAXONOMY_PATH)


# path (resolved) -> (sha256 of file text, parsed taxonomy). No lock: dict get/set
# are GIL-atomic, so a concurrent double-parse on a cache miss is harmless (last
# write wins, both results are valid for identical input text).
_taxonomy_cache: dict[str, tuple[str, "LayeredTaxonomy"]] = {}


def load_layered_taxonomy(path: str | Path) -> LayeredTaxonomy:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    cache_key = str(p.resolve())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cached = _taxonomy_cache.get(cache_key)
    if cached is not None and cached[0] == digest:
        return cached[1]

    taxonomy = _parse_layered_taxonomy(text)
    _taxonomy_cache[cache_key] = (digest, taxonomy)
    return taxonomy


def _parse_layered_taxonomy(text: str) -> LayeredTaxonomy:
    data = yaml.safe_load(text) or {}
    version = str(data.get("taxonomy_version") or data.get("version") or "").strip()
    if not version:
        raise ValueError("Layered taxonomy is missing version")

    if "records" in data:
        taxonomy = _structured_taxonomy_from_data(data, version)
        _validate_taxonomy(taxonomy)
        return taxonomy

    genres = tuple(_genre_from_row(row, version) for row in data.get("genres", []))
    facets = tuple(_facet_from_row(row) for row in data.get("facets", []))
    aliases = tuple(_alias_from_row(row) for row in data.get("aliases", []))
    edges = tuple(_edge_from_row(row) for row in data.get("edges", []))
    bridge_rules = tuple(_bridge_rule_from_row(row) for row in data.get("bridge_rules", []))
    rejected_terms = tuple(_rejected_term_from_row(row) for row in data.get("rejected_terms", []))

    taxonomy = LayeredTaxonomy(
        version=version,
        genres=genres,
        aliases=aliases,
        edges=edges,
        facets=facets,
        bridge_rules=bridge_rules,
        rejected_terms=rejected_terms,
    )
    _validate_taxonomy(taxonomy)
    return taxonomy


def _genre_from_row(row: dict[str, Any], version: str) -> CanonicalGenre:
    return CanonicalGenre(
        genre_id=str(row["genre_id"]),
        name=str(row["name"]),
        kind=str(row["kind"]),
        specificity_score=float(row["specificity_score"]),
        status=str(row["status"]),
        taxonomy_version=version,
        role=str(row.get("role") or ("family" if str(row["kind"]) == FAMILY_KIND else "leaf")),
    )


def _alias_from_row(row: dict[str, Any]) -> GenreAlias:
    return GenreAlias(
        alias=str(row["alias"]),
        canonical_genre_id=str(row["canonical_genre_id"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
    )


def _edge_from_row(row: dict[str, Any]) -> GenreEdge:
    return GenreEdge(
        source_genre_id=str(row["source_genre_id"]),
        target_genre_id=str(row["target_genre_id"]),
        edge_type=str(row["edge_type"]),
        weight=float(row["weight"]),
        confidence=float(row["confidence"]),
        source=str(row["source"]),
        notes=row.get("notes"),
    )


def _facet_from_row(row: dict[str, Any]) -> CanonicalFacet:
    return CanonicalFacet(
        facet_id=str(row["facet_id"]),
        name=str(row["name"]),
        facet_type=str(row["facet_type"]),
        status=str(row["status"]),
    )


def _bridge_rule_from_row(row: dict[str, Any]) -> BridgeRule:
    mode_allowed = row["mode_allowed"]
    if isinstance(mode_allowed, str):
        modes = tuple(part.strip() for part in mode_allowed.split(",") if part.strip())
    else:
        modes = tuple(str(part) for part in mode_allowed)
    return BridgeRule(
        source_genre_id=str(row["source_genre_id"]),
        target_genre_id=str(row["target_genre_id"]),
        required_family_min=float(row["required_family_min"]),
        required_facet_overlap=float(row["required_facet_overlap"]),
        required_sonic_similarity=float(row["required_sonic_similarity"]),
        required_transition_quality=float(row["required_transition_quality"]),
        mode_allowed=modes,
        notes=row.get("notes"),
        required_facets_any=tuple(str(item) for item in row.get("required_facets_any", ())),
    )


def _rejected_term_from_row(row: dict[str, Any]) -> RejectedTerm:
    return RejectedTerm(
        term=str(row["term"]),
        reason=str(row["reason"]),
        notes=row.get("notes"),
    )


def _structured_taxonomy_from_data(data: dict[str, Any], version: str) -> LayeredTaxonomy:
    records = data.get("records") or []
    if not isinstance(records, list):
        raise ValueError("Layered taxonomy records must be a list")

    enums = data.get("enums") or {}
    allowed_reject_reasons = set(enums.get("reject_reason") or ())
    allowed_facet_types = set(enums.get("facet_type") or ())

    genres: list[CanonicalGenre] = []
    facets: list[CanonicalFacet] = []
    aliases: list[GenreAlias] = []
    edges: list[GenreEdge] = []
    rejected_terms: list[RejectedTerm] = []

    genre_by_name: dict[str, CanonicalGenre] = {}
    facet_by_name: dict[str, CanonicalFacet] = {}
    alias_records: list[dict[str, Any]] = []
    source_edges: list[tuple[dict[str, Any], dict[str, Any]]] = []

    seen_active_names: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Layered taxonomy record must be an object")
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError("Layered taxonomy record is missing name")
        kind = str(row.get("kind") or "").strip()
        role = str(row.get("role") or "").strip()
        status = str(row.get("status") or "").strip()
        normalized = normalize_taxonomy_name(name)

        if kind == "alias":
            alias_records.append(row)
            continue
        if kind == "reject" or status == "rejected" or role == "reject":
            reason = str(row.get("reject_reason") or "").strip()
            if not reason:
                raise ValueError(f"Rejected record is missing reject_reason: {name}")
            if allowed_reject_reasons and reason not in allowed_reject_reasons:
                raise ValueError(f"Unsupported reject_reason for {name}: {reason}")
            rejected_terms.append(RejectedTerm(term=name, reason=reason, notes=row.get("notes")))
            continue
        if kind == "facet":
            facet_type = str(row.get("facet_type") or "").strip()
            if not facet_type:
                raise ValueError(f"Facet record is missing facet_type: {name}")
            if allowed_facet_types and facet_type not in allowed_facet_types:
                raise ValueError(f"Unsupported facet_type for {name}: {facet_type}")
            facet = CanonicalFacet(
                facet_id=_record_id(name),
                name=name,
                facet_type=facet_type,
                status=status or "active",
            )
            facets.append(facet)
            facet_by_name[normalized] = facet
            continue

        if normalized in seen_active_names:
            raise ValueError(f"Duplicate canonical taxonomy name after normalization: {name}")
        seen_active_names.add(normalized)
        genre = CanonicalGenre(
            genre_id=_record_id(name),
            name=name,
            kind=kind,
            specificity_score=_float_or_default(row.get("specificity_score"), 0.0),
            status=status or "active",
            taxonomy_version=version,
            role=role or ("family" if kind == FAMILY_KIND else "leaf"),
        )
        genres.append(genre)
        genre_by_name[normalized] = genre
        source_edges.append((row, {"source_id": genre.genre_id, "source_name": name}))

    for row in alias_records:
        name = str(row.get("name") or "").strip()
        target_name = str(row.get("canonical_target") or "").strip()
        if not target_name:
            raise ValueError(f"Alias is missing canonical_target: {name}")
        target_genre = genre_by_name.get(normalize_taxonomy_name(target_name))
        target_facet = facet_by_name.get(normalize_taxonomy_name(target_name))
        if target_genre is None and target_facet is None:
            raise ValueError(f"Alias points to unknown genre: {name}")
        alias_policy = row.get("alias_policy")
        aliases.append(
            GenreAlias(
                alias=name,
                canonical_genre_id=(target_genre.genre_id if target_genre is not None else target_facet.facet_id),
                source=str(row.get("source") or "reviewed_taxonomy"),
                confidence=_float_or_default(row.get("confidence"), 1.0),
                alias_policy=alias_policy if isinstance(alias_policy, dict) else None,
                target_kind="genre" if target_genre is not None else "facet",
            )
        )

    for row, source in source_edges:
        for edge_row in row.get("parent_edges") or ():
            if not isinstance(edge_row, dict):
                raise ValueError(f"Parent edge must be an object: {source['source_name']}")
            target_name = str(edge_row.get("target") or "").strip()
            target = genre_by_name.get(normalize_taxonomy_name(target_name))
            target_facet = facet_by_name.get(normalize_taxonomy_name(target_name))
            if target is None:
                if target_facet is None:
                    raise ValueError(f"Parent edge points to unknown genre: {source['source_name']}->{target_name}")
                continue
            edges.append(
                GenreEdge(
                    source_genre_id=str(source["source_id"]),
                    target_genre_id=target.genre_id,
                    edge_type=str(edge_row.get("edge_type") or "is_a"),
                    weight=_float_or_default(edge_row.get("weight"), 1.0),
                    confidence=_float_or_default(edge_row.get("confidence"), 1.0),
                    source=str(edge_row.get("source") or "reviewed_taxonomy"),
                    notes=edge_row.get("notes"),
                )
            )

    bridge_rules = tuple(_structured_bridge_rules_from_data(data, genre_by_name, facet_by_name))
    return LayeredTaxonomy(
        version=version,
        genres=tuple(genres),
        aliases=tuple(aliases),
        edges=tuple(edges),
        facets=tuple(facets),
        bridge_rules=bridge_rules,
        rejected_terms=tuple(rejected_terms),
    )


def _structured_bridge_rules_from_data(
    data: dict[str, Any],
    genre_by_name: dict[str, CanonicalGenre],
    facet_by_name: dict[str, CanonicalFacet],
) -> list[BridgeRule]:
    bridge_rules: list[BridgeRule] = []
    for row in data.get("bridge_rules") or ():
        if not isinstance(row, dict):
            raise ValueError("Bridge rule must be an object")
        source_name = str(row.get("source") or "").strip()
        target_name = str(row.get("target") or "").strip()
        source = genre_by_name.get(normalize_taxonomy_name(source_name))
        target = genre_by_name.get(normalize_taxonomy_name(target_name))
        if source is None or target is None:
            raise ValueError(f"Bridge rule points to unknown genre: {source_name}->{target_name}")
        required_facets_any = []
        for facet_name in row.get("required_facets_any") or ():
            facet = facet_by_name.get(normalize_taxonomy_name(str(facet_name)))
            if facet is None:
                raise ValueError(f"Bridge rule points to unknown facet: {source_name}->{target_name}:{facet_name}")
            required_facets_any.append(facet.facet_id)
        modes = row.get("mode_allowed") or ()
        if isinstance(modes, str):
            mode_allowed = tuple(part.strip() for part in modes.split(",") if part.strip())
        else:
            mode_allowed = tuple(str(part) for part in modes)
        bridge_rules.append(
            BridgeRule(
                source_genre_id=source.genre_id,
                target_genre_id=target.genre_id,
                required_family_min=_float_or_default(row.get("required_family_min"), 0.0),
                required_facet_overlap=_float_or_default(row.get("required_facet_overlap"), 0.0),
                required_sonic_similarity=_threshold_value(row.get("required_sonic_similarity")),
                required_transition_quality=_threshold_value(row.get("required_transition_quality")),
                mode_allowed=mode_allowed,
                notes=row.get("notes"),
                required_facets_any=tuple(required_facets_any),
            )
        )
    return bridge_rules


def _record_id(name: str) -> str:
    normalized = normalize_taxonomy_name(name)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _threshold_value(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("dynamic", "narrow", "strict", "discover"):
            if key in value:
                return _float_or_default(value[key], 0.0)
        if value:
            return _float_or_default(next(iter(value.values())), 0.0)
        return 0.0
    return _float_or_default(value, 0.0)


def _is_plain_alias(alias: GenreAlias) -> bool:
    policy = alias.alias_policy or {}
    alias_type = str(policy.get("type") or policy.get("alias_type") or "").casefold()
    if alias_type == "conditional":
        return False
    return not policy.get("requires_any_context") and not policy.get("requires_all_context")


def _group_edges(edges: tuple[GenreEdge, ...]) -> dict[str, tuple[GenreEdge, ...]]:
    grouped: dict[str, list[GenreEdge]] = {}
    for edge in edges:
        grouped.setdefault(edge.source_genre_id, []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_parent_edges(edges: tuple[GenreEdge, ...]) -> dict[str, tuple[GenreEdge, ...]]:
    grouped: dict[str, list[GenreEdge]] = {}
    parent_edge_types = {"is_a", "family_context"}
    for edge in edges:
        if edge.edge_type in parent_edge_types:
            grouped.setdefault(edge.source_genre_id, []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_is_a_children(edges: tuple[GenreEdge, ...]) -> dict[str, tuple[str, ...]]:
    """parent genre_id -> its direct ``is_a`` children. The reverse of the edge
    direction stored in the YAML, which records child -> parent."""
    grouped: dict[str, list[str]] = {}
    for edge in edges:
        if edge.edge_type == "is_a":
            grouped.setdefault(edge.target_genre_id, []).append(edge.source_genre_id)
    return {key: tuple(value) for key, value in grouped.items()}


def _validate_taxonomy(taxonomy: LayeredTaxonomy) -> None:
    genre_ids = {genre.genre_id for genre in taxonomy.genres}
    facet_ids = {facet.facet_id for facet in taxonomy.facets}
    if len(genre_ids) != len(taxonomy.genres):
        raise ValueError("Layered taxonomy has duplicate genre ids")
    if len(facet_ids) != len(taxonomy.facets):
        raise ValueError("Layered taxonomy has duplicate facet ids")

    for alias in taxonomy.aliases:
        valid_ids = facet_ids if alias.target_kind == "facet" else genre_ids
        if alias.canonical_genre_id not in valid_ids:
            raise ValueError(f"Alias points to unknown genre: {alias.alias}")
        if not _is_plain_alias(alias):
            policy = alias.alias_policy or {}
            if not policy.get("requires_any_context") and not policy.get("requires_all_context"):
                raise ValueError(f"Conditional alias lacks context requirements: {alias.alias}")
    for edge in taxonomy.edges:
        if edge.source_genre_id not in genre_ids or edge.target_genre_id not in genre_ids:
            raise ValueError(f"Edge points to unknown genre: {edge.source_genre_id}->{edge.target_genre_id}")
    for rule in taxonomy.bridge_rules:
        if rule.source_genre_id not in genre_ids or rule.target_genre_id not in genre_ids:
            raise ValueError(f"Bridge rule points to unknown genre: {rule.source_genre_id}->{rule.target_genre_id}")
        for facet_id in rule.required_facets_any:
            if facet_id not in facet_ids:
                raise ValueError(f"Bridge rule points to unknown facet: {facet_id}")
