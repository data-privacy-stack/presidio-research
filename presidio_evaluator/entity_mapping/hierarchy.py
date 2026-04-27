# ---------------------------------------------------------------------------
# EntityHierarchy
# ---------------------------------------------------------------------------


import copy
import difflib
import re

from presidio_evaluator.entity_mapping.definitions import (
    COUNTRIES,
    HIERARCHY,
    EntityNotMappedError,
)

# ---------------------------------------------------------------------------
# BIO/BIOES/BILOU/BILUO prefix/suffix stripping
# ---------------------------------------------------------------------------

_BIO_PREFIX_RE = re.compile(r"^[BIOELSU]-(.+)$", re.IGNORECASE)
_BIO_SUFFIX_RE = re.compile(r"^(.+)-[BIOELSU]$", re.IGNORECASE)


class EntityHierarchy:
    """
    PII entity taxonomy with canonicalization, branch lookup, and customization.

    Wraps a (deep-copied) taxonomy dict and exposes methods to resolve raw
    labels to canonical names, look up their branch in the tree, and add
    aliases to existing entities.

    Example — create a custom variant::

        h = EntityHierarchy()
        h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
        h.canonicalize("ELECTRONIC_MAIL")   # -> 'EMAIL_ADDRESS'
    """

    # TODO: add :params: to docstring

    def __init__(
        self,
        hierarchy: dict | None = None,
        canonical_depth: int = 3,
    ) -> None:
        self.hierarchy: dict = copy.deepcopy(
            hierarchy if hierarchy is not None else HIERARCHY,
        )
        self.countries: set[str] = COUNTRIES
        self.canonical_depth: int = canonical_depth
        self.country_prefixed_doc_types: dict[str, str] = {}
        self._rebuild()

    @staticmethod
    def normalize(label: str) -> str:
        """Normalize a label: strip BIO prefix/suffix, uppercase, remove underscores and dashes."""
        return (
            EntityHierarchy._strip_bio(label).upper().replace("_", "").replace("-", "")
        )

    def canonicalize(self, raw_label: str, threshold: float = 0.80) -> str:
        """
        Return the canonical entity name for a raw label.

        Resolution order: exact alias map → country-prefix → fuzzy fallback.
        Raises EntityNotMappedError if the label cannot be resolved.
        """
        norm = self.normalize(raw_label)
        if norm in self.raw_to_canonical:
            return self.raw_to_canonical[norm]
        country_match = self._country_prefix_canonical(raw_label, threshold=threshold)
        if country_match:
            return country_match
        fuzzy_match = self._fuzzy_resolve(raw_label, threshold)
        if fuzzy_match:
            return fuzzy_match
        raise EntityNotMappedError(f"Unknown entity label: {raw_label!r}")

    def get_branch(self, raw_label: str) -> list[str]:
        """Return the full ancestor path for a raw (or canonical) entity label.

        Example: get_branch("GERMANY_PASSPORT_NUMBER") -> ['PII', 'GOVERNMENT_ID', 'PASSPORT']
        """
        canonical = self.canonicalize(raw_label)
        branch = self.canonical_to_branch.get(canonical)
        if branch is None:
            raise EntityNotMappedError(
                f"Canonical entity {canonical!r} has no branch in hierarchy",
            )
        return branch

    def get_depth(self, entity: str) -> int:
        """Return the depth of a canonical entity in the hierarchy tree.

        Depth is defined as the length of the ancestor path:
        PII=1, PERSON=2, NAME=3, FIRST_NAME=4.

        :param entity: A canonical entity name.
        :return: depth (int >= 1).
        :raises EntityNotMappedError: if entity is not found.
        """
        branch = self.canonical_to_branch.get(entity)
        if branch is None:
            raise EntityNotMappedError(
                f"Canonical entity {entity!r} has no branch in hierarchy"
            )
        return len(branch)

    def add_alias(self, entity_name: str, alias: str) -> None:
        """Add a raw alias for an existing entity."""
        found = self._find_node(entity_name)
        if found is None:
            raise KeyError(f"Entity {entity_name!r} not found in hierarchy")
        parent_dict, key = found
        value = parent_dict[key]
        if isinstance(value, list):
            if alias not in value:
                value.append(alias)
        else:
            value[alias] = []
        self._rebuild()

    @staticmethod
    def _strip_bio(label: str) -> str:
        """Strip a single BIO/BIOES/BILOU/BILUO prefix or suffix (e.g. B-PERSON → PERSON)."""
        m = _BIO_PREFIX_RE.match(label)
        if m:
            return m.group(1)
        m = _BIO_SUFFIX_RE.match(label)
        if m:
            return m.group(1)
        return label

    @staticmethod
    def _collect_all_raw(value) -> list[str]:
        """Recursively collect all raw alias strings from a hierarchy value (list or nested dict)."""
        items: list[str] = []
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                items.append(k)
                items.extend(EntityHierarchy._collect_all_raw(v))
        return items

    @staticmethod
    def _build_alias_map(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> dict[str, str]:
        """Build a normalized-label → canonical-name lookup dict by walking the hierarchy tree."""
        mapping: dict[str, str] = {}
        for key, value in node.items():
            if depth >= canonical_depth:
                mapping[EntityHierarchy.normalize(key)] = key
                for alias in EntityHierarchy._collect_all_raw(value):
                    mapping[EntityHierarchy.normalize(alias)] = key
            elif isinstance(value, list):
                mapping[EntityHierarchy.normalize(key)] = key
                for alias in value:
                    mapping[EntityHierarchy.normalize(alias)] = key
            elif isinstance(value, dict):
                mapping[EntityHierarchy.normalize(key)] = key
                mapping.update(
                    EntityHierarchy._build_alias_map(value, canonical_depth, depth + 1),
                )
        return mapping

    @staticmethod
    def _collect_canonical_nodes(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> list[str]:
        """Collect the names of all canonical-depth leaf nodes from the hierarchy tree."""
        result: list[str] = []
        for key, value in node.items():
            if depth >= canonical_depth:
                result.append(key)
            elif isinstance(value, list):
                result.append(key)
            elif isinstance(value, dict):
                result.extend(
                    EntityHierarchy._collect_canonical_nodes(
                        value,
                        canonical_depth,
                        depth + 1,
                    ),
                )
        return result

    @staticmethod
    def _build_branch_map(
        node: dict,
        canonical_depth: int = 3,
        current_path: list[str] | None = None,
        depth: int = 1,
    ) -> dict[str, list[str]]:
        """Build a canonical-name → ancestor-path dict by walking the hierarchy tree."""
        if current_path is None:
            current_path = []
        result: dict[str, list[str]] = {}
        for key, value in node.items():
            path = current_path + [key]
            if depth >= canonical_depth:
                result[key] = path
            elif isinstance(value, list):
                result[key] = path
            elif isinstance(value, dict):
                result[key] = path
                result.update(
                    EntityHierarchy._build_branch_map(
                        value,
                        canonical_depth,
                        path,
                        depth + 1,
                    ),
                )
        return result

    def _rebuild(self) -> None:
        """Recompute raw_to_canonical, all_canonical_entities, and canonical_to_branch from the current hierarchy."""
        self.raw_to_canonical: dict[str, str] = self._build_alias_map(
            self.hierarchy,
            self.canonical_depth,
        )
        self.all_canonical_entities: list[str] = self._collect_canonical_nodes(
            self.hierarchy,
            self.canonical_depth,
        )
        self.canonical_to_branch: dict[str, list[str]] = self._build_branch_map(
            self.hierarchy,
            self.canonical_depth,
        )

    def _resolve_remainder(self, remainder: str, threshold: float) -> str:
        """Canonicalize the document-type portion of a country-prefixed label, falling back to NATIONAL_ID."""
        override = self.country_prefixed_doc_types.get(remainder.upper())
        if override:
            return override
        try:
            return self.canonicalize(remainder, threshold=threshold)
        except EntityNotMappedError:
            return "NATIONAL_ID"

    def _country_prefix_canonical(self, raw: str, threshold: float = 1.0) -> str | None:
        """Return the canonical entity for a country-prefixed label using exact country matching, or None."""
        upper = raw.upper()
        parts3 = upper.split("_", 2)
        if len(parts3) == 3 and f"{parts3[0]}_{parts3[1]}" in self.countries:
            return self._resolve_remainder(parts3[2], threshold)
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        country, remainder = parts
        if country not in self.countries:
            return None
        return self._resolve_remainder(remainder, threshold)

    def _fuzzy_country_prefix_canonical(self, raw: str, threshold: float) -> str | None:
        """Return the canonical entity for a country-prefixed label using fuzzy country matching, or None."""
        upper = raw.upper()
        parts3 = upper.split("_", 2)
        if len(parts3) == 3:
            two_token = f"{parts3[0]}_{parts3[1]}"
            two_token_cutoff = max(threshold, 0.90)
            if difflib.get_close_matches(
                two_token,
                self.countries,
                n=1,
                cutoff=two_token_cutoff,
            ):
                return self._resolve_remainder(parts3[2], threshold)
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        prefix, remainder = parts
        if difflib.get_close_matches(prefix, self.countries, n=1, cutoff=threshold):
            return self._resolve_remainder(remainder, threshold)
        return None

    def _fuzzy_resolve(self, raw_label: str, threshold: float) -> str | None:
        """Attempt fuzzy matching against the alias map and fuzzy country prefixes, returning a canonical name or None."""
        country_match = self._fuzzy_country_prefix_canonical(raw_label, threshold)
        if country_match:
            return country_match
        norm = self.normalize(raw_label)
        matches = difflib.get_close_matches(
            norm,
            self.raw_to_canonical,
            n=1,
            cutoff=threshold,
        )
        if matches:
            return self.raw_to_canonical[matches[0]]
        return None

    def _find_node(
        self,
        name: str,
        tree: dict | None = None,
    ) -> tuple[dict, str] | None:
        """Return (parent_dict, key) for the first node matching name in the hierarchy tree, or None."""
        if tree is None:
            tree = self.hierarchy
        for key, value in tree.items():
            if key == name:
                return (tree, key)
            if isinstance(value, dict):
                result = self._find_node(name, value)
                if result:
                    return result
        return None
