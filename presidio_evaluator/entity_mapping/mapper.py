"""
Entity mapping utilities for aligning entity types across different models and datasets.
"""

from typing import Callable, Dict, List, Optional, Protocol, Any, Tuple
import logging
import re

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


class EntityMapper(Protocol):
    """
    Protocol for entity type mappers.

    All entity mappers must implement a map() method that takes a source entity
    and list of target entities, returning the best match or None.
    """

    def map(self, source_entity: str, target_entities: List[str]) -> Optional[str]:
        """
        Map a source entity to the most similar target entity.

        :param source_entity: Entity type to map from
        :param target_entities: List of candidate entity types to map to
        :return: The best matching target entity or None if no match found
        """
        ...


class DictEntityMapper:
    """
    Simple dictionary-based entity mapper.

    Provides a consistent interface for deterministic dict-based mappings.

    Examples:
        >>> mapper = DictEntityMapper({'FIRST_NAME': 'PERSON', 'SSN': 'ID'})
        >>> mapper.map('FIRST_NAME', ['PERSON', 'LOCATION'])
        'PERSON'
        >>> mapper.map('SSN', ['ID', 'PERSON'])
        'ID'
        >>> mapper.map('UNKNOWN', ['PERSON'])
        None
    """

    def __init__(self, mappings: Dict[str, str]):
        """
        Initialize with entity mappings.

        :param mappings: Dictionary mapping source entities to target entities
        """
        self.mappings = mappings

    def map(self, source_entity: str, target_entities: List[str]) -> Optional[str]:
        """
        Map source entity using dictionary lookup.

        :param source_entity: Entity type to map from
        :param target_entities: List of candidate entity types to map to

        :returns: Mapped entity if found in mappings AND present in target_entities, else None
        """
        mapped = self.mappings.get(source_entity)
        if mapped and mapped in target_entities:
            return mapped
        return None


class SemanticEntityMapper:
    """
    Map entity types using semantic similarity via embeddings.

    Uses sentence transformers to compute semantic similarity between entity type names,
    enabling automatic mapping even when entity names differ slightly (typos, abbreviations,
    different languages, etc.).

    Examples:
        >>> mapper = SemanticEntityMapper(threshold=0.85)
        >>> mapper.map('FIRST_NAME', ['PERSON', 'LOCATION', 'ORG'])
        'PERSON'
        >>> mapper.map('SSN', ['PERSON', 'ID', 'ORG'])
        'ID'
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        threshold: float = 0.35,
        cache_embeddings: bool = True,
        full_name_weight: float = 0.6,
    ):
        """
        Initialize semantic entity mapper.

        :param model_name: Name of the sentence transformer model to use.
                       Default is all-mpnet-base-v2 for best quality entity mapping.

                       Alternative models to try:
                       - "sentence-transformers/all-MiniLM-L6-v2" (faster, lighter weight)
                       - "sentence-transformers/paraphrase-MiniLM-L6-v2" (paraphrase-focused)
                       - "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (QA-focused)

                       Note: For short entity names (single words), word embeddings like
                       Word2Vec or GloVe might work better than sentence transformers,
                       as they capture direct word relationships rather than sentence-level
                       semantics.
        :param threshold: Minimum similarity score (0-1) for a match. Higher values are stricter.
        :param cache_embeddings: Whether to cache computed embeddings for reuse.
        :param full_name_weight: Weight for full entity name similarity (0-1).
                                (e.g. "EMAIL_ADDRESS" instead of "EMAIL" and "ADDRESS")
                                The remaining weight (1 - full_name_weight) is used for
                                word-level similarity. Higher values emphasize the full name.
                                Default 0.6 means 60% full name, 40% word-level.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.cache_embeddings = cache_embeddings
        self.full_name_weight = full_name_weight
        self._model = None
        self._embedding_cache: Dict[str, Any] = {}

    def _load_model(self):
        """Lazy load the sentence transformer model."""

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SemanticEntityMapper. "
                "Install it with: pip install sentence-transformers"
            )

        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic entity mapping. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text, using cache if enabled."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        model = self._load_model()
        embedding = model.encode(text, convert_to_tensor=False)

        if self.cache_embeddings:
            self._embedding_cache[text] = embedding

        return embedding

    def _compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text strings using their embeddings.

        :param text1: First text (entity name or word)
        :param text2: Second text (entity name or word)
        :returns: Cosine similarity score between 0 and 1
        """
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)

        return float(
            np.dot(emb1_np, emb2_np)
            / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
        )

    def compute_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute semantic similarity between two entity types.

        Uses a hybrid approach combining:
        1. Full entity name similarity (e.g., "EMAIL_ADDRESS" as a whole)
        2. Word-level pairwise similarity (e.g., comparing individual words)

        :param entity1: First entity type name
        :param entity2: Second entity type name

        :returns: Similarity score between 0 and 1 (weighted combination of full name
            and word-level similarities)
        """
        if self.full_name_weight == 1.0:
            full_name_similarity = self._compute_cosine_similarity(entity1, entity2)

            logger.info(
                f"Full name only (weight=1.0): '{entity1}' <-> '{entity2}' = {full_name_similarity:.4f}"
            )
            return full_name_similarity

        # Optimization: If full_name_weight is 0.0, only compute word-level similarity
        if self.full_name_weight == 0.0:
            words1 = self._split_entity_name(entity1)
            words2 = self._split_entity_name(entity2)

            logger.info(
                f"Word-level only (weight=0.0): '{entity1}' -> {words1}, '{entity2}' -> {words2}"
            )

            # Compute pairwise similarities between all word pairs
            word_similarities = []
            for word1 in words1:
                for word2 in words2:
                    sim = self._compute_cosine_similarity(word1, word2)
                    word_similarities.append(sim)
                    logger.info(f"  Word pair: '{word1}' <-> '{word2}' = {sim:.4f}")

            word_level_similarity = (
                float(np.max(word_similarities)) if word_similarities else 0.0
            )
            logger.info(f"Word-level average similarity: {word_level_similarity:.4f}")
            return word_level_similarity

        # Hybrid approach: compute both and combine with weights
        # 1. Compute full entity name similarity
        full_name_similarity = self._compute_cosine_similarity(entity1, entity2)

        logger.info(
            f"Full name comparison: '{entity1}' <-> '{entity2}' = {full_name_similarity:.4f}"
        )

        # 2. Check if word-level splitting is needed
        words1 = self._split_entity_name(entity1)
        words2 = self._split_entity_name(entity2)

        # If both entities are single words (no splits), skip word-level calculation
        if len(words1) == 1 and len(words2) == 1:
            logger.info(
                f"No splits needed: '{entity1}' and '{entity2}' are single words"
            )
            return float(full_name_similarity)

        logger.info(
            f"Word-level split: '{entity1}' -> {words1}, '{entity2}' -> {words2}"
        )

        # Get embeddings for all words and compute pairwise similarities
        word_similarities = []
        for word1 in words1:
            for word2 in words2:
                sim = self._compute_cosine_similarity(word1, word2)
                word_similarities.append(sim)
                logger.info(f"  Word pair: '{word1}' <-> '{word2}' = {sim:.4f}")

        word_level_similarity = (
            float(np.max(word_similarities)) if word_similarities else 0.0
        )

        logger.info(f"Word-level average similarity: {word_level_similarity:.4f}")

        # 3. Combine with weighted average
        combined_similarity = (
            self.full_name_weight * full_name_similarity
            + (1 - self.full_name_weight) * word_level_similarity
        )

        logger.info(
            f"Combined similarity (weight={self.full_name_weight:.2f}): {combined_similarity:.4f}"
        )

        return float(combined_similarity)

    def _split_entity_name(self, entity_name: str) -> List[str]:
        """
        Split an entity name into individual words.

        Handles underscore-separated (FIRST_NAME) and camelCase (FirstName) formats.

        :param entity_name: Entity name to split
        :return: List of individual words

        Examples:
            >>> _split_entity_name("FIRST_NAME")
            ["FIRST", "NAME"]
            >>> _split_entity_name("EMAIL_ADDRESS")
            ["EMAIL", "ADDRESS"]
            >>> _split_entity_name("SSN")
            ["SSN"]
        """
        # First, split on underscores
        parts = entity_name.split("_")

        # Then split camelCase within each part
        words = []
        for part in parts:
            # Split camelCase (e.g., FirstName -> First Name)
            camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
            camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split)
            words.extend(camel_split.split())

        # Filter out empty strings and return
        return [word for word in words if word]

    def _best_match_with_score(
        self, source_entity: str, target_entities: List[str]
    ) -> tuple[Optional[str], Optional[float]]:
        """Return the best matching target entity alongside its similarity score."""
        if not target_entities:
            return None, None

        similarities = [
            (target, self.compute_similarity(source_entity, target))
            for target in target_entities
        ]

        best_match, best_score = max(similarities, key=lambda item: item[1])
        return best_match, float(best_score)

    def map(self, source_entity: str, target_entities: List[str]) -> Optional[str]:
        """
        Map a source entity to the most similar target entity.

        :param source_entity: Entity type to map from
        :param target_entities: List of candidate entity types to map to
        :return: The most similar target entity if similarity exceeds threshold, else None

        Examples:
            >>> mapper = SemanticEntityMapper(threshold=0.85)
            >>> mapper.map('FIRST_NAME', ['PERSON', 'LOCATION'])
            'PERSON'
            >>> mapper.map('CIUDAD', ['CITY', 'PERSON'])  # Spanish for city
            'CITY'
        """
        best_match, best_score = self._best_match_with_score(
            source_entity, target_entities
        )

        if best_match is None or best_score is None:
            return None

        logger.debug(
            f"Entity '{source_entity}' -> '{best_match}' (score: {best_score:.3f}, "
            f"threshold: {self.threshold})"
        )

        if best_score >= self.threshold:
            return best_match

        logger.debug(
            f"No match found for '{source_entity}': best score {best_score:.3f} "
            f"below threshold {self.threshold}"
        )
        return None

    def map_multiple_entities(
        self, source_entities: List[str], target_entities: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Map multiple source entities to target entities.

        :param source_entities: List of entity types to map from
        :param target_entities: List of candidate entity types to map to
        :return: Dictionary mapping source entities to best matching target entities

        Examples:
            >>> mapper = SemanticEntityMapper()
            >>> result = mapper.map_multiple_entities(
            ...     ['FIRST_NAME', 'SSN', 'CITY'],
            ...     ['PERSON', 'ID', 'LOCATION']
            ... )
            >>> result
            {'FIRST_NAME': 'PERSON', 'SSN': 'ID', 'CITY': 'LOCATION'}
        """
        mappings = {}
        for source in source_entities:
            mappings[source] = self.map(source, target_entities)
        return mappings

    def create_mapping_dict(
        self,
        source_entities: List[str],
        target_entities: List[str],
        include_unmapped: bool = False,
    ) -> Dict[str, str]:
        """
        Create a mapping dictionary from source to target entities.

        This is useful for creating entity_mapping configurations.

        :param source_entities: List of entity types to map from
        :param target_entities: List of candidate entity types to map to
        :param include_unmapped: If True, include source entities that couldn't be mapped
                            (mapped to themselves)
        :return: Dictionary suitable for use as entity_mapping parameter

        Examples:
            >>> mapper = SemanticEntityMapper()
            >>> mapping = mapper.create_mapping_dict(
            ...     ['FIRST_NAME', 'LAST_NAME', 'SSN'],
            ...     ['PERSON', 'ID']
            ... )
            >>> mapping
            {'FIRST_NAME': 'PERSON', 'LAST_NAME': 'PERSON', 'SSN': 'ID'}
        """
        result = {}
        mappings = self.map_multiple_entities(source_entities, target_entities)

        for source, target in mappings.items():
            if target is not None:
                result[source] = target
            elif include_unmapped:
                result[source] = source

        return result

    def analyze_mappings(
        self, source_entities: List[str], target_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze mapping quality and provide detailed statistics.

        :param source_entities: List of entity types to map from
        :param target_entities: List of candidate entity types to map to
        :return: Dictionary with mapping statistics and details
        """
        matches: Dict[str, Optional[str]] = {}
        scores: Dict[str, Optional[float]] = {}

        for source in source_entities:
            match, score = self._best_match_with_score(source, target_entities)
            if match is not None and score is not None and score >= self.threshold:
                matches[source] = match
                scores[source] = score
            else:
                matches[source] = None
                scores[source] = score

        mapped = {s: m for s, m in matches.items() if m is not None}
        unmapped = [s for s, m in matches.items() if m is None]

        stats: Dict[str, Any] = {
            "total_source_entities": len(source_entities),
            "total_target_entities": len(target_entities),
            "mapped_count": len(mapped),
            "unmapped_count": len(unmapped),
            "mapping_rate": len(mapped) / len(source_entities)
            if source_entities
            else 0,
            "threshold": self.threshold,
            "mappings": mapped,
            "unmapped_entities": unmapped,
            "scores": {s: scores[s] for s in mapped if scores[s] is not None},
        }

        mapped_scores = [scores[s] for s in mapped if scores[s] is not None]
        if mapped_scores:
            stats["avg_score"] = float(sum(mapped_scores) / len(mapped_scores))
            stats["min_score"] = min(mapped_scores)
            stats["max_score"] = max(mapped_scores)

        return stats

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Cleared embedding cache")


def create_hierarchical_mapper(
    exact_mappings: Optional[Dict[str, str]] = None,
    substring_mappings: Optional[Dict[str, str]] = None,
    semantic_threshold: float = 0.40,  # Lower threshold as fallback only
    use_semantic_fallback: bool = True,
    exact_match_confidence: float = 1.0,
    pattern_match_confidence: float = 0.6,
) -> Callable[[str, List[str]], Optional[Tuple[str, float]]]:
    """
    Create a robust entity mapper that combines multiple strategies.

    Strategy order:
    1. Exact match (case-insensitive) - configurable confidence (default 1.0)
    2. Substring/contains matching - configurable confidence (default 0.6)
    3. Pattern-based matching (e.g., FIRST_NAME -> PERSON) - configurable confidence (default 0.6)
    4. Semantic similarity (fallback only) - variable confidence from model

    :param exact_mappings: Direct mappings to try first
    :param substring_mappings: Mappings based on substring matching
    :param semantic_threshold: Minimum similarity for semantic matching (low as fallback)
    :param use_semantic_fallback: Whether to use semantic similarity as last resort
    :param exact_match_confidence: Confidence for exact match strategies (default 1.0)
    :param pattern_match_confidence: Confidence for substring/pattern match strategies (default 0.6)
    :return: Function that maps source entity to target entity with confidence score (entity, confidence)

    Example:
        >>> mapper = create_enhanced_mapper(
        ...     exact_mappings={'SSN': 'ID', 'EMAIL': 'CONTACT'},
        ...     substring_mappings={'NAME': 'PERSON', 'ID': 'ID'}
        ... )
        >>> mapper('FIRST_NAME', ['PERSON', 'ID', 'LOCATION'])
        'PERSON'
        >>> mapper('DRIVER_LICENSE', ['PERSON', 'ID', 'LOCATION'])
        'ID'
    """
    exact_mappings = exact_mappings or {}
    substring_mappings = substring_mappings or {}

    semantic_mapper = (
        SemanticEntityMapper(threshold=semantic_threshold)
        if use_semantic_fallback
        else None
    )

    # Normalize exact mappings to uppercase
    exact_map_normalized = {k.upper(): v for k, v in exact_mappings.items()}

    def mapper(
        source_entity: str, target_entities: List[str]
    ) -> Optional[Tuple[str, float]]:
        source_upper = source_entity.upper()

        # Strategy 1: Exact match - high confidence (predefined mapping)
        if source_upper in exact_map_normalized:
            mapped = exact_map_normalized[source_upper]
            if mapped in target_entities:
                return (mapped, exact_match_confidence)

        # Strategy 2: Substring matching - medium confidence (generic patterns)
        for substring, mapped_to in substring_mappings.items():
            if substring.upper() in source_upper:
                if mapped_to in target_entities:
                    return (mapped_to, pattern_match_confidence)

        # Strategy 3: Semantic similarity as last resort - variable confidence
        if semantic_mapper:
            mapped = semantic_mapper.map(source_entity, target_entities)
            if mapped:
                score = semantic_mapper.compute_similarity(source_entity, mapped)
                return (mapped, float(score))

        return None

    return mapper


def create_presidio_mapper() -> Callable[[str, List[str]], Optional[Tuple[str, float]]]:
    """
    Create a mapper specifically tuned for Presidio entity types.

    :return: Mapper function optimized for common Presidio entities, returns (entity, confidence) tuples

    Example:
        >>> mapper = create_presidio_mapper()
        >>> mapper('FIRST_NAME', ['PERSON', 'LOCATION', 'ID'])
        ('PERSON', 1.0)
        >>> mapper('US_SSN', ['PERSON', 'US_SSN', 'FINANCIAL'])
        ('US_SSN', 1.0)
    """
    # Explicit mappings for Presidio entity types
    exact_mappings = {
        # Person-related
        "FIRST_NAME": "PERSON",
        "LAST_NAME": "PERSON",
        "FULL_NAME": "PERSON",
        "NAME": "PERSON",
        "PATIENT": "PERSON",
        "DOCTOR": "PERSON",
        # IDs
        "SSN": "US_SSN",
        "US_SSN": "US_SSN",
        "UK_NHS": "ID",
        "US_PASSPORT": "US_PASSPORT",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "MEDICAL_LICENSE": "MEDICAL_LICENSE",
        "US_ITIN": "US_ITIN",
        "SG_NRIC_FIN": "SG_NRIC_FIN",
        "AU_TFN": "AU_TFN",
        "AU_ABN": "AU_ABN",
        "AU_ACN": "AU_ACN",
        # Financial
        "CREDIT_CARD": "CREDIT_CARD",
        "CRYPTO": "CRYPTO",
        "IBAN_CODE": "IBAN_CODE",
        "US_BANK_NUMBER": "US_BANK_NUMBER",
        # Contact
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "EMAIL": "EMAIL",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "PHONE": "PHONE_NUMBER",
        "FAX_NUMBER": "PHONE_NUMBER",
        "FAX": "PHONE_NUMBER",
        # ORGANIZATION
        "ORGANIZATION": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "NORP": "ORGANIZATION",
        "NRP": "ORGANIZATION",
        "INSTITUTE": "ORGANIZATION",
        # LOCATION
        "ADDRESS": "LOCATION",
        "HOSPITAL": "LOCATION",
        "FACILITY": "LOCATION",
        "STREET_ADDRESS": "LOCATION",
        "CITY": "LOCATION",
        "ZIP": "LOCATION",
        "ZIP_CODE": "LOCATION",
        "ZIPCODE": "LOCATION",
        "STATE": "LOCATION",
        "COUNTRY": "LOCATION",
        "GPE": "LOCATION",
        # Technical
        "IP_ADDRESS": "IP_ADDRESS",
        "URL": "URL",
        "DOMAIN_NAME": "URL",
        # Date/Time
        "DATE": "DATE_TIME",
        "DATE_TIME": "DATE_TIME",
        "TIME": "DATE_TIME",
        "DOB": "DATE_TIME",
        "DATE_OF_BIRTH": "DATE_TIME",
        "BIRTH_DATE": "DATE_TIME",
    }

    substring_mappings = {
        "NAME": "PERSON",
        "PERSON": "PERSON",
        "PATIENT": "PERSON",
        "DOCTOR": "PERSON",
        "LICENSE": "LICENSE",
        "PASSPORT": "PASSPORT",
        "ADDRESS": "LOCATION",
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "CARD": "CREDIT_CARD",
        "BANK": "US_BANK_NUMBER",
    }

    # Add target entity types as exact mappings to themselves
    # (e.g., if someone passes "PERSON" as input, it should map to "PERSON")
    target_entity_types = set(exact_mappings.values())
    for entity_type in target_entity_types:
        if entity_type not in exact_mappings:
            exact_mappings[entity_type] = entity_type

    return create_hierarchical_mapper(
        exact_mappings=exact_mappings,
        substring_mappings=substring_mappings,
        semantic_threshold=0.40,
        use_semantic_fallback=True,
    )


class HybridEntityMapper:
    """
    A hybrid entity mapper that intelligently combines multiple strategies.

    This provides a more reliable alternative to pure semantic similarity.

    Example:
        >>> mapper = HybridEntityMapper()
        >>> # Add custom mappings
        >>> mapper.add_exact_mapping('CUSTOM_ID', 'ID')
        >>> mapper.add_pattern('NAME', 'PERSON')
        >>>
        >>> # Map entities
        >>> result = mapper.map('FIRST_NAME', ['PERSON', 'ID'])
        >>> assert result == 'PERSON'
    """

    def __init__(self, semantic_threshold: float = 0.40):
        """
        Initialize hybrid mapper.

        :param semantic_threshold: Minimum similarity for semantic fallback (low)
        """
        self.exact_mappings: Dict[str, str] = {}
        self.pattern_mappings: Dict[str, str] = {}
        self.semantic_mapper = SemanticEntityMapper(threshold=semantic_threshold)

    def add_exact_mapping(self, source: str, target: str):
        """Add an exact entity mapping."""
        self.exact_mappings[source.upper()] = target

    def add_exact_mappings(self, mappings: Dict[str, str]):
        """Add multiple exact mappings."""
        for source, target in mappings.items():
            self.add_exact_mapping(source, target)

    def add_pattern(self, pattern: str, target: str):
        """Add a substring pattern mapping."""
        self.pattern_mappings[pattern.upper()] = target

    def map(self, source_entity: str, target_entities: List[str]) -> Optional[str]:
        """
        Map source entity to best matching target entity.

        Uses multiple strategies in order of reliability:
        1. Exact mappings
        2. Pattern/substring matching
        3. Semantic similarity (fallback)
        """
        source_upper = source_entity.upper()

        # Try exact mapping
        if source_upper in self.exact_mappings:
            mapped = self.exact_mappings[source_upper]
            if mapped in target_entities:
                return mapped

        # Try pattern matching
        for pattern, mapped_to in self.pattern_mappings.items():
            if pattern in source_upper:
                if mapped_to in target_entities:
                    return mapped_to

        # Fall back to semantic similarity
        return self.semantic_mapper.map(source_entity, target_entities)

    def map_multiple_entities(
        self, source_entities: List[str], target_entities: List[str]
    ) -> Dict[str, Optional[str]]:
        """Map multiple entities."""
        return {source: self.map(source, target_entities) for source in source_entities}
