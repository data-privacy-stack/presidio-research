"""
Comprehensive tests for entity mapping functionality.

This module tests:
1. Entity mappers (semantic, hierarchical, hybrid, presidio)
2. EntityMappingHelper for interactive mapping
3. Entity mapping during evaluation comparison
4. End-to-end metrics with entity mapping
"""

import pytest
from unittest.mock import Mock

from presidio_evaluator import InputSample
from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation import TokenEvaluator
from presidio_evaluator.entity_mapping import (
    SemanticEntityMapper,
    create_hierarchical_mapper,
    create_presidio_mapper,
    EntityMappingHelper,
)
from tests.mocks import MockTokensModel


# ============================================================================
# PART 1: Entity Mapper Tests (Semantic, Hierarchical, Hybrid, Presidio)
# ============================================================================


class TestEntityMappers:
    """Tests for various entity mapper implementations."""

    def test_semantic_mapper_basic_functionality(self):
        """Test basic SemanticEntityMapper functionality."""
        mapper = SemanticEntityMapper(threshold=0.60)

        assert mapper.threshold == 0.60
        assert mapper.cache_embeddings is True
        assert mapper.full_name_weight == 0.6  # Default value

        result = mapper.map("EMAIL", ["CONTACT"])
        assert result == "CONTACT"

        result = mapper.map("PERSON", [])
        assert result is None

    def test_semantic_mapper_caching(self):
        """Test that embedding caching works."""
        mapper = SemanticEntityMapper(cache_embeddings=True)

        mapper._get_embedding("PERSON")
        assert "PERSON" in mapper._embedding_cache

        mapper.clear_cache()
        assert len(mapper._embedding_cache) == 0

    def test_enhanced_mapper_exact_match(self):
        """Test that exact mappings work correctly."""
        mapper = create_hierarchical_mapper(
            exact_mappings={"SSN": "ID", "EMAIL": "CONTACT"}
        )

        entity, confidence = mapper("SSN", ["PERSON", "ID", "LOCATION"])
        assert entity == "ID"
        assert confidence == 1.0

        entity, confidence = mapper("EMAIL", ["PERSON", "CONTACT"])
        assert entity == "CONTACT"
        assert confidence == 1.0

    def test_enhanced_mapper_substring_match(self):
        """Test that substring matching works."""
        mapper = create_hierarchical_mapper(
            substring_mappings={"NAME": "PERSON", "LICENSE": "ID"}
        )

        entity, confidence = mapper("FIRST_NAME", ["PERSON", "ID"])
        assert entity == "PERSON"
        assert confidence == 0.6  # Substring match confidence

        entity, confidence = mapper("DRIVER_LICENSE", ["PERSON", "ID"])
        assert entity == "ID"
        assert confidence == 0.6

    def test_semantic_mapper_full_name_weight(self):
        """Test that full_name_weight parameter works correctly."""
        mapper_full = SemanticEntityMapper(threshold=0.35, full_name_weight=1.0)
        mapper_word = SemanticEntityMapper(threshold=0.35, full_name_weight=0.0)
        mapper_balanced = SemanticEntityMapper(threshold=0.35, full_name_weight=0.6)

        assert mapper_full.full_name_weight == 1.0
        assert mapper_word.full_name_weight == 0.0
        assert mapper_balanced.full_name_weight == 0.6

    def test_enhanced_mapper_strategy_order(self):
        """Test that strategies are applied in correct order (exact before substring)."""
        mapper = create_hierarchical_mapper(
            exact_mappings={"NAME": "EXACT_MATCH"},
            substring_mappings={"NAME": "SUBSTRING_MATCH"},
        )

        entity, confidence = mapper(
            "NAME", ["EXACT_MATCH", "SUBSTRING_MATCH", "PATTERN"]
        )
        assert entity == "EXACT_MATCH"
        assert confidence == 1.0  # Exact match takes precedence

    def test_presidio_mapper(self):
        """Test pre-configured Presidio mapper."""
        mapper = create_presidio_mapper()

        # Person entities
        entity, confidence = mapper("FIRST_NAME", ["PERSON", "LOCATION"])
        assert entity == "PERSON"
        assert confidence == 1.0

        # IDs
        entity, confidence = mapper("US_SSN", ["PERSON", "US_SSN", "LOCATION"])
        assert entity == "US_SSN"
        assert confidence == 1.0

        # Financial
        entity, confidence = mapper("CREDIT_CARD", ["CREDIT_CARD", "PERSON"])
        assert entity == "CREDIT_CARD"
        assert confidence == 1.0

        # Contact
        entity, confidence = mapper("EMAIL_ADDRESS", ["EMAIL_ADDRESS", "PERSON"])
        assert entity == "EMAIL_ADDRESS"
        assert confidence == 1.0

        # Location
        entity, confidence = mapper("ADDRESS", ["LOCATION", "PERSON"])
        assert entity == "LOCATION"
        assert confidence == 1.0

    def test_presidio_mapper_entity_naming_variations(self):
        """Test that entity naming variations (camel case, hyphens, etc.) are handled correctly."""
        mapper = create_presidio_mapper()

        entity, confidence = mapper("CreditCard", ["CREDIT_CARD", "PERSON", "LOCATION"])
        assert entity == "CREDIT_CARD"
        assert confidence > 0

        entity, confidence = mapper(
            "credit-card", ["CREDIT_CARD", "PERSON", "LOCATION"]
        )
        assert entity == "CREDIT_CARD"
        assert confidence > 0

        entity, confidence = mapper("PHONE", ["PHONE_NUMBER", "PERSON", "LOCATION"])
        assert entity == "PHONE_NUMBER"
        assert confidence > 0

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        mapper = create_hierarchical_mapper(exact_mappings={"ssn": "ID"})

        entity, confidence = mapper("SSN", ["ID"])
        assert entity == "ID"
        assert confidence == 1.0

    def test_no_match_returns_none(self):
        """Test that no match returns None."""
        mapper = create_hierarchical_mapper(use_semantic_fallback=False)

        result = mapper("UNKNOWN_ENTITY", ["PERSON", "ID"])
        assert result is None


# ============================================================================
# PART 2: EntityMappingHelper Tests
# ============================================================================


def create_sample_dataset():
    """Create a small dataset for testing."""
    return [
        InputSample(
            full_text="John Doe",
            spans=[Span("PERSON", "John Doe", 0, 8)],
            masked="[PERSON]",
            tokens=["John", "Doe"],
            tags=["PERSON", "PERSON"],
        ),
        InputSample(
            full_text="john@example.com",
            spans=[Span("EMAIL", "john@example.com", 0, 16)],
            masked="[EMAIL]",
            tokens=["john@example.com"],
            tags=["EMAIL"],
        ),
        InputSample(
            full_text="123-45-6789",
            spans=[Span("SSN", "123-45-6789", 0, 11)],
            masked="[SSN]",
            tokens=["123-45-6789"],
            tags=["SSN"],
        ),
    ]


def create_mock_analyzer():
    """Create a mock analyzer that supports specific entities."""
    mock = Mock()
    mock.get_supported_entities.return_value = [
        "PERSON",
        "EMAIL_ADDRESS",
        "US_SSN",
        "LOCATION",
    ]
    return mock


class TestEntityMappingHelper:
    """Tests for EntityMappingHelper class."""

    def test_initialization_with_dataset_list(self):
        """Test initialization with a list of samples."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()

        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        assert helper.dataset == dataset
        assert len(helper._all_dataset_entities) == 3  # PERSON, EMAIL, SSN
        assert len(helper._all_model_entities) == 4

    def test_set_mapping(self):
        """Test manually setting a mapping."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()
        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        helper.set_mapping("EMAIL", "EMAIL_ADDRESS")

        assert helper._manual_mappings["EMAIL"] == "EMAIL_ADDRESS"
        assert helper._suggested_mapping["EMAIL"] == "EMAIL_ADDRESS"

    def test_exclude_dataset_entities(self):
        """Test excluding dataset entities."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()
        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        helper.exclude_dataset_entities("SSN")

        assert "SSN" in helper._excluded_dataset_entities
        assert "SSN" not in helper._suggested_mapping

    def test_get_mapping_success(self):
        """Test get_mapping when all entities are mapped."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()
        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        helper.set_mapping("PERSON", "PERSON")
        helper.set_mapping("EMAIL", "EMAIL_ADDRESS")
        helper.set_mapping("SSN", "US_SSN")

        mapping = helper.get_mapping()

        assert isinstance(mapping, dict)
        assert "PERSON" in mapping
        assert "EMAIL" in mapping
        assert "SSN" in mapping
        assert None not in mapping.values()

    def test_get_model_entities_to_use(self):
        """Test getting model entities to use."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()
        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        helper.set_mapping("PERSON", "PERSON")
        helper.set_mapping("EMAIL", "EMAIL_ADDRESS")
        helper.set_mapping("SSN", "US_SSN")

        entities = helper.get_model_entities_to_use()

        assert isinstance(entities, list)
        assert "PERSON" in entities
        assert "EMAIL_ADDRESS" in entities
        assert "US_SSN" in entities

    def test_get_model_entities_to_use_includes_none_mappings(self):
        """Test that entities mapped to None are included using their original names."""
        dataset = create_sample_dataset()
        analyzer = create_mock_analyzer()
        helper = EntityMappingHelper(dataset=dataset, model=analyzer)

        helper.set_mapping("PERSON", "PERSON")
        helper.set_mapping("EMAIL", None)  # None means use as-is
        helper.set_mapping("SSN", "US_SSN")

        entities = helper.get_model_entities_to_use()

        assert isinstance(entities, list)
        assert "PERSON" in entities
        assert "EMAIL" in entities  # Should include original name when mapped to None
        assert "US_SSN" in entities


# ============================================================================
# PART 3: Entity Mapping During Comparison Tests
# ============================================================================


class TestEntityMappingComparison:
    """Test that entity mapping is applied during comparison, preserving original data."""

    def test_normalize_entity_for_comparison_unprefixed(self):
        """Test normalization of unprefixed entities."""
        from presidio_evaluator.evaluation import BaseEvaluator

        mapping = {"STREET_ADDRESS": "LOCATION", "GPE": "LOCATION"}

        assert (
            BaseEvaluator._normalize_entity_for_comparison("STREET_ADDRESS", mapping)
            == "LOCATION"
        )
        assert (
            BaseEvaluator._normalize_entity_for_comparison("GPE", mapping) == "LOCATION"
        )
        assert (
            BaseEvaluator._normalize_entity_for_comparison("PHONE_NUMBER", mapping)
            == "PHONE_NUMBER"
        )
        assert BaseEvaluator._normalize_entity_for_comparison("O", mapping) == "O"

    def test_normalize_entity_for_comparison_prefixed(self):
        """Test normalization of prefixed entities (B-, I-, L-, U-)."""
        from presidio_evaluator.evaluation import BaseEvaluator

        mapping = {"STREET_ADDRESS": "LOCATION", "GPE": "LOCATION"}

        assert (
            BaseEvaluator._normalize_entity_for_comparison("B-STREET_ADDRESS", mapping)
            == "B-LOCATION"
        )
        assert (
            BaseEvaluator._normalize_entity_for_comparison("I-GPE", mapping)
            == "I-LOCATION"
        )
        assert (
            BaseEvaluator._normalize_entity_for_comparison("L-STREET_ADDRESS", mapping)
            == "L-LOCATION"
        )
        assert (
            BaseEvaluator._normalize_entity_for_comparison("B-PHONE_NUMBER", mapping)
            == "B-PHONE_NUMBER"
        )

    def test_evaluator_requires_entity_mapping(self):
        """Test that evaluator raises error if entity_mapping is not provided."""
        model = MockTokensModel(prediction=["O", "O", "O"])

        # entity_mapping=None should raise an error
        with pytest.raises(ValueError, match="entity_mapping is required"):
            TokenEvaluator(model=model, entity_mapping=None)

        # entity_mapping={} is valid (identity mapping - no translation needed)
        evaluator = TokenEvaluator(model=model, entity_mapping={})
        assert evaluator.entity_mapping == {}

    def test_model_entity_mapping_raises_error(self):
        """Test that passing entity_mapping to model raises deprecation error."""
        from tests.mocks import MockModel

        with pytest.raises(
            ValueError, match="entity_mapping.*deprecated.*removed from BaseModel"
        ):
            _ = MockModel(entity_mapping={"STREET_ADDRESS": "LOCATION"})

    def test_comparison_uses_mapped_entities(self):
        """Test that comparison uses mapped entities while preserving original in errors."""
        tokens = ["123", "Main", "St"]
        tags = ["B-STREET_ADDRESS", "I-STREET_ADDRESS", "I-STREET_ADDRESS"]
        prediction = ["B-LOCATION", "I-LOCATION", "I-LOCATION"]

        sample = InputSample(full_text="123 Main St", tokens=tokens, tags=tags)
        model = MockTokensModel(prediction=prediction)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={"STREET_ADDRESS": "LOCATION"},
            compare_by_io=True,
        )

        results, errors = evaluator.compare(sample, prediction)

        assert results[("LOCATION", "LOCATION")] == 3
        assert len(errors) == 0

    def test_comparison_preserves_original_entities_in_errors(self):
        """Test that error reports show original entity names, not mapped ones."""
        tokens = ["123", "Main", "Blvd"]
        tags = ["B-STREET_ADDRESS", "I-STREET_ADDRESS", "I-STREET_ADDRESS"]
        prediction = ["B-PERSON", "I-PERSON", "I-PERSON"]

        sample = InputSample(full_text="123 Main Blvd", tokens=tokens, tags=tags)
        model = MockTokensModel(prediction=prediction)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={"STREET_ADDRESS": "LOCATION"},
            compare_by_io=True,
            skip_words=[],
        )

        results, errors = evaluator.compare(sample, prediction)

        assert len(errors) == 3
        assert all(error.annotation == "STREET_ADDRESS" for error in errors)
        assert ("LOCATION", "PERSON") in results

    def test_unmapped_entities_kept_as_is(self):
        """Test that entities not in mapping are kept unchanged."""
        tokens = ["555-1234", "John"]
        tags = ["PHONE_NUMBER", "PERSON"]
        prediction = ["PHONE_NUMBER", "PERSON"]

        sample = InputSample(full_text="555-1234 John", tokens=tokens, tags=tags)
        model = MockTokensModel(prediction=prediction)

        evaluator = TokenEvaluator(
            model=model, entity_mapping={"STREET_ADDRESS": "LOCATION"}
        )

        results, errors = evaluator.compare(sample, prediction)

        assert results[("PHONE_NUMBER", "PHONE_NUMBER")] == 1
        assert results[("PERSON", "PERSON")] == 1
        assert len(errors) == 0

    def test_multiple_entities_map_to_same_target(self):
        """Test that multiple dataset entities can map to the same model entity."""
        tokens = ["Seattle", ",", "WA", ",", "123", "Main", "St", ",", "98101"]
        tags = [
            "GPE",
            "O",
            "GPE",
            "O",
            "STREET_ADDRESS",
            "STREET_ADDRESS",
            "STREET_ADDRESS",
            "O",
            "ZIP_CODE",
        ]
        prediction = [
            "LOCATION",
            "O",
            "LOCATION",
            "O",
            "LOCATION",
            "LOCATION",
            "LOCATION",
            "O",
            "LOCATION",
        ]

        sample = InputSample(
            full_text="Seattle, WA, 123 Main St, 98101", tokens=tokens, tags=tags
        )
        model = MockTokensModel(prediction=prediction)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={
                "GPE": "LOCATION",
                "STREET_ADDRESS": "LOCATION",
                "ZIP_CODE": "LOCATION",
            },
        )

        results, errors = evaluator.compare(sample, prediction)

        assert results[("LOCATION", "LOCATION")] == 6
        assert results[("O", "O")] == 3
        assert len(errors) == 0

    def test_none_mapping_means_identity(self):
        """Test that mapping to None means identity mapping (use entity as-is)."""
        tokens = ["123", "Main", "St", "Seattle"]
        tags = ["B-STREET_ADDRESS", "I-STREET_ADDRESS", "I-STREET_ADDRESS", "GPE"]
        # Model predicts STREET_ADDRESS (same as dataset)
        prediction = [
            "B-STREET_ADDRESS",
            "I-STREET_ADDRESS",
            "I-STREET_ADDRESS",
            "ORGANIZATION",
        ]

        sample = InputSample(full_text="123 Main St Seattle", tokens=tokens, tags=tags)
        model = MockTokensModel(prediction=prediction)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={
                "STREET_ADDRESS": None,  # None means use as-is
                "GPE": "ORGANIZATION",
            },
            compare_by_io=True,
        )

        results, errors = evaluator.compare(sample, prediction)

        # STREET_ADDRESS should match STREET_ADDRESS (not become "None")
        assert results[("STREET_ADDRESS", "STREET_ADDRESS")] == 3
        assert results[("ORGANIZATION", "ORGANIZATION")] == 1
        assert len(errors) == 0


# ============================================================================
# PART 4: End-to-End Tests with Metrics
# ============================================================================


class TestEntityMappingE2EMetrics:
    """End-to-end tests for entity mapping with full metric calculation."""

    def test_e2e_entity_mapping_with_metrics_shows_mapped_entities(self):
        """
        Test that when multiple dataset entities map to one model entity,
        metrics are calculated for the MAPPED entity (LOCATION), not originals.
        """
        # Create dataset with different entity types that all map to LOCATION
        sample1 = InputSample(
            full_text="123 Main Street",
            tokens=["123", "Main", "Street"],
            tags=["STREET_ADDRESS", "STREET_ADDRESS", "STREET_ADDRESS"],
        )

        sample2 = InputSample(
            full_text="Seattle WA", tokens=["Seattle", "WA"], tags=["GPE", "GPE"]
        )

        sample3 = InputSample(full_text="98101", tokens=["98101"], tags=["ZIP_CODE"])

        sample4 = InputSample(
            full_text="555-1234", tokens=["555-1234"], tags=["PHONE_NUMBER"]
        )

        dataset = [sample1, sample2, sample3, sample4]

        predictions = [
            ["LOCATION", "LOCATION", "LOCATION"],
            ["LOCATION", "O"],
            ["PERSON"],
            ["PHONE_NUMBER"],
        ]

        class BatchMockModel(MockTokensModel):
            def __init__(self, predictions):
                super().__init__(prediction=None)
                self.predictions = predictions

            def batch_predict(self, dataset, **kwargs):
                return self.predictions

        model = BatchMockModel(predictions)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={
                "STREET_ADDRESS": "LOCATION",
                "GPE": "LOCATION",
                "ZIP_CODE": "LOCATION",
                "PHONE_NUMBER": "PHONE_NUMBER",
            },
            skip_words=[],
            compare_by_io=True,
        )

        evaluation_results = evaluator.evaluate_all(dataset)
        scores = evaluator.calculate_score(evaluation_results)

        entity_results = scores.entity_recall_dict
        entity_precision = scores.entity_precision_dict

        # Metrics should show LOCATION (mapped), not STREET_ADDRESS/GPE/ZIP_CODE (original)
        assert "LOCATION" in entity_results
        assert "PHONE_NUMBER" in entity_results
        assert "STREET_ADDRESS" not in entity_results
        assert "GPE" not in entity_results
        assert "ZIP_CODE" not in entity_results

        location_recall = entity_results["LOCATION"]
        location_precision = entity_precision["LOCATION"]

        assert location_recall == 4 / 6
        assert location_precision == 4 / 4

    def test_e2e_shows_granular_errors_but_aggregated_metrics(self):
        """
        Test that error reports preserve original entity names for analysis,
        but metrics aggregate at the mapped entity level.
        """
        samples = [
            InputSample(
                full_text="123 Main",
                tokens=["123", "Main"],
                tags=["STREET_ADDRESS", "STREET_ADDRESS"],
            ),
            InputSample(
                full_text="Seattle WA", tokens=["Seattle", "WA"], tags=["GPE", "GPE"]
            ),
        ]

        predictions = [["O", "O"], ["LOCATION", "LOCATION"]]

        class BatchMockModel(MockTokensModel):
            def __init__(self, predictions):
                super().__init__(prediction=None)
                self.predictions = predictions

            def batch_predict(self, dataset, **kwargs):
                return self.predictions

        model = BatchMockModel(predictions)

        evaluator = TokenEvaluator(
            model=model,
            entity_mapping={"STREET_ADDRESS": "LOCATION", "GPE": "LOCATION"},
            skip_words=[],
            compare_by_io=True,
        )

        evaluation_results = evaluator.evaluate_all(samples)

        # Errors preserve ORIGINAL entity names
        all_errors = []
        for result in evaluation_results:
            all_errors.extend(result.model_errors)

        street_errors = [e for e in all_errors if e.annotation == "STREET_ADDRESS"]
        assert len(street_errors) == 2

        # But metrics aggregate at LOCATION level
        scores = evaluator.calculate_score(evaluation_results)

        location_recall = scores.entity_recall_dict["LOCATION"]
        expected_recall = 2 / 4
        assert location_recall == expected_recall

        # Cannot get separate metrics for STREET_ADDRESS vs GPE
        assert "STREET_ADDRESS" not in scores.entity_recall_dict
        assert "GPE" not in scores.entity_recall_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
