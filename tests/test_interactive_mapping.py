"""
Tests for interactive entity mapping utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json
import tempfile

from presidio_evaluator.entity_mapping import (
    get_model_entities,
    get_dataset_entities,
    suggest_mapping,
    print_mapping_summary,
    save_mapping_to_file,
    load_mapping_from_file,
)
from presidio_evaluator.data_objects import InputSample, Span


class TestGetModelEntities:
    """Tests for get_model_entities function."""
    
    def test_presidio_analyzer_engine(self):
        """Test with Presidio AnalyzerEngine that has get_supported_entities."""
        mock_analyzer = Mock()
        mock_analyzer.get_supported_entities.return_value = ["PERSON", "LOCATION", "EMAIL"]
        
        entities = get_model_entities(mock_analyzer, language="en")
        
        assert entities == ["PERSON", "LOCATION", "EMAIL"]
        mock_analyzer.get_supported_entities.assert_called_once_with("en")
    
    def test_wrapped_presidio_analyzer(self):
        """Test with wrapped Presidio analyzer."""
        mock_wrapper = Mock()
        mock_analyzer = Mock()
        mock_analyzer.get_supported_entities.return_value = ["PERSON", "ID"]
        mock_wrapper.analyzer_engine = mock_analyzer
        mock_wrapper.get_supported_entities = None  # Wrapper doesn't have it directly
        
        entities = get_model_entities(mock_wrapper, language="en")
        
        assert entities == ["PERSON", "ID"]
    
    def test_model_with_entities_attribute(self):
        """Test with model that has entities attribute."""
        mock_model = Mock()
        mock_model.entities = ["PERSON", "LOCATION", "ORG"]
        
        entities = get_model_entities(mock_model)
        
        assert entities == ["PERSON", "LOCATION", "ORG"]
    
    def test_transformers_model_with_id2label(self):
        """Test with Transformers model config.id2label."""
        mock_config = Mock()
        mock_config.id2label = {
            0: "O",
            1: "B-PERSON",
            2: "I-PERSON",
            3: "B-LOCATION",
            4: "I-LOCATION",
        }
        mock_config.label2id = None
        
        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.entities = None
        mock_model.get_supported_entities = None
        
        entities = get_model_entities(mock_model)
        
        assert set(entities) == {"PERSON", "LOCATION"}
        assert "O" not in entities
    
    def test_transformers_model_with_label2id(self):
        """Test with Transformers model config.label2id."""
        mock_config = Mock()
        mock_config.id2label = None
        mock_config.label2id = {
            "O": 0,
            "B-ORG": 1,
            "I-ORG": 2,
            "B-LOCATION": 3,
        }
        
        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.entities = None
        mock_model.get_supported_entities = None
        
        entities = get_model_entities(mock_model)
        
        assert set(entities) == {"ORG", "LOCATION"}
    
    def test_spacy_model(self):
        """Test with SpaCy model."""
        mock_ner = Mock()
        mock_ner.labels = ["PERSON", "ORG", "GPE"]
        
        mock_model = Mock()
        mock_model.pipe_names = ["tok2vec", "tagger", "parser", "ner"]
        mock_model.get_pipe.return_value = mock_ner
        mock_model.entities = None
        mock_model.get_supported_entities = None
        
        entities = get_model_entities(mock_model)
        
        assert entities == ["PERSON", "ORG", "GPE"]
        mock_model.get_pipe.assert_called_once_with("ner")
    
    def test_unknown_model_type(self):
        """Test with unknown model type returns empty list."""
        mock_model = Mock(spec=[])  # Empty spec - no attributes
        
        entities = get_model_entities(mock_model)
        
        assert entities == []


class TestTransformersNlpEngineExtraction:
    """Tests for extracting entities from TransformersNlpEngine."""
    
    def test_transformers_nlp_engine_with_user_mapping(self):
        """Test extracting entities from TransformersNlpEngine with NerModelConfiguration."""
        from presidio_evaluator.entity_mapping.interactive import _get_transformers_nlp_engine_entities
        
        # Mock NerModelConfiguration with user-provided mapping
        mock_config = Mock()
        mock_config.model_to_presidio_entity_mapping = {
            "PATIENT": "PERSON",
            "STAFF": "PERSON",
            "HOSPITAL": "ORGANIZATION",
            "AGE": "AGE",
            "DATE": "DATE_TIME"
        }
        
        # Mock TransformersNlpEngine
        mock_nlp_engine = Mock()
        mock_nlp_engine.ner_model_configuration = mock_config
        mock_nlp_engine._engines = {}
        
        entities = _get_transformers_nlp_engine_entities(mock_nlp_engine)
        
        # Should get unique values from the mapping
        assert set(entities) == {"PERSON", "ORGANIZATION", "AGE", "DATE_TIME"}
    
    def test_transformers_nlp_engine_from_model_config(self):
        """Test extracting entities from TransformersNlpEngine's underlying model."""
        from presidio_evaluator.entity_mapping.interactive import _get_transformers_nlp_engine_entities
        
        # Mock transformer model config
        mock_transformer_config = Mock()
        mock_transformer_config.id2label = {
            0: "O",
            1: "B-PATIENT",
            2: "I-PATIENT",
            3: "B-HOSPITAL",
            4: "I-HOSPITAL",
            5: "B-AGE",
        }
        mock_transformer_config.label2id = None
        
        mock_transformer_model = Mock()
        mock_transformer_model.config = mock_transformer_config
        
        mock_engine = Mock()
        mock_engine.model = mock_transformer_model
        
        # Mock TransformersNlpEngine
        mock_nlp_engine = Mock()
        mock_nlp_engine.ner_model_configuration = None
        mock_nlp_engine._engines = {"en": mock_engine}
        
        entities = _get_transformers_nlp_engine_entities(mock_nlp_engine)
        
        # Should extract and strip BIO prefixes
        assert set(entities) == {"PATIENT", "HOSPITAL", "AGE"}
    
    def test_transformers_nlp_engine_user_mapping_takes_precedence(self):
        """Test that user-provided mapping takes precedence over model config."""
        from presidio_evaluator.entity_mapping.interactive import _get_transformers_nlp_engine_entities
        
        # Mock user mapping (Presidio entity names)
        mock_config = Mock()
        mock_config.model_to_presidio_entity_mapping = {
            "PATIENT": "PERSON",
            "HOSPITAL": "LOCATION"  # User chose to map to LOCATION instead of ORG
        }
        
        # Mock transformer model config (original entity names)
        mock_transformer_config = Mock()
        mock_transformer_config.id2label = {
            0: "O",
            1: "B-PATIENT",
            2: "B-HOSPITAL",
            3: "B-ORGANIZATION"  # This won't be used
        }
        
        mock_transformer_model = Mock()
        mock_transformer_model.config = mock_transformer_config
        
        mock_engine = Mock()
        mock_engine.model = mock_transformer_model
        
        # Mock TransformersNlpEngine
        mock_nlp_engine = Mock()
        mock_nlp_engine.ner_model_configuration = mock_config
        mock_nlp_engine._engines = {"en": mock_engine}
        
        entities = _get_transformers_nlp_engine_entities(mock_nlp_engine)
        
        # Should use user's mapping, not model's original labels
        assert set(entities) == {"PERSON", "LOCATION"}
        assert "ORGANIZATION" not in entities  # From model config, but not in user mapping


class TestGetDatasetEntities:
    """Tests for get_dataset_entities function."""
    
    def test_with_loaded_dataset_with_counts(self):
        """Test with loaded dataset, returning counts."""
        samples = [
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
                full_text="Jane Smith",
                spans=[Span("PERSON", "Jane Smith", 0, 10)],
                masked="[PERSON]",
                tokens=["Jane", "Smith"],
                tags=["PERSON", "PERSON"],
            ),
        ]
        
        result = get_dataset_entities(samples, include_counts=True)
        
        assert result == {"PERSON": 2, "EMAIL": 1}
    
    def test_with_loaded_dataset_without_counts(self):
        """Test with loaded dataset, returning list."""
        samples = [
            InputSample(
                full_text="Test",
                spans=[
                    Span("PERSON", "Test", 0, 4),
                    Span("EMAIL", "test@test.com", 5, 18),
                ],
                masked="",
                tokens=["Test"],
                tags=["PERSON"],
            ),
        ]
        
        result = get_dataset_entities(samples, include_counts=False)
        
        assert set(result) == {"PERSON", "EMAIL"}
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        result = get_dataset_entities([], include_counts=True)
        assert result == {}
        
        result = get_dataset_entities([], include_counts=False)
        assert result == []


class TestSuggestMapping:
    """Tests for suggest_mapping function."""
    
    def test_basic_mapping(self):
        """Test basic entity mapping."""
        from presidio_evaluator.entity_mapping import create_presidio_mapper
        
        dataset_entities = ["FIRST_NAME", "EMAIL"]
        model_entities = ["PERSON", "CONTACT", "ID", "LOCATION"]
        
        mapping = suggest_mapping(
            dataset_entities,
            model_entities,
            mapper=create_presidio_mapper(),
            return_scores=False
        )
        
        assert mapping["FIRST_NAME"] == "PERSON"
        assert mapping["EMAIL"] == "CONTACT"
    
    def test_mapping_with_scores(self):
        """Test mapping that returns scores."""
        from presidio_evaluator.entity_mapping import SemanticEntityMapper
        
        dataset_entities = ["FIRST_NAME"]
        model_entities = ["PERSON", "LOCATION"]
        
        mapper = SemanticEntityMapper(threshold=0.3)
        mapping, scores = suggest_mapping(
            dataset_entities,
            model_entities,
            mapper=mapper,
            return_scores=True
        )
        
        assert "FIRST_NAME" in mapping
        assert "FIRST_NAME" in scores
        assert isinstance(scores["FIRST_NAME"], float)
    
    def test_presidio_mapper_confidence_scores(self):
        """Test that exact matches get 1.0, pattern matches get 0.6, semantic gets variable."""
        from presidio_evaluator.entity_mapping import create_presidio_mapper
        
        # Test exact matches - should get confidence 1.0
        exact_match_entities = [
            "PERSON",          # Exact match to target
            "EMAIL_ADDRESS",   # Exact match to target
            "CREDIT_CARD",     # Exact match to target
            "FIRST_NAME",      # In exact_mappings dict -> PERSON
            "US_SSN",          # Exact match to target
            "STREET_ADDRESS",  # In exact_mappings dict -> LOCATION
        ]
        model_entities = ["PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "LOCATION", "US_SSN"]
        
        mapper = create_presidio_mapper()
        mapping, scores = suggest_mapping(
            exact_match_entities,
            model_entities,
            mapper=mapper,
            return_scores=True
        )
        
        # All exact matches should have confidence 1.0
        for entity in exact_match_entities:
            assert entity in mapping, f"{entity} should be mapped"
            assert entity in scores, f"{entity} should have a score"
            assert scores[entity] == 1.0, f"{entity} (exact match) should have confidence 1.0, got {scores[entity]}"
    
    def test_unmapped_entities(self):
        """Test entities that don't map."""
        from presidio_evaluator.entity_mapping import DictEntityMapper
        
        dataset_entities = ["UNKNOWN_TYPE", "CUSTOM_ENTITY"]
        model_entities = ["PERSON", "LOCATION"]
        
        mapper = DictEntityMapper({})  # Empty mapping
        mapping = suggest_mapping(
            dataset_entities,
            model_entities,
            mapper=mapper,
            return_scores=False
        )
        
        assert mapping["UNKNOWN_TYPE"] is None
        assert mapping["CUSTOM_ENTITY"] is None


class TestSaveLoadMapping:
    """Tests for save and load mapping functions."""
    
    def test_save_and_load_mapping(self):
        """Test saving and loading a mapping."""
        mapping = {
            "FIRST_NAME": "PERSON",
            "EMAIL": "CONTACT",
            "SSN": "ID"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            save_mapping_to_file(mapping, temp_path)
            
            # Load
            loaded_mapping = load_mapping_from_file(temp_path)
            
            assert loaded_mapping == mapping
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_mapping_from_file("nonexistent_file.json")


class TestPrintMappingSummary:
    """Tests for print_mapping_summary function."""
    
    def test_print_mapping_summary_basic(self, capsys):
        """Test basic summary printing."""
        dataset_entities = ["FIRST_NAME", "EMAIL", "UNKNOWN"]
        model_entities = ["PERSON", "CONTACT"]
        mapping = {
            "FIRST_NAME": "PERSON",
            "EMAIL": "CONTACT",
            "UNKNOWN": None
        }
        
        print_mapping_summary(
            dataset_entities,
            model_entities,
            mapping
        )
        
        captured = capsys.readouterr()
        assert "ENTITY MAPPING SUMMARY" in captured.out
        assert "FIRST_NAME" in captured.out
        assert "EMAIL" in captured.out
        assert "UNKNOWN" in captured.out
        assert "PERSON" in captured.out
        assert "CONTACT" in captured.out
    
    def test_print_mapping_summary_with_counts(self, capsys):
        """Test summary printing with counts."""
        dataset_entities = {"FIRST_NAME": 100, "EMAIL": 50}
        model_entities = ["PERSON", "CONTACT"]
        mapping = {
            "FIRST_NAME": "PERSON",
            "EMAIL": "CONTACT",
        }
        
        print_mapping_summary(
            dataset_entities,
            model_entities,
            mapping
        )
        
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "50" in captured.out
    
    def test_print_mapping_summary_with_scores(self, capsys):
        """Test summary printing with scores."""
        dataset_entities = ["FIRST_NAME"]
        model_entities = ["PERSON"]
        mapping = {"FIRST_NAME": "PERSON"}
        scores = {"FIRST_NAME": 0.95}
        
        print_mapping_summary(
            dataset_entities,
            model_entities,
            mapping,
            scores
        )
        
        captured = capsys.readouterr()
        assert "0.95" in captured.out
