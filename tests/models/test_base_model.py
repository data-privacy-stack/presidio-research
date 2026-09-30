import pandas as pd
import pytest

from presidio_evaluator import InputSample, tags_to_span_ids
from presidio_evaluator.entity_mapping.data_objects import (
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
)
from tests.mocks import MockModel, MockTokensModel


@pytest.fixture(scope="session")
def mock_model():
    return MockModel(entities_to_keep=["name"])


# test_align_entity_types and test_align_prediction removed
# These methods have been deprecated and removed from BaseModel.
# Entity mapping is now handled by the evaluator.


def test_to_log(mock_model):
    log_dict = mock_model.to_log()

    assert log_dict["labeling_scheme"] == mock_model.labeling_scheme
    assert log_dict["entities_to_keep"] == mock_model.entities


# ── predict_dataset() tests ──────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "sentence_id",
    "token",
    "annotation",
    "prediction",
    "start_indices",
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
]


def _make_sample(tokens, tags, start_indices, sample_id=None):
    """Build an InputSample with pre-set tokens/tags (no spaCy tokenisation)."""
    sample = InputSample(full_text=" ".join(tokens))
    sample.tokens = tokens
    sample.tags = tags
    sample.start_indices = start_indices
    sample.sample_id = sample_id
    return sample


def test_predict_dataset_schema():
    """predict_dataset() returns a DataFrame with exactly the 7 required columns."""
    tokens = ["Hello", "John"]
    tags = ["O", "PERSON"]
    predictions = ["O", "PERSON"]
    start_indices = [0, 6]

    model = MockTokensModel(prediction=predictions)
    sample = _make_sample(tokens, tags, start_indices, sample_id=0)

    df = model.predict_dataset([sample])

    assert list(df.columns) == EXPECTED_COLUMNS
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_predict_dataset_values():
    """predict_dataset() assembles token/annotation/prediction values correctly."""
    tokens = ["Alice", "lives", "in", "Paris"]
    annotations = ["PERSON", "O", "O", "LOCATION"]
    predictions = ["PERSON", "O", "O", "GPE"]
    start_indices = [0, 6, 12, 15]

    model = MockTokensModel(prediction=predictions)
    sample = _make_sample(tokens, annotations, start_indices, sample_id=42)

    df = model.predict_dataset([sample])

    assert list(df["token"]) == tokens
    assert list(df["annotation"]) == annotations
    assert list(df["prediction"]) == predictions
    assert list(df["start_indices"]) == start_indices
    # sentence_id should come from sample_id
    assert all(df["sentence_id"] == 42)


def test_predict_dataset_uses_index_when_no_sample_id():
    """When sample_id is None, predict_dataset() falls back to the loop index."""
    tokens = ["foo"]
    model = MockTokensModel(prediction=["O"])
    sample = _make_sample(tokens, ["O"], [0], sample_id=None)

    df = model.predict_dataset([sample])

    assert df["sentence_id"].iloc[0] == 0


def test_predict_dataset_no_entity_mapping():
    """predict_dataset() must NOT remap entity names — raw predictions pass through."""
    # Model predicts "FIRST_NAME"; no mapping should change it.
    tokens = ["Bob"]
    annotations = ["PERSON"]
    raw_prediction = ["FIRST_NAME"]

    model = MockTokensModel(prediction=raw_prediction)
    sample = _make_sample(tokens, annotations, [0], sample_id=1)

    df = model.predict_dataset([sample])

    assert df["prediction"].iloc[0] == "FIRST_NAME"
    assert df["annotation"].iloc[0] == "PERSON"


def test_predict_dataset_multi_sample():
    """predict_dataset() handles multiple samples, assigning correct sentence_ids."""
    samples = [
        _make_sample(["Alice"], ["PERSON"], [0], sample_id=10),
        _make_sample(["Bob", "Smith"], ["PERSON", "PERSON"], [0, 4], sample_id=11),
    ]
    model = MockTokensModel(prediction=["O"])

    df = model.predict_dataset(samples)

    assert len(df) == 3  # 1 token + 2 tokens
    assert list(df["sentence_id"]) == [10, 11, 11]


# ── span-id columns ──────────────────────────────────────────────────────────


def _make_sample_with_span_ids(tokens, tags, start_indices, span_ids, sample_id=None):
    """Sample whose tags were created from spans: span_to_tag set span_ids."""
    sample = _make_sample(tokens, tags, start_indices, sample_id=sample_id)
    sample.span_ids = span_ids
    return sample


def test_predict_dataset_attaches_annotation_span_ids():
    """Each token carries the index of the gold span covering it, None for O."""
    tokens = ["Ana", "Ruiz", "29", "here"]
    tags = ["NAME", "NAME", "AGE", "O"]
    start_indices = [0, 4, 9, 12]
    span_ids = [0, 0, 1, None]

    model = MockTokensModel(prediction=["O"] * 4)
    sample = _make_sample_with_span_ids(
        tokens, tags, start_indices, span_ids, sample_id=0
    )

    df = model.predict_dataset([sample])

    assert list(df.columns) == EXPECTED_COLUMNS
    assert list(df[ANNOTATION_SPAN_ID]) == [0, 0, 1, None]
    assert list(df[PREDICTION_SPAN_ID]) == [None] * 4


def test_predict_dataset_prediction_span_ids_from_io_tags():
    """A tag-only model gets label-run ids: touching same-type spans read as one."""
    tokens = ["Ana", "Ruiz", "29", "here"]
    sample = _make_sample(tokens, ["O"] * 4, [0, 4, 9, 12], sample_id=0)
    model = MockTokensModel(prediction=["NAME", "NAME", "AGE", "O"])

    df = model.predict_dataset([sample])

    assert list(df["prediction"]) == ["NAME", "NAME", "AGE", "O"]
    assert list(df[PREDICTION_SPAN_ID]) == [0, 0, 1, None]


def test_predict_dataset_prediction_span_ids_from_bio_tags():
    """BIO prefixes give exact prediction boundaries; tags are flattened to IO."""
    tokens = ["Ana", "Ruiz", "Bob"]
    sample = _make_sample(tokens, ["O"] * 3, [0, 4, 9], sample_id=0)
    model = MockTokensModel(prediction=["B-NAME", "I-NAME", "B-NAME"])

    df = model.predict_dataset([sample])

    assert list(df["prediction"]) == ["NAME", "NAME", "NAME"]
    assert list(df[PREDICTION_SPAN_ID]) == [0, 0, 1]


def test_predict_dataset_span_ids_from_directly_tagged_sample():
    """A sample built from tags gets label-run gold ids."""
    model = MockTokensModel(prediction=["O"])
    sample = _make_sample(["foo"], ["PERSON"], [0], sample_id=0)
    sample.span_ids = tags_to_span_ids(sample.tags)

    df = model.predict_dataset([sample])

    assert list(df.columns) == EXPECTED_COLUMNS
    assert list(df[ANNOTATION_SPAN_ID]) == [0]


def test_predict_dataset_span_ids_none_for_idless_sample_in_mixed_dataset():
    """A sample without span ids gets None when others in the dataset have them."""
    with_ids = _make_sample_with_span_ids(["Bob"], ["PERSON"], [0], [0], sample_id=0)
    without_ids = _make_sample(["Ann"], ["PERSON"], [0], sample_id=1)

    model = MockTokensModel(prediction=["O"])
    df = model.predict_dataset([with_ids, without_ids])

    assert list(df[ANNOTATION_SPAN_ID]) == [0, None]


def test_predict_dataset_span_ids_stay_ints():
    """int + None must not be coerced to float64 (0.0, NaN) on construction."""
    tokens = ["Ana", "Ruiz", "x"]
    tags = ["NAME", "NAME", "O"]
    sample = _make_sample_with_span_ids(tokens, tags, [0, 4, 9], [0, 0, None])
    df = MockTokensModel(prediction=["NAME", "O", "O"]).predict_dataset([sample])
    for column in (ANNOTATION_SPAN_ID, PREDICTION_SPAN_ID):
        values = list(df[column])
        assert all(type(v) is int for v in values if v is not None), column
    assert list(df[ANNOTATION_SPAN_ID]) == [0, 0, None]
    assert list(df[PREDICTION_SPAN_ID]) == [0, None, None]


def test_tags_to_spans_uses_character_offsets():
    """Default batch_predict_spans turns tag runs into character spans."""
    sample = _make_sample(["Ana", "Ruiz", ",", "29"], ["O"] * 4, [0, 4, 9, 11])
    sample.full_text = "Ana Ruiz , 29"
    model = MockTokensModel(prediction=["NAME", "NAME", "O", "AGE"])

    [spans] = model.batch_predict_spans([sample])

    assert [(s.entity_type, s.start_position, s.end_position) for s in spans] == [
        ("NAME", 0, 8),
        ("AGE", 11, 13),
    ]
    assert [s.entity_value for s in spans] == ["Ana Ruiz", "29"]
