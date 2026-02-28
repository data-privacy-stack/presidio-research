import pytest

from presidio_evaluator import InputSample, Span
from tests.mocks import MockModel


@pytest.fixture(scope="session")
def mock_model():
    return MockModel(entities_to_keep=["name"])


# test_align_entity_types and test_align_prediction removed
# These methods have been deprecated and removed from BaseModel.
# Entity mapping is now handled by the evaluator.


@pytest.mark.parametrize(
    "tags, expected_tags",
    [
        (["O", "O"], ["O", "O"]),
        (["name", "O"], ["name", "O"]),
        (["O", "credit_card"], ["O", "O"]),
    ],
)
def test_filter_tags_in_supported_entities(mock_model, tags, expected_tags):
    actual_tags = mock_model.filter_tags_in_supported_entities(tags=tags)
    assert actual_tags == expected_tags


@pytest.mark.parametrize(
    "tags, expected_tags, scheme",
    [
        (["O", "O"], ["O", "O"], "BILUO"),
        (["B-name", "I-name", "L-name"], ["B-name", "I-name", "I-name"], "BIO"),
        (["B-name", "I-name", "I-name"], ["B-name", "I-name", "L-name"], "BILUO"),
    ],
)
def test_to_scheme(mock_model, tags, expected_tags, scheme):
    mock_model.labeling_scheme = scheme
    actual_tags = mock_model.to_scheme(tags=tags)
    assert actual_tags == expected_tags


def test_to_log(mock_model):
    log_dict = mock_model.to_log()

    assert log_dict['labeling_scheme'] == mock_model.labeling_scheme
    assert log_dict['entities_to_keep'] == mock_model.entities
