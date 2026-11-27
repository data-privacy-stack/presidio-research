import pytest
from faker.providers import DynamicProvider
from presidio_analyzer import RecognizerResult

from presidio_evaluator.data_generator import PresidioPseudonymization


@pytest.fixture(scope="session")
def faker_providers():
    person_provider = DynamicProvider("person", ["James"])
    location_provider = DynamicProvider("location", ["Africa"])

    return [person_provider, location_provider]


@pytest.mark.parametrize(
    # fmt: off
    "text, entity1, entity2, start1, end1, start2, end2, value1, value2",
    [
        (
            "Hi I live in South Africa and my name is Toma",
            "location",
            "person",
            13,
            25,
            41,
            45,
            "Africa",
            "James",
        ),
        (
            "Africa is my continent, James",
            "location",
            "person",
            0,
            6,
            24,
            29,
            "Africa",
            "James",
        ),
    ],
    # fmt: on
)
def test_presidio_pseudonymize_two_entities(
    text, entity1, entity2, start1, end1, start2, end2, value1, value2, faker_providers
):
    presidio_response = [
        RecognizerResult(entity_type=entity1, start=start1, end=end1, score=0.85),
        RecognizerResult(entity_type=entity2, start=start2, end=end2, score=0.85),
    ]
    presidio_pseudonymizer = PresidioPseudonymization(
        lower_case_ratio=0.0, map_to_presidio_entities=False
    )
    presidio_pseudonymizer.add_provider(faker_providers[0])
    presidio_pseudonymizer.add_provider(faker_providers[1])
    pseudonyms = presidio_pseudonymizer.pseudonymize(
        original_text=text, presidio_response=presidio_response, count=5
    )

    assert len(pseudonyms) == 5
    for pseudonym in pseudonyms:
        assert value1 in pseudonym
        assert value2 in pseudonym
        assert text[:start1].lower() in pseudonym.lower()
        assert text[end1:start2].lower() in pseudonym.lower()


def test_presidio_pseudonymize_presidio_entity_names():
    text = "My name is David Johnson and my email is david.johnson@example.com"
    presidio_response = [
        RecognizerResult(entity_type="PERSON", start=11, end=24, score=0.85),
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=41, end=66, score=1.0),
    ]
    presidio_pseudonymizer = PresidioPseudonymization(lower_case_ratio=0.0)
    pseudonyms = presidio_pseudonymizer.pseudonymize(
        original_text=text, presidio_response=presidio_response, count=5
    )

    assert len(pseudonyms) == 5
    for pseudonym in pseudonyms:
        assert pseudonym.startswith("My name is ")
        assert " and my email is " in pseudonym
        assert "David Johnson" not in pseudonym
        assert "david.johnson@example.com" not in pseudonym
        assert "{{" not in pseudonym


def test_presidio_pseudonymize_keeps_angle_brackets_in_text(faker_providers):
    text = "From: Toma <toma@example.com>"
    presidio_response = [
        RecognizerResult(entity_type="person", start=6, end=10, score=0.85),
        RecognizerResult(entity_type="email", start=12, end=28, score=1.0),
    ]
    presidio_pseudonymizer = PresidioPseudonymization(
        lower_case_ratio=0.0, map_to_presidio_entities=False
    )
    presidio_pseudonymizer.add_provider(faker_providers[0])
    pseudonyms = presidio_pseudonymizer.pseudonymize(
        original_text=text, presidio_response=presidio_response, count=5
    )

    for pseudonym in pseudonyms:
        assert pseudonym.startswith("From: James <")
        assert pseudonym.endswith(">")
        assert "toma@example.com" not in pseudonym
        assert "{" not in pseudonym
        assert "}" not in pseudonym
