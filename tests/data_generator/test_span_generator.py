import pytest
from faker import Faker
from faker.providers import BaseProvider

from presidio_evaluator import InputSample, Span
from presidio_evaluator.data_generator.faker_extensions import SpanGenerator


@pytest.fixture(scope="session")
def test_provider():
    class TestProvider(BaseProvider):
        def foo(self) -> str:
            return "bar"

        def foofoofoo(self) -> str:
            return "bar"

        def foo2(self) -> str:
            return "barbar"

        def foo3(self) -> str:
            return "barbarbar"

    return TestProvider


@pytest.fixture(scope="session")
def span_faker(test_provider):
    generator = SpanGenerator()
    Faker.seed(42)
    faker = Faker(generator=generator)
    faker.add_provider(test_provider)

    return faker


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("My name is {{foo}}", "My name is bar"),
        ("My name is {{  foo   }}", "My name is bar"),
        ("my name is {{foofoofoo}}", "my name is bar"),
        ("my name is {{FOO}}", "my name is bar"),
    ],
)
def test_one_replacement(span_faker, pattern, expected):
    res = span_faker.parse(pattern, add_spans=True)

    assert res.full_text == expected
    assert res.masked == pattern
    assert res.spans[0].start_position == expected.index("bar")
    assert res.spans[0].end_position == len(expected)
    assert res.spans[0].entity_value == "bar"


def test_multiple_replacements(span_faker):
    pattern = "{{foo}} and then {{foo2}}, {{foofoofoo}} and finally {{foo3}}."
    expected = "bar and then barbar, bar and finally barbarbar."
    expected_spans = [
        Span(entity_value="bar", start_position=0, end_position=3, entity_type="foo"),
        Span(
            entity_value="barbar",
            start_position=13,
            end_position=19,
            entity_type="foo2",
        ),
        Span(
            entity_value="bar",
            start_position=21,
            end_position=24,
            entity_type="foofoofoo",
        ),
        Span(
            entity_value="barbarbar",
            start_position=37,
            end_position=46,
            entity_type="foo3",
        ),
    ]

    res = span_faker.parse(pattern, add_spans=True)

    actual_spans = sorted(res.spans, key=lambda x: x.start_position)

    assert res.full_text == expected
    assert res.masked == pattern
    for expected, actual in zip(expected_spans, actual_spans):
        assert expected == actual


def test_spans_result_repr():
    sr = InputSample(
        full_text="momo",
        spans=[
            Span(
                entity_value="momo",
                start_position=0,
                end_position=4,
                entity_type="name",
            )
        ],
    )
    expected = """Full text: momo
Spans: [Span(type: name, value: momo, char_span: [0: 4])]
"""

    assert sr.__repr__() == expected


def test_no_replacements(span_faker):
    pattern = "this is a sentence with no fields"

    res = span_faker.parse(pattern, add_spans=True)

    assert res.full_text == pattern
    assert len(res.spans) == 0


def test_without_spans(span_faker):
    pattern = "this is a sentence with {{foo}}"
    expected = "this is a sentence with bar"
    res = span_faker.parse(pattern)

    assert type(res) is str
    assert res == expected


def test_generated_text_contains_spans_text(span_faker):
    pattern = "My name is {{name}} and i live in {{address}}."

    res = span_faker.parse(pattern, add_spans=True)

    for span in res.spans:
        assert span.entity_value in res.full_text


@pytest.mark.parametrize(
    "pattern, non_element_text",
    [
        ("{{name}} My name is {{name}}", " My name is "),
        ("a b {{name}}{{name}}{{name}}", "a b "),
        ("...{{name}}{{name}} {{name}}...", "... ..."),
    ],
)
def test_generated_text_duplicate_types_returns_different_results(
    span_faker, pattern, non_element_text
):
    res = span_faker.parse(pattern, add_spans=True)

    # assert that span values exist in the text
    for span in res.spans:
        assert span.entity_value in res.full_text

    # assert that the non-element text is identical
    substring_indices = list(range(len(res.full_text)))
    for span in res.spans:
        substring_indices = [
            ind
            for ind in substring_indices
            if ind not in range(span.start_position, span.end_position)
        ]

    actual_non_element_text = "".join(
        [res.full_text[i] for i in range(len(res.full_text)) if i in substring_indices]
    )
    assert actual_non_element_text == non_element_text

    # assert that names are different from each other
    for i in range(len(res.spans)):
        for j in range(i + 1, len(res.spans)):
            assert res.spans[i].entity_value != res.spans[j].entity_value
            assert res.spans[i].start_position != res.spans[j].start_position
            assert res.spans[i].end_position != res.spans[j].end_position
