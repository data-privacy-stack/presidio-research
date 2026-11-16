import pytest
from faker import Faker

from presidio_evaluator import InputSample, Span
from presidio_evaluator.data_generator.faker_extensions import (
    RecordGenerator,
    SentenceFaker,
)
from presidio_evaluator.data_generator.faker_extensions.sentences import (
    lower_case_sample,
)
from presidio_evaluator.data_generator.presidio_data_generator import (
    PresidioDataGenerator,
)

TEMPLATE = "My name is {{name}} and I live in {{city}}"


def _assert_spans_aligned(sample):
    for span in sample.spans:
        assert span.entity_value == span.entity_value.lower()
        assert sample.full_text[span.start_position : span.end_position] == (
            span.entity_value
        )


def test_new_provider_no_alias_raises_attribute_error():
    data_generator = SentenceFaker(lower_case_ratio=0.0)
    with pytest.raises(AttributeError):
        data_generator.parse("My doctor is {{doc_name}}", 0)


def test_new_provider_with_alias():
    data_generator = SentenceFaker(lower_case_ratio=0.0)
    data_generator.add_provider_alias("name", "doc_name")
    start_of_sentence = "My doctor is "
    res = data_generator.parse(
        template=f"{start_of_sentence}{{{{doc_name}}}}", template_id=0
    )
    assert res
    assert len(res.full_text) > len(start_of_sentence)
    assert start_of_sentence in res.full_text


@pytest.mark.parametrize(
    "record",
    [
        {"name": "John Smith", "city": "Boston"},
        # "İ".lower() is two characters long, which shifts the later span
        {"name": "İsmail Yılmaz", "city": "İzmir"},
    ],
)
def test_lower_case_ratio_lowers_text_and_keeps_spans_aligned(record):
    data_generator = SentenceFaker(lower_case_ratio=1.0, records=[record])
    res = data_generator.parse(TEMPLATE, 0)

    # the text around and inside the spans is lowered piece by piece
    name, city = record["name"].lower(), record["city"].lower()
    assert res.full_text == f"my name is {name} and i live in {city}"
    assert res.masked == "my name is {{name}} and i live in {{city}}"
    for span in res.spans:
        assert span.entity_value == record[span.entity_type].lower()
    _assert_spans_aligned(res)


def test_lower_case_ratio_without_records():
    res = SentenceFaker(lower_case_ratio=1.0).parse(TEMPLATE, 0)

    assert res.full_text.startswith("my name is ")
    assert res.full_text == res.full_text.lower()
    assert res.masked == "my name is {{name}} and i live in {{city}}"
    assert len(res.spans) == 2
    _assert_spans_aligned(res)


def test_lower_case_ratio_without_spans():
    res = SentenceFaker(lower_case_ratio=1.0).parse(TEMPLATE, 0, add_spans=False)

    assert isinstance(res, str)
    assert res.startswith("my name is ")
    assert res == res.lower()


def test_presidio_data_generator_lower_case_ratio_keeps_the_sample():
    records = [{"name": "İsmail Yılmaz", "city": "İzmir"}]
    faker = Faker(generator=RecordGenerator(records=records))
    with pytest.warns(DeprecationWarning):
        data_generator = PresidioDataGenerator(custom_faker=faker, lower_case_ratio=1.0)
    res = data_generator.parse(TEMPLATE, 0)

    assert res.full_text == "my name is i̇smail yılmaz and i live in i̇zmir"
    _assert_spans_aligned(res)


def test_lower_case_sample_rejects_overlapping_spans():
    sample = InputSample(
        full_text="Hello World",
        spans=[Span("A", "Hello World", 0, 11), Span("B", "World", 6, 11)],
    )
    with pytest.raises(ValueError, match="overlapping"):
        lower_case_sample(sample)
