import pytest

from presidio_evaluator.data_generator.faker_extensions import SentenceFaker


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
    res = data_generator.parse("My name is {{name}} and I live in {{city}}", 0)

    expected = f"My name is {record['name']} and I live in {record['city']}"
    assert res.full_text == expected.lower()
    for span in res.spans:
        value = record[span.entity_type].lower()
        assert span.entity_value == value
        assert res.full_text[span.start_position : span.end_position] == value
