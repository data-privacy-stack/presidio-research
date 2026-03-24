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
