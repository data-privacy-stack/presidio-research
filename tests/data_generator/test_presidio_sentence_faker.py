import inspect
from copy import deepcopy

import pytest
from faker import Faker

from presidio_evaluator.data_generator import PresidioSentenceFaker
from presidio_evaluator.data_generator.faker_extensions import providers
from presidio_evaluator.data_generator.faker_extensions.datasets import (
    load_fake_person_df,
)


def _get_classes_from_module(module):
    return [
        member[1]
        for member in inspect.getmembers(module, inspect.isclass)
        if member[1].__module__ == module.__name__
    ]


@pytest.mark.parametrize("num_sentences", (1500, 3000))
def test_generate_new_fake_sentences(num_sentences: int):
    standard_faker = Faker()
    default_faker_providers = standard_faker.providers
    presidio_providers = _get_classes_from_module(providers)

    sentence_faker = PresidioSentenceFaker(locale="en", lower_case_ratio=0)
    assert len(sentence_faker._sentence_templates) > 0, (
        "Did not load default sentence templates"
    )

    expected_providers = deepcopy(default_faker_providers)
    expected_providers.extend(presidio_providers)
    expected_providers.extend(
        [
            standard_faker.__getattr__(alias[0])
            for alias in PresidioSentenceFaker.PROVIDER_ALIASES
        ]
    )
    actual_providers = sentence_faker._sentence_faker.providers
    num_aliases = len(PresidioSentenceFaker.PROVIDER_ALIASES)
    actual_num_providers = len(actual_providers)
    expected_aliases = {
        getattr(standard_faker, provider_name[0])
        for provider_name in PresidioSentenceFaker.PROVIDER_ALIASES
    }
    assert actual_num_providers == len(expected_providers), (
        f"Expected {len(presidio_providers)} presidio providers to be used and {num_aliases} aliases. "
        f"Faker has been extended with {actual_num_providers - len(default_faker_providers)} providers/aliases. "
        f"Expected Providers: {[provider.__name__ for provider in presidio_providers]} "
        f"Expected Aliases: {expected_aliases} "
    )

    fake_sentence_results = sentence_faker.generate_new_fake_sentences(num_sentences)
    assert len(fake_sentence_results) == num_sentences
    for fake_sentence_result in fake_sentence_results:
        assert fake_sentence_result.full_text
        assert fake_sentence_result.masked
        assert fake_sentence_result.template_id >= 0


@pytest.mark.parametrize(
    "template_before, template_after",
    [
        (
            "I just moved to {{CiTY}} from {{Country}}",
            "I just moved to {{city}} from {{country}}",
        ),
        (
            "I just moved to <city> from <country>.",
            "I just moved to {{city}} from {{country}}.",
        ),
    ],
)
def test_preprocess_template(template_before: str, template_after: str):
    sentence_faker = PresidioSentenceFaker(locale="en", lower_case_ratio=0)
    assert sentence_faker._preprocess_template(template_before) == template_after


def _record_pool(sentence_faker: PresidioSentenceFaker) -> list[dict]:
    return sentence_faker._sentence_faker.factories[0].records


def test_match_records_to_locale_uses_only_records_from_that_country():
    records = load_fake_person_df()

    sentence_faker = PresidioSentenceFaker(
        locale="de_DE", lower_case_ratio=0, match_records_to_locale=True
    )

    pool = _record_pool(sentence_faker)
    assert len(pool) == (records["country_code"] == "DE").sum()
    assert {record["country_code"] for record in pool} == {"DE"}


def test_records_are_not_matched_to_locale_by_default():
    sentence_faker = PresidioSentenceFaker(locale="de_DE", lower_case_ratio=0)

    assert len(_record_pool(sentence_faker)) == len(load_fake_person_df())


def test_match_records_to_locale_fills_empty_fields():
    # No GB record has a state, and some have no mothers_maiden,
    # which Faker has no provider for.
    sentence_faker = PresidioSentenceFaker(
        locale="en_GB",
        lower_case_ratio=0,
        sentence_templates=[
            "I live in {{city}}, {{state}} {{state_abbr}}. My mother was {{mothers_maiden}}"
        ],
        random_seed=42,
        match_records_to_locale=True,
    )

    pool = _record_pool(sentence_faker)
    assert all("mothers_maiden" in record for record in pool)
    assert not any("state" in record for record in pool)

    samples = sentence_faker.generate_new_fake_sentences(200)
    values = [span.entity_value for sample in samples for span in sample.spans]
    assert "nan" not in values


@pytest.mark.parametrize(
    "locale, country",
    [
        ("de_DE", "DE"),
        ("de-AT", "AT"),
        ("sr_Latn_RS", "RS"),
        ("en_US.UTF-8", "US"),
        ("es_CA", "ES"),
        ("fr_QC", "CA"),
        ("en_MS", "MY"),
    ],
)
def test_locale_country(locale, country):
    assert PresidioSentenceFaker._locale_country(locale) == country


@pytest.mark.parametrize("locale", [["de_DE"], {"de_DE": 1.0}])
def test_match_records_to_locale_accepts_locale_list_and_dict(locale):
    sentence_faker = PresidioSentenceFaker(
        locale=locale, lower_case_ratio=0, match_records_to_locale=True
    )

    assert {record["country_code"] for record in _record_pool(sentence_faker)} == {"DE"}


def test_match_records_to_locale_ignores_case_and_spaces_in_country_code():
    sentence_faker = PresidioSentenceFaker(
        locale="de_DE",
        lower_case_ratio=0,
        base_records=[
            {"country_code": " de", "city": "Berlin"},
            {"country_code": "FR", "city": "Paris"},
        ],
        match_records_to_locale=True,
    )

    assert _record_pool(sentence_faker) == [{"country_code": " de", "city": "Berlin"}]


@pytest.mark.parametrize(
    "locale, base_records, error",
    [
        ("de", None, "has no country"),
        ("ar_AA", None, "has no country"),
        ("ja_JP", None, "No base records have country_code 'JP'"),
        ("de_DE", [], "got no base records"),
        ("de_DE", [{"city": "Berlin"}], "needs a country_code field"),
        (
            "de_DE",
            [{"country_code": "FR"}, {"country_code": float("nan")}],
            "Available: FR$",
        ),
    ],
)
def test_match_records_to_locale_errors(locale, base_records, error):
    with pytest.raises(ValueError, match=error):
        PresidioSentenceFaker(
            locale=locale,
            lower_case_ratio=0,
            base_records=base_records,
            match_records_to_locale=True,
        )
