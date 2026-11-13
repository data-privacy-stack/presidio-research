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


def _cities(sentence_faker: PresidioSentenceFaker, num_samples: int) -> set[str]:
    samples = sentence_faker.generate_new_fake_sentences(num_samples)
    return {sample.spans[0].entity_value for sample in samples}


def test_match_records_to_locale_uses_only_records_from_that_country():
    records = load_fake_person_df()
    german_cities = set(records.loc[records["country_code"] == "DE", "city"])

    sentence_faker = PresidioSentenceFaker(
        locale="de_DE",
        lower_case_ratio=0,
        sentence_templates=["Bitte die Rechnung nach {{city}} schicken!"],
        random_seed=42,
        match_records_to_locale=True,
    )

    assert _cities(sentence_faker, 100) <= german_cities


def test_records_are_not_matched_to_locale_by_default():
    records = load_fake_person_df()
    german_cities = set(records.loc[records["country_code"] == "DE", "city"])

    sentence_faker = PresidioSentenceFaker(
        locale="de_DE",
        lower_case_ratio=0,
        sentence_templates=["Bitte die Rechnung nach {{city}} schicken!"],
        random_seed=42,
    )

    assert not _cities(sentence_faker, 100) <= german_cities


@pytest.mark.parametrize(
    "locale, base_records, error",
    [
        ("de", None, "has no country part"),
        ("ja_JP", None, "No base records have country_code 'JP'"),
        ("de_DE", [{"city": "Berlin"}], "needs a country_code field"),
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
