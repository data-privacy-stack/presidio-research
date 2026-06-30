import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider
from faker.typing import SeedType
from tqdm import tqdm

from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import raw_data_dir
from presidio_evaluator.data_generator.faker_extensions import (
    AddressProviderNew,
    AgeProvider,
    HospitalProvider,
    IpAddressProvider,
    NationalityProvider,
    OrganizationProvider,
    PhoneNumberProviderNew,
    ReligionProvider,
    UsDriverLicenseProvider,
)
from presidio_evaluator.data_generator.faker_extensions.datasets import (
    load_fake_person_df,
)
from presidio_evaluator.data_generator.faker_extensions.sentences import SentenceFaker

presidio_templates_file_path = raw_data_dir / "templates.txt"
presidio_additional_entity_providers = [
    IpAddressProvider,
    NationalityProvider,
    OrganizationProvider,
    UsDriverLicenseProvider,
    AgeProvider,
    AddressProviderNew,
    PhoneNumberProviderNew,
    ReligionProvider,
    HospitalProvider,
]


class PresidioSentenceFaker:
    """
    A high level interface for generating fake sentences with entity metadata.
    By default, this leverages all the existing templates and additional providers in this library.
    :param: locale: The faker locale to use e.g. 'en_US'
    :param lower_case_ratio: Percentage of names that should start with lower case
    :param: sentence_templates: Defaults to presidio_templates_file_path, a provided argument overrides this
    :param: entity_providers: Defaults to presidio_additional_entity_providers, a provided argument overrides this
    :param: base_records: A DataFrame with entity types as columns and each row corresponding to a fake individual.
    Defaults to presidio_evaluator.data_generator.faker_extensions.datasets.load_fake_person_df()
    :param: entity_type_mapping: A dictionary mapping entity types to Presidio entity types
    :param: provider_aliases: A dictionary mapping provider names to the given entity types.
    Useful if the templates contain a different name for the entity type than the one supported by Faker or PresidioSentenceFaker.
    :param: random_seed: A seed to make results reproducible between runs
    """

    PROVIDER_ALIASES = [
        ("name", "person"),
        ("credit_card_number", "credit_card"),
        ("date_of_birth", "birthday"),
    ]
    ENTITY_TYPE_MAPPING = {
        "person": "PERSON",
        "ip_address": "IP_ADDRESS",
        "us_driver_license": "US_DRIVER_LICENSE",
        "organization": "ORGANIZATION",
        "name_female": "PERSON",
        "address": "STREET_ADDRESS",
        "country": "GPE",
        "state": "GPE",
        "credit_card_number": "CREDIT_CARD",
        "city": "GPE",
        "street_name": "STREET_ADDRESS",
        "building_number": "STREET_ADDRESS",
        "name": "PERSON",
        "iban": "IBAN_CODE",
        "last_name": "PERSON",
        "last_name_male": "PERSON",
        "last_name_female": "PERSON",
        "first_name": "PERSON",
        "first_name_male": "PERSON",
        "first_name_female": "PERSON",
        "phone_number": "PHONE_NUMBER",
        "url": "DOMAIN_NAME",
        "ssn": "US_SSN",
        "email": "EMAIL_ADDRESS",
        "date_time": "DATE_TIME",
        "date_of_birth": "DATE_TIME",
        "day_of_week": "DATE_TIME",
        "year": "DATE_TIME",
        "name_male": "PERSON",
        "prefix_male": "TITLE",
        "prefix_female": "TITLE",
        "prefix": "TITLE",
        "nationality": "NRP",
        "nation_woman": "NRP",
        "nation_man": "NRP",
        "nation_plural": "NRP",
        "first_name_nonbinary": "PERSON",
        "postcode": "STREET_ADDRESS",
        "secondary_address": "STREET_ADDRESS",
        "job": "TITLE",
        "zipcode": "ZIP_CODE",
        "state_abbr": "GPE",
        "age": "AGE",
    }

    def __init__(
        self,
        locale: str,
        lower_case_ratio: float,
        sentence_templates: list[str] | None = None,
        entity_providers: list[BaseProvider] | None = None,
        base_records: pd.DataFrame | list[dict] | None = None,
        entity_type_mapping: dict[str, str] | None = None,
        provider_aliases: list[tuple[str, str]] | None = None,
        random_seed: SeedType | None = None,
    ) -> None:
        self._sentence_templates = sentence_templates
        if not self._sentence_templates:
            self._sentence_templates = [
                line.strip()
                for line in presidio_templates_file_path.read_text().splitlines()
            ]
        if entity_providers is None:
            print("Using default entity providers")
            entity_providers = presidio_additional_entity_providers
        if base_records is None:
            base_records = load_fake_person_df()

        self._sentence_faker = SentenceFaker(
            records=base_records,
            locale=locale,
            lower_case_ratio=lower_case_ratio,
        )
        for entity_provider in entity_providers:
            self._sentence_faker.add_provider(entity_provider)

        self.seed(random_seed)

        if not entity_type_mapping:
            print(
                "Using default entity mapping between the entities "
                "in the templates and the ones in the output dataset",
            )
            entity_type_mapping = self.ENTITY_TYPE_MAPPING

        self._entity_type_mapping = entity_type_mapping

        if not provider_aliases:
            print("Using default provider aliases")
            provider_aliases = self.PROVIDER_ALIASES

        for provider, alias in provider_aliases:
            self._sentence_faker.add_provider_alias(
                provider_name=provider,
                new_name=alias,
            )
        self.fake_sentence_results = None

    def generate_new_fake_sentences(self, num_samples: int) -> list[InputSample]:
        """Generate fake sentences based on the templates, input data and entity providers."""
        self.fake_sentence_results = []
        # Map faker generated entity types to Presidio entity types
        for _ in tqdm(range(num_samples), desc="Sampling"):
            template_id = random.choice(range(len(self._sentence_templates)))
            template = self._sentence_templates[template_id]
            template = self._preprocess_template(template)
            fake_sentence_result = self._sentence_faker.parse(template, template_id)
            for span in fake_sentence_result.spans:
                if span.entity_type in self._entity_type_mapping.keys():
                    # Use the mapped entity type if exists
                    span.entity_type = self._entity_type_mapping[span.entity_type]
                else:
                    # Otherwise, capitalize the entity type and add to the mapping
                    print(
                        f"Warning: Non-mapped entity type found: {span.entity_type}. "
                        f"Non-mapped entities will be mapped to {span.entity_type.upper()} "
                        f"in the output dataset. If you prefer a different mapping, "
                        f"pass the `entity_type_mapping` argument with a mapping for this entity type.",
                    )
                    self._entity_type_mapping[span.entity_type] = (
                        span.entity_type.upper()
                    )
            for key, value in self._entity_type_mapping.items():
                fake_sentence_result.masked = fake_sentence_result.masked.replace(
                    f"{{{{{key}}}}}",
                    f"{{{{{value}}}}}",
                )
            self.fake_sentence_results.append(fake_sentence_result)
        return self.fake_sentence_results

    @staticmethod
    def seed(seed_value=42) -> None:
        """Seed the faker and random modules for reproducibility."""
        Faker.seed(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def add_provider(self, provider: BaseProvider) -> None:
        """
        Add a provider to the sentence faker
        :param provider: A faker provider inheriting from BaseProvider
        """
        self._sentence_faker.add_provider(provider)

    def add_provider_alias(self, provider_name: str, new_name: str) -> None:
        """
        Adds a copy of a provider, with a different name
        :param provider_name: Name of original provider
        :param new_name: New name
        :example:
        >>>self.add_provider_alias(provider_name="name", new_name="person")
        >>>self.person()
        """
        self._sentence_faker.add_provider_alias(
            provider_name=provider_name,
            new_name=new_name,
        )

    def add_entity_type_mapping(
        self,
        input_entity_type: str,
        output_entity_type: str,
    ) -> None:
        self._entity_type_mapping[input_entity_type] = output_entity_type

    @staticmethod
    def _preprocess_template(template: str):
        """Lowercase the entity names within double curly braces in the template, and replace < and > with {{ and }}."""  # noqa: E501

        def lowercase_within_braces(s):
            return re.sub(
                r"{{(.*?)}}",
                lambda match: f"{{{{{match.group(1).lower()}}}}}",
                s,
            )

        template = template.replace("<", "{{").replace(">", "}}")
        template = lowercase_within_braces(template)

        return template


if __name__ == "__main__":
    sentence_faker = PresidioSentenceFaker(
        locale="en_US",
        lower_case_ratio=0.05,
        random_seed=42,
    )
    fake_sentence_results = sentence_faker.generate_new_fake_sentences(
        num_samples=10000,
    )
    repo_root = Path(__file__).parent.parent.parent
    output_file = repo_root / "data/presidio_data_generator_data.json"
    to_json = [result.to_dict() for result in fake_sentence_results]
    with open(f"{output_file}", "w+", encoding="utf-8") as f:
        json.dump(to_json, f, ensure_ascii=False, indent=2)
