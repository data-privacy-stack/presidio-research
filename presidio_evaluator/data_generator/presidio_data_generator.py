import random
import re
import warnings
from collections.abc import Generator

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider
from pandas import DataFrame
from tqdm import tqdm

from presidio_evaluator.data_generator.faker_extensions import (
    SpanGenerator,
)
from presidio_evaluator.data_generator.faker_extensions.data_objects import (
    FakerSpansResult,
)


class PresidioDataGenerator:
    def __init__(
        self,
        custom_faker: Faker = None,
        locale: list[str] | None = None,
        lower_case_ratio: float = 0.05,
    ) -> None:
        """
        Fake data generator.
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}}."
        :param custom_faker: A Faker object provided by the user
        :param locale: A locale object to create our own Faker instance if a custom one was not provided.
        :param lower_case_ratio: Percentage of names that should start with lower case

        :example:

        >>>sentence_templates = [
        >>>    "My name is {{name}}",
        >>>    "Please send it to {{address}}",
        >>>    "I just moved to {{city}} from {{country}}"
        >>>]


        >>>data_generator = PresidioDataGenerator()
        >>>fake_records = data_generator.generate_fake_data(
        >>>    templates=sentence_templates, n_samples=10
        >>>)

        >>>fake_records = list(fake_records)

        >>># Print the spans of the first sample
        >>>print(fake_records[0].fake)
        I just moved to North Kim from Ukraine

        >>>print(fake_records[0].spans)
        [{"value": "Ukraine", "start": 31, "end": 38, "type": "country"}, {"value": "North Kim", "start": 16, "end": 25, "type": "city"}]

        """

        warnings.warn(
            "PresidioDataGenerator is deprecated and will be removed in future versions."
            "Use PresidioSentenceFaker instead",
            category=DeprecationWarning,
            stacklevel=2,
        )

        if custom_faker and locale:
            raise ValueError(
                "If a custom faker is passed, it's expected to have its locales loaded",
            )

        if custom_faker:
            self.faker = custom_faker
        else:
            generator = (
                SpanGenerator()
            )  # To allow PresidioDataGenerator to return spans and not just strings
            self.faker = Faker(locale=locale, generator=generator)
        self.lower_case_ratio = lower_case_ratio

    def parse(
        self,
        template: str,
        template_id: int | None = None,
        sample_id: int | None = None,
        add_spans: bool = True,
    ) -> FakerSpansResult | str:
        """
        This function replaces known PII {{tokens}} in a template sentence
        with a fake value for each token and returns a sentence with fake PII.

        Examples:
            1. "My name is {{first_name_female}} {{last_name}}".
            2. "I want to increase limit on my card # {{credit_card_number}}
                for certain duration of time. is it possible?"


        :param template: str with token(s) that needs to be replaced by fake PII
        :param template_id: The identifier of the specific template
        :param add_spans: Whether to return the spans or just the fake text

        :returns: Fake sentence.

        """
        try:
            if isinstance(self.faker.factories[0], SpanGenerator):
                fake_pattern = self.faker.parse(
                    template,
                    add_spans=add_spans,
                    template_id=template_id,
                    sample_id=sample_id,
                )
            else:
                fake_pattern = self.faker.parse(template)
            if random.random() < self.lower_case_ratio:
                fake_pattern = self._lower_pattern(fake_pattern)
            return fake_pattern
        except Exception as err:
            raise AttributeError(
                f'Failed to generate fake data based on template "{template}".'
                f"You might need to add a new Faker provider! "
                f"{err}",
            ) from err

    @staticmethod
    def read_template_file(templates_file):
        with open(templates_file) as f:
            lines = f.readlines()
            lines = [line.replace("\\n", "\n") for line in lines]
            return lines

    def generate_fake_data(
        self,
        templates: list[str],
        n_samples: int,
    ) -> Generator[FakerSpansResult, None, None] | Generator[str, None, None]:
        """
        Generates fake PII data whenever it encounters known faker entities in a template.
        :param templates: A list of strings containing templates
        :param n_samples: Number of samples to generate
        """
        if not templates:
            templates = None

        for i in tqdm(range(n_samples), desc="Sampling"):
            template_id = random.choice(range(len(templates)))
            template = templates[template_id]
            yield self.parse(template=template, template_id=template_id, sample_id=i)

    @staticmethod
    def _lower_pattern(pattern: str | FakerSpansResult):
        if isinstance(pattern, str):
            return pattern.lower()
        elif isinstance(pattern, FakerSpansResult):
            pattern.fake = pattern.fake.lower()
            for span in pattern.spans:
                span.value = str(span.value).lower()
            return pattern

    @staticmethod
    def seed(seed_value=42) -> None:
        Faker.seed(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def add_provider_alias(self, provider_name: str, new_name: str) -> None:
        """
        Adds a copy of a provider, with a different name
        :param provider_name: Name of original provider
        :param new_name: New name
        :example:
        >>>add_provider_alias(provider_name="name", new_name="person")
        >>>self.faker.person()
        """
        original = getattr(self.faker, provider_name)

        new_provider = BaseProvider(self.faker)
        setattr(new_provider, new_name, original)
        setattr(new_provider, new_name.lower(), original)  # avoid case sensitivity
        setattr(new_provider, new_name.upper(), original)  # avoid case sensitivity
        self.faker.add_provider(new_provider)

    @staticmethod
    def update_fake_name_generator_df(fake_data: pd.DataFrame) -> DataFrame:
        """
        Adapts the csv from FakeNameGenerator to fit the data generation process used here.
        :param fake_data: a pd.DataFrame with loaded data from FakeNameGenerator.com
        :return: None
        """

        def camel_to_snake(name):
            # Borrowed from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        def full_name(row):
            if random.random() > 0.2:
                return str(row.first_name) + " " + str(row.last_name)
            else:
                space_after_initials = " " if random.random() > 0.5 else ". "
                return (
                    str(row.first_name)
                    + " "
                    + str(row.middle_initial)
                    + space_after_initials
                    + str(row.last_name)
                )

        def name_gendered(row):
            first_name_female, prefix_female, last_name_female = (
                (row.first_name, row.prefix, row.last_name)
                if row.gender == "female"
                else ("", "", "")
            )
            first_name_male, prefix_male, last_name_male = (
                (row.first_name, row.prefix, row.last_name)
                if row.gender == "male"
                else ("", "", "")
            )
            return (
                first_name_female,
                first_name_male,
                prefix_female,
                prefix_male,
                last_name_female,
                last_name_male,
            )

        fake_data.columns = [camel_to_snake(col) for col in fake_data.columns]

        # Update some column names to fit Faker
        fake_data.rename(
            columns={"country": "country_code", "state": "state_abbr"},
            inplace=True,
        )

        fake_data.rename(
            columns={
                "country_full": "country",
                "name_set": "nationality",
                "street_address": "street_name",
                "state_full": "state",
                "given_name": "first_name",
                "surname": "last_name",
                "title": "prefix",
                "email_address": "email",
                "telephone_number": "phone_number",
                "telephone_country_code": "country_calling_code",
                "birthday": "date_of_birth",
                "cc_number": "credit_card_number",
                "cc_type": "credit_card_provider",
                "cc_expires": "credit_card_expire",
                "occupation": "job",
                "domain": "domain_name",
                "username": "user_name",
                "zip_code": "zipcode",
            },
            inplace=True,
        )
        fake_data["person"] = fake_data.apply(full_name, axis=1)
        fake_data["name"] = fake_data["person"]
        genderized = fake_data.apply(
            lambda x: pd.Series(
                name_gendered(x),
                index=[
                    "first_name_female",
                    "first_name_male",
                    "prefix_female",
                    "prefix_male",
                    "last_name_female",
                    "last_name_male",
                ],
            ),
            axis=1,
            result_type="expand",
        )

        # Remove credit card data, rely on Faker's as it is more realistic
        del fake_data["credit_card_number"]

        fake_data = pd.concat([fake_data, genderized], axis="columns")
        return fake_data
