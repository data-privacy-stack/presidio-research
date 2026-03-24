import random
from typing import Any

import pandas as pd
from faker import Faker
from faker.generator import _re_token
from faker.providers import BaseProvider, DynamicProvider

from presidio_evaluator import InputSample, Span
from presidio_evaluator.data_generator.faker_extensions import SpanGenerator


class RecordGenerator(SpanGenerator):
    """Prioritizes the sampling of values found in input records,
    in order to allow semantically similar elements to be drawn together.

    For example, for the input template
    "My first name is {{name}} and my email is {{email}}",
    assuming the user added a list of records containing name and email
    (e.g. `[{"name": "Stephane Smith", "email": "stephane.smith@gmail.com"}]`),
    this generator will sample the name and email from the same record
    and not independently.
    In case the template contains entities not found in the record,
    the generator will use regular Faker providers.
    In case the template contains the same type multiple times,
    the first would be taken from the record and the next ones
    from regular Faker providers.

    :example:
    >>>from faker import Faker
    >>>from presidio_evaluator.data_generator.faker_extensions import RecordGenerator

    >>>records = [
    >>>     {"name": "Alan", "email": "alan@a.com"},
    >>>     {"name": "Barry", "email": "barry@b.com"},
    >>>     {"name": "Cynthia", "email": "cynthia@c.com"},
    >>>     ]

    >>>generator = RecordGenerator(records=records)
    >>>faker = Faker(generator=generator)

    >>># Sample fake values from the same record
    >>>faker.parse("I'm {{name}} and my email is {{email}}.")

    I'm Alan and my email is alan@a.com.

    >>># Using more than one type will use regular Faker providers:
    >>>faker.parse("{{name}}, {{name}} and {{name}} will email {{email}}.")

    Cynthia, Manuel Gonzales and Jillian Riley will email cynthia@c.com.

    >>># Sample from a Pandas DataFrame
    >>>import pandas as pd
    >>>df = pd.DataFrame({"name":["a","b","c"],"email":["a@a","b@b","c@c"]}) # or read from file
    >>>records = df.to_dict(orient="records")
    >>>generator = RecordGenerator(records=records)
    >>>faker = Faker(generator=generator)
    >>>faker.parse("I'm {{name}} and my email is {{email}}")

    I'm a and my email is a@a

    >>># Return spans of fake values
    >>>res = faker.parse("I'm {{name}} and my email is {{email}}",add_spans=True)

    {"fake": "I'm c and my email is c@c",
     "spans": "[{\"entity_value\": \"c@c\", \"start_position\": 22, \"end_position\": 25, \"entity_type\": \"email\"},
     {\"entity_value\": \"c\", \"start_position\": 4, \"end_position\": 5, \"entity_type\": \"name\"}]"
     }

    """

    def __init__(self, records: list[dict] | None = None) -> None:
        super().__init__()
        self.records = records

        if self.records:
            for record in self.records:
                if not isinstance(record, dict):
                    raise TypeError("Each element should be of type Dict")

        # Use an internal provider to sample from the input elements
        self.dynamic_record_provider = DynamicProvider(
            provider_name="",
            elements=records,
            generator=self,
        )

    def _get_random_record(self):
        return self.dynamic_record_provider.get_random_value().copy()

    def _match_to_span(self, text: str, **kwargs) -> list[Span]:
        """Adds logic for sampling from input records if possible."""
        matches = _re_token.finditer(text)
        # Sample one record (Dict containing fake values)
        record = self._get_random_record()

        results: list[Span] = []
        for match in matches:
            formatter = match.group()[2:-2]
            stripped = formatter.strip()

            value = str(self.format(formatter=stripped, record=record))
            if stripped in record:
                del record[stripped]  # Remove in order not to sample twice

            results.append(
                Span(
                    entity_type=formatter,
                    start_position=match.start(),
                    end_position=match.end(),
                    entity_value=value,
                ),
            )

        return results

    def format(self, formatter: str, *args: Any, **kwargs: Any) -> str:
        """Fill in fake data. If the input record has the requested entity, return its value."""
        record = kwargs.get("record")
        if not record or not record.get(
            formatter,
        ):  # type not in record, go to default faker
            return super().format(formatter)

        return record[formatter]


class SentenceFaker(Faker):
    def __init__(
        self,
        lower_case_ratio: float = 0.05,
        records: pd.DataFrame | list[dict] | None = None,
        **faker_kwargs,
    ) -> None:
        """
        Leverages Faker to create fake PII entities into predefined templates of structure: a b c {{PII}} d e f,
        e.g. "My name is {{first_name}} and returning span information in addition to fake text."
        :param lower_case_ratio: Percentage of names that should start with lower case
        :param records: fake PII values to draw from, in order to maintain semantic relationship between elements.
        :param faker_kwargs: Arguments for the parent class (faker.Faker)

        :example:

        >>>from presidio_evaluator.data_generator.faker_extensions import SentenceFaker

        >>>template = "I just moved to {{city}} from {{country}}"
        >>>fake_sentence_result = SentenceFaker().parse(template)
        >>>print(fake_sentence_result.full_text)
        I just moved to North Kim from Ukraine
        >>>print(fake_sentence_result.spans)
        [{"entity_value": "Ukraine", "start_position": 31, "end_position": 38, "entity_type": "country"}, {"entity_value": "North Kim", "start_position": 16, "end_position": 25, "entity_type": "city"}]
        """

        if records is not None:
            if isinstance(records, pd.DataFrame):
                records = records.to_dict(orient="records")
            generator = RecordGenerator(records=records)
        else:
            generator = SpanGenerator()

        super().__init__(generator=generator, **faker_kwargs)
        self.lower_case_ratio = lower_case_ratio

    def parse(
        self,
        template: str,
        template_id: int | None = None,
        add_spans: bool | None = True,
    ) -> InputSample | str:
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
            if isinstance(self.factories[0], SpanGenerator):
                span_generator: SpanGenerator = self.factories[0]
                fake_pattern = span_generator.parse(
                    template,
                    add_spans=add_spans,
                    template_id=template_id,
                )
            else:
                fake_pattern = self.factories[0].parse(template)
            if random.random() < self.lower_case_ratio:
                fake_pattern = self._lower_pattern(fake_pattern)
            return fake_pattern
        except Exception as err:
            raise AttributeError(
                f'Failed to generate fake data based on template "{template}". '
                f"Add a new Faker provider or create an alias "
                f"for the entity name. {err}",
            ) from err

    @staticmethod
    def _lower_pattern(pattern: str | InputSample):
        if isinstance(pattern, str):
            return pattern.lower()
        elif isinstance(pattern, InputSample):
            pattern.fake = pattern.full_text.lower()
            for span in pattern.spans:
                span.entity_value = str(span.entity_value).lower()
            return pattern

    def add_provider_alias(self, provider_name: str, new_name: str) -> None:
        """
        Adds a copy of a provider, with a different name
        :param provider_name: Name of original provider
        :param new_name: New name
        :example:
        >>>self.add_provider_alias(provider_name="name", new_name="person")
        >>>self.person()
        """
        original = getattr(self, provider_name)

        new_provider = BaseProvider(self)
        setattr(new_provider, new_name, original)
        self.add_provider(new_provider)
