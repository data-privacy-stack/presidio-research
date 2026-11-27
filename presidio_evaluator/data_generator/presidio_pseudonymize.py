from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from presidio_evaluator.data_generator.faker_extensions import SentenceFaker


class PresidioPseudonymization(SentenceFaker):
    def __init__(self, map_to_presidio_entities: bool = True, **kwargs) -> None:
        """
        Create pseudoanoymized data using Presidio by identifying real entities
        and replacing them with fake ones.
        :param kwargs: kwargs for PresidioDataGenerator
        :param map_to_presidio_entities:
        Whether to creating a mapping between Faker's providers and Presidio's entities
        """

        super().__init__(**kwargs)
        if map_to_presidio_entities:
            # Template placeholders are looked up in lower case. PHONE_NUMBER,
            # URL and DATE_TIME already match Faker providers of the same name.
            self.add_provider_alias("name", "person")
            self.add_provider_alias("ipv4", "ip_address")
            self.add_provider_alias("company", "organization")
            self.add_provider_alias("country", "location")
            self.add_provider_alias("credit_card_number", "credit_card")
            self.add_provider_alias("iban", "iban_code")
            self.add_provider_alias("ssn", "us_ssn")
            self.add_provider_alias("email", "email_address")

    def pseudonymize(
        self,
        original_text: str,
        presidio_response: list[RecognizerResult],
        count: int,
    ):
        """

        :param original_text: str containing the original text
        :param presidio_response: list of results from Presidio, to be used to know where entities are
        :param count: number of perturbations to return
        :return: List[str] with fake perturbations of original text
        """

        presidio_response = sorted(presidio_response, key=lambda resp: resp.start)

        # Replace each entity with a {{ENTITY_TYPE}} placeholder directly,
        # so "<" and ">" in the original text are left as they are
        operators = {
            resp.entity_type: OperatorConfig(
                "replace", {"new_value": "{{" + resp.entity_type + "}}"}
            )
            for resp in presidio_response
        }
        anonymizer_engine = AnonymizerEngine()
        anonymized_result = anonymizer_engine.anonymize(
            text=original_text,
            analyzer_results=presidio_response,
            operators=operators,
        )

        templated_text = anonymized_result.text
        fake_texts = [self.parse(templated_text, add_spans=False) for _ in range(count)]
        return fake_texts
