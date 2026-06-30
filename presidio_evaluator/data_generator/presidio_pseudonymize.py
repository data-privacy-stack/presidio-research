from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

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
            self.add_provider_alias("name", "PERSON")
            self.add_provider_alias("ipv4", "IP_ADDRESS")
            self.add_provider_alias("company", "ORGANIZATION")
            self.add_provider_alias("country", "LOCATION")
            self.add_provider_alias("credit_card_number", "CREDIT_CARD")
            self.add_provider_alias("iban", "IBAN_CODE")
            self.add_provider_alias("phone_number", "PHONE_NUMBER")
            self.add_provider_alias("url", "URL")
            self.add_provider_alias("ssn", "US_SSN")
            self.add_provider_alias("email", "EMAIL_ADDRESS")
            self.add_provider_alias("date_time", "DATE_TIME")

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

        anonymizer_engine = AnonymizerEngine()
        anonymized_result = anonymizer_engine.anonymize(
            text=original_text,
            analyzer_results=presidio_response,
        )

        templated_text = anonymized_result.text
        templated_text = templated_text.replace(">", "}}").replace("<", "{{")
        fake_texts = [self.parse(templated_text, add_spans=False) for _ in range(count)]
        return fake_texts
