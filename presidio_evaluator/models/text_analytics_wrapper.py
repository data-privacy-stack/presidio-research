import warnings
from typing import List, Optional, Dict

from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel

try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    TextAnalyticsClient = None
    AzureKeyCredential = None


class TextAnalyticsWrapper(BaseModel):
    def __init__(
        self,
        ta_client: Optional[TextAnalyticsClient] = None,
        ta_key: Optional[str] = "",
        ta_endpoint: Optional[str] = "",
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        score_threshold: float = 0.4,
        language: str = "en",
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Evaluation wrapper for the Azure Text Analytics
        :param ta_client: object of type TextAnalyticsClient
        :param ta_key: Azure cognitive Services for Language key
        :param ta_endpoint: Azure cognitive Services for Language endpoint
        :param entity_mapping: Mapping between input dataset entities and entities
        expected by Azure cognitive Services for Language
        """
        super().__init__(
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future version. "
            f"Please use the dedicated recognizer within Presidio Analyzer instead. "
            DeprecationWarning,
            stacklevel=2,
        )

        self.score_threshold = score_threshold
        self.language = language
        self.ta_key = ta_key
        self.ta_endpoint = ta_endpoint

        if not TextAnalyticsClient:
            raise ImportError("azure.ai.textanalytics is not installed")

        if not ta_client:
            ta_client = self.__authenticate_client(ta_key, ta_endpoint)
        self.ta_client = ta_client

    def __authenticate_client(self, key: str, endpoint: str):
        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, credential=ta_credential
        )
        return text_analytics_client

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        documents = [sample.full_text]
        response = self.ta_client.recognize_pii_entities(documents, language="en")
        results = [doc for doc in response if not doc.is_error]
        starts = []
        ends = []
        scores = []
        tags = []
        #
        for res in results:
            for entity in res.entities:
                if entity.confidence_score < self.score_threshold:
                    continue
                else:
                    starts.append(entity.offset)
                    ends.append(entity.offset + len(entity.text))
                    tags.append(entity.category)
                    scores.append(entity.confidence_score)

        response_tags = span_to_tag(
            scheme="IO",
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tokens=sample.tokens,
            scores=scores,
            tags=tags,
        )
        return response_tags

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        predictions = []
        for sample in dataset:
            predictions.append(self.predict(sample, **kwargs))

        return predictions
