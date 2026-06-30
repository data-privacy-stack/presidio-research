from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngine

from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel
from presidio_evaluator.span_to_tag import span_to_tag


class PresidioRecognizerWrapper(BaseModel):
    """
    Class wrapper for one specific PII recognizer
    To evaluate the entire set of recognizers, refer to PresidioAnaylzerWrapper
    :param recognizer: An object of type EntityRecognizer (in presidio-analyzer)
    :param nlp_engine: An object of type NlpEngine, e.g. SpacyNlpEngine (in presidio-analyzer)
    :param entities_to_keep: List of entity types to focus on while ignoring all the rest.
    Default=None would look at all entity types
    :param with_nlp_artifacts: Whether NLP artifacts should be obtained
        (faster if not, but some recognizers need it)
    """

    def __init__(
        self,
        recognizer: EntityRecognizer,
        nlp_engine: NlpEngine,
        entities_to_keep: list[str] = None,
        labeling_scheme: str = "IO",
        with_nlp_artifacts: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
        )
        self.with_nlp_artifacts = with_nlp_artifacts
        self.recognizer = recognizer
        self.nlp_engine = nlp_engine

        if not self.nlp_engine.is_loaded():
            self.nlp_engine.load()

    def __make_nlp_artifacts(self, text: str):
        return self.nlp_engine.process_text(text, "en")

    def predict(self, sample: InputSample, **kwargs) -> list[str]:
        nlp_artifacts = None
        if self.with_nlp_artifacts:
            nlp_artifacts = self.__make_nlp_artifacts(sample.full_text)

        results = self.recognizer.analyze(
            sample.full_text,
            self.entities,
            nlp_artifacts,
        )
        starts = []
        ends = []
        tags = []
        scores = []
        for res in results:
            if not res.start:
                res.start = 0
            starts.append(res.start)
            ends.append(res.end)
            tags.append(res.entity_type)
            scores.append(res.score)
        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tags=tags,
            tokens=sample.tokens,
            scores=scores,
        )
        return response_tags

    def batch_predict(self, dataset: list[InputSample], **kwargs) -> list[list[str]]:
        return [self.predict(sample, **kwargs) for sample in dataset]
