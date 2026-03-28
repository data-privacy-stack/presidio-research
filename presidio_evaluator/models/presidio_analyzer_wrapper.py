from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    EntityRecognizer,
    RecognizerResult,
)

from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel


class PresidioAnalyzerWrapper(BaseModel):
    def __init__(
        self,
        analyzer_engine: AnalyzerEngine | None = None,
        entities_to_keep: list[str] | None = None,
        verbose: bool = False,
        labeling_scheme: str = "IO",
        score_threshold: float = 0.4,
        language: str = "en",
        entity_mapping: dict[str, str] | None = None,
        ad_hoc_recognizers: list[EntityRecognizer] | None = None,
        context: list[str] | None = None,
        allow_list: list[str] | None = None,
    ) -> None:
        """
        Evaluation wrapper for the Presidio Analyzer
        :param analyzer_engine: object of type AnalyzerEngine (from presidio-analyzer)
        :param entities_to_keep: Which entities should be evaluated? if None, all are kept
        :param verbose: Whether to print more debug info
        :param labeling_scheme: Used to translate (if needed)
        the prediction to a specific scheme (IO, BIO/IOB, BILUO)
        :param score_threshold: Minimum score for an entity to be considered
        :param language: Language of the text
        :param entity_mapping: DEPRECATED. Entity mapping is now handled by
        CanonicalMapper before evaluation.
        :param ad_hoc_recognizers: List of ad-hoc recognizers to be used in the analyze method
        :param context: List of context words to be passed to the analyze method
        :param allow_list: List of allowed values to be passed to the analyze method
        """
        if entity_mapping is not None:
            raise DeprecationWarning(
                "The 'entity_mapping' parameter is deprecated and has been removed. "
                "Entity mapping is now handled by CanonicalMapper before evaluation. "
                "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for examples.",
            )

        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
        )
        self.name = "Presidio Analyzer"
        self.score_threshold = score_threshold
        self.language = language
        self.ad_hoc_recognizers = ad_hoc_recognizers
        self.context = context
        self.allow_list = allow_list

        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()

        self.analyzer_engine = analyzer_engine

        self.print_discrepancies()

    def predict(self, sample: InputSample, **kwargs) -> list[str]:
        self.__update_kwargs(kwargs)

        results = self.analyzer_engine.analyze(
            text=sample.full_text,
            **kwargs,
        )
        response_tags = self.__recognizer_results_to_tags(results, sample)
        return response_tags

    def batch_predict(self, dataset: list[InputSample], **kwargs) -> list[list[str]]:
        self.__update_kwargs(kwargs)
        texts = [sample.full_text for sample in dataset]
        batch_analyzer = BatchAnalyzerEngine(analyzer_engine=self.analyzer_engine)
        analyzer_results = batch_analyzer.analyze_iterator(texts=texts, **kwargs)

        predictions = []
        for prediction, sample in zip(analyzer_results, dataset, strict=False):
            predictions.append(self.__recognizer_results_to_tags(prediction, sample))

        return predictions

    @staticmethod
    def __recognizer_results_to_tags(
        results: list[RecognizerResult],
        sample: InputSample,
    ) -> list[str]:
        starts = []
        ends = []
        scores = []
        tags = []
        for res in results:
            starts.append(res.start)
            ends.append(res.end)
            tags.append(res.entity_type)
            scores.append(res.score)
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

    def __update_kwargs(self, kwargs) -> None:
        kwargs["language"] = kwargs.get("language", self.language)
        kwargs["score_threshold"] = kwargs.get("score_threshold", self.score_threshold)
        kwargs["ad_hoc_recognizers"] = kwargs.get(
            "ad_hoc_recognizers",
            self.ad_hoc_recognizers,
        )
        kwargs["context"] = kwargs.get("context", self.context)
        kwargs["allow_list"] = kwargs.get("allow_list", self.allow_list)
        kwargs["entities"] = kwargs.get("entities", self.entities)

    def print_discrepancies(self) -> None:
        supported_entities = self.analyzer_engine.get_supported_entities(
            language=self.language,
        )

        if not self.entities:
            self.entities = supported_entities

        for entity in self.entities:
            if entity not in supported_entities:
                print(
                    f"Warning: Entity {entity} is not supported by this instance of Presidio Analyzer Engine",
                )
        print("--------")
        print("Entities supported by this Presidio Analyzer instance:")
        print(", ".join(supported_entities))

    def _update_recognizers_based_on_entities_to_keep(self) -> None:
        """Add ORGANIZATION as it is removed by default."""

        supported_entities = self.analyzer_engine.get_supported_entities(
            language=self.language,
        )

        if "ORGANIZATION" in self.entities and "ORGANIZATION" not in supported_entities:
            recognizers = self.analyzer_engine.get_recognizers()
            spacy_recognizer = [
                rec
                for rec in recognizers
                if rec.name in {"SpacyRecognizer", "StanzaRecognizer"}
            ]
            if spacy_recognizer:
                spacy_recognizer = spacy_recognizer[0]
                spacy_recognizer.supported_entities.append("ORGANIZATION")
                self.entities.append("ORGANIZATION")
                print("Added ORGANIZATION as a supported entity from spaCy/Stanza")
