import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Union, Tuple
import logging

import numpy as np
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from spacy.tokens import Token

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult, ModelError, ErrorType
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.models import BaseModel, PresidioAnalyzerWrapper

logger = logging.getLogger(__name__)

GENERIC_ENTITIES = ("PII", "ID", "PII", "PHI", "ID_NUM", "NUMBER", "NUM", "GENERIC_PII")


class DeprecationError(RuntimeError):
    """Raised when a deprecated method that has been fully removed is called."""


class BaseEvaluator(ABC):
    def __init__(
        self,
        model: Optional[Union[BaseModel, AnalyzerEngine]],
        verbose: bool = False,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
    ):
        """
        Evaluate a PII detection model or a Presidio analyzer / recognizer

        :param model: Instance of a fitted model (of base type BaseModel),
        or an instance of Presidio Analyzer
        :param verbose: Whether to print debug information
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param generic_entities: List of entities that are not considered an error if
        detected instead of something other entity. For example: PII, ID, number
        :param skip_words: List of words to skip. If None, the default list would be used.
        """

        if model is None:
            warnings.warn(
                "Using the evaluator without a model only supports comparing actual vs. existing "
                "predicted tags. It will not run the model to generate predictions."
            )
            self.model = None

        elif isinstance(model, AnalyzerEngine):
            num_languages = len(model.supported_languages)
            if num_languages > 1:
                warnings.warn(
                    f"Presidio Analyzer supports multiple languages ({num_languages}). "
                    "Using the first language in the list for evaluation."
                )

            self.model = PresidioAnalyzerWrapper(
                analyzer_engine=model,
                entities_to_keep=entities_to_keep,
                score_threshold=model.default_score_threshold,
                language=model.supported_languages[0],
            )

        elif isinstance(model, BaseModel):
            self.model = model

        else:
            raise ValueError(
                "Model should be an instance of BaseModel or Presidio Analyzer, or None."
            )

        self.verbose = verbose
        self.entities_to_keep = entities_to_keep
        if self.entities_to_keep is None and self.model and self.model.entities:
            self.entities_to_keep = self.model.entities

        self.generic_entities = (
            generic_entities if generic_entities else GENERIC_ENTITIES
        )

        if skip_words is None:
            warnings.warn(
                "skip words not provided, using default skip words. "
                "If you want the evaluation to not use skip words, pass skip_words=[]"
            )
            self.skip_words = get_skip_words()
        else:
            self.skip_words = skip_words

    def compare(
        self, input_sample: InputSample, prediction: List[str]
    ) -> Tuple[Counter, List[ModelError]]:
        """
        Compares ground truth tags (annotation) and predicted (prediction)
        :param input_sample: input sample containing list of tags
        :param prediction: predicted value for each token
        """
        annotation = list(input_sample.tags)
        tokens = input_sample.tokens

        if len(annotation) != len(prediction):
            logger.warning(
                "Annotation and prediction do not have the"
                "same length. Sample={}".format(input_sample)
            )
            return Counter(), []

        results = Counter()
        mistakes = []

        if self.entities_to_keep:
            prediction = self._adjust_per_entities(prediction)
            annotation = self._adjust_per_entities(annotation)

        for i in range(0, len(annotation)):
            cur_token = tokens[i]
            cur_prediction = prediction[i]
            cur_annotation = annotation[i]

            results[(cur_annotation, cur_prediction)] += 1

            if self.verbose:
                logger.info("Annotation: %s", cur_annotation)
                logger.info("Prediction: %s", cur_prediction)
                logger.info("Results: %s", results)

            is_error = cur_annotation != cur_prediction

            if is_error:
                reverted = self.__revert_known_errors(
                    cur_annotation, cur_prediction, cur_token, results
                )
                if reverted:
                    continue

                if prediction[i] == "O":
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )
                elif annotation[i] == "O":
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.FP,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )
                else:
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.WrongEntity,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                        )
                    )

        return results, mistakes

    def __revert_known_errors(
        self,
        current_annotation: str,
        current_prediction: str,
        current_token: Union[str, Token],
        results: Counter[Tuple[str, str]],
    ) -> bool:
        reverted = False

        if str(current_token).lower().strip() in self.skip_words:
            # Ignore cases where the token is a skip word
            results[(current_annotation, current_prediction)] -= 1
            reverted = True

        if current_prediction in self.generic_entities and current_annotation != "O":
            # Ignore cases where the prediction is generic
            results[(current_annotation, current_prediction)] -= 1
            # Add a result which assumes the generic equals the specific
            results[(current_annotation, current_annotation)] += 1
            reverted = True

        elif current_annotation in self.generic_entities and current_prediction != "O":
            # Ignore cases where the prediction is generic
            results[(current_annotation, current_prediction)] -= 1
            # Add a result which assumes the generic equals the specific
            results[(current_prediction, current_prediction)] += 1
            reverted = True

        # Remove temporary keys which should not be counted
        if results[(current_annotation, current_prediction)] == 0:
            del results[(current_annotation, current_prediction)]

        return reverted

    def _adjust_per_entities(self, tags: List[str]) -> List[str]:
        if self.entities_to_keep:
            return [tag if tag in self.entities_to_keep else "O" for tag in tags]
        else:
            return tags

    def evaluate_sample(
        self, sample: InputSample, prediction: List[str]
    ) -> EvaluationResult:
        if self.verbose:
            logger.debug("Input sentence: {}".format(sample.full_text))

        if not self.model:
            raise ValueError(
                "Model is not set. Please instantiate the evaluator with a model to evaluate the dataset."
            )

        results, model_errors = self.compare(input_sample=sample, prediction=prediction)

        return EvaluationResult(
            results=results,
            model_errors=model_errors,
            text=sample.full_text,
            tokens=[str(token) for token in sample.tokens],
            actual_tags=sample.tags,
            predicted_tags=prediction,
            start_indices=sample.start_indices,
        )

    def evaluate_all(
        self, dataset: List[InputSample], **kwargs
    ) -> List[EvaluationResult]:
        """REMOVED. Use the new 3-step pipeline instead.

        .. deprecated::
            ``evaluate_all()`` has been removed. Migrate to::

                # Step 1: predict
                results_df = model.predict_dataset(dataset)

                # Step 2: map entities to a common namespace
                from presidio_evaluator.entity_mapping.mapper import CanonicalMapper
                mapper = CanonicalMapper()
                mapped_df = mapper.get_mapped_results_dataframe(results_df)

                # Step 3: evaluate
                from presidio_evaluator.evaluation import SpanEvaluator
                evaluator = SpanEvaluator(model=None)
                result_per_type = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)
                global_df = SpanEvaluator.create_global_entities_df(mapped_df)
                result = evaluator.calculate_score_on_df(per_type=False, results_df=global_df, evaluation_result=result_per_type)
        """
        raise DeprecationError(
            "evaluate_all() has been removed. Use the new 3-step pipeline:\n"
            "  1. results_df = model.predict_dataset(dataset)\n"
            "  2. mapper = CanonicalMapper(); mapped_df = mapper.get_mapped_results_dataframe(results_df)\n"
            "  3. result = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)\n"
            "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example."
        )

    @abstractmethod
    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        """
        Compares the evaluation results (predicted vs. actual) and calculates evaluation scores
        """

        pass

    def get_results_dataframe(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with the results of the evaluation.

        :param evaluation_results: List of EvaluationResult objects containing the evaluation results.
        :param entities: Optional list of entities to filter the results. If None, all entities are included.
        Note: entities should be model entity names (e.g., LOCATION), not dataset entity names.

        :return: A pandas DataFrame with the following columns
            - sentence_id
            - token text
            - annotation (normalized to model entity types)
            - prediction
            - start_indices
        """

        if not evaluation_results or not evaluation_results[0].tokens:
            raise ValueError(
                "The evaluation results should not be empty and must contain tokens. "
                "Ensure that the input samples have tokens."
            )

        rows_list = []
        for i, res in enumerate(evaluation_results):
            tokens = res.tokens
            annotations = list(res.actual_tags)
            predictions = list(res.predicted_tags)
            if self.entities_to_keep:
                annotations = self._adjust_per_entities(annotations)
                predictions = self._adjust_per_entities(predictions)

            # Filter to the requested entity subset (e.g. for per-entity scoring)
            annotations = self._filter_entities(annotations, entities)
            predictions = self._filter_entities(predictions, entities)
            start_indices = res.start_indices
            for j in range(len(tokens)):
                rows_list.append(
                    {
                        "sentence_id": i,
                        "token": tokens[j],
                        "annotation": annotations[j],
                        "prediction": predictions[j],
                        "start_indices": start_indices[j],
                    }
                )

        results_df = pd.DataFrame(rows_list)
        return results_df

    @staticmethod
    def _filter_entities(
        tags: List[str], entities: Optional[List[str]] = None
    ) -> List[str]:
        """
        Filter the tags to only include the specified entities.
        If entities is None, return all tags.
        """
        if entities is None:
            return tags
        return [tag if tag in entities else "O" for tag in tags]

    @staticmethod
    def precision(
        tp: int, fp: Optional[int] = None, num_predicted: Optional[int] = None
    ) -> float:
        """
        Calculate precision based on true positives (tp), false positives (fp), or total predicted entities (num_predicted).

        :param tp: Number of true positives
        :param fp: Number of false positives (optional, if num_predicted is not provided)
        :param num_predicted: Total number of predicted entities (optional, if fp is not provided)
        :return: Precision value as a float
        """
        if fp and num_predicted:
            raise ValueError(
                "Both fp and num_predicted should not be provided. "
                "Use either fp or num_predicted, but not both."
            )
        if fp is None and num_predicted is None:
            raise ValueError(
                "Either fp or num_predicted should be provided to calculate precision."
            )
        if fp:
            num_predicted = fp + tp

        return tp / num_predicted if num_predicted > 0 else np.nan

    @staticmethod
    def recall(
        tp: int, fn: Optional[int] = None, num_annotated: Optional[int] = None
    ) -> float:
        """
        Calculate recall based on true positives (tp), false negatives (fn), or total annotated entities (num_annotated).

        :param tp: Number of true positives
        :param fn: Number of false negatives (optional, if num_annotated is not provided)
        :param num_annotated: Total number of annotated entities (optional, if fn is not provided)
        :return: Recall value as a float
        """
        if fn and num_annotated:
            raise ValueError(
                "Both fn and num_annotated should not be provided. "
                "Use either fn or num_annotated, but not both."
            )
        if fn is None and num_annotated is None:
            raise ValueError(
                "Either fn or num_annotated should be provided to calculate recall."
            )
        if fn:
            num_annotated = fn + tp

        return tp / num_annotated if num_annotated > 0 else np.nan

    @staticmethod
    def f_beta(precision: float, recall: float, beta: float) -> float:
        """
        Returns the F score for precision, recall and a beta parameter
        :param precision: a float with the precision value
        :param recall: a float with the recall value
        :param beta: a float with the beta parameter of the F measure,
        which gives more or less weight to precision
        vs. recall
        :return: a float value of the f(beta) measure.
        """
        if np.isnan(precision) or np.isnan(recall) or (precision == 0 and recall == 0):
            return np.nan

        return ((1 + beta**2) * precision * recall) / (((beta**2) * precision) + recall)
