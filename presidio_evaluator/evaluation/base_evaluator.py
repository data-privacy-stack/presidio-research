import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Dict, Union, Tuple
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


class BaseEvaluator(ABC):
    def __init__(
        self,
        model: Optional[Union[BaseModel, AnalyzerEngine]],
        entity_mapping: Dict[str, Optional[str]],
        verbose: bool = False,
        compare_by_io: bool = True,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
    ):
        """
        Evaluate a PII detection model or a Presidio analyzer / recognizer

        :param model: Instance of a fitted model (of base type BaseModel),
        or an instance of Presidio Analyzer
        :param entity_mapping: REQUIRED. Dictionary mapping dataset entity types to model entity types.
        Keys are dataset entities (e.g., STREET_ADDRESS, GPE), values are model entities (e.g., LOCATION).
        This mapping is used during comparison to match dataset labels with model predictions.
        :param verbose: Whether to print debug information
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param generic_entities: List of entities that are not considered an error if
        detected instead of something other entity. For example: PII, ID, number
        :param skip_words: List of words to skip. If None, the default list would be used.
        """

        # Validate entity_mapping is provided
        if entity_mapping is None:
            raise ValueError(
                "entity_mapping is required. Please provide a dictionary mapping dataset entity types "
                "to model entity types. Example: {'STREET_ADDRESS': 'LOCATION', 'GPE': 'LOCATION'}. "
                "If no mapping is needed, pass an empty dict: entity_mapping={}"
            )

        self.entity_mapping = entity_mapping

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
        self.compare_by_io = compare_by_io
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
        :param input_sample: input sample containing list of tags with scheme
        :param prediction: predicted value for each token
        self.labeling_scheme

        """
        annotation = input_sample.tags
        tokens = input_sample.tokens

        if len(annotation) != len(prediction):
            logger.warning(
                "Annotation and prediction do not have the"
                "same length. Sample={}".format(input_sample)
            )
            return Counter(), []

        results = Counter()
        mistakes = []

        new_annotation = annotation.copy()

        if self.compare_by_io:
            new_annotation = self._to_io(new_annotation)
            prediction = self._to_io(prediction)

        # Keep a copy for error reporting (after IO conversion but before normalization)
        original_annotation = new_annotation.copy()

        # Normalize annotation entities to model entities BEFORE filtering
        # This ensures dataset entities (like GPE) are mapped to model entities (like LOCATION)
        # before checking if they should be kept
        normalized_annotation = [
            self._normalize_entity_for_comparison(tag, self.entity_mapping)
            for tag in new_annotation
        ]

        # Now filter by entities_to_keep (which contains model entity names)
        if self.entities_to_keep:
            prediction = self._adjust_per_entities(prediction)
            normalized_annotation = self._adjust_per_entities(normalized_annotation)

        for i in range(0, len(normalized_annotation)):
            cur_token = tokens[i]
            cur_prediction = prediction[i]
            cur_annotation = original_annotation[i]  # Use original for error reporting
            cur_normalized_annotation = normalized_annotation[i]

            # Use normalized annotation for counting and comparison
            results[(cur_normalized_annotation, cur_prediction)] += 1

            if self.verbose:
                logger.info("Annotation: %s", cur_annotation)
                logger.info("Normalized Annotation: %s", cur_normalized_annotation)
                logger.info("Prediction: %s", cur_prediction)
                logger.info("Results: %s", results)

            # check if there was an error (using normalized annotation)
            is_error = cur_normalized_annotation != cur_prediction

            if is_error:
                reverted = self.__revert_known_errors(
                    cur_normalized_annotation, cur_prediction, cur_token, results
                )
                if reverted:
                    # This isn't really an error, continue.
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
                elif normalized_annotation[i] == "O":
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

    @staticmethod
    def _to_io(tags: List[str]) -> List[str]:
        """
        Translates BILUO/BIO/IOB to IO - only In or Out of entity.
        ['B-PERSON','I-PERSON','L-PERSON'] is translated into
        ['PERSON','PERSON','PERSON']
        :param tags: the input tags in BILUO/IOB/BIO format
        :return: a new list of IO tags
        """
        return [tag[2:] if "-" in tag else tag for tag in tags]

    @staticmethod
    def _normalize_entity_for_comparison(
        entity: str, entity_mapping: Dict[str, str]
    ) -> str:
        """
        Normalize an entity name using mapping for comparison.
        Handles prefixed tags (B-STREET_ADDRESS -> B-LOCATION).

        Mapping to None means identity mapping (use entity as-is).

        :param entity: Entity tag (e.g., "STREET_ADDRESS" or "B-STREET_ADDRESS")
        :param entity_mapping: Dict mapping dataset entities to model entities
        :return: Normalized entity for comparison
        """
        if entity == "O":
            return entity

        # Handle prefixed tags (B-, I-, L-, U-)
        if "-" in entity:
            prefix, clean_entity = entity.split("-", 1)
            mapped_entity = entity_mapping.get(clean_entity, clean_entity)
            # If mapped to None, use original entity (identity mapping)
            if mapped_entity is None:
                mapped_entity = clean_entity
            return f"{prefix}-{mapped_entity}"
        else:
            mapped_entity = entity_mapping.get(entity, entity)
            # If mapped to None, use original entity (identity mapping)
            return entity if mapped_entity is None else mapped_entity

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
        """Evaluate a dataset given a model and labels.

        :param dataset: A list of InputSample samples, containing the ground truth tags
        :param kwargs: Additional arguments for the model's predict method
        """

        if not self.model:
            raise ValueError(
                "Model is not set. Please instantiate the evaluator with a model to evaluate the dataset."
            )

        evaluation_results = []

        logger.info("Using entity mapping for comparison: %s", self.entity_mapping)

        logger.info("Running model %s on dataset...", self.model.__class__.__name__)
        predictions = self.model.batch_predict(dataset, **kwargs)
        logger.info("Finished running model on dataset")

        for prediction, sample in zip(predictions, dataset):
            # Remove entities not requested (in model.entities_to_keep))
            prediction = self.model.filter_tags_in_supported_entities(prediction)

            # Switch to requested labeling scheme (IO/BIO/BILUO)
            prediction = self.model.to_scheme(prediction)

            evaluation_result = self.evaluate_sample(
                sample=sample, prediction=prediction
            )
            evaluation_results.append(evaluation_result)

        return evaluation_results

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
            # Normalize annotation entities (dataset namespace) to model entities
            # so that entity names are comparable to predictions and to entities_to_keep.
            # Predictions are already in model namespace and need no remapping.
            annotations = res.actual_tags
            predictions = res.predicted_tags
            if self.compare_by_io:
                annotations = self._to_io(annotations)
                predictions = self._to_io(predictions)
            annotations = [
                self._normalize_entity_for_comparison(tag, self.entity_mapping)
                for tag in annotations
            ]
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
