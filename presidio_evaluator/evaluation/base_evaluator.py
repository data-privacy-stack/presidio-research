import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Dict, Union, Tuple
import copy

import numpy as np
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from spacy.tokens import Token

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult, ModelError, ErrorType
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.models import BaseModel, PresidioAnalyzerWrapper

GENERIC_ENTITIES = ("PII", "ID", "PII", "PHI", "ID_NUM", "NUMBER", "NUM", "GENERIC_PII")


class BaseEvaluator(ABC):
    def __init__(
        self,
        model: Optional[Union[BaseModel, AnalyzerEngine]],
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
        :param compare_by_io: True if comparison should be done on the entity
        level and not the sub-entity level
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities. If the provided model has a list of entities to keep,
        this list would be used for evaluation.
        :param generic_entities: List of entities that are not considered an error if
        detected instead of something other entity. For example: PII, ID, number
        :param skip_words: List of words to skip. If None, the default list would be used.
        """

        if model is None:
            warnings.warn("Using the evaluator without a model only supports comparing actual vs. existing "
                          "predicted tags. It will not run the model to generate predictions.")
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
            warnings.warn("skip words not provided, using default skip words. "
                          "If you want the evaluation to not use skip words, pass skip_words=[]")
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
            print(
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

        # Ignore annotations that aren't in the list of
        # requested entities.
        if self.entities_to_keep:
            prediction = self._adjust_per_entities(prediction)
            new_annotation = self._adjust_per_entities(new_annotation)

        for i in range(0, len(new_annotation)):
            cur_token = tokens[i]
            cur_prediction = prediction[i]
            cur_annotation = new_annotation[i]
            results[(cur_annotation, cur_prediction)] += 1

            if self.verbose:
                print("Annotation:", cur_annotation)
                print("Prediction:", cur_prediction)
                print(results)

            # check if there was an error
            is_error = cur_annotation != cur_prediction

            if is_error:
                reverted = self.__revert_known_errors(
                    cur_annotation, cur_prediction, cur_token, results
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
                elif new_annotation[i] == "O":
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
        if not self.entities_to_keep:
            return tags

        filtered_tags = []
        for tag in tags:
            if tag == "O":
                filtered_tags.append(tag)
                continue

            base = tag.split("-", 1)[1] if "-" in tag else tag
            filtered_tags.append(tag if base in self.entities_to_keep else "O")

        return filtered_tags

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

    def evaluate_sample(
        self, sample: InputSample, prediction: List[str]
    ) -> EvaluationResult:
        if self.verbose:
            print("Input sentence: {}".format(sample.full_text))

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
        if self.model.entity_mapping:
            print(
                f"Mapping entity values using this dictionary: {self.model.entity_mapping}"
            )

        print(f"Running model {self.model.__class__.__name__} on dataset...")
        predictions = self.model.batch_predict(dataset, **kwargs)
        print("Finished running model on dataset")

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

    @staticmethod
    def align_entity_types(
        input_samples: List[InputSample],
        entities_mapping: Dict[str, str] = None,
        allow_missing_mappings: bool = False,
    ) -> List[InputSample]:
        """
        Change input samples to conform with the provided entity mappings
        :return: new list of InputSample
        """
        if not entities_mapping:
            raise ValueError("entities_mapping must be provided and non-empty")

        # Work on copies to avoid mutating the original dataset
        new_input_samples = [copy.deepcopy(sample) for sample in input_samples]

        # Pre-compute a normalized mapping for case/underscore-insensitive lookups
        normalized_mapping = {
            key.upper().replace("_", ""): value for key, value in entities_mapping.items()
        }

        def resolve_mapping_key(search_key: str) -> Optional[str]:
            """Return the mapped entity value for a search key, ignoring case/underscores."""
            return normalized_mapping.get(search_key.upper().replace("_", ""))

        # A list that will contain updated input samples,
        new_list = []
        missing_entities = set()

        for input_sample in new_input_samples:
            contains_field_in_mapping = False
            new_spans = []
            # Update spans to match the entity types in the values of entities_mapping
            for span in input_sample.spans:
                new_name = resolve_mapping_key(span.entity_type)
                if new_name:
                    span.entity_type = new_name
                    contains_field_in_mapping = True

                    new_spans.append(span)
                else:
                    missing_entities.add(span.entity_type)
                    if not allow_missing_mappings:
                        raise ValueError(
                            f"Key {span.entity_type} cannot be found in the provided entities_mapping"
                        )
            input_sample.spans = new_spans

            # Update tags in case this sample has relevant entities for evaluation
            if contains_field_in_mapping:
                for i, tag in enumerate(input_sample.tags):
                    has_prefix = "-" in tag
                    if has_prefix:
                        prefix = tag[:2]
                        clean = tag[2:]
                    else:
                        prefix = ""
                        clean = tag

                    new_name = resolve_mapping_key(clean)
                    if new_name:
                        input_sample.tags[i] = f"{prefix}{new_name}"
                    else:
                        input_sample.tags[i] = "O"
            else:
                # No mapped entities in this sample; keep spans empty and zero out tags
                input_sample.tags = ["O" for _ in input_sample.tags]

            new_list.append(input_sample)

        if allow_missing_mappings and missing_entities:
            print(
                "Entities without mapping (filtered out): "
                f"{sorted(missing_entities)}"
            )

        return new_list
        # Iterate on all samples

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

    @staticmethod
    def get_results_dataframe(
        evaluation_results: List[EvaluationResult], entities: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Return a DataFrame with the results of the evaluation.

        :param evaluation_results: List of EvaluationResult objects containing the evaluation results.
        :param entities: Optional list of entities to filter the results. If None, all entities are included.

        :return: A pandas DataFrame with the following columns
            - sentence_id
            - token text
            - annotation
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
            annotations = BaseEvaluator._filter_entities(res.actual_tags, entities)
            predictions = BaseEvaluator._filter_entities(res.predicted_tags, entities)
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
        filtered_tags = []
        for tag in tags:
            if tag == "O":
                filtered_tags.append(tag)
                continue

            if "-" in tag:
                _, base = tag.split("-", 1)
            else:
                base = tag

            filtered_tags.append(tag if base in entities else "O")

        return filtered_tags

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
