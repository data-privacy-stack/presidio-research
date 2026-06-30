import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from presidio_evaluator import InputSample
from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy
from presidio_evaluator.evaluation import EvaluationResult
from presidio_evaluator.evaluation.skipwords import get_skip_words
from presidio_evaluator.models import BaseModel

if TYPE_CHECKING:
    from presidio_evaluator.entity_mapping.data_objects import MappedResults

logger = logging.getLogger(__name__)

GENERIC_ENTITIES = ("PII", "ID", "PII", "PHI", "ID_NUM", "NUMBER", "NUM", "GENERIC_PII")


def _to_l1(entity: str, hierarchy: EntityHierarchy) -> str:
    """Map an entity label to its depth-2 (branch) ancestor."""
    if entity == "O":
        return "O"
    branch = hierarchy.canonical_to_branch.get(entity)
    if branch is None:
        return entity  # unknown entity — pass through unchanged
    if len(branch) >= 2:
        return branch[1]  # e.g. ['PII', 'PERSON', 'NAME'] -> 'PERSON'
    return branch[0]  # depth-1 node (PII itself)


def _to_l0(entity: str) -> str:
    """Map any non-O entity label to 'PII'."""
    return "O" if entity == "O" else "PII"


class DeprecationError(RuntimeError):
    """Raised when a deprecated method that has been fully removed is called."""


class BaseEvaluator(ABC):
    def __init__(
        self,
        model: BaseModel | None = None,
        verbose: bool = False,
        entities_to_keep: list[str] | None = None,
        generic_entities: list[str] | None = None,
        skip_words: list | None = None,
    ) -> None:
        """
        Evaluate PII detection results.

        :param model: Deprecated. Must be None. Passing a model is no longer supported.
        Use model.predict_dataset(dataset) to obtain a results DataFrame,
        then pass it to calculate_score_on_df().
        :param verbose: Whether to print debug information
        :param entities_to_keep: List of entity names to focus the evaluator on (and ignore the rest).
        Default is None = all entities.
        :param generic_entities: List of entities that are not considered an error if
        detected instead of something other entity. For example: PII, ID, number
        :param skip_words: List of words to skip. If None, the default list would be used.
        """

        if model is not None:
            raise DeprecationError(
                "Passing a model to the evaluator constructor is no longer supported.\n"
                "Use the new 3-step pipeline instead:\n"
                "  1. results_df = model.predict_dataset(dataset)\n"
                "  2. mapper = CanonicalMapper()\n"
                "     mapped_df = mapper.get_mapped_results_dataframe(results_df)\n"
                "  3. result = evaluator.calculate_score_on_df(results_df=mapped_df)\n"
                "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
            )

        self.model = None
        self.verbose = verbose
        self.entities_to_keep = entities_to_keep

        self.generic_entities = (
            generic_entities if generic_entities else GENERIC_ENTITIES
        )

        if skip_words is None:
            logger.warning(
                "skip words not provided, using default skip words. "
                "If you want the evaluation to not use skip words, pass skip_words=[]"
            )
            self.skip_words = get_skip_words()
        else:
            self.skip_words = skip_words

    def evaluate_all(
        self,
        dataset: list[InputSample],
        **kwargs,
    ) -> list[EvaluationResult]:
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
                evaluator = SpanEvaluator()
                result_per_type = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)
                global_df = SpanEvaluator.create_global_entities_df(mapped_df)
                result = evaluator.calculate_score_on_df(per_type=False, results_df=global_df, evaluation_result=result_per_type)
        """
        raise DeprecationError(
            "evaluate_all() has been removed. Use the new 3-step pipeline:\n"
            "  1. results_df = model.predict_dataset(dataset)\n"
            "  2. mapper = CanonicalMapper(); mapped_df = mapper.get_mapped_results_dataframe(results_df)\n"
            "  3. result = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)\n"
            "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
        )

    def calculate_score(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        raise DeprecationError(
            "calculate_score() has been removed. Use calculate_score_on_df() instead:\n"
            "  result = evaluator.calculate_score_on_df(results_df=mapped_df)\n"
            "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
        )

    @abstractmethod
    def calculate_score_on_df(
        self,
        results_df: pd.DataFrame,
        beta: float = 2.0,
        level: Literal["entity", "pii", "both"] = "both",
        **kwargs,
    ) -> EvaluationResult:
        """
        Primary entry point. Evaluate predictions against ground truth annotations.

        :param results_df: DataFrame with columns sentence_id, token, annotation,
            prediction, start_indices — as produced by model.predict_dataset() and
            optionally processed by CanonicalMapper.get_mapped_results_dataframe().
        :param beta: F-beta parameter for score calculation (default 2.0).
        :param level: Which metrics to compute. One of ``"entity"``, ``"pii"``,
            or ``"both"`` (default).
        :return: EvaluationResult with the requested precision/recall/F metrics.
        """
        pass

    def get_results_dataframe(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
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

        raise DeprecationError(
            "get_results_dataframe() has been removed. Use the new 3-step pipeline:\n"
            "  1. results_df = model.predict_dataset(dataset)\n"
            "  2. mapped_df = mapper.get_mapped_results_dataframe(results_df)\n"
            "  3. result = evaluator.calculate_score_on_df(results_df=mapped_df)\n"
            "The returned DataFrame from model.predict_dataset() is already in the same format.",
        )

    @staticmethod
    def _filter_entities(
        tags: list[str],
        entities: list[str] | None = None,
    ) -> list[str]:
        """
        Filter the tags to only include the specified entities.
        If entities is None, return all tags.
        """
        if entities is None:
            return tags
        return [tag if tag in entities else "O" for tag in tags]

    @staticmethod
    def precision(
        tp: int,
        fp: int | None = None,
        num_predicted: int | None = None,
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
                "Use either fp or num_predicted, but not both.",
            )
        if fp is None and num_predicted is None:
            raise ValueError(
                "Either fp or num_predicted should be provided to calculate precision.",
            )
        if fp:
            num_predicted = fp + tp

        return tp / num_predicted if num_predicted > 0 else np.nan

    @staticmethod
    def recall(
        tp: int,
        fn: int | None = None,
        num_annotated: int | None = None,
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
                "Use either fn or num_annotated, but not both.",
            )
        if fn is None and num_annotated is None:
            raise ValueError(
                "Either fn or num_annotated should be provided to calculate recall.",
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

    def calculate_hierarchical_scores(
        self,
        results: "MappedResults",
        beta: float = 2.0,
    ) -> dict[str, EvaluationResult]:
        """Evaluate model performance at three granularity levels simultaneously.

        Accepts the output of :meth:`CanonicalMapper.get_mapped_results_dataframe`
        and produces scores at:

        - **binary**: PII vs. O — privacy-critical binary detection signal.
        - **branch**: Branch-level (PERSON, LOCATION, …) — category accuracy.
        - **detailed**: Canonical surface (NAME, STREET_ADDRESS, …) — granularity accuracy.

        :param results: :class:`~presidio_evaluator.entity_mapping.MappedResults`
            as returned by ``CanonicalMapper.get_mapped_results_dataframe()``.
        :param beta: F-beta parameter (default ``2.0``).
        :return: ``dict[str, EvaluationResult]`` with keys ``"binary"``,
            ``"branch"``, ``"detailed"``.
        """
        from presidio_evaluator.entity_mapping.data_objects import (
            MappedResults,  # noqa: PLC0415
        )

        if not isinstance(results, MappedResults):
            raise TypeError(
                f"results must be a MappedResults instance, got {type(results).__name__}. "
                "Use CanonicalMapper.get_mapped_results_dataframe() to produce it."
            )

        return {
            "binary": self.calculate_score_on_df(results.binary, beta=beta),
            "branch": self.calculate_score_on_df(results.branch, beta=beta),
            "detailed": self.calculate_score_on_df(results.detailed, beta=beta),
        }
