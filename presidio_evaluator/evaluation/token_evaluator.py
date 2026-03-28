import logging
import warnings
from collections import Counter
from typing import Literal

import pandas as pd
from spacy.tokens import Token

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import (
    BaseEvaluator,
    DeprecationError,
    ErrorType,
    EvaluationResult,
    ModelError,
)
from presidio_evaluator.evaluation.evaluation_result import PIIEvaluationMetrics

logger = logging.getLogger(__name__)


class TokenEvaluator(BaseEvaluator):
    """
    Evaluates the performance of a token-based Named Entity Recognition (NER) model.
    This class is designed to assess the model's ability to correctly identify and classify tokens in text.
    """

    def compare(
        self,
        input_sample: InputSample,
        prediction: list[str],
    ) -> tuple[Counter, list[ModelError]]:
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
                f"same length. Sample={input_sample}",
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
                    cur_annotation,
                    cur_prediction,
                    cur_token,
                    results,
                )
                if reverted:
                    continue

                token_start = (
                    input_sample.start_indices[i]
                    if input_sample.start_indices
                    else None
                )
                token_end = (
                    token_start + len(str(cur_token))
                    if token_start is not None
                    else None
                )

                if prediction[i] == "O":
                    mistakes.append(
                        ModelError(
                            error_type=ErrorType.FN,
                            annotation=cur_annotation,
                            prediction=cur_prediction,
                            token=cur_token,
                            full_text=input_sample.full_text,
                            metadata=input_sample.metadata,
                            start=token_start,
                            end=token_end,
                        ),
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
                            start=token_start,
                            end=token_end,
                        ),
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
                            start=token_start,
                            end=token_end,
                        ),
                    )

        return results, mistakes

    def __revert_known_errors(
        self,
        current_annotation: str,
        current_prediction: str,
        current_token: str | Token,
        results: Counter[tuple[str, str]],
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

    def _adjust_per_entities(self, tags: list[str]) -> list[str]:
        if self.entities_to_keep:
            return [tag if tag in self.entities_to_keep else "O" for tag in tags]
        else:
            return tags

    def evaluate_sample(
        self,
        sample: InputSample,
        prediction: list[str],
    ) -> EvaluationResult:
        warnings.warn(
            "evaluate_sample() is deprecated. Use predict_dataset() + calculate_score_on_df() instead:\n"
            "  results_df = model.predict_dataset(dataset)\n"
            "  result = evaluator.calculate_score_on_df(results_df=results_df)",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.verbose:
            logger.debug(f"Input sentence: {sample.full_text}")

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

    def calculate_score_on_df(
        self,
        results_df: pd.DataFrame,
        beta: float = 2.0,
        level: Literal["entity", "pii", "both"] = "both",
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth using a pre-mapped DataFrame.

        Accepts the same 5-column schema as SpanEvaluator.calculate_score_on_df()
        (sentence_id, token, annotation, prediction, start_indices).  For each
        sentence the tokens, annotations, and predictions are extracted and fed
        into the token-level :meth:`compare` method; then
        :meth:`calculate_score` aggregates the results.

        :param results_df: DataFrame with columns sentence_id, token, annotation,
            prediction, start_indices — as produced by model.predict_dataset() and
            optionally processed by CanonicalMapper.get_mapped_results_dataframe().
        :param beta: F-beta parameter for score calculation (default 2.0).
        :param level: Which metrics to compute. One of ``"entity"``, ``"pii"``,
            or ``"both"`` (default).
        :return: EvaluationResult with the requested precision/recall/F metrics.
        """
        evaluation_results: list[EvaluationResult] = []

        for _, sentence_df in results_df.groupby("sentence_id", sort=False):
            tokens = sentence_df["token"].tolist()
            annotations = sentence_df["annotation"].tolist()
            predictions = sentence_df["prediction"].tolist()
            start_indices = sentence_df["start_indices"].tolist()

            input_sample = InputSample(
                full_text=" ".join(tokens),
                tokens=tokens,
                tags=annotations,
                start_indices=start_indices,
            )
            results, errors = self.compare(
                input_sample=input_sample,
                prediction=predictions,
            )
            evaluation_results.append(
                EvaluationResult(
                    results=results,
                    model_errors=errors,
                    text=" ".join(tokens),
                    tokens=tokens,
                    actual_tags=annotations,
                    predicted_tags=predictions,
                    start_indices=start_indices,
                ),
            )

        return self._calculate_score(evaluation_results, beta=beta, level=level)

    def calculate_score(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
        beta: float = 2.0,
        level: Literal["entity", "pii", "both"] = "both",
    ) -> EvaluationResult:
        raise DeprecationError(
            "calculate_score() has been removed. Use calculate_score_on_df() instead:\n"
            "  result = evaluator.calculate_score_on_df(results_df=mapped_df)\n"
            "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
        )

    def _calculate_score(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
        beta: float = 2.0,
        level: Literal["entity", "pii", "both"] = "both",
    ) -> EvaluationResult:
        """Aggregate evaluation results into an EvaluationResult."""
        # Default to entities_to_keep if no explicit entities provided
        if entities is None:
            entities = self.entities_to_keep

        for res in evaluation_results:
            if not res.results:
                # token evaluation works on the results object, so run the compare method if not done yet
                input_sample = InputSample(
                    full_text=res.text or "",
                    tokens=res.tokens,
                    tags=res.actual_tags,
                )
                results, errors = self.compare(
                    input_sample=input_sample,
                    prediction=res.predicted_tags or [],
                )
                res.results = results
                res.model_errors = errors

        # aggregate results
        all_results = sum([er.results for er in evaluation_results], Counter())

        # per-entity metrics
        per_type: dict[str, PIIEvaluationMetrics] = {}
        if level in ("entity", "both"):
            if not entities:
                entities1 = list({x[0] for x in all_results.keys() if x[0] != "O"})
                entities2 = list({x[1] for x in all_results.keys() if x[1] != "O"})
                entities = list(set(entities1).union(set(entities2)))

            for entity in entities:
                annotated = sum([all_results[x] for x in all_results if x[0] == entity])
                predicted = sum([all_results[x] for x in all_results if x[1] == entity])
                tp = all_results[(entity, entity)]
                recall = self.recall(tp=tp, num_annotated=annotated)
                precision = self.precision(tp=tp, num_predicted=predicted)
                per_type[entity] = PIIEvaluationMetrics(
                    precision=precision,
                    recall=recall,
                    f_beta=self.f_beta(precision=precision, recall=recall, beta=beta),
                    num_predicted=predicted,
                    num_annotated=annotated,
                    true_positives=tp,
                    false_positives=predicted - tp,
                    false_negatives=annotated - tp,
                )

        # global PII metrics
        pii_recall = None
        pii_precision = None
        pii_f_beta = None
        pii_tp = pii_predicted = pii_annotated = pii_fp = pii_fn = 0
        if level in ("pii", "both"):
            pii_annotated = sum([all_results[x] for x in all_results if x[0] != "O"])
            pii_predicted = sum([all_results[x] for x in all_results if x[1] != "O"])
            pii_tp = sum(
                [all_results[x] for x in all_results if (x[0] != "O" and x[1] != "O")]
            )
            pii_fp = pii_predicted - pii_tp
            pii_fn = pii_annotated - pii_tp
            pii_recall = self.recall(tp=pii_tp, num_annotated=pii_annotated)
            pii_precision = self.precision(tp=pii_tp, num_predicted=pii_predicted)
            pii_f_beta = self.f_beta(pii_precision, pii_recall, beta)

        # aggregate errors
        errors = []
        for res in evaluation_results:
            if res.model_errors:
                errors.extend(res.model_errors)

        evaluation_result = EvaluationResult(
            results=all_results,
            model_errors=errors,
            pii_precision=pii_precision,
            pii_recall=pii_recall,
            per_type=per_type if per_type else None,
            pii_f=pii_f_beta,
            n=sum(m.num_annotated for m in per_type.values()) if per_type else None,
            pii_true_positives=pii_tp,
            pii_false_positives=pii_fp,
            pii_false_negatives=pii_fn,
            pii_predicted=pii_predicted,
            pii_annotated=pii_annotated,
        )

        return evaluation_result


class Evaluator(TokenEvaluator):
    """DEPRECATED: Use Token/Span Evaluator instead."""

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "The 'Evaluator' class is deprecated. "
            "Please update your code to use either "
            "SpanEvaluator or TokenEvaluator instead.",
        )
