from collections import defaultdict
from typing import List, Optional, Union, Set, Tuple, Dict
import pandas as pd

from presidio_analyzer import AnalyzerEngine

from presidio_evaluator.evaluation import BaseEvaluator, ModelError, ErrorType
from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation.evaluation_result import (
    EvaluationResult,
)
from presidio_evaluator.models import BaseModel


class SpanEvaluator(BaseEvaluator):
    """
    Evaluates PII detection using span-based fuzzy matching with character-level Intersection over Union (IoU).
    """

    def __init__(
        self,
        model: Optional[Union[BaseModel, AnalyzerEngine]],
        entity_mapping: Dict[str, str],
        verbose: bool = False,
        compare_by_io: bool = True,
        entities_to_keep: Optional[List[str]] = None,
        generic_entities: Optional[List[str]] = None,
        skip_words: Optional[List] = None,
        iou_threshold: float = 0.9,
        char_based: bool = True,
    ):
        """
        Initialize the SpanEvaluator for evaluating pii entities detection results.

        :param model: Instance of a fitted model or Presidio Analyzer
        :param entity_mapping: REQUIRED. Dictionary mapping dataset entity types to model entity types.
        :param verbose: Whether to print debug information
        :param compare_by_io: Whether to compare on entity level only
        :param entities_to_keep: List of entity names to focus on
        :param generic_entities: List of entities that are not considered errors
        :param skip_words: List of words to skip during evaluation
        :param iou_threshold: Minimum Intersection over Union (IoU) threshold for considering spans as matching.
                            Value between 0 and 1, where higher values require more overlap (default: 0.5)
        :param skip_words: Optional list of custom skip words to ignore during token normalization,
                            should also include punctuation marks.
                         If None, uses skip words from skipwords.py (default: None).
                         Pass an empty list ([]) to disable skip word removal entirely.
        :param char_based: If True, calculate IoU at the character-level, else, calculate iou at the token-level.
        """
        super().__init__(
            model=model,
            entity_mapping=entity_mapping,
            verbose=verbose,
            compare_by_io=compare_by_io,
            entities_to_keep=entities_to_keep,
            generic_entities=generic_entities,
            skip_words=skip_words,
        )

        self.iou_threshold = iou_threshold
        self.char_based = char_based

    def _normalize_tokens(
        self, tokens: List[str], start_indices: Optional[List[int]] = None
    ) -> Union[List[str], Tuple[List[str], List[int]]]:
        """
        Normalize tokens by:
        1. Converting to lowercase
        2. Removing stop words
        3. Removing standalone punctuation
        4. Removing skip words (common words that shouldn't affect entity matching)

        :param tokens: List of token strings to normalize
        :return: List of normalized tokens
        """

        if not start_indices:
            start_indices = [None] * len(tokens)  # placeholder
        normalized = []
        normalized_indices = []
        for token, start in zip(tokens, start_indices):
            token = token.lower()
            # Skip if token is in skip words list
            if token in self.skip_words:
                continue
            normalized.append(token)
            normalized_indices.append(start)

        if not start_indices:
            return normalized

        return normalized, normalized_indices

    def _merge_adjacent_spans(self, spans: List[Span], df: pd.DataFrame) -> List[Span]:
        """
        Merge adjacent spans of the same entity type if separated only by skip words / punctuation.

        :param spans: List of Span objects to potentially merge
        :param df: DataFrame containing the tokens and their positions
        :return: List of merged Span objects
        """
        if not spans:
            return []
        spans = sorted(spans, key=lambda x: x.start_position)
        merged = []
        current = spans[0]

        for next_span in spans[1:]:
            if (
                current.entity_type == next_span.entity_type
                and self._are_spans_adjacent(current, next_span, df)
            ):
                merged_tokens = [current.entity_value, next_span.entity_value]
                merged_normalized_text = (
                    current.normalized_tokens + next_span.normalized_tokens
                )
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=" ".join(merged_tokens),
                    start_position=current.start_position,
                    end_position=next_span.end_position,
                    normalized_start_index=min(
                        current.normalized_start_index, next_span.normalized_start_index
                    ),
                    normalized_end_index=max(
                        current.normalized_end_index, next_span.normalized_end_index
                    ),
                    normalized_tokens=merged_normalized_text,
                    token_start=current.token_start,
                    token_end=next_span.token_end,
                )
            else:
                merged.append(current)
                current = next_span

        merged.append(current)
        return merged

    def _are_spans_adjacent(self, span1: Span, span2: Span, df: pd.DataFrame) -> bool:
        """
        Check if two spans are adjacent, i.e., separated only by skipwords / punctuation or whitespace tokens.

        :param span1: First Span object
        :param span2: Second Span object
        :param df: DataFrame containing the tokens
        :return: True if spans are adjacent, False otherwise
        """
        # Slice tokens between span1 and span2 using the row indices
        between_tokens = df.loc[
            span1.token_end : span2.token_start - 1, "token"
        ].tolist()
        non_skip_tokens = [
            tok for tok in between_tokens if tok.lower().strip() not in self.skip_words
        ]
        return len(non_skip_tokens) == 0

    @staticmethod
    def calculate_iou(
        span1: Span,
        span2: Span,
        ignore_entity_type: bool = True,
        use_normalized_indices: bool = True,
        char_based: bool = True,
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) between two spans at character or token level.

        :param span1: First Span object
        :param span2: Second Span object
        :param ignore_entity_type: If True, ignores the entity type when calculating IoU
        :param use_normalized_indices: If True, uses normalized indices for IoU calculation
        :param char_based: If True, calculates IoU at character level, else at token level

        """
        if char_based:
            iou = span1.iou(
                other=span2,
                ignore_entity_type=ignore_entity_type,
                use_normalized_indices=use_normalized_indices,
            )
        else:
            range1 = set(span1.normalized_tokens)
            range2 = set(span2.normalized_tokens)

            intersection = len(range1.intersection(range2))
            union = len(range1.union(range2))

            iou = intersection / union if union > 0 else 0.0

        return iou

    def _process_sentence_spans(
        self, sentence_df: pd.DataFrame
    ) -> Tuple[List[Span], List[Span]]:
        annotation_spans = self._create_spans(
            df=sentence_df, column="annotation", apply_entity_mapping=True
        )
        prediction_spans = self._create_spans(
            df=sentence_df, column="prediction", apply_entity_mapping=False
        )

        annotation_spans = self._merge_adjacent_spans(
            spans=annotation_spans, df=sentence_df
        )
        prediction_spans = self._merge_adjacent_spans(
            spans=prediction_spans, df=sentence_df
        )

        return annotation_spans, prediction_spans

    @staticmethod
    def _handle_unmatched_predictions(
        prediction_spans: List[Span],
        matched_preds: Set[Tuple[str, int, int]],
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """
        Handle predictions that weren't matched to any annotation.

        :param prediction_spans: List of prediction Span objects
        :param matched_preds: Set of already matched prediction spans
        :param evaluation_result: EvaluationResult object to update

        """
        if not evaluation_result.model_errors:
            evaluation_result.model_errors = []

        for pred_span in prediction_spans:
            pred_span_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )
            if pred_span_key not in matched_preds:
                evaluation_result.results[("O", pred_span.entity_type)] += 1
                evaluation_result.pii_false_positives += 1
                evaluation_result.per_type[pred_span.entity_type].false_positives += 1
                model_error = ModelError(
                    error_type=ErrorType.FP,
                    annotation="O",
                    prediction=pred_span.entity_type,
                    full_text=pred_span.entity_value,
                    token=" ".join(pred_span.normalized_tokens),
                    explanation=f"False positive for {pred_span}",
                )
                evaluation_result.model_errors.append(model_error)

        return evaluation_result

    @staticmethod
    def _check_if_matched_already(
        pred_span: Span, ann_span: Span, matched_preds: set
    ) -> bool:
        """
        Check if a prediction span is valid for matching with an annotation span.

        A prediction is valid if:
        1. It hasn't already been matched to another annotation
        2. Its entity type matches the annotation's entity type

        Args:
            pred_span: The prediction Span to check
            ann_span: The annotation Span being matched against
            matched_preds: Set of already matched prediction spans

        Returns:
            bool: True if the prediction is valid for matching, False otherwise
        """
        # Create unique key for the prediction span
        pred_span_key = (
            pred_span.entity_type,
            pred_span.start_position,
            pred_span.end_position,
        )

        # Check if prediction is already matched
        if pred_span_key in matched_preds:
            return False

        return True

    def _find_best_match(
        self,
        ann_span: Span,
        prediction_spans: List[Span],
        matched_preds: Set[Tuple[str, int, int]],
    ) -> Tuple[Optional[Span], float]:
        """
        Find the best matching prediction span for a given annotation span.

        :param ann_span: The annotation Span to match against
        :param prediction_spans: List of prediction Span objects
        :param matched_preds: Set of already matched prediction spans to avoid duplicates
        """
        best_match = None
        best_iou = 0.0

        for pred_span in prediction_spans:
            if self._check_if_matched_already(
                pred_span=pred_span, ann_span=ann_span, matched_preds=matched_preds
            ):
                iou = self.calculate_iou(
                    span1=ann_span,
                    span2=pred_span,
                    ignore_entity_type=True,
                    use_normalized_indices=True,
                    char_based=self.char_based,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_span

        return best_match, best_iou

    def _update_result_with_overall_metrics(
        self,
        evaluation_result: EvaluationResult,
        beta: float,
    ) -> None:
        """
        Update the evaluation result with overall metrics and per-type metrics.

        :param evaluation_result: EvaluationResult object to update
        :param beta: The beta parameter for F-beta score calculation.
        """

        precision, recall, f_beta = self._calculate_metrics(
            evaluation_result.pii_true_positives,
            evaluation_result.pii_predicted,
            evaluation_result.pii_annotated,
            beta,
        )
        evaluation_result.pii_recall = recall
        evaluation_result.pii_precision = precision
        evaluation_result.pii_f = f_beta

    def _update_per_type_metrics(
        self,
        evaluation_result: EvaluationResult,
        beta: float,
    ) -> None:
        """
        Update per-type metrics in the evaluation result.

        :param evaluation_result: EvaluationResult object containing per-type metrics
        :param beta: F-beta parameter

        """

        for entity_type, pii_metrics in evaluation_result.per_type.items():
            # Calculate metrics for this entity type
            precision, recall, f_beta = self._calculate_metrics(
                pii_metrics.true_positives,
                pii_metrics.num_predicted,
                pii_metrics.num_annotated,
                beta,
            )
            pii_metrics.precision = precision
            pii_metrics.recall = recall
            pii_metrics.f_beta = f_beta

    @staticmethod
    def create_global_entities_df(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame containing global PII entities from the results DataFrame.

        :param results_df: DataFrame containing the evaluation results
        :return: DataFrame with global entities and their counts
        """
        # Create a deep copy to avoid modifying the original DataFrame
        global_df = results_df.copy(deep=True)
        global_df["annotation"] = global_df["annotation"].apply(
            lambda x: "O" if x == "O" else "PII"
        )
        global_df["prediction"] = global_df["prediction"].apply(
            lambda x: "O" if x == "O" else "PII"
        )
        return global_df

    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        """
        Calculate the evaluation score based on the provided evaluation results (evaluation run).
        :param evaluation_results: List of EvaluationResult objects containing the results of the evaluation run,
        specifically `actual_tags` and `predicted_tags`.
        :param entities: Optional list of entities to filter the evaluation results by.
        If None, defaults to the evaluator's entities_to_keep (set in constructor).
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        """
        # Default to entities_to_keep if no explicit entities provided
        if entities is None:
            entities = self.entities_to_keep
        
        evaluation_result = EvaluationResult()
        df = self.get_results_dataframe(
            evaluation_results=evaluation_results, entities=entities
        )
        evaluation_result = self.calculate_score_on_df(
            per_type=True, results_df=df, beta=beta, evaluation_result=evaluation_result
        )
        global_pii_df = self.create_global_entities_df(results_df=df)
        evaluation_result = self.calculate_score_on_df(
            per_type=False,
            results_df=global_pii_df,
            beta=beta,
            evaluation_result=evaluation_result,
        )
        return evaluation_result

    def calculate_score_on_df(
        self,
        per_type: bool,
        results_df: pd.DataFrame,
        beta: float = 2,
        evaluation_result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate the predictions against ground truth annotations.
        This method processes a DataFrame containing evaluation results and calculates metrics
        using span-based fuzzy matching with IoU threshold. It can operate in two modes:
        per-entity type evaluation or global PII evaluation.

        :param per_type: If True, performs per-entity type evaluation; if False, performs
                    global PII vs non-PII evaluation
        :param results_df: DataFrame containing sentence_id, tokens, token start indices,
                        annotations and predictions columns
        :param beta: The beta parameter for F-beta score calculation. Higher values weight
                    recall more than precision. Default is 2.
        :param evaluation_result: Optional existing EvaluationResult to update. If None,
                                creates a new one.
        :return: EvaluationResult object containing computed metrics, counts, and error analysis

        """
        if not evaluation_result:
            evaluation_result = EvaluationResult()

        # Process each sentence
        for _, sentence_df in results_df.groupby("sentence_id"):
            # Get and process spans for the sentence
            evaluation_result = self._compare_one_sentence(
                sentence_df=sentence_df,
                per_type=per_type,
                evaluation_result=evaluation_result,
            )
        # Create and return the final evaluation result
        if per_type:
            self._update_per_type_metrics(evaluation_result, beta)
        else:
            self._update_result_with_overall_metrics(
                evaluation_result,
                beta,
            )

        return evaluation_result

    def _compare_one_sentence(
        self,
        sentence_df: pd.DataFrame,
        per_type: bool,
        evaluation_result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Compare one sentence's annotations and predictions, updating the evaluation result.

        :param per_type: If True, performs per-entity type evaluation; if False, performs
                global PII vs non-PII evaluation
        :param sentence_df: DataFrame containing sentence_id, tokens, token start indices,
                annotations and predictions columns
        :param evaluation_result: Optional existing EvaluationResult to update. If None,
                creates a new one.

        """
        if not evaluation_result:
            evaluation_result = EvaluationResult()

        annotation_spans, prediction_spans = self._process_sentence_spans(sentence_df)
        # Match predictions with annotations and update metrics
        evaluation_result = self._match_predictions_with_annotations(
            annotation_spans, prediction_spans, evaluation_result, per_type
        )
        return evaluation_result

    def _create_spans(
        self, df: pd.DataFrame, column: str, apply_entity_mapping: bool = False
    ) -> List[Span]:
        """
        Create spans from a DataFrame column.

        :param df: DataFrame containing the spans.
        :param column: Name of the column to extract spans from.
        :param apply_entity_mapping: If True, applies entity mapping to translate dataset entities to model entities.

        Returns:
            List[Span]: List of Span objects created from the DataFrame.
        """
        spans = []
        current_entity_type = None
        current_tokens = []
        current_start_indices = []
        current_token_start = None  # Add token position tracking
        curr_char_position = 0
        token_position = 0  # Add token position counter

        for idx, (_, row) in enumerate(df.iterrows()):
            entity_type = row[column]
            if self.compare_by_io:
                entity_type = self._to_io([entity_type])[0]

            # Apply entity mapping for annotation column to translate dataset entities to model entities
            if apply_entity_mapping and entity_type != "O":
                entity_type = self._normalize_entity_for_comparison(
                    entity_type, self.entity_mapping
                )
            token = row["token"]
            token_start = row["start_indices"]
            token_length = len(token)
            # If this isn't the first token, add space before it
            if idx > df.index[0]:
                curr_char_position += 1  # Account for space between tokens

            token_end = curr_char_position + token_length

            if entity_type == "O":
                if current_entity_type and current_tokens:
                    normalized_tokens, normalized_start_indices = (
                        self._normalize_tokens(current_tokens, current_start_indices)
                    )
                    if normalized_tokens:
                        spans.append(
                            self.__create_span(
                                entity_type=current_entity_type,
                                start_indices=current_start_indices,
                                token_start=current_token_start,
                                current_tokens=current_tokens,
                                idx=idx,
                                normalized_satrt_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            )
                        )
                    current_entity_type = None
                    current_tokens = []
                    current_start_indices = []
                    current_token_start = None

                curr_char_position = token_end
                token_position += 1  # Increment token position
                continue

            if entity_type != current_entity_type:
                if current_entity_type and current_tokens:
                    normalized_tokens, normalized_start_indices = (
                        self._normalize_tokens(current_tokens, current_start_indices)
                    )
                    if normalized_tokens:
                        spans.append(
                            self.__create_span(
                                entity_type=current_entity_type,
                                start_indices=current_start_indices,
                                token_start=current_token_start,
                                current_tokens=current_tokens,
                                idx=idx,
                                normalized_satrt_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            )
                        )
                current_entity_type = entity_type
                current_tokens = [token]
                current_start_indices = [token_start]
                current_token_start = idx  # Set token start position

            else:
                current_tokens.append(token)
                current_start_indices.append(token_start)
            curr_char_position = token_end
            token_position += 1  # Increment token position

        # Handle final span
        if current_entity_type and current_tokens:
            normalized_tokens, normalized_start_indices = self._normalize_tokens(
                current_tokens, current_start_indices
            )
            if normalized_tokens:
                spans.append(
                    self.__create_span(
                        entity_type=current_entity_type,
                        start_indices=current_start_indices,
                        token_start=current_token_start,
                        current_tokens=current_tokens,
                        idx=df.index[-1] + 1,
                        normalized_satrt_indices=normalized_start_indices,
                        normalized_tokens=normalized_tokens,
                    )
                )
        return spans

    def __create_span(
        self,
        entity_type: str,
        start_indices: List[int],
        token_start: int,
        current_tokens: List[str],
        idx: int,
        normalized_satrt_indices: List[int],
        normalized_tokens: List[str],
    ):
        return Span(
            entity_type=entity_type,
            entity_value=" ".join(current_tokens),
            start_position=start_indices[0],
            end_position=start_indices[-1] + len(current_tokens[-1]),
            normalized_tokens=normalized_tokens,
            normalized_start_index=min(normalized_satrt_indices),
            normalized_end_index=self._get_normalized_end_index(
                normalized_tokens, normalized_satrt_indices
            ),
            token_start=token_start,
            token_end=idx,
        )

    @staticmethod
    def _get_normalized_end_index(
        normalized_tokens: List[str], normalized_indices: List[int]
    ) -> int:
        """Calculate the end character index of the last token in the normalized tokens list."""  # noqa: E501
        return max(
            [
                len(tok) + start
                for tok, start in zip(normalized_tokens, normalized_indices)
            ]
        )

    def _calculate_metrics(
        self,
        true_positives: int,
        num_predicted: int,
        num_annotated: int,
        beta: float = 2,
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F-beta score using the new logic.

        :param true_positives: Number of true positives
        :param num_predicted: Number of predicted spans
        :param num_annotated: Number of annotated (gold) spans
        :param beta: The beta parameter for F-beta score calculation. Default is 2.
        :return: Dictionary containing precision, recall, and f-beta metrics
        """
        precision = self.precision(tp=true_positives, num_predicted=num_predicted)
        recall = self.recall(tp=true_positives, num_annotated=num_annotated)
        f_beta = self.f_beta(precision=precision, recall=recall, beta=beta)
        return precision, recall, f_beta

    def _match_predictions_with_annotations(
        self,
        annotation_spans: List[Span],
        prediction_spans: List[Span],
        evaluation_result: EvaluationResult,
        per_type: bool = True,
    ) -> EvaluationResult:
        if not evaluation_result.model_errors:
            evaluation_result.model_errors = []

        # Track which prediction spans have been processed
        processed_predictions = set()

        for ann_span in annotation_spans:
            ann_type = ann_span.entity_type
            self._add_to_annotated(evaluation_result, per_type, ann_type)

            # Find all overlapping prediction spans with IoU > 0, regardless of type
            overlapping_preds = self._get_all_overlapping(ann_span, prediction_spans)

            self._add_to_processed_predictions(
                processed_predictions,
                overlapping_preds,
            )

            if not overlapping_preds:  # scenario 2- there is no prediction
                if per_type:
                    evaluation_result.per_type[ann_type].false_negatives += 1
                    evaluation_result.results[(ann_type, "O")] += 1
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span, pred_span=None, error_type=ErrorType.FN
                        )
                    )
                else:
                    evaluation_result.pii_false_negatives += 1

            elif (
                len(overlapping_preds) == 1
            ):  # scenario group 1 (single overlap of same/different type and below/above threshold)
                self._compare_single_overlaps(
                    evaluation_result=evaluation_result,
                    ann_span=ann_span,
                    overlapping_preds=overlapping_preds,
                    per_type=per_type,
                )

            # Handle pred_span aggregation cases
            else:  # Scenario group 2
                self._compare_multiple_overlaps(
                    evaluation_result=evaluation_result,
                    ann_span=ann_span,
                    overlapping_preds=overlapping_preds,
                    per_type=per_type,
                )

        # Handle prediction spans that don't overlap with any annotation span
        for pred_span in prediction_spans:
            pred_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )

            # If this prediction has not been processed (no overlap with any annotation)
            if pred_key not in processed_predictions:
                if per_type:
                    evaluation_result.per_type[
                        pred_span.entity_type
                    ].false_positives += 1
                    evaluation_result.per_type[pred_span.entity_type].num_predicted += 1
                else:
                    evaluation_result.pii_false_positives += 1
                    evaluation_result.pii_predicted += 1

                # Add to confusion matrix
                evaluation_result.results[("O", pred_span.entity_type)] = (
                    evaluation_result.results.get(("O", pred_span.entity_type), 0) + 1
                )

                # Add error
                evaluation_result.model_errors.append(
                    ModelError(
                        error_type=ErrorType.FP,
                        annotation="O",
                        prediction=pred_span.entity_type,
                        full_text=pred_span.entity_value,
                        token=" ".join(pred_span.normalized_tokens),
                        explanation=f"False prediction with no overlap: {pred_span.entity_type}",
                    )
                )

        return evaluation_result

    def _compare_single_overlaps(
        self,
        evaluation_result: EvaluationResult,
        ann_span: Span,
        overlapping_preds: List[Tuple[Span, float]],
        per_type: bool,
    ):
        """Calculate metrics for a single overlapping prediction span (Scenario group 1)."""

        ann_type = ann_span.entity_type
        pred_span, iou = overlapping_preds[0]
        pred_type = pred_span.entity_type
        if iou >= self.iou_threshold:  # Scenarios 1 (TP) or 4 (Wrong Entity)
            if pred_type == ann_type:  # scenario 1 (TP)
                if per_type:
                    evaluation_result.per_type[ann_type].true_positives += 1
                    evaluation_result.per_type[pred_type].num_predicted += 1
                    evaluation_result.results[(ann_type, pred_type)] += 1
                else:
                    evaluation_result.pii_true_positives += 1
                    evaluation_result.pii_predicted += 1
            else:  # Scenario 4 (Wrong Entity)
                if per_type:
                    evaluation_result.per_type[ann_type].false_negatives += 1
                    evaluation_result.per_type[pred_type].false_positives += 1
                    evaluation_result.per_type[pred_type].num_predicted += 1
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.WrongEntity,
                            iou=iou,
                        )
                    )
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FN,
                            iou=iou,
                        )
                    )
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FP,
                            iou=iou,
                        )
                    )
                    evaluation_result.results[(ann_type, pred_type)] += 1
                else:
                    evaluation_result.pii_false_negatives += 1
                    evaluation_result.pii_false_positives += 1
                    evaluation_result.pii_predicted += 1

        else:  # Scenario 5 (low IoU overlap)
            if (
                ann_type == pred_type
            ):  # Scenario 5a (FN and FP - treat as separate entities)
                if per_type:
                    evaluation_result.per_type[ann_type].false_negatives += 1
                    evaluation_result.per_type[pred_type].false_positives += 1
                    evaluation_result.per_type[pred_type].num_predicted += 1
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FN,
                            iou=iou,
                        )
                    )
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FP,
                            iou=iou,
                        )
                    )
                    evaluation_result.results[(ann_type, "O")] += 1
                    evaluation_result.results[("O", pred_type)] += 1
                else:
                    evaluation_result.pii_false_negatives += 1
                    evaluation_result.pii_false_positives += 1
                    evaluation_result.pii_predicted += 1

            else:  # Scenario 5b - different types (FN and FP)
                if per_type:
                    evaluation_result.per_type[ann_type].false_negatives += 1
                    evaluation_result.per_type[pred_type].false_positives += 1
                    evaluation_result.per_type[pred_type].num_predicted += 1

                    # Add two errors, one as FP and the other as FN (not WrongEntity due to low IoU)
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FN,
                            iou=iou,
                        )
                    )
                    evaluation_result.model_errors.append(
                        self._get_model_error(
                            ann_span=ann_span,
                            pred_span=pred_span,
                            error_type=ErrorType.FP,
                            iou=iou,
                        )
                    )

                    evaluation_result.results[(ann_type, "O")] += 1
                    evaluation_result.results[("O", pred_type)] += 1
                else:
                    evaluation_result.pii_false_negatives += 1
                    evaluation_result.pii_false_positives += 1
                    evaluation_result.pii_predicted += 1

    def _compare_multiple_overlaps(
        self,
        evaluation_result: EvaluationResult,
        ann_span: Span,
        overlapping_preds: List[Tuple[Span, float]],
        per_type: bool,
    ):
        """Calculate metrics for an annotation span overlapping with multiple pred spans."""

        annotation_was_counted = (
            False  # Only count as FN once if matched multiple times
        )
        ann_type = ann_span.entity_type
        # Group overlapping spans by entity type
        spans_by_type, iou_by_type = self._group_spans_by_type(overlapping_preds)

        # Calculate cumulative IoU per type
        cumulative_iou_by_type = {}
        for entity_type, spans in spans_by_type.items():
            cumulative_iou_by_type[entity_type] = self._calculate_combined_iou(
                ann_span, spans
            )

        for cumulative_type, iou_per_type in cumulative_iou_by_type.items():
            # Check if there are spans of the same type as the annotation (Scenario 6)
            if ann_type == cumulative_type:
                same_type_spans = spans_by_type[cumulative_type]

                if iou_per_type >= self.iou_threshold:
                    # Scenario 6A: Cumulative IoU with spans of the same type > threshold
                    if per_type:
                        if not annotation_was_counted:
                            annotation_was_counted = True
                        evaluation_result.per_type[
                            ann_span.entity_type
                        ].true_positives += 1
                        evaluation_result.per_type[cumulative_type].num_predicted += 1
                        evaluation_result.results[(ann_type, ann_type)] += 1
                    else:
                        evaluation_result.pii_true_positives += 1
                        evaluation_result.pii_predicted += 1
                else:
                    # Scenario 6B: Cumulative IoU with spans of the same type < threshold
                    if per_type:
                        if not annotation_was_counted:
                            evaluation_result.per_type[
                                ann_span.entity_type
                            ].false_negatives += 1
                            evaluation_result.model_errors.append(
                                self._get_model_error(
                                    ann_span=ann_span,
                                    pred_span=same_type_spans[0],
                                    error_type=ErrorType.FN,
                                    iou=iou_per_type,
                                )
                            )
                            evaluation_result.results[(ann_type, "O")] += 1
                            annotation_was_counted = True
                        # For low IoU same-type cases, treat predictions as false positives
                        evaluation_result.per_type[cumulative_type].false_positives += 1
                        evaluation_result.per_type[cumulative_type].num_predicted += 1
                        evaluation_result.model_errors.append(
                            self._get_model_error(
                                ann_span=ann_span,
                                pred_span=same_type_spans[0],
                                error_type=ErrorType.FP,
                                iou=iou_per_type,
                            )
                        )
                        evaluation_result.results[("O", cumulative_type)] += 1

                    else:
                        if not annotation_was_counted:
                            evaluation_result.pii_false_negatives += 1
                            annotation_was_counted = True
                        evaluation_result.pii_false_positives += 1
                        evaluation_result.pii_predicted += 1

            else:
                # Scenarios 7a,b: Cumulative IoU with spans of a different type
                different_type_spans = spans_by_type[cumulative_type]

                if iou_per_type >= self.iou_threshold:
                    # Scenario 7A: Cumulative IoU with spans of a different type > threshold
                    if per_type:
                        if not annotation_was_counted:
                            evaluation_result.per_type[ann_type].false_negatives += 1
                            evaluation_result.model_errors.append(
                                self._get_model_error(
                                    ann_span=ann_span,
                                    pred_span=different_type_spans[0],
                                    error_type=ErrorType.FN,
                                    iou=iou_per_type,
                                )
                            )
                            annotation_was_counted = True

                        evaluation_result.per_type[cumulative_type].false_positives += 1
                        evaluation_result.per_type[cumulative_type].num_predicted += 1
                        evaluation_result.model_errors.append(
                            self._get_model_error(
                                ann_span=ann_span,
                                pred_span=different_type_spans[0],
                                error_type=ErrorType.WrongEntity,
                                iou=iou_per_type,
                            )
                        )

                        evaluation_result.model_errors.append(
                            self._get_model_error(
                                ann_span=ann_span,
                                pred_span=different_type_spans[0],
                                error_type=ErrorType.FP,
                                iou=iou_per_type,
                            )
                        )
                        evaluation_result.results[(ann_type, cumulative_type)] += 1
                    else:
                        if not annotation_was_counted:
                            evaluation_result.pii_false_negatives += 1
                            annotation_was_counted = True
                        evaluation_result.pii_false_positives += 1
                        evaluation_result.pii_predicted += 1

                else:
                    # Scenario 7B: Cumulative IoU with spans of a different type < threshold
                    if per_type:
                        if not annotation_was_counted:
                            evaluation_result.per_type[ann_type].false_negatives += 1
                            evaluation_result.model_errors.append(
                                self._get_model_error(
                                    ann_span=ann_span,
                                    pred_span=different_type_spans[0],
                                    error_type=ErrorType.FN,
                                    iou=iou_per_type,
                                )
                            )
                            annotation_was_counted = True

                        evaluation_result.per_type[cumulative_type].false_positives += 1
                        evaluation_result.per_type[cumulative_type].num_predicted += 1

                        # Add two errors, one as FP and the other as FN (not WrongEntity due to low IoU)

                        evaluation_result.model_errors.append(
                            self._get_model_error(
                                ann_span=ann_span,
                                pred_span=different_type_spans[0],
                                error_type=ErrorType.FP,
                                iou=iou_per_type,
                            )
                        )
                        evaluation_result.results[(ann_type, "O")] += 1
                        evaluation_result.results[("O", cumulative_type)] += 1
                    else:
                        if not annotation_was_counted:
                            evaluation_result.pii_false_negatives += 1
                            annotation_was_counted = True
                        evaluation_result.pii_false_positives += 1
                        evaluation_result.pii_predicted += 1

    @staticmethod
    def _group_spans_by_type(
        overlapping_preds: List[Tuple[Span, float]],
    ) -> Tuple[Dict[str, List[Span]], Dict[str, List[float]]]:
        """
        Group spans by entity type and their corresponding IoU values.

        :param overlapping_preds: List of spans to group, with their corresponding IoU values.
        """
        spans_by_type = defaultdict(list)
        iou_by_type = defaultdict(list)

        for pred_span, iou in overlapping_preds:
            spans_by_type[pred_span.entity_type].append(pred_span)
            iou_by_type[pred_span.entity_type].append(iou)
        return spans_by_type, iou_by_type

    @staticmethod
    def _add_to_processed_predictions(
        processed_predictions: Set[Tuple[str, int, int]],
        overlapping_preds: List[Tuple[Span, float]],
    ) -> None:
        """
        Update processed predictions set with all overlapping predictions.
        :param processed_predictions: Set of already processed prediction spans
        :param overlapping_preds: List of tuples containing overlapping prediction spans and their IoU scores

        """
        # Add all overlapping predictions to processed set
        for pred_span, _ in overlapping_preds:
            pred_key = (
                pred_span.entity_type,
                pred_span.start_position,
                pred_span.end_position,
            )
            processed_predictions.add(pred_key)

    def _get_model_error(
        self,
        ann_span: Optional[Span],
        pred_span: Optional[Span],
        error_type: ErrorType,
        iou: float = 0.0,
    ):
        def get_explanation():
            match error_type:
                case ErrorType.FP:
                    return (
                        f"Entity {pred_span.entity_type} falsely detected, iou={iou:.2f} "
                        f"compared to threshold={self.iou_threshold}"
                    )
                case ErrorType.FN:
                    if (
                        pred_span and pred_span.entity_type == ann_span.entity_type
                    ):  # FN due to low IoU
                        return (
                            f"Entity {ann_span.entity_type} not detected due to low iou={iou:.2f} "
                            f"compared to threshold={self.iou_threshold}"
                        )
                    elif pred_span and pred_span.entity_type != ann_span.entity_type:
                        return (
                            f"Entity {ann_span.entity_type} not detected. "
                            f"iou with {pred_span.entity_type}={iou:.2f} "
                            f"compared to threshold={self.iou_threshold}"
                        )
                    else:
                        return f"Entity {ann_span.entity_type} not detected."
                case ErrorType.WrongEntity:
                    return (
                        f"Wrong entity type: {ann_span.entity_type} detected as "
                        f"{pred_span.entity_type}, iou={iou:.2f} "
                        f"compared to threshold={self.iou_threshold}"
                    )

            return ValueError(f"Unknown or missing error type: {error_type}")

        prediction = "O" if error_type == ErrorType.FN else pred_span.entity_type
        annotation = "O" if error_type == ErrorType.FP else ann_span.entity_type
        explanation = get_explanation()

        return ModelError(
            error_type=error_type,
            annotation=annotation,
            prediction=prediction,
            full_text=pred_span.entity_value
            if error_type == ErrorType.FP
            else ann_span.entity_value,
            token=" ".join(
                pred_span.normalized_tokens
                if error_type == ErrorType.FP
                else ann_span.normalized_tokens
            ),
            explanation=explanation,
        )

    def _get_all_overlapping(
        self, ann_span: Span, prediction_spans: List[Span]
    ) -> List[Tuple[Span, float]]:
        """Get all prediction spans that overlap with the annotation span, regardless of type.

        :param ann_span: The annotation Span to match against
        :param prediction_spans: List of all prediction Span objects
        :return: List of tuples containing overlapping prediction spans and their IoU scores
        """

        overlapping_preds = []
        for pred_span in prediction_spans:
            iou = self.calculate_iou(ann_span, pred_span, char_based=self.char_based)
            if iou > 0:
                overlapping_preds.append((pred_span, iou))

        overlapping_preds.sort(key=lambda x: x[0].start_position)

        return overlapping_preds

    @staticmethod
    def _update_wrong_entities(
        overlapping_preds,
        annotated_entity_type,
        matched_predictions,
        ann_span,
        evaluation_result,
    ) -> EvaluationResult:
        non_type_matching_preds = [
            (p, iou)
            for p, iou in overlapping_preds
            if p.entity_type != annotated_entity_type
            and (p.entity_type, p.start_position, p.end_position)
            not in matched_predictions
        ]

        for non_matching_pred, iou in non_type_matching_preds:
            # Record entity type mismatches in error analysis
            if non_matching_pred.entity_type != ann_span.entity_type:
                evaluation_result.model_errors.append(
                    ModelError(
                        error_type=ErrorType.WrongEntity,
                        annotation=ann_span.entity_type,
                        prediction=non_matching_pred.entity_type,
                        full_text=ann_span.entity_value,
                        token=" ".join(ann_span.normalized_tokens),
                        explanation=f"Wrong entity type: {ann_span.entity_type} detected as {non_matching_pred.entity_type}, iou={iou:.2f}",
                    )
                )
            evaluation_result.results[
                (ann_span.entity_type, non_matching_pred.entity_type)
            ] = (
                evaluation_result.results.get(
                    (ann_span.entity_type, non_matching_pred.entity_type), 0
                )
                + 1
            )

        return evaluation_result

    def _add_to_annotated(self, evaluation_result, per_type, entity_type):
        if per_type:
            evaluation_result.per_type[entity_type].num_annotated += 1
        else:
            evaluation_result.pii_annotated += 1

    def _calculate_combined_iou(
        self, annotation_span: Span, prediction_spans: List[Span]
    ) -> float:
        """
        Calculate the combined IoU of multiple prediction spans against an annotation span.

        :param annotation_span: The annotation span to match against
        :param prediction_spans: List of prediction spans that potentially overlap
        :return: Combined IoU value between 0 and 1
        """
        if not prediction_spans:
            return 0.0

        if self.char_based:
            # Character-based IoU
            ann_chars = set(
                range(
                    annotation_span.normalized_start_index,
                    annotation_span.normalized_end_index + 1,
                )
            )
            pred_chars = set()
            for i, pred_span in enumerate(prediction_spans):
                if i == 0:
                    pred_chars.update(
                        range(
                            pred_span.normalized_start_index,
                            pred_span.normalized_end_index + 1,
                        )
                    )
                else:
                    pred_chars.update(
                        range(
                            pred_span.normalized_start_index - 1,
                            pred_span.normalized_end_index,
                        )
                    )
            intersection = len(ann_chars.intersection(pred_chars))
            union = len(ann_chars.union(pred_chars))
        else:
            # Token-based IoU
            ann_tokens = set(annotation_span.normalized_tokens)
            pred_tokens = set()
            for pred_span in prediction_spans:
                pred_tokens.update(pred_span.normalized_tokens)

            intersection = len(ann_tokens.intersection(pred_tokens))
            union = len(ann_tokens.union(pred_tokens))
        return intersection / union if union > 0 else 0.0
