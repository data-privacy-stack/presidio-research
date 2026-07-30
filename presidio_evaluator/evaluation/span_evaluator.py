from collections import defaultdict
from typing import Literal

import pandas as pd

from presidio_evaluator.data_objects import Span
from presidio_evaluator.evaluation import (
    BaseEvaluator,
    DeprecationError,
    ErrorType,
    EvaluationResult,
    ModelError,
)
from presidio_evaluator.models import BaseModel


class SpanEvaluator(BaseEvaluator):
    """
    Evaluates PII detection using span-based fuzzy matching with character-level Intersection over Union (IoU).
    """

    def __init__(
        self,
        verbose: bool = False,
        model: BaseModel | None = None,
        entities_to_keep: list[str] | None = None,
        generic_entities: list[str] | None = None,
        skip_words: list | None = None,
        iou_threshold: float = 0.9,
        char_based: bool = True,
    ) -> None:
        """
        Initialize the SpanEvaluator for evaluating pii entities detection results.

        :param model: Instance of a fitted model or Presidio Analyzer
        :param verbose: Whether to print debug information
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
            verbose=verbose,
            entities_to_keep=entities_to_keep,
            generic_entities=generic_entities,
            skip_words=skip_words,
        )

        self.iou_threshold = iou_threshold
        self.char_based = char_based

    def _normalize_tokens(
        self,
        tokens: list[str],
        start_indices: list[int] | None = None,
    ) -> tuple[list[str], list[int]]:
        """
        Normalize tokens by:
        1. Converting to lowercase
        2. Removing stop words
        3. Removing standalone punctuation
        4. Removing skip words (common words that shouldn't affect entity matching)

        :param tokens: (list[str]) Token strings to normalize.
        :param start_indices: (list[int] | None) Character start offset of each
            token, parallel to ``tokens``; defaults to zeros when omitted.
        :return: (tuple[list[str], list[int]]) ``(normalized_tokens,
            normalized_start_indices)`` — the surviving tokens (lowercased,
            skip words removed) and their start offsets, kept parallel.
        """

        if not start_indices:
            start_indices = [0] * len(tokens)
        normalized = []
        normalized_indices = []
        for token, start in zip(tokens, start_indices, strict=False):
            token = token.lower()  # noqa: PLW2901 — intentional: normalize token to lowercase for skip-word matching
            # Skip if token is in skip words list
            if token in self.skip_words:
                continue
            normalized.append(token)
            normalized_indices.append(start)

        return normalized, normalized_indices

    def _merge_adjacent_spans(self, spans: list[Span], df: pd.DataFrame) -> list[Span]:
        """
        Merge adjacent spans of the same entity type if separated only by skip words / punctuation.

        :param spans: (list[Span]) Span objects to potentially merge.
        :param df: (pd.DataFrame) The sentence's rows, used to inspect the
            tokens between two candidate spans.
        :return: (list[Span]) Spans sorted by start position, with same-type
            neighbors separated only by skip words fused into single spans.
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
                merged_normalized_text = (current.normalized_tokens or []) + (
                    next_span.normalized_tokens or []
                )
                if (
                    current.normalized_start_indices is not None
                    and next_span.normalized_start_indices is not None
                ):
                    merged_normalized_indices = (
                        current.normalized_start_indices
                        + next_span.normalized_start_indices
                    )
                else:
                    merged_normalized_indices = None
                current = Span(
                    entity_type=current.entity_type,
                    entity_value=" ".join(merged_tokens),
                    start_position=current.start_position,
                    end_position=next_span.end_position,
                    normalized_start_index=min(
                        current.normalized_start_index or 0,
                        next_span.normalized_start_index or 0,
                    ),
                    normalized_end_index=max(
                        current.normalized_end_index or 0,
                        next_span.normalized_end_index or 0,
                    ),
                    normalized_tokens=merged_normalized_text,
                    normalized_start_indices=merged_normalized_indices,
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

        :param span1: (Span) The earlier span (by start position).
        :param span2: (Span) The later span.
        :param df: (pd.DataFrame) The sentence's rows, sliced positionally via
            the spans' sentence-relative ``token_end``/``token_start``.
        :return: (bool) True if every token between the spans is a skip word.
        """
        # token_start/token_end are positions within the sentence, so slice
        # positionally — the DataFrame's index labels are caller-defined
        # (e.g. global across sentences) and must not be used as positions.
        if span1.token_end is None or span2.token_start is None:
            raise ValueError(
                "Spans must have token_start/token_end set to check adjacency",
            )
        between_tokens = df["token"].iloc[span1.token_end : span2.token_start].tolist()
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

        :param span1: (Span) First Span object.
        :param span2: (Span) Second Span object.
        :param ignore_entity_type: (bool) If True, ignores the entity type when calculating IoU.
        :param use_normalized_indices: (bool) If True, uses normalized indices for IoU calculation.
        :param char_based: (bool) If True, calculates IoU at character level, else at token level.
        :return: (float) IoU value between 0 and 1.
        """
        if char_based:
            iou = span1.iou(
                other=span2,
                ignore_entity_type=ignore_entity_type,
                use_normalized_indices=use_normalized_indices,
            )
        else:
            iou = SpanEvaluator._token_iou(span1, [span2])

        return iou

    @staticmethod
    def _token_iou(ann_span: Span, pred_spans: list[Span]) -> float:
        """
        Calculate the token-level IoU between an annotation span and one or more
        prediction spans.

        When every span carries per-token start indices, tokens are compared as
        (start index, token) pairs, so identical words at different positions
        (e.g. both occurrences in "Michael met Michael") are distinct tokens.
        Spans without per-token indices fall back to comparing token strings,
        ignoring prediction spans with no positional overlap so that identical
        words at disjoint positions can't match.

        Known limitation of the fallback path: once spans do overlap
        positionally, token strings can't be disambiguated without positions,
        so a repeated word may over-count the intersection (e.g. annotation
        "Michael Smith" vs prediction "Smith met Michael" yields 1.0 instead
        of 1/3). Spans built by _create_spans always carry per-token indices
        and take the exact position-aware path.

        :param ann_span: (Span) The annotation Span to match against.
        :param pred_spans: (list[Span]) One or more prediction Spans; their
            tokens are pooled before computing the IoU.
        :return: (float) IoU value between 0 and 1.
        """
        ann_tokens = SpanEvaluator._positional_tokens(ann_span)
        pred_token_sets = [
            SpanEvaluator._positional_tokens(pred_span) for pred_span in pred_spans
        ]

        if ann_tokens is not None and all(
            tokens is not None for tokens in pred_token_sets
        ):
            pred_tokens: set = set()
            for tokens in pred_token_sets:
                pred_tokens.update(tokens or set())
        else:
            ann_tokens = set(ann_span.normalized_tokens or [])
            pred_tokens = set()
            for pred_span in pred_spans:
                if (
                    pred_span.intersect(
                        ann_span,
                        ignore_entity_type=True,
                        use_normalized_indices=True,
                    )
                    > 0
                ):
                    pred_tokens.update(pred_span.normalized_tokens or [])

        intersection = len(ann_tokens.intersection(pred_tokens))
        union = len(ann_tokens.union(pred_tokens))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _positional_tokens(span: Span) -> set[tuple[int, str]] | None:
        """
        Build a position-aware token set of (start index, token) pairs for a span.

        Comparing tokens together with their positions prevents identical words at
        different positions (e.g. both occurrences in "Michael met Michael") from
        being treated as the same token.

        :param span: (Span) Span to extract positional tokens from.
        :return: (set[tuple[int, str]] | None) Set of (start index, token)
            pairs, or None if the span does not carry per-token start indices
        :raises ValueError: If the span carries per-token start indices whose
            length does not match the number of normalized tokens
        """
        tokens = span.normalized_tokens or []
        indices = span.normalized_start_indices
        if indices is None:
            return None
        if len(indices) != len(tokens):
            raise ValueError(
                f"Inconsistent Span: {len(tokens)} normalized tokens but "
                f"{len(indices)} normalized start indices ({span})",
            )
        return set(zip(indices, tokens, strict=True))

    def _process_sentence_spans(
        self,
        sentence_df: pd.DataFrame,
    ) -> tuple[list[Span], list[Span]]:
        """
        Build the gold and predicted spans for one sentence.

        :param sentence_df: (pd.DataFrame) Rows of a single sentence with
            ``token``, ``annotation``, ``prediction`` and ``start_indices`` columns.
        :return: (tuple[list[Span], list[Span]]) ``(annotation_spans,
            prediction_spans)`` — each built from the corresponding tag column
            and merged across skip-word gaps by ``_merge_adjacent_spans``.
        """
        annotation_spans = self._create_spans(df=sentence_df, column="annotation")
        prediction_spans = self._create_spans(df=sentence_df, column="prediction")

        annotation_spans = self._merge_adjacent_spans(
            spans=annotation_spans,
            df=sentence_df,
        )
        prediction_spans = self._merge_adjacent_spans(
            spans=prediction_spans,
            df=sentence_df,
        )

        return annotation_spans, prediction_spans

    def _update_result_with_overall_metrics(
        self,
        evaluation_result: EvaluationResult,
        beta: float,
    ) -> None:
        """
        Update the evaluation result with overall metrics and per-type metrics.

        :param evaluation_result: (EvaluationResult) Result to update in place —
            fills ``pii_precision``, ``pii_recall`` and ``pii_f`` from the
            ``pii_*`` counters.
        :param beta: (float) The beta parameter for F-beta score calculation.
        """

        precision, recall, f_beta = self._calculate_metrics(
            evaluation_result.pii_true_positives or 0,
            evaluation_result.pii_predicted or 0,
            evaluation_result.pii_annotated or 0,
            beta,
            false_positives=evaluation_result.pii_false_positives or 0,
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

        :param evaluation_result: (EvaluationResult) Result whose
            ``per_type`` dict (``dict[str, PIIEvaluationMetrics]``) gets its
            ``precision``/``recall``/``f_beta`` fields computed from the counts.
        :param beta: (float) F-beta parameter.
        """

        for _entity_type, pii_metrics in evaluation_result.per_type.items():
            # Calculate metrics for this entity type
            precision, recall, f_beta = self._calculate_metrics(
                pii_metrics.true_positives,
                pii_metrics.num_predicted,
                pii_metrics.num_annotated,
                beta,
                false_positives=pii_metrics.false_positives,
            )
            pii_metrics.precision = precision
            pii_metrics.recall = recall
            pii_metrics.f_beta = f_beta

    @staticmethod
    def create_global_entities_df(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame containing global PII entities from the results DataFrame.

        :param results_df: (pd.DataFrame) Token-level evaluation results with
            ``annotation`` and ``prediction`` columns.
        :return: (pd.DataFrame) Copy of ``results_df`` with every non-"O"
            annotation and prediction label replaced by ``"PII"``, for the
            global PII-vs-O scoring pass.
        """
        # Create a deep copy to avoid modifying the original DataFrame
        global_df = results_df.copy(deep=True)
        global_df["annotation"] = global_df["annotation"].apply(
            lambda x: "O" if x == "O" else "PII",
        )
        global_df["prediction"] = global_df["prediction"].apply(
            lambda x: "O" if x == "O" else "PII",
        )
        return global_df

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

    def calculate_score_on_df(
        self,
        results_df: pd.DataFrame,
        beta: float = 2,
        level: Literal["entity", "pii", "both"] = "both",
        evaluation_result: EvaluationResult | None = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth annotations.

        :param results_df: (pd.DataFrame) DataFrame containing sentence_id, tokens,
                        token start indices, annotations and predictions columns —
                        as produced by model.predict_dataset() and optionally
                        processed by CanonicalMapper.get_mapped_results_dataframe().
        :param level: (Literal["entity", "pii", "both"]) Which metrics to compute:

                    - ``"entity"`` — per-entity-type precision/recall/F only
                    - ``"pii"`` — global PII (everything vs ``"O"`` ) metrics only
                    - ``"both"`` (default) — both passes; the returned
                      ``EvaluationResult`` contains per-type **and** global PII metrics.
        :param beta: (float) F-beta parameter (default 2).
        :param evaluation_result: (EvaluationResult | None) Optional existing
                        EvaluationResult to accumulate into.
        :return: (EvaluationResult) Result with the requested metrics populated —
                        ``per_type`` for "entity", the ``pii_*`` fields for "pii",
                        both for "both".
        """
        if level in ("entity", "both"):
            evaluation_result = self._run_score_pass(
                per_type=True,
                results_df=results_df,
                beta=beta,
                evaluation_result=evaluation_result,
            )
        if level in ("pii", "both"):
            global_pii_df = self.create_global_entities_df(results_df)
            evaluation_result = self._run_score_pass(
                per_type=False,
                results_df=global_pii_df,
                beta=beta,
                evaluation_result=evaluation_result,
            )
        if not evaluation_result:
            evaluation_result = EvaluationResult()
        return evaluation_result

    def _run_score_pass(
        self,
        per_type: bool,
        results_df: pd.DataFrame,
        beta: float = 2,
        evaluation_result: EvaluationResult | None = None,
    ) -> EvaluationResult:
        """
        Run a single scoring pass over the results DataFrame.

        :param per_type: (bool) If True, performs per-entity type evaluation; if False,
                    performs global PII vs non-PII evaluation
        :param results_df: (pd.DataFrame) DataFrame containing sentence_id, tokens,
                        token start indices, annotations and predictions columns
        :param beta: (float) The beta parameter for F-beta score calculation. Higher
                    values weight recall more than precision. Default is 2.
        :param evaluation_result: (EvaluationResult | None) Optional existing
                        EvaluationResult to update. If None, creates a new one.
        :return: (EvaluationResult) Result containing computed metrics, counts,
                        and error analysis
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
            evaluation_result.n = sum(
                m.num_annotated for m in evaluation_result.per_type.values()
            )
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
        evaluation_result: EvaluationResult | None = None,
    ) -> EvaluationResult:
        """
        Compare one sentence's annotations and predictions, updating the evaluation result.

        :param per_type: (bool) If True, performs per-entity type evaluation; if False,
                performs global PII vs non-PII evaluation
        :param sentence_df: (pd.DataFrame) DataFrame containing sentence_id, tokens,
                token start indices, annotations and predictions columns
        :param evaluation_result: (EvaluationResult | None) Optional existing
                EvaluationResult to update. If None, creates a new one.
        :return: (EvaluationResult) The updated result.
        """
        if not evaluation_result:
            evaluation_result = EvaluationResult()

        annotation_spans, prediction_spans = self._process_sentence_spans(sentence_df)
        # Match predictions with annotations and update metrics
        evaluation_result = self._match_predictions_with_annotations(
            annotation_spans,
            prediction_spans,
            evaluation_result,
            per_type,
        )
        return evaluation_result

    def _create_spans(self, df: pd.DataFrame, column: str) -> list[Span]:
        """
        Create spans from a DataFrame column.

        :param df: (pd.DataFrame) One sentence's rows with ``token``,
            ``start_indices`` and the tag column to read.
        :param column: (str) Name of the tag column to extract spans from
            (``"annotation"`` or ``"prediction"``).
        :return: (list[Span]) One Span per maximal run of identically-tagged
            tokens; runs whose tokens are all skip words are dropped.
        """
        spans = []
        current_entity_type = None
        current_tokens = []
        current_start_indices = []
        current_token_start: int = 0

        for idx, (_, row) in enumerate(df.iterrows()):
            entity_type = row[column]
            token = row["token"]
            token_start = row["start_indices"]

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
                                normalized_start_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            ),
                        )
                    current_entity_type = None
                    current_tokens = []
                    current_start_indices = []
                    current_token_start = 0

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
                                normalized_start_indices=normalized_start_indices,
                                normalized_tokens=normalized_tokens,
                            ),
                        )
                current_entity_type = entity_type
                current_tokens = [token]
                current_start_indices = [token_start]
                current_token_start = idx  # Set token start position

            else:
                current_tokens.append(token)
                current_start_indices.append(token_start)

        # Handle final span
        if current_entity_type and current_tokens:
            normalized_tokens, normalized_start_indices = self._normalize_tokens(
                current_tokens,
                current_start_indices,
            )
            if normalized_tokens:
                spans.append(
                    self.__create_span(
                        entity_type=current_entity_type,
                        start_indices=current_start_indices,
                        token_start=current_token_start,
                        current_tokens=current_tokens,
                        idx=len(df),
                        normalized_start_indices=normalized_start_indices,
                        normalized_tokens=normalized_tokens,
                    ),
                )
        return spans

    def __create_span(
        self,
        entity_type: str,
        start_indices: list[int],
        token_start: int,
        current_tokens: list[str],
        idx: int,
        normalized_start_indices: list[int],
        normalized_tokens: list[str],
    ) -> Span:
        """
        Assemble a Span from the tokens accumulated by ``_create_spans``.

        :param entity_type: (str) Entity type of the span.
        :param start_indices: (list[int]) Character start offset of each raw token.
        :param token_start: (int) Sentence-relative position of the first token.
        :param current_tokens: (list[str]) The raw token strings of the span.
        :param idx: (int) Sentence-relative position one past the last token.
        :param normalized_start_indices: (list[int]) Character start offsets of
            the tokens that survived skip-word normalization.
        :param normalized_tokens: (list[str]) The normalized (lowercased,
            skip-words removed) token strings.
        :return: (Span) Span carrying both raw offsets and normalized
            token/offset views used by IoU calculations.
        """
        return Span(
            entity_type=entity_type,
            entity_value=" ".join(current_tokens),
            start_position=start_indices[0],
            end_position=start_indices[-1] + len(current_tokens[-1]),
            normalized_tokens=normalized_tokens,
            normalized_start_index=min(normalized_start_indices),
            normalized_end_index=self._get_normalized_end_index(
                normalized_tokens,
                normalized_start_indices,
            ),
            normalized_start_indices=normalized_start_indices,
            token_start=token_start,
            token_end=idx,
        )

    @staticmethod
    def _get_normalized_end_index(
        normalized_tokens: list[str],
        normalized_indices: list[int],
    ) -> int:
        """Calculate the end character index of the last token in the normalized tokens list."""  # noqa: E501
        return max(
            [
                len(tok) + start
                for tok, start in zip(
                    normalized_tokens, normalized_indices, strict=False
                )
            ],
        )

    def _calculate_metrics(
        self,
        true_positives: int,
        num_predicted: int,
        num_annotated: int,
        beta: float = 2,
        false_positives: int | None = None,
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F-beta score.

        Counting is two-sided: recall is true_positives / num_annotated
        (annotations covered), while precision is
        (num_predicted - false_positives) / num_predicted (prediction spans
        credited with a match). The numerators may differ — e.g. one wide
        prediction covering two annotations is two recall hits but a single
        credited prediction.

        :param true_positives: (int) Number of annotations covered at IoU >= threshold.
        :param num_predicted: (int) Number of predicted spans.
        :param num_annotated: (int) Number of annotated (gold) spans.
        :param beta: (float) The beta parameter for F-beta score calculation. Default is 2.
        :param false_positives: (int | None) Number of predicted spans with no successful
            match. If None, precision falls back to true_positives / num_predicted.
        :return: (tuple[float, float, float]) ``(precision, recall, f_beta)``;
            precision/recall are ``np.nan`` when their denominator is 0.
        """
        precision_hits = (
            true_positives
            if false_positives is None
            else num_predicted - false_positives
        )
        precision = self.precision(tp=precision_hits, num_predicted=num_predicted)
        recall = self.recall(tp=true_positives, num_annotated=num_annotated)
        f_beta = self.f_beta(precision=precision, recall=recall, beta=beta)
        return precision, recall, f_beta

    @staticmethod
    def _span_key(span: Span) -> tuple[str, int, int]:
        """Identity of a span for match bookkeeping.

        :param span: (Span) The span to identify.
        :return: (tuple[str, int, int]) ``(entity_type, start_position,
            end_position)`` — hashable identity used in the pass-tracking sets.
        """
        return (span.entity_type, span.start_position, span.end_position)

    def _group_iou(
        self,
        ann_span: Span,
        spans: list[Span],
        pairwise_ious: list[float],
    ) -> float:
        """Coverage of an annotation by a group of same-type spans.

        :param ann_span: (Span) The annotation being covered.
        :param spans: (list[Span]) The overlapping prediction spans of one type.
        :param pairwise_ious: (list[float]) IoU of each span in ``spans``
            against ``ann_span``, parallel to ``spans`` (one entry per span).
        :return: (float) Exact pairwise IoU when ``spans`` has a single span,
            else the combined IoU of the whole group.
        """
        if len(spans) == 1:
            return pairwise_ious[0]
        return self._calculate_combined_iou(ann_span, spans)

    def _match_predictions_with_annotations(
        self,
        annotation_spans: list[Span],
        prediction_spans: list[Span],
        evaluation_result: EvaluationResult,
        per_type: bool = True,
    ) -> EvaluationResult:
        """Match spans and update counts using two-sided counting.

        Recall side (per annotation): each annotation independently asks
        whether the predictions of its type cover it at IoU >= threshold —
        individually or combined. Covered -> true positive (recall hit),
        otherwise false negative.

        Precision side (per prediction): each prediction span enters
        num_predicted exactly once, no matter how many annotations it
        overlaps. A prediction that participated in at least one successful
        same-type match is credited; any other prediction is a false
        positive.

        Precision is therefore (num_predicted - false_positives) /
        num_predicted while recall is true_positives / num_annotated; the two
        numerators may legitimately differ (one wide prediction covering two
        annotations is two recall hits but a single credited prediction).

        :param annotation_spans: (list[Span]) Gold spans of one sentence.
        :param prediction_spans: (list[Span]) Predicted spans of the same sentence.
        :param evaluation_result: (EvaluationResult) Accumulator updated in
            place — counts, confusion-matrix ``results`` and ``model_errors``.
        :param per_type: (bool) If True, update ``per_type`` metrics per entity
            type; if False, update the global ``pii_*`` counters only.
        :return: (EvaluationResult) The same ``evaluation_result``, updated.
        """
        if not evaluation_result.model_errors:
            evaluation_result.model_errors = []

        # Prediction spans that participated in >= 1 successful same-type match.
        successful_predictions: set[tuple[str, int, int]] = set()
        # Prediction spans overlapping >= 1 annotation (for FP explanations).
        overlapping_predictions: set[tuple[str, int, int]] = set()
        # Prediction spans already represented in the confusion matrix by a
        # wrong-entity cell (ann_type, pred_type) — each span appears in the
        # matrix once, so these skip the ("O", pred_type) row.
        wrong_entity_predictions: set[tuple[str, int, int]] = set()

        # --- Recall pass: one verdict per annotation ---
        for ann_span in annotation_spans:
            ann_type = ann_span.entity_type
            self._add_to_annotated(evaluation_result, per_type, ann_type)

            overlapping_preds = self._get_all_overlapping(ann_span, prediction_spans)
            for pred_span, _ in overlapping_preds:
                overlapping_predictions.add(self._span_key(pred_span))
            spans_by_type, iou_by_type = self._group_spans_by_type(overlapping_preds)

            same_type_spans = spans_by_type.get(ann_type, [])
            iou = (
                self._group_iou(ann_span, same_type_spans, iou_by_type[ann_type])
                if same_type_spans
                else 0.0
            )

            # Wrong-entity analysis: another type covering this annotation at
            # IoU >= threshold. Collected first so the FN branch can put the
            # wrong type (rather than "O") in the confusion matrix.
            wrong_type_hits: list[tuple[str, list[Span], float]] = []
            if per_type:
                for other_type, other_spans in spans_by_type.items():
                    if other_type == ann_type:
                        continue
                    other_iou = self._group_iou(
                        ann_span, other_spans, iou_by_type[other_type]
                    )
                    if other_iou >= self.iou_threshold:
                        wrong_type_hits.append((other_type, other_spans, other_iou))

            if same_type_spans and iou >= self.iou_threshold:
                if per_type:
                    evaluation_result.per_type[ann_type].true_positives += 1
                    evaluation_result.results[(ann_type, ann_type)] += 1
                else:
                    evaluation_result.pii_true_positives += 1
                for pred_span in same_type_spans:
                    successful_predictions.add(self._span_key(pred_span))
            elif per_type:
                evaluation_result.per_type[ann_type].false_negatives += 1
                if not wrong_type_hits:
                    evaluation_result.results[(ann_type, "O")] += 1
                # Attach the closest evidence to the FN record: a same-type
                # prediction below threshold, else any overlapping prediction.
                fn_pred = same_type_spans[0] if same_type_spans else None
                fn_iou = iou
                if fn_pred is None and overlapping_preds:
                    fn_pred, fn_iou = overlapping_preds[0]
                evaluation_result.model_errors.append(
                    self._get_model_error(
                        ann_span=ann_span,
                        pred_span=fn_pred,
                        error_type=ErrorType.FN,
                        iou=fn_iou,
                    ),
                )
            else:
                evaluation_result.pii_false_negatives += 1

            for other_type, other_spans, other_iou in wrong_type_hits:
                evaluation_result.results[(ann_type, other_type)] += 1
                for wrong_span in other_spans:
                    wrong_entity_predictions.add(self._span_key(wrong_span))
                evaluation_result.model_errors.append(
                    self._get_model_error(
                        ann_span=ann_span,
                        pred_span=other_spans[0],
                        error_type=ErrorType.WrongEntity,
                        iou=other_iou,
                    ),
                )

        # --- Precision pass: each prediction span is counted exactly once ---
        for pred_span in prediction_spans:
            pred_key = self._span_key(pred_span)
            if per_type:
                evaluation_result.per_type[pred_span.entity_type].num_predicted += 1
            else:
                evaluation_result.pii_predicted += 1

            if pred_key in successful_predictions:
                continue

            if per_type:
                evaluation_result.per_type[pred_span.entity_type].false_positives += 1
            else:
                evaluation_result.pii_false_positives += 1

            if pred_key not in wrong_entity_predictions:
                evaluation_result.results[("O", pred_span.entity_type)] = (
                    evaluation_result.results.get(("O", pred_span.entity_type), 0) + 1
                )
            if pred_key in overlapping_predictions:
                explanation = (
                    f"Entity {pred_span.entity_type} falsely detected: overlaps "
                    f"annotation(s) but no match reached "
                    f"threshold={self.iou_threshold}"
                )
            else:
                explanation = (
                    f"False prediction with no overlap: {pred_span.entity_type}"
                )
            evaluation_result.model_errors.append(
                ModelError(
                    error_type=ErrorType.FP,
                    annotation="O",
                    prediction=pred_span.entity_type,
                    full_text=pred_span.entity_value,
                    token=" ".join(pred_span.normalized_tokens or []),
                    explanation=explanation,
                    start=pred_span.start_position,
                    end=pred_span.end_position,
                ),
            )

        return evaluation_result

    @staticmethod
    def _group_spans_by_type(
        overlapping_preds: list[tuple[Span, float]],
    ) -> tuple[dict[str, list[Span]], dict[str, list[float]]]:
        """
        Group spans by entity type and their corresponding IoU values.

        :param overlapping_preds: (list[tuple[Span, float]]) Prediction spans to
            group, each paired with its IoU against the annotation, as produced
            by ``_get_all_overlapping``.
        :return: (tuple[dict[str, list[Span]], dict[str, list[float]]]) Two
            parallel dicts keyed by entity type: ``spans_by_type[t][i]`` is a
            prediction span of type ``t`` and ``iou_by_type[t][i]`` is that same
            span's IoU. Keys are exactly the types present in
            ``overlapping_preds``; both are defaultdicts, so missing keys yield
            empty lists (prefer ``.get`` to avoid inserting keys on lookup).
        """
        spans_by_type = defaultdict(list)
        iou_by_type = defaultdict(list)

        for pred_span, iou in overlapping_preds:
            spans_by_type[pred_span.entity_type].append(pred_span)
            iou_by_type[pred_span.entity_type].append(iou)
        return spans_by_type, iou_by_type

    def _get_model_error(
        self,
        ann_span: Span | None,
        pred_span: Span | None,
        error_type: ErrorType,
        iou: float = 0.0,
    ) -> ModelError:
        """
        Build a ModelError record for error analysis.

        :param ann_span: (Span | None) The gold span involved, or None for a
            standalone false positive.
        :param pred_span: (Span | None) The predicted span involved, or None
            for a clean miss.
        :param error_type: (ErrorType) FN, FP or WrongEntity.
        :param iou: (float) The IoU that drove the verdict; quoted in the
            explanation text.
        :return: (ModelError) Record with annotation/prediction labels, the
            active span's text and offsets, and a human-readable explanation.
        """

        def get_explanation():
            pred_type = pred_span.entity_type if pred_span else "O"
            ann_type = ann_span.entity_type if ann_span else "O"
            match error_type:
                case ErrorType.FP:
                    return (
                        f"Entity {pred_type} falsely detected, iou={iou:.2f} "
                        f"compared to threshold={self.iou_threshold}"
                    )
                case ErrorType.FN:
                    if (
                        pred_span and pred_span.entity_type == ann_type
                    ):  # FN due to low IoU
                        return (
                            f"Entity {ann_type} not detected due to low iou={iou:.2f} "
                            f"compared to threshold={self.iou_threshold}"
                        )
                    elif pred_span and pred_span.entity_type != ann_type:
                        return (
                            f"Entity {ann_type} not detected. "
                            f"iou with {pred_span.entity_type}={iou:.2f} "
                            f"compared to threshold={self.iou_threshold}"
                        )
                    else:
                        return f"Entity {ann_type} not detected."
                case ErrorType.WrongEntity:
                    return (
                        f"Wrong entity type: {ann_type} detected as "
                        f"{pred_type}, iou={iou:.2f} "
                        f"compared to threshold={self.iou_threshold}"
                    )

            return ValueError(f"Unknown or missing error type: {error_type}")

        prediction = (
            "O"
            if error_type == ErrorType.FN or not pred_span
            else pred_span.entity_type
        )
        annotation = (
            "O" if error_type == ErrorType.FP or not ann_span else ann_span.entity_type
        )
        explanation = get_explanation()

        active_span = pred_span if error_type == ErrorType.FP else ann_span
        if active_span is None:
            raise ValueError(f"No active span for error type {error_type}")
        return ModelError(
            error_type=error_type,
            annotation=annotation,
            prediction=prediction,
            full_text=active_span.entity_value,
            token=" ".join(active_span.normalized_tokens or []),
            explanation=explanation,
            start=active_span.start_position,
            end=active_span.end_position,
        )

    def _get_all_overlapping(
        self,
        ann_span: Span,
        prediction_spans: list[Span],
    ) -> list[tuple[Span, float]]:
        """Get all prediction spans that overlap with the annotation span, regardless of type.

        :param ann_span: (Span) The annotation Span to match against.
        :param prediction_spans: (list[Span]) All prediction spans of the sentence.
        :return: (list[tuple[Span, float]]) ``(prediction span, IoU)`` pairs for
            every prediction with IoU > 0 against ``ann_span``, sorted by the
            prediction's start position.
        """

        overlapping_preds = []
        for pred_span in prediction_spans:
            iou = self.calculate_iou(ann_span, pred_span, char_based=self.char_based)
            if iou > 0:
                overlapping_preds.append((pred_span, iou))

        overlapping_preds.sort(key=lambda x: x[0].start_position)

        return overlapping_preds

    def _add_to_annotated(
        self,
        evaluation_result: EvaluationResult,
        per_type: bool,
        entity_type: str,
    ) -> None:
        """
        Count one annotation in the recall denominator.

        :param evaluation_result: (EvaluationResult) Result object to update.
        :param per_type: (bool) If True, increment the type's ``num_annotated``;
            otherwise increment the global ``pii_annotated``.
        :param entity_type: (str) Entity type of the annotation.
        """
        if per_type:
            evaluation_result.per_type[entity_type].num_annotated += 1
        else:
            evaluation_result.pii_annotated += 1

    def _calculate_combined_iou(
        self,
        annotation_span: Span,
        prediction_spans: list[Span],
    ) -> float:
        """
        Calculate the combined IoU of multiple prediction spans against an annotation span.

        :param annotation_span: (Span) The annotation span to match against.
        :param prediction_spans: (list[Span]) Prediction spans whose coverage
            is pooled before computing the IoU.
        :return: (float) Combined IoU value between 0 and 1; 0.0 when
            ``prediction_spans`` is empty.
        """
        if not prediction_spans:
            return 0.0

        if self.char_based:
            # Character-based IoU
            ann_chars = set(
                range(
                    annotation_span.normalized_start_index or 0,
                    (annotation_span.normalized_end_index or 0) + 1,
                ),
            )
            pred_chars: set[int] = set()
            for i, pred_span in enumerate(prediction_spans):
                if i == 0:
                    pred_chars.update(
                        range(
                            pred_span.normalized_start_index or 0,
                            (pred_span.normalized_end_index or 0) + 1,
                        ),
                    )
                else:
                    pred_chars.update(
                        range(
                            (pred_span.normalized_start_index or 0) - 1,
                            pred_span.normalized_end_index or 0,
                        ),
                    )
            intersection = len(ann_chars.intersection(pred_chars))
            union = len(ann_chars.union(pred_chars))
            return intersection / union if union > 0 else 0.0

        return self._token_iou(annotation_span, prediction_spans)
