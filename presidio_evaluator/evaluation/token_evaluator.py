import warnings
from collections import Counter
from typing import Optional, List

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import BaseEvaluator, EvaluationResult


class TokenEvaluator(BaseEvaluator):
    """
    Evaluates the performance of a token-based Named Entity Recognition (NER) model.
    This class is designed to assess the model's ability to correctly identify and classify tokens in text.
    """

    def calculate_score(
        self,
        evaluation_results: List[EvaluationResult],
        entities: Optional[List[str]] = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        """
        Calculates the evaluation score based on the provided evaluation results.

        :param evaluation_results: List of EvaluationResult objects containing the results of the evaluation.
        :param entities: Optional list of entities to filter the evaluation results.
        If None, defaults to the evaluator's entities_to_keep (set in constructor).
        :return: An EvaluationResult object containing the aggregated results.

        Returns the pii_precision, pii_recall, f_measure either and number of records for each entity
        or for all entities (ignore_entity_type = True)
        :param evaluation_results: List of EvaluationResult
        :param entities: List of entities to calculate score to. Default is None: all entities
        :param beta: DEPRECATED. F measure beta value between different entity types,
        or to treat these as misclassifications
        Please use the beta value defined in the constructor of the Evaluator class.

        :return: EvaluationResult with precision, recall and f measures
        """

        # Default to entities_to_keep if no explicit entities provided
        if entities is None:
            entities = self.entities_to_keep

        for res in evaluation_results:
            if not res.results:
                # token evaluation works on the results object, so run the compare method if not done yet
                input_sample = InputSample(full_text = res.text, tokens=res.tokens, tags=res.actual_tags)
                results, errors = self.compare(input_sample=input_sample, prediction=res.predicted_tags)
                res.results = results
                res.errors = errors

        # aggregate results
        all_results = sum([er.results for er in evaluation_results], Counter())

        # compute pii_recall per entity
        entity_recall = {}
        entity_precision = {}
        n = {}
        if not entities:
            entities1 = list(set([x[0] for x in all_results.keys() if x[0] != "O"]))
            entities2 = list(set([x[1] for x in all_results.keys() if x[1] != "O"]))
            entities = list(set(entities1).union(set(entities2)))

        for entity in entities:
            # all annotation of given type
            annotated = sum([all_results[x] for x in all_results if x[0] == entity])
            predicted = sum([all_results[x] for x in all_results if x[1] == entity])
            n[entity] = annotated
            tp = all_results[(entity, entity)]

            entity_recall[entity] = self.recall(tp=tp, num_annotated=annotated)

            entity_precision[entity] = self.precision(tp=tp, num_predicted=predicted)

        # compute pii_precision and pii_recall
        annotated_all = sum([all_results[x] for x in all_results if x[0] != "O"])
        predicted_all = sum([all_results[x] for x in all_results if x[1] != "O"])

        tp = sum([all_results[x] for x in all_results if (x[0] != "O" and x[1] != "O")])
        pii_recall = self.recall(tp=tp, num_annotated=annotated_all)
        pii_precision = self.precision(tp=tp, num_predicted=predicted_all)

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
            entity_recall_dict=entity_recall,
            entity_precision_dict=entity_precision,
            n_dict=n,
            pii_f=pii_f_beta,
            n=sum(n.values()),
        )

        return evaluation_result


class Evaluator(TokenEvaluator):
    """
    Alias for TokenEvaluator to maintain backward compatibility.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Evaluator is deprecated and will be removed in a future version. "
            "Use SpanEvaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
