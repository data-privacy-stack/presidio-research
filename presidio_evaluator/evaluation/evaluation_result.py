import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from presidio_evaluator.evaluation import ModelError


@dataclass
class PIIEvaluationMetrics:
    """Metrics for a specific entity type.

    :param precision: Precision for the entity type (true_positives / (true_positives + false_positives))
    :param recall: Recall for the entity type (true_positives / (true_positives + false_negatives))
    :param f_beta: F-beta score for the entity type, where beta is typically set to 1 (harmonic mean of precision and recall)
    :param num_predicted: Total number of predicted entities of this type
    :param num_annotated: Total number of annotated entities of this type
    :param true_positives: Number of true positive predictions for this entity type
    :param false_positives: Number of false positive predictions for this entity type
    :param false_negatives: Number of false negative predictions for this entity type
    """

    precision: float = 0.0
    recall: float = 0.0
    f_beta: float = 0.0
    num_predicted: int = 0
    num_annotated: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class EvaluationResult:
    def __init__(
        self,
        results: Counter | None = None,
        model_errors: list[ModelError] | None = None,
        text: str | None = None,
        pii_recall: float | None = None,
        pii_precision: float | None = None,
        pii_f: float | None = None,
        n: int | None = None,
        entity_recall_dict: dict[str, float] | None = None,
        entity_precision_dict: dict[str, float] | None = None,
        n_dict: dict[str, int] | None = None,
        per_type: dict[str, PIIEvaluationMetrics] | None = None,
        pii_predicted: int | None = 0,
        pii_annotated: int | None = 0,
        pii_true_positives: int | None = 0,
        pii_false_positives: int | None = 0,
        pii_false_negatives: int | None = 0,
        tokens: list[str] | None = None,
        actual_tags: list[str] | None = None,
        predicted_tags: list[str] | None = None,
        start_indices: list[int] = None,
    ) -> None:
        """
        Holds the output of a comparison between ground truth and predicted
        :param results: Represents the confusion matrix as a Counter object,
        where the key is a tuple of (actual, predicted) and the value is the count of occurrences.
        with structure {(actual, predicted) : count}
        :param model_errors: List of specific model errors for further inspection
        :param text: sample's full text (if used for one sample)
        :param pii_recall: Recall for all entities (PII or not)
        :param pii_precision: Precision for all entities (PII or not)
        :param pii_f: F measure for all entities (PII or not)
        :param n: Number of total entity tokens
        :param entity_recall_dict: Recall per entity
        :param entity_precision_dict: Precision per entity
        :param n_dict: Number of tokens per entity
        :param per_type: EntityTypeMetrics for each entity type
        :param pii_predicted: Total number of predicted entities (agnostic to type: PII yes/no)
        :param pii_annotated: Total number of annotated entities (agnostic to type: PII yes/no)
        :param pii_true_positives: Total number of true positives (agnostic to type: PII yes/no)
        :param pii_false_positives: Total number of false positives (agnostic to type: PII yes/no)
        :param pii_false_negatives: Total number of false negatives (agnostic to type: PII yes/no)
        :param tokens: List of tokens
        :param actual_tags: List of actual tags
        :param predicted_tags: List of predicted tags
        :param start_indices: List of start indices of tokens in the text
        """

        self.results = results if results else Counter()
        self.model_errors = model_errors
        self.text = text

        if per_type and entity_recall_dict:
            raise ValueError(
                "Cannot provide both per_type and entity_recall_dict. Use one of them.",
            )
        if per_type and entity_precision_dict:
            raise ValueError(
                "Cannot provide both per_type and entity_precision_dict. "
                "Use one of them.",
            )
        if per_type and n_dict:
            raise ValueError(
                "Cannot provide both per_type and n_dict. Use one of them.",
            )

        if per_type and not entity_recall_dict:
            entity_recall_dict = {
                ent: metrics.recall for ent, metrics in per_type.items()
            }
        if per_type and not entity_precision_dict:
            entity_precision_dict = {
                ent: metrics.precision for ent, metrics in per_type.items()
            }
        if per_type and not n_dict:
            n_dict = {ent: metrics.num_predicted for ent, metrics in per_type.items()}

        self.per_type = per_type if per_type else defaultdict(PIIEvaluationMetrics)
        self.pii_recall = pii_recall
        self.pii_precision = pii_precision
        self.pii_f = pii_f
        self.n = n
        self._entity_recall_dict = entity_recall_dict if entity_recall_dict else {}
        self._entity_precision_dict = (
            entity_precision_dict if entity_precision_dict else {}
        )
        self._n_dict = n_dict if n_dict else {}

        self.pii_predicted = pii_predicted
        self.pii_annotated = pii_annotated
        self.pii_true_positives = pii_true_positives
        self.pii_false_positives = pii_false_positives
        self.pii_false_negatives = pii_false_negatives

        self.tokens = tokens
        self.actual_tags = actual_tags
        self.predicted_tags = predicted_tags
        self.start_indices = start_indices if start_indices is not None else []

    @property
    def entity_precision_dict(self) -> dict[str, float]:
        """
        Property that returns precision values per entity type.
        Prioritizes deriving from per_type but falls back to stored _entity_precision_dict for backward compatibility.
        """
        if self.per_type:
            return {ent: metrics.precision for ent, metrics in self.per_type.items()}
        return self._entity_precision_dict

    @entity_precision_dict.setter
    def entity_precision_dict(self, value: dict[str, float]) -> None:
        """
        Setter for entity_precision_dict to maintain backward compatibility.
        """
        self._entity_precision_dict = value if value else {}

    @property
    def entity_recall_dict(self) -> dict[str, float]:
        """
        Property that returns recall values per entity type.
        Prioritizes deriving from per_type but falls back to stored _entity_recall_dict for backward compatibility.
        """
        if self.per_type:
            return {ent: metrics.recall for ent, metrics in self.per_type.items()}
        return self._entity_recall_dict

    @entity_recall_dict.setter
    def entity_recall_dict(self, value: dict[str, float]) -> None:
        """
        Setter for entity_recall_dict to maintain backward compatibility.
        """
        self._entity_recall_dict = value if value else {}

    @property
    def n_dict(self) -> dict[str, int]:
        """
        Property that returns the number of annotated tokens per entity type.
        Prioritizes deriving from per_type but falls back to stored _n_dict for backward compatibility.
        """
        if self.per_type:
            return {
                ent: metrics.num_annotated for ent, metrics in self.per_type.items()
            }
        return self._n_dict

    @n_dict.setter
    def n_dict(self, value: dict[str, int]) -> None:
        """
        Setter for n_dict to maintain backward compatibility.
        """
        self._n_dict = value if value else {}

    def __str__(self) -> str:
        return_str = ""
        if not self.entity_precision_dict or not self.entity_recall_dict:
            return json.dumps(self.results)

        entities = self.n_dict.keys()

        row_format = "{:>20}{:>20.2%}{:>20.2%}{:>20}"
        header_format = "{:>20}" * 4
        return_str += str(
            header_format.format(
                *("Entity", "Precision", "Recall", "Number of samples"),
            ),
        )
        for entity in entities:
            return_str += "\n" + row_format.format(
                entity,
                self.entity_precision_dict[entity],
                self.entity_recall_dict[entity],
                self.n_dict[entity],
            )

        # add PII values
        return_str += "\n" + row_format.format(
            "PII",
            self.pii_precision,
            self.pii_recall,
            self.n,
        )

        return_str += f"\nPII F measure: {self.pii_f:.2%}"
        return return_str

    def __repr__(self) -> str:
        return f"stats={self.results}"

    def to_log(self) -> dict:
        metrics_dict = {
            "pii_f": self.pii_f,
            "pii_recall": self.pii_recall,
            "pii_precision": self.pii_precision,
            "n": self.n,
        }
        if self.entity_precision_dict:
            metrics_dict.update(
                {
                    f"{ent}_precision": v
                    for (ent, v) in self.entity_precision_dict.items()
                },
            )
        if self.entity_recall_dict:
            metrics_dict.update(
                {f"{ent}_recall": v for (ent, v) in self.entity_recall_dict.items()},
            )
        if self.n:
            metrics_dict.update(self.n_dict)
        return metrics_dict

    def to_confusion_matrix(self) -> tuple[list[str], list[list[int]]]:
        entities = list(self.n_dict.keys())
        if "O" in entities:
            entities = [ent for ent in entities if ent != "O"]
        entities = sorted(entities)
        entities.append("O")
        confusion_matrix = [[0] * len(entities) for _ in range(len(entities))]
        for i, actual in enumerate(entities):
            for j, predicted in enumerate(entities):
                confusion_matrix[i][j] = self.results[(actual, predicted)]

        return entities, confusion_matrix

    def to_confusion_df(self) -> pd.DataFrame:
        entities, confmatrix = self.to_confusion_matrix()

        conf_df = pd.DataFrame(confmatrix, columns=entities).set_axis(entities)

        precision_df = pd.DataFrame(self.entity_precision_dict, index=["precision"])
        recall_series = pd.Series(self.entity_recall_dict)

        # add precision numbers as the last row
        conf_df = pd.concat([conf_df, precision_df], axis=0)

        # add recall numbers as the last column
        conf_df["recall"] = recall_series
        # conf_df.set_index(["recall"], drop=False)
        return conf_df
