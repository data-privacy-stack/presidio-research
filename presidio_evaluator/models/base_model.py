from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import pandas as pd

from presidio_evaluator import InputSample, io_to_scheme


class BaseModel(ABC):
    def __init__(
        self,
        labeling_scheme: str = "IO",
        entities_to_keep: List[str] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ):
        """
        Abstract class for evaluating NER models and others
        :param entities_to_keep: Which entities should be evaluated? All other
        entities are ignored. If None, none are filtered
        :param labeling_scheme: Used to translate (if needed)
        the prediction to a specific scheme (IO, BIO/IOB, BILUO)
        :param entity_mapping: DEPRECATED. This parameter is no longer supported.
        Entity mapping should be passed to the evaluator instead.
        :param verbose: Whether to print more debug info
        """
        
        if entity_mapping is not None:
            raise ValueError(
                "The 'entity_mapping' parameter is deprecated and has been removed from BaseModel.\n"
                "Entity mapping is now handled by the evaluator for better separation of concerns.\n"
                "Please pass entity_mapping to the evaluator constructor instead:\n"
                "  evaluator = Evaluator(model=model, entity_mapping={...})\n"
                "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for examples."
            )
        
        self.entities = entities_to_keep
        self.labeling_scheme = labeling_scheme
        self.verbose = verbose
        self.name = self.__class__.__name__

    @abstractmethod
    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        """
        Abstract. Returns the predicted tokens/spans from the evaluated model
        :param sample: Sample to be evaluated
        :return: List of tags in self.labeling_scheme format
        """
        pass

    @abstractmethod
    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        """Perform batch prediction if the model supports it."""

    def predict_dataset(self, dataset: List[InputSample]) -> pd.DataFrame:
        """Predict entities for a dataset and return results as a DataFrame.

        Calls batch_predict() internally and assembles the result into a
        flat DataFrame.  No entity mapping is applied — that is the mapper's job.

        :param dataset: List of InputSample objects (must have tokens and tags set).
        :return: DataFrame with exactly 5 columns:
            sentence_id, token, annotation, prediction, start_indices
        """
        predictions = self.batch_predict(dataset)

        rows: List[Dict] = []
        for i, (sample, pred_tags) in enumerate(zip(dataset, predictions)):
            sentence_id = sample.sample_id if sample.sample_id is not None else i
            tokens = sample.tokens
            annotations = sample.tags
            start_indices = sample.start_indices
            for j in range(len(tokens)):
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "token": str(tokens[j]),
                        "annotation": annotations[j] if j < len(annotations) else "O",
                        "prediction": pred_tags[j] if j < len(pred_tags) else "O",
                        "start_indices": start_indices[j] if j < len(start_indices) else 0,
                    }
                )

        return pd.DataFrame(
            rows,
            columns=["sentence_id", "token", "annotation", "prediction", "start_indices"],
        )

    def filter_tags_in_supported_entities(self, tags: List[str]) -> List[str]:
        """
        Replaces tags of unwanted entities with O.
        :param tags: Lits of tags
        :return: List of tags where tags not in self.entities are considered "O"
        """
        if not self.entities:
            return tags
        return [tag if self._tag_in_entities(tag) else "O" for tag in tags]

    def to_scheme(self, tags: List[str]):

        """
        Translates IO tags to BIO/BILUO based on the input labeling_scheme
        :param tags: Current tags in IO
        :return: Tags in labeling scheme
        """

        io_tags = [self._to_io(tag) for tag in tags]

        return io_to_scheme(io_tags=io_tags, scheme=self.labeling_scheme)

    @staticmethod
    def _to_io(tag):
        if "-" in tag:
            return tag[2:]
        return tag

    def to_log(self) -> Dict:
        """
        Returns a dictionary of parameters for logging purposes.
        :return:
        """
        return {
            "labeling_scheme": self.labeling_scheme,
            "entities_to_keep": self.entities,
        }

    def _tag_in_entities(self, tag: str) -> bool:
        """True if the tag is in the entities to keep."""

        if not self.entities:
            return True

        if tag == "O":
            return True

        if tag[1] == "-":  # BIO/BILUO
            return tag[2:] in self.entities
        else:  # IO
            return tag in self.entities
