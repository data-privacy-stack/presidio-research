from typing import List

import spacy

from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel


class SpacyModel(BaseModel):
    def __init__(
        self,
        model: spacy.language.Language = None,
        model_name: str = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
    ):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
        )

        if model is None:
            if model_name is None:
                raise ValueError("Either model_name or model object must be supplied")
            self.model = spacy.load(model_name)
        else:
            self.model = model

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        """
        Predict a list of tags for an inpuit sample.
        :param sample: InputSample
        :return: list of tags
        """
        doc = self.model(sample.full_text)
        tags = self._get_tags_from_doc(doc)
        if len(doc) != len(sample.tokens):
            print("mismatch between input tokens and new tokens")

        return tags

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        texts = [sample.full_text for sample in dataset]

        docs = self.model.pipe(texts=texts)
        predictions = []
        for doc in docs:
            tags = self._get_tags_from_doc(doc)
            predictions.append(tags)
        return predictions

    @staticmethod
    def _get_tags_from_doc(doc):
        tags = [token.ent_type_ if token.ent_type_ != "" else "O" for token in doc]
        return tags
