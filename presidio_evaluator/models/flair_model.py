from typing import List, Dict
import warnings

import spacy

from presidio_evaluator.data_objects import PRESIDIO_SPACY_ENTITIES

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    from flair.tokenization import SpacyTokenizer
except ImportError:
    print("Flair is not installed by default")

from presidio_evaluator import InputSample, tokenize, span_to_tag
from presidio_evaluator.models import BaseModel


class FlairModel(BaseModel):
    """
    Evaluator for Flair models

    .. deprecated::
        FlairModel is deprecated. Use Flair models through Presidio Analyzer instead.
        See: https://microsoft.github.io/presidio/analyzer/nlp_engines/flair/

    :param model: model of type SequenceTagger
    :param model_path:
    :param entities_to_keep:
    :param verbose:
    and model expected entity types
    """

    def __init__(
        self,
        model=None,
        model_path: str = None,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        entity_mapping: Dict[str, str] = PRESIDIO_SPACY_ENTITIES,
    ):
        warnings.warn(
            "FlairModel is deprecated and will be removed in a future version. "
            "Use Flair models through Presidio Analyzer instead. "
            "See: https://microsoft.github.io/presidio/analyzer/nlp_engines/flair/",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            entity_mapping=entity_mapping,
        )
        if model is None:
            if model_path is None:
                raise ValueError("Either model_path or model object must be supplied")
            self.model = SequenceTagger.load(model_path)
        else:
            self.model = model

        self.spacy_tokenizer = SpacyTokenizer(model=spacy.load("en_core_web_sm"))

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        sentence = Sentence(text=sample.full_text, use_tokenizer=self.spacy_tokenizer)
        self.model.predict(sentence)

        ents = sentence.get_spans("ner")
        if ents:
            tags, texts, start, end = zip(
                *[
                    (ent.tag, ent.text, ent.start_position, ent.end_position)
                    for ent in ents
                ]
            )

            tags = [
                tag if tag != "PER" else "PERSON" for tag in tags
            ]  # Flair's tag for PERSON is PER

            # Flair tokens might not be consistent with spaCy's tokens (even when using spacy tokenizer)
            # Use spacy tokenization and not stanza to maintain consistency with other models:
            if not sample.tokens:
                sample.tokens = tokenize(sample.full_text)

            # Create tags (label per token) based on stanza spans and spacy tokens
            tags = span_to_tag(
                scheme="IO",
                text=sample.full_text,
                starts=start,
                ends=end,
                tags=tags,
                tokens=sample.tokens,
            )
        else:
            tags = ["O" for _ in range(len(sample.tokens))]

        if len(tags) != len(sample.tokens):
            print("mismatch between input tokens and new tokens")

        return tags

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        predictions = []
        for sample in dataset:
            predictions.append(self.predict(sample, **kwargs))

        return predictions
