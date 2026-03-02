from typing import List, Optional, Dict
import warnings

import spacy

from presidio_evaluator import InputSample, span_to_tag, tokenize
from presidio_evaluator.data_objects import PRESIDIO_SPACY_ENTITIES

try:
    import spacy_stanza
    import stanza  # noqa: F401
except ImportError:
    print("stanza and spacy_stanza are not installed")
from presidio_evaluator.models import SpacyModel


class StanzaModel(SpacyModel):
    """
    Class wrapping Stanza models, using spacy_stanza.

    .. deprecated::
        StanzaModel is deprecated. Use Stanza models through Presidio Analyzer instead.

    :param model: spaCy Language object representing a stanza model
    :param model_name: Name of model, e.g. "en"
    :param entities_to_keep: List of entities to predict on
    :param verbose: Whether to print more
    :param labeling_scheme: Whether to return IO, BIO or BILUO tags
    :param entity_mapping: Mapping between input dataset entities and entities expected by the model
    """

    def __init__(
        self,
        model: spacy.language.Language = None,
        model_name: str = "en",
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        entity_mapping: Optional[Dict[str, str]] = PRESIDIO_SPACY_ENTITIES,
    ):
        warnings.warn(
            "StanzaModel is deprecated and will be removed in a future version. "
            "Use Stanza models through Presidio Analyzer instead. ",
            DeprecationWarning,
            stacklevel=2,
        )
        if not model and not model_name:
            raise ValueError("Either model_name or model object must be supplied")
        if not model:
            model = spacy_stanza.load_pipeline(
                model_name,
                processors="tokenize,pos,lemma,ner",
            )

        super().__init__(
            model=model,
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        """
        Predict the tags using a stanza model.

        :param sample: InputSample with text
        :return: list of tags
        """

        doc = self.model(sample.full_text)
        if doc.ents:
            tags, texts, start, end = zip(
                *[(s.label_, s.text, s.start_char, s.end_char) for s in doc.ents]
            )

            # Stanza tokens might not be consistent with spaCy's tokens.
            # Use spacy tokenization and not stanza
            # to maintain consistency with other models:
            if not sample.tokens:
                sample.tokens = tokenize(sample.full_text)

            # Create tags (label per token) based on stanza spans and spacy tokens
            tags = span_to_tag(
                scheme=self.labeling_scheme,
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
