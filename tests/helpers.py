"""Shared helpers for building evaluator DataFrames in tests."""

import pandas as pd

from presidio_evaluator import tags_to_span_ids
from presidio_evaluator.entity_mapping.data_objects import (
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
)


def with_span_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach span-id columns derived from the label columns' runs.

    ``BaseModel.predict_dataset`` produces these columns from the source spans.
    Hand-built frames only have IO labels, so ids are label runs, one per
    sentence, which matches what a tag-only model would produce.
    """
    df = df.copy()
    for label_column, id_column in (
        ("annotation", ANNOTATION_SPAN_ID),
        ("prediction", PREDICTION_SPAN_ID),
    ):
        if label_column not in df.columns or id_column in df.columns:
            continue
        ids: list[int | None] = []
        for _, sentence in df.groupby("sentence_id", sort=False):
            ids += tags_to_span_ids(sentence[label_column].tolist())
        df[id_column] = pd.Series(ids, dtype="object", index=df.index)
    return df


def make_results_df(tokens, annotations, predictions, sentence_id=0) -> pd.DataFrame:
    """One-sentence results frame with whitespace-joined start indices."""
    starts, position = [], 0
    for token in tokens:
        starts.append(position)
        position += len(token) + 1
    return with_span_ids(
        pd.DataFrame(
            {
                "sentence_id": [sentence_id] * len(tokens),
                "token": tokens,
                "start_indices": starts,
                "annotation": annotations,
                "prediction": predictions,
            }
        )
    )
