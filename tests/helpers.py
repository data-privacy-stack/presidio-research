"""Shared helpers for building evaluator DataFrames in tests."""

import pandas as pd

from presidio_evaluator import ensure_span_ids


def with_span_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach span-id columns derived from label runs (see ensure_span_ids)."""
    return ensure_span_ids(df)


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
