from pathlib import Path

from . import raw_data

_raw_data_path = raw_data.__path__
if not hasattr(_raw_data_path, "__getitem__"):
    _raw_data_path = _raw_data_path._path
raw_data_dir = Path(_raw_data_path[0])

from typing import Never  # noqa: E402

from .presidio_pseudonymize import PresidioPseudonymization  # noqa: E402
from .presidio_sentence_faker import (  # noqa: E402
    PresidioSentenceFaker,
    presidio_additional_entity_providers,
    presidio_templates_file_path,
)


def read_synth_dataset() -> Never:
    raise DeprecationWarning(
        "read_synth_dataset is deprecated. Please use InputSample.read_dataset_json",
    )


__all__ = [
    "PresidioSentenceFaker",
    "PresidioPseudonymization",
    "read_synth_dataset",
    "raw_data_dir",
    "presidio_templates_file_path",
    "presidio_additional_entity_providers",
]
