import dataclasses
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(eq=True)
class FakerSpan:
    """FakerSpan holds the start, end, value and type of every element replaced."""

    value: str
    start: int
    end: int
    type: str

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "FakerSpan is deprecated and will be removed in future versions."
            "Use Span instead",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return super().__new__(cls)

    def __repr__(self) -> str:
        return json.dumps(dataclasses.asdict(self))


@dataclass()
class FakerSpansResult:
    """FakerSpansResult holds the full fake sentence, the original template
    and a list of spans for each element replaced."""

    fake: str
    spans: list[FakerSpan]
    template: str | None = None
    template_id: int | None = None
    sample_id: int | None = None

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "FakerSpansResult is deprecated and will be removed in future versions."
            "Use InputSample instead",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return super().__new__(cls)

    def __str__(self) -> str:
        return self.fake

    def __repr__(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def toJSON(self):  # noqa: N802
        spans_dict = json.dumps([dataclasses.asdict(span) for span in self.spans])
        return json.dumps(
            {
                "fake": self.fake,
                "spans": spans_dict,
                "template": self.template,
                "template_id": self.template_id,
                "sample_id": self.sample_id,
            },
        )

    @classmethod
    def fromJSON(cls, json_string):  # noqa: N802
        """Load a single FakerSpansResult from a JSON string."""
        json_dict = json.loads(json_string)
        converted_spans = []
        for span_dict in json.loads(json_dict["spans"]):
            converted_spans.append(FakerSpan(**span_dict))
        json_dict["spans"] = converted_spans
        return cls(**json_dict)

    @classmethod
    def count_entities(cls, fake_records: list["FakerSpansResult"]) -> Counter:
        """Count frequency of entity types in a list of FakerSpansResult."""
        count_per_entity_new = Counter()
        for record in fake_records:
            for span in record.spans:
                count_per_entity_new[span.type] += 1
        return count_per_entity_new.most_common()

    @classmethod
    def load_dataset_from_file(
        cls,
        filename: Path | str,
    ) -> list["FakerSpansResult"]:
        """Load a dataset of FakerSpansResult from a JSON file."""
        with open(filename, encoding="utf-8") as f:
            return [cls.fromJSON(line) for line in f.readlines()]

    @classmethod
    def update_entity_types(
        cls,
        dataset: list["FakerSpansResult"],
        entity_mapping: dict[str, str],
    ) -> None:
        """Replace entity types using a translator dictionary."""
        for sample in dataset:
            # update entity types on spans
            for span in sample.spans:
                span.type = entity_mapping[span.type]
            # update entity types on the template string
            for key, value in entity_mapping.items():
                sample.template = sample.template.replace(
                    "{{" + key + "}}",
                    "{{" + value + "}}",
                )
