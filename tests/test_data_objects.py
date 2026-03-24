from copy import deepcopy
from pathlib import Path

import pytest
import spacy
from spacy.tokens import DocBin

from presidio_evaluator import InputSample, Span


@pytest.fixture(scope="session")
def small_dataset():
    dir_path = Path(__file__).parent
    input_samples = InputSample.read_dataset_json(
        Path(dir_path, "data", "generated_small.json")
    )
    return input_samples


@pytest.fixture(scope="session")
def input_sample_result():
    return InputSample(
        full_text="Dan is my name.",
        spans=[
            Span(
                entity_type="name", entity_value="Dan", start_position=0, end_position=3
            )
        ],
        masked="{{name}} is my name.",
        template_id=3,
        create_tags_from_span=True,
    )


@pytest.fixture(scope="session")
def input_sample_result_2():
    return InputSample(
        full_text="Dan is my name. Tel Aviv is my home.",
        spans=[
            Span(
                entity_type="name", entity_value="Dan", start_position=0, end_position=3
            ),
            Span(
                entity_type="city",
                entity_value="Tel Aviv",
                start_position=16,
                end_position=24,
            ),
        ],
        masked="{{name}} is my name. {{city}} is my home.",
        template_id=4,
        create_tags_from_span=True,
    )


@pytest.fixture(scope="session")
def pair_of_spans():
    span1 = Span(
        entity_type="PERSON",
        entity_value="Alice is my",
        start_position=3,
        end_position=14,
        normalized_tokens=["Alice"],
        normalized_start_index=3,
        normalized_end_index=8,
    )
    span2 = Span(
        entity_type="ORG",
        entity_value="Acme Corp",
        start_position=0,
        end_position=9,
        normalized_tokens=["Acme", "Corp"],
        normalized_start_index=0,
        normalized_end_index=9,
    )

    return span1, span2


def test_update_entity_types(input_sample_result):
    records = [deepcopy(input_sample_result)]
    [record.translate_input_sample_tags({"name": "PERSON"}) for record in records]
    assert records[0].spans[0].entity_type == "PERSON"


def test_load_dataset_from_file(input_sample_result_2):
    dir_path = Path(__file__).parent

    records = InputSample.read_dataset_json(
        Path(dir_path, "data", "mock_input_samples.json")
    )
    assert len(records) == 2
    assert records[0].masked == input_sample_result_2.masked
    assert records[0].full_text == input_sample_result_2.full_text
    assert records[0].spans == input_sample_result_2.spans


def test_count_entities(input_sample_result, input_sample_result_2):
    counts = InputSample.count_entities([input_sample_result, input_sample_result_2])
    assert len(counts) == 2
    assert all(ent[0] == "name" or ent[0] == "city" for ent in counts)


def test_remove_unsupported_entities(input_sample_result, input_sample_result_2):

    filtered = InputSample.remove_unsupported_entities(
        [input_sample_result, input_sample_result_2], {"name": "PERSON"}
    )
    assert len(filtered) == 1
    assert filtered[0].spans[0].entity_type == "name"


def test_to_conll():
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )

    conll = InputSample.create_conll_dataset(input_samples)

    sentences = conll["sentence"].unique()
    assert len(sentences) == len(input_samples)


def test_to_spacy_all_entities(small_dataset):
    spacy_ver = InputSample.create_spacy_dataset(small_dataset)

    assert len(spacy_ver) == len(small_dataset)


def test_to_spacy_all_entities_specific_entities(small_dataset):
    spacy_ver = InputSample.create_spacy_dataset(small_dataset, entities=["PERSON"])

    spacy_ver_with_labels = [
        sample for sample in spacy_ver if len(sample[1]["entities"])
    ]

    assert len(spacy_ver_with_labels) < len(small_dataset)
    assert len(spacy_ver_with_labels) > 0


def test_to_spacy_file_and_back(small_dataset):
    spacy_pipeline = spacy.load("en_core_web_sm")
    InputSample.create_spacy_dataset(
        small_dataset,
        output_path="dataset.spacy",
        translate_tags=False,
        spacy_pipeline=spacy_pipeline,
        alignment_mode="strict",
    )

    db = DocBin()
    db.from_disk("dataset.spacy")
    docs = db.get_docs(vocab=spacy_pipeline.vocab)
    for doc, input_sample in zip(docs, small_dataset):
        input_ents = sorted(input_sample.spans, key=lambda x: x.start_position)
        spacy_ents = sorted(doc.ents, key=lambda x: x.start_char)
        for spacy_ent, input_span in zip(spacy_ents, input_ents):
            assert spacy_ent.start_char == input_span.start_position
            assert spacy_ent.end_char == input_span.end_position


def test_from_spacy_doc():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Nice to meet you Mr. Perkins.")

    sample = InputSample.from_spacy_doc(doc)
    assert sample.spans[0].entity_type == "PERSON"
    assert sample.tags == ["O", "O", "O", "O", "O", "U-PERSON", "O"]


@pytest.mark.parametrize(
    "start1, end1, start2, end2, intersection_length, ignore_entity_type",
    [
        (150, 153, 160, 165, 0, True),
        (150, 153, 150, 153, 3, True),
        (150, 153, 152, 154, 1, True),
        (150, 153, 100, 151, 1, True),
        (150, 153, 100, 151, 0, False),
    ],
)
def test_spans_intersection(
    start1, end1, start2, end2, intersection_length, ignore_entity_type
):
    span1 = Span(
        entity_type="A", entity_value="123", start_position=start1, end_position=end1
    )
    span2 = Span(
        entity_type="B", entity_value="123", start_position=start2, end_position=end2
    )

    intersection = span1.intersect(span2, ignore_entity_type=ignore_entity_type)
    assert intersection == intersection_length


def test_start_indices_in_create_tags_from_span():
    """Test that start_indices are correctly set when using create_tags_from_span=True"""
    sample = InputSample(
        full_text="John works at Microsoft.",
        spans=[
            Span(
                entity_type="PERSON",
                entity_value="John",
                start_position=0,
                end_position=4,
            ),
            Span(
                entity_type="ORG",
                entity_value="Microsoft",
                start_position=14,
                end_position=23,
            ),
        ],
        create_tags_from_span=True,
    )

    # Verify start_indices are set correctly
    assert len(sample.tokens) == len(sample.start_indices)

    assert sample.tokens[0].text == "John"
    assert sample.start_indices[0] == 0

    assert sample.tokens[3].text == "Microsoft"
    assert sample.start_indices[3] == 14


def test_manually_set_start_indices():
    """Test manually setting start_indices during initialization"""
    sample = InputSample(
        full_text="Alex and Bob are friends.",
        spans=[
            Span(
                entity_type="PERSON",
                entity_value="Alex",
                start_position=0,
                end_position=4,
            ),
            Span(
                entity_type="PERSON",
                entity_value="Bob",
                start_position=9,
                end_position=12,
            ),
        ],
    )

    # When creating tags from spans, the start_indices should be updated
    tokens, labels, start_indices = sample.get_tags()
    assert start_indices == [0, 5, 9, 13, 17, 24]


def test_span_intersection(pair_of_spans):
    """Test that spans with different entity types do not intersect"""
    span1 = pair_of_spans[0]
    span2 = pair_of_spans[1]

    intersection_strict = span1.intersect(span2, ignore_entity_type=False)
    assert intersection_strict == 0  # Different types should not intersect

    intersection_type = span1.intersect(span2, ignore_entity_type=True)
    assert intersection_type == 6

    intersection_strict_normalized = span1.intersect(
        span2, ignore_entity_type=False, use_normalized_indices=True
    )
    assert intersection_strict_normalized == 0  # Different types should not intersect

    intersection_type_normalized = span1.intersect(
        span2, ignore_entity_type=True, use_normalized_indices=True
    )
    assert intersection_type_normalized == 5


def test_span_union(pair_of_spans):
    """Test that spans with different entity types do not intersect"""
    span1 = pair_of_spans[0]
    span2 = pair_of_spans[1]

    union_strict = span1.union(span2, ignore_entity_type=False)
    assert union_strict == 0  # Different types should not union

    union_type = span1.union(span2, ignore_entity_type=True)
    assert union_type == 14

    union_strict_normalized = span1.union(
        span2, ignore_entity_type=False, use_normalized_indices=True
    )
    assert (
        union_strict_normalized == 0
    )  # Different types should not union in normalized form

    union_type_normalized = span1.union(
        span2, ignore_entity_type=True, use_normalized_indices=True
    )
    assert union_type_normalized == 9


def test_span_iou(pair_of_spans):
    """Test that spans with different entity types do not intersect"""
    span1 = pair_of_spans[0]
    span2 = pair_of_spans[1]

    iou_strict = span1.iou(span2, ignore_entity_type=False)
    assert iou_strict == 0

    iou_type = span1.iou(span2, ignore_entity_type=True)
    assert iou_type == 6 / 14.0

    iou_strict_normalized = span1.iou(
        span2, ignore_entity_type=False, use_normalized_indices=True
    )
    assert iou_strict_normalized == 0

    iou_type_normalized = span1.iou(
        span2, ignore_entity_type=True, use_normalized_indices=True
    )
    assert iou_type_normalized == 5 / 9.0


def test_extract_entity_types_from_spans():
    """Test extracting entity types from span-based samples"""
    samples = [
        InputSample(
            full_text="John works at Microsoft.",
            spans=[
                Span(
                    entity_type="PERSON",
                    entity_value="John",
                    start_position=0,
                    end_position=4,
                ),
                Span(
                    entity_type="ORG",
                    entity_value="Microsoft",
                    start_position=14,
                    end_position=23,
                ),
            ],
        ),
        InputSample(
            full_text="Alice lives in Seattle.",
            spans=[
                Span(
                    entity_type="PERSON",
                    entity_value="Alice",
                    start_position=0,
                    end_position=5,
                ),
                Span(
                    entity_type="LOCATION",
                    entity_value="Seattle",
                    start_position=15,
                    end_position=22,
                ),
            ],
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON", "ORG", "LOCATION"}


def test_extract_entity_types_from_tags_bio():
    """Test extracting entity types from BIO-tagged samples"""
    samples = [
        InputSample(
            full_text="John Smith works here.",
            tags=["B-PERSON", "I-PERSON", "O", "O"],
            tokens=[],
            create_tags_from_span=False,
        ),
        InputSample(
            full_text="Microsoft Corporation in Seattle.",
            tags=["B-ORG", "I-ORG", "O", "B-LOCATION"],
            tokens=[],
            create_tags_from_span=False,
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON", "ORG", "LOCATION"}


def test_extract_entity_types_from_tags_biluo():
    """Test extracting entity types from BILUO-tagged samples"""
    samples = [
        InputSample(
            full_text="John works at Microsoft.",
            tags=["U-PERSON", "O", "O", "U-ORG"],
            tokens=[],
            create_tags_from_span=False,
        ),
        InputSample(
            full_text="Bob Smith at Google.",
            tags=["B-PERSON", "L-PERSON", "O", "U-ORG"],
            tokens=[],
            create_tags_from_span=False,
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON", "ORG"}


def test_extract_entity_types_from_tags_io():
    """Test extracting entity types from IO-tagged samples"""
    samples = [
        InputSample(
            full_text="John works.",
            tags=["PERSON", "O"],
            tokens=[],
            create_tags_from_span=False,
        ),
        InputSample(
            full_text="Microsoft here.",
            tags=["ORG", "O"],
            tokens=[],
            create_tags_from_span=False,
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON", "ORG"}


def test_extract_entity_types_mixed():
    """Test extracting entity types from both spans and tags"""
    samples = [
        InputSample(
            full_text="John works at Microsoft.",
            spans=[
                Span(
                    entity_type="PERSON",
                    entity_value="John",
                    start_position=0,
                    end_position=4,
                ),
                Span(
                    entity_type="ORG",
                    entity_value="Microsoft",
                    start_position=14,
                    end_position=23,
                ),
            ],
        ),
        InputSample(
            full_text="Seattle is nice.",
            tags=["B-LOCATION", "O", "O"],
            tokens=[],
            create_tags_from_span=False,
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON", "ORG", "LOCATION"}


def test_extract_entity_types_empty_dataset():
    """Test extracting entity types from empty dataset"""
    entity_types = InputSample.extract_entity_types([])
    assert entity_types == set()


def test_extract_entity_types_no_entities():
    """Test extracting entity types when samples have no entities"""
    samples = [
        InputSample(
            full_text="Hello world.",
            tags=["O", "O"],
            tokens=[],
            create_tags_from_span=False,
        )
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == set()


def test_extract_entity_types_duplicate_entities():
    """Test that duplicate entity types are handled correctly"""
    samples = [
        InputSample(
            full_text="John and Alice.",
            spans=[
                Span(
                    entity_type="PERSON",
                    entity_value="John",
                    start_position=0,
                    end_position=4,
                ),
                Span(
                    entity_type="PERSON",
                    entity_value="Alice",
                    start_position=9,
                    end_position=14,
                ),
            ],
        ),
        InputSample(
            full_text="Bob here.",
            spans=[
                Span(
                    entity_type="PERSON",
                    entity_value="Bob",
                    start_position=0,
                    end_position=3,
                )
            ],
        ),
    ]

    entity_types = InputSample.extract_entity_types(samples)
    assert entity_types == {"PERSON"}  # Should only appear once in the set
