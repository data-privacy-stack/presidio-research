import pytest

from presidio_evaluator import io_to_scheme, span_to_tag

BILUO_SCHEME = "BILUO"
BIO_SCHEME = "BIO"
IO_SCHEME = "IO"


# fmt: off
def test_span_to_bio_multiple_tokens():
    text = "My Address is 409 Bob st. Manhattan NY. I just moved in"
    start = 14
    end = 38
    tag = "ADDRESS"

    bio = span_to_tag(BIO_SCHEME, text, [start], [end], [tag])

    print(bio)

    expected = ['O', 'O', 'O', 'B-ADDRESS', 'I-ADDRESS', 'I-ADDRESS',
                'I-ADDRESS', 'I-ADDRESS', 'I-ADDRESS', 'O', 'O', 'O', 'O', 'O']
    assert bio == expected


def test_span_to_bio_single_at_end():
    text = "My name is Josh"
    start = 11
    end = 15
    tag = "NAME"

    biluo = span_to_tag(BIO_SCHEME, text, [start], [end], [tag] )

    expected = ['O', 'O', 'O', 'B-NAME']
    assert biluo == expected


def test_span_to_biluo_multiple_tokens():
    text = "My Address is 409 Bob st. Manhattan NY. I just moved in"
    start = 14
    end = 38
    tag = "ADDRESS"

    biluo = span_to_tag(BILUO_SCHEME, text, [start], [end], [tag])

    expected = ['O', 'O', 'O', 'B-ADDRESS', 'I-ADDRESS', 'I-ADDRESS',
                'I-ADDRESS', 'I-ADDRESS', 'L-ADDRESS', 'O', 'O', 'O', 'O', 'O']
    assert biluo == expected


def test_span_to_biluo_adjacent_entities():
    text = "Mr. Tree"
    start1 = 0
    end1 = 2
    start2 = 4
    end2 = 8

    start = [start1, start2]
    end = [end1, end2]

    tag = ["TITLE", "NAME"]

    biluo = span_to_tag(BILUO_SCHEME, text, start, end, tag)

    print(biluo)

    expected = ['U-TITLE', 'U-NAME']
    assert biluo == expected


def test_span_to_biluo_single_at_end():
    text = "My name is Josh"
    start = 11
    end = 15
    tag = "NAME"

    biluo = span_to_tag(BILUO_SCHEME, text, [start], [end], [tag])

    print(biluo)

    expected = ['O', 'O', 'O', 'U-NAME']
    assert biluo == expected


def test_span_to_biluo_multiple_entities():
    text = "My name is Josh or David"
    start1 = 11
    end1 = 15
    start2 = 19
    end2 = 26

    start = [start1, start2]
    end = [end1, end2]

    tag = ["NAME", "NAME"]

    biluo = span_to_tag(BILUO_SCHEME, text, start, end, tag)

    print(biluo)

    expected = ['O', 'O', 'O', 'U-NAME', 'O', 'U-NAME']
    assert biluo == expected


def test_span_to_bio_multiple_entities():
    text = "My name is Josh or David"
    start1 = 11
    end1 = 15
    start2 = 19
    end2 = 26

    start = [start1, start2]
    end = [end1, end2]

    tag = ["NAME", "NAME"]

    biluo = span_to_tag(scheme=BIO_SCHEME, text=text, starts=start,
                        ends=end, tags=tag)

    print(biluo)

    expected = ['O', 'O', 'O', 'B-NAME', 'O', 'B-NAME']
    assert biluo == expected


def test_span_to_bio_specific_input():
    text = "Someone stole my credit card. The number is 5277716201469117 and " \
           "the my name is Mary Anguiano"
    start = 80
    end = 93
    expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O', 'O', 'B-PERSON', 'I-PERSON']
    tag = ["PERSON"]
    biluo = span_to_tag(BIO_SCHEME, text, [start], [end], tag)
    assert biluo == expected


def test_span_to_biluo_specific_input():
    text = "Someone stole my credit card. The number is 5277716201469117 and " \
           "the my name is Mary Anguiano"
    start = 80
    end = 93
    expected = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O', 'O', 'B-PERSON', 'L-PERSON']
    tag = ["PERSON"]
    biluo = span_to_tag(BILUO_SCHEME, text, [start], [end], tag)
    assert biluo == expected


def test_span_to_biluo_adjecent_identical_entities():
    text = "May I get access to Jessica Gump's account?"
    start = 20
    end = 32
    expected = ['O', 'O', 'O', 'O', 'O', 'B-PERSON', 'L-PERSON', 'O', 'O', 'O']
    tag = ["PERSON"]
    biluo = span_to_tag(BILUO_SCHEME, text, [start], [end], tag)
    assert biluo == expected


def test_overlapping_entities_first_ends_in_mid_second():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [22, 25]
    end = [37, 37]
    scores = [0.6, 0.6]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'US_PHONE_NUMBER',
                'US_PHONE_NUMBER', 'US_PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(IO_SCHEME, text, start, end, tag, scores)
    assert io == expected


def test_overlapping_entities_second_embedded_in_first_with_lower_score():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [22, 25]
    end = [37, 33]
    scores = [0.6, 0.5]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'PHONE_NUMBER',
                'PHONE_NUMBER', 'PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(scheme=IO_SCHEME, text=text, starts=start, ends=end, tags=tag, scores=scores)
    assert io == expected


def test_overlapping_entities_second_embedded_in_first_has_higher_score():
    text = "My new phone number is 1 705 774 8720. Thanks, man"
    start = [23, 25]
    end = [37, 28]
    scores = [0.6, 0.7]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'US_PHONE_NUMBER',
                'PHONE_NUMBER', 'PHONE_NUMBER',
                 'O', 'O', 'O', 'O']
    io = span_to_tag(scheme=IO_SCHEME, text=text, starts=start, ends=end, tags=tag, scores=scores)
    assert io == expected


def test_overlapping_entities_second_embedded_in_first_has_lower_score():
    text= "My new phone number is 1 705 774 8720. Thanks, man"
    start = [23, 25]
    end = [37, 28]
    scores = [0.6, 0.3]
    tag = ["PHONE_NUMBER", "US_PHONE_NUMBER"]
    expected = ['O', 'O', 'O', 'O', 'O', 'PHONE_NUMBER', 'PHONE_NUMBER',
                'PHONE_NUMBER', 'PHONE_NUMBER',
                'O', 'O', 'O', 'O']
    io = span_to_tag(scheme=IO_SCHEME, text=text, starts=start, ends=end, tags=tag, scores=scores)
    assert io == expected


def test_overlapping_entities_pyramid():
    text = "My new phone number is 1 705 999 774 8720. Thanks, cya"
    start = [23, 25, 29]
    end = [41, 36, 32]
    scores = [0.6, 0.7, 0.8]
    tag = ["A1", "B2", "C3"]
    expected = ['O', 'O', 'O', 'O', 'O', 'A1', 'B2', 'C3', 'B2',
                 'A1', 'O', 'O', 'O', 'O']
    io = span_to_tag(scheme=IO_SCHEME, text=text, starts=start, ends=end, tags=tag, scores=scores)
    assert io == expected


def test_token_contains_span():
    # The last token here (https://www.gmail.com/) contains the span (www.gmail.com).
    # In this case the token should be tagged as the span tag, even if not all of it is covered by the span.

    text = "My website is https://www.gmail.com/"
    start = [22]
    end = [35]
    scores = [1.0]
    tag = ["DOMAIN_NAME"]
    expected = ["O", "O", "O", "DOMAIN_NAME"]
    io = span_to_tag(scheme=IO_SCHEME, text=text, starts=start, ends=end, tags=tag, scores=scores)
    assert io == expected


@pytest.mark.parametrize(
    "tags, expected_tags, scheme",
    [
        (["O", "O"], ["O", "O"], "IO"),
        (["O", "O"], ["O", "O"], "BIO"),
        (["O", "O"], ["O", "O"], "BILUO"),
        (["name", "name", "name"], ["name", "name", "name"], "IO"),
        (["name", "name", "name"], ["B-name", "I-name", "I-name"], "BIO"),
        (["name", "name", "name"], ["B-name", "I-name", "L-name"], "BILUO"),
        (["name", "name", "name", "O", "phone"], ["B-name", "I-name", "L-name", "O", "U-phone"], "BILUO"),
        (["O", "name", "O"], ["O", "B-name", "O"], "BIO"),
        (["O", "name", "O"], ["O", "U-name", "O"], "BILUO"),
        (["O", "name", "O"], ["O", "U-name", "O"], "BILOU"),
    ],
)
def test_io_to_scheme(tags, expected_tags, scheme):
    actual_tags = io_to_scheme(io_tags=tags, scheme=scheme)
    assert actual_tags == expected_tags

# fmt: on


# ── return_span_ids ──────────────────────────────────────────────────────────


def test_span_to_tag_returns_span_ids_aligned_with_tags():
    text = "Ana Ruiz 29"
    starts = [0, 9]
    ends = [8, 11]
    tags = ["NAME", "AGE"]

    io, span_ids = span_to_tag(
        IO_SCHEME, text, starts, ends, tags, return_span_ids=True
    )

    assert io == ["NAME", "NAME", "AGE"]
    assert span_ids == [0, 0, 1]


def test_span_to_tag_span_ids_are_none_for_o_tokens():
    text = "I am Josh"
    io, span_ids = span_to_tag(
        IO_SCHEME, text, [5], [9], ["NAME"], return_span_ids=True
    )

    assert io == ["O", "O", "NAME"]
    assert span_ids == [None, None, 0]


def test_span_to_tag_adjacent_same_type_spans_get_distinct_ids():
    """Two neighbouring entities of the same type are distinguishable by id."""
    text = "Paris London"
    starts = [0, 6]
    ends = [5, 12]
    tags = ["LOCATION", "LOCATION"]

    io, span_ids = span_to_tag(
        IO_SCHEME, text, starts, ends, tags, return_span_ids=True
    )

    assert io == ["LOCATION", "LOCATION"]
    assert span_ids == [0, 1]


def test_span_to_tag_split_span_keeps_one_id():
    """A span split by a higher-score overlap is still one entity."""
    text = "one two three"
    starts = [0, 4]
    ends = [13, 7]
    tags = ["X", "Y"]
    scores = [0.5, 0.9]

    io, span_ids = span_to_tag(
        IO_SCHEME, text, starts, ends, tags, scores, return_span_ids=True
    )

    assert io == ["X", "Y", "X"]
    assert span_ids == [0, 1, 0]


def test_span_to_tag_default_return_is_tags_only():
    tags = span_to_tag(IO_SCHEME, "I am Josh", [5], [9], ["NAME"])
    assert tags == ["O", "O", "NAME"]
