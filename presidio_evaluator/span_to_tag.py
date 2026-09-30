import spacy
from spacy.tokens import Doc

loaded_spacy = {}


def get_spacy(loaded_spacy=loaded_spacy, model_version="en_core_web_sm"):
    if model_version not in loaded_spacy:
        print(f"loading model {model_version}")
        loaded_spacy[model_version] = spacy.load(model_version)
    return loaded_spacy[model_version]


def tokenize(text, model_version="en_core_web_sm") -> Doc:
    return get_spacy(model_version=model_version)(text)


def _get_detailed_tags_for_span(scheme: str, cur_tags: list[str]) -> list[str]:
    """
    Replace IO tags (e.g. O PERSON PERSON) with BIO/BILUO tags.
    """

    if all(tag == "O" for tag in cur_tags):
        return cur_tags

    return_tags = []
    if len(cur_tags) == 1:
        if scheme == "BILUO":
            return_tags.append(f"U-{cur_tags[0]}")
        else:
            return_tags.append(f"B-{cur_tags[0]}")
    elif len(cur_tags) > 0:
        tg = cur_tags[0]
        for j in range(0, len(cur_tags)):
            if j == 0:
                return_tags.append(f"B-{tg}")
            elif j == len(cur_tags) - 1:
                if scheme == "BILUO":
                    return_tags.append(f"L-{tg}")
                else:
                    return_tags.append(f"I-{tg}")
            else:
                return_tags.append(f"I-{tg}")
    return return_tags


def _sort_spans(start, end, tag, score, span_id):
    if len(start) > 0:
        tpl = [
            (a, b, c, d, e)
            for a, b, c, d, e in sorted(
                zip(start, end, tag, score, span_id, strict=False),
                key=lambda pair: pair[0],
            )
        ]
        start, end, tag, score, span_id = [
            [x[i] for x in tpl] for i in range(len(tpl[0]))
        ]
    return start, end, tag, score, span_id


def _handle_overlaps(start, end, tag, score, span_id):
    start, end, tag, score, span_id = _sort_spans(start, end, tag, score, span_id)
    if len(start) == 0:
        return start, end, tag, score, span_id
    number_of_spans = len(start)
    i = 0
    while i < number_of_spans - 1:
        for j in range(i + 1, number_of_spans):
            # Span j intersects with span i
            if start[i] <= start[j] <= end[i]:
                # i's score is higher, remove intersecting part
                if score[i] > score[j]:
                    # j is contained within i but has lower score, remove
                    if start[i] <= end[j] <= end[i]:
                        score[j] = 0
                    # else, j continues after i ended:
                    else:
                        start[j] = end[i] + 1
                # j's score is higher, break i
                # If i finishes after j ended, split i
                elif end[j] < end[i]:
                    # create new span at the end; the segment is still the same
                    # input span, so it keeps i's identity
                    start.append(end[j] + 1)
                    end.append(end[i])
                    score.append(score[i])
                    tag.append(tag[i])
                    span_id.append(span_id[i])
                    number_of_spans += 1
                    # truncate the current i to end at start(j)
                    end[i] = start[j] - 1
                # else, i finishes before j ended. truncate i
                else:
                    end[i] = start[j] - 1

        i += 1
    start, end, tag, score, span_id = _sort_spans(start, end, tag, score, span_id)
    return start, end, tag, score, span_id


def span_to_tag(
    scheme: str,
    text: str,
    starts: list[int],
    ends: list[int],
    tags: list[str],
    scores: list[float] | None = None,
    tokens: Doc | list[str] | None = None,
    token_model_version: str = "en_core_web_sm",  # noqa: S107
    return_span_ids: bool = False,
    start_indices: list[int] | None = None,
) -> list[str] | tuple[list[str], list[int | None]]:
    """
    Turns a list of start and end values with corresponding labels, into a NER
    tagging (BILUO,BIO/IOB)
    :param scheme: labeling scheme, either BILUO, BIO/IOB or IO
    :param text: input text
    :param tokens: text tokenized to tokens: a spaCy Doc, or plain strings
        together with ``start_indices``
    :param starts: list of indices where entities in the text start
    :param ends: list of indices where entities in the text end
    :param tags: list of entity names
    :param scores: score of tag (confidence)
    :param token_model_version: model used for tokenization if no tokens provided
    :param return_span_ids: when True, also return one entry per token holding
        the index of the input span that tagged it (None for O tokens). Ids
        refer to the caller's span order; a span split by overlap resolution
        keeps one id for all its segments.
    :param start_indices: character offset of each token; required when
        ``tokens`` are plain strings, ignored for a spaCy Doc
    :return: list of strings, representing either BILUO or BIO for the input;
        with return_span_ids, a (tags, span_ids) tuple instead
    """

    if not scores:
        # assume all scores are of equal weight
        scores = [0.5 for start in starts]

    span_ids = list(range(len(starts)))
    starts, ends, tags, scores, span_ids = _handle_overlaps(
        starts, ends, tags, scores, span_ids
    )

    if not tokens:
        tokens = tokenize(text, token_model_version)
    if isinstance(tokens, Doc):
        offsets = [(token.idx, len(token.text)) for token in tokens]
    else:
        if start_indices is None or len(start_indices) != len(tokens):
            raise ValueError(
                "start_indices must be given, one per token, when tokens are strings"
            )
        offsets = [(idx, len(str(token))) for token, idx in zip(tokens, start_indices)]

    io_tags = []
    token_span_ids: list[int | None] = []
    for token_idx, token_len in offsets:
        found = False
        for span_index in range(0, len(starts)):
            span_start_in_token = (
                token_idx <= starts[span_index] <= token_idx + token_len
            )
            span_end_in_token = token_idx <= ends[span_index] <= token_idx + token_len
            if (
                starts[span_index] <= token_idx < ends[span_index]
            ):  # token start is between start and end
                io_tags.append(tags[span_index])
                found = True
            elif (
                span_start_in_token and span_end_in_token
            ):  # span is within token boundaries (special case)
                io_tags.append(tags[span_index])
                found = True
            if found:
                token_span_ids.append(span_ids[span_index])
                break

        if not found:
            io_tags.append("O")
            token_span_ids.append(None)

    out_tags = io_tags if scheme == "IO" else io_to_scheme(io_tags, scheme)
    if return_span_ids:
        return out_tags, token_span_ids
    return out_tags


def tags_to_span_ids(tags: list[str]) -> list[int | None]:
    """Derive a per-token entity instance id from a tag sequence.

    A new instance starts on a ``B-`` or ``U-`` prefix, on a change of entity
    type, or after an ``O`` token; ``I-``/``L-`` tokens continue the current
    one. Ids are sequential from 0 and ``None`` for ``O``.

    BIO/BILUO tags give exact instance boundaries. Plain IO tags carry no
    boundary information, so two adjacent entities of the same type read as one
    instance; callers that know their spans should use :func:`span_to_tag` with
    ``return_span_ids=True`` instead.

    :param tags: per-token tags in IO, BIO or BILUO scheme.
    :return: one id per token.
    """
    ids: list[int | None] = []
    current_id = -1
    current_type: str | None = None
    for raw_tag in tags:
        tag = str(raw_tag)
        if tag == "O":
            ids.append(None)
            current_type = None
            continue
        prefix, sep, entity_type = tag.partition("-")
        if not sep:
            prefix, entity_type = "", tag
        if prefix in ("B", "U") or entity_type != current_type:
            current_id += 1
        ids.append(current_id)
        current_type = entity_type
    return ids


def io_to_scheme(io_tags: list[str], scheme: str) -> list[str]:
    """Set tagging based on scheme (BIO or BILUO).
    :param io_tags: List of tags in IO (e.g. O O O PERSON PERSON O)
    :param scheme: Requested scheme (IO, BILUO or BIO)
    """

    if scheme == "IO":
        return io_tags

    if scheme == "BILOU":
        scheme = "BILUO"

    current_tag = ""
    span_index = 0
    changes = []
    for io_tag in io_tags:
        if io_tag != current_tag:
            changes.append(span_index)
        span_index += 1
        current_tag = io_tag
    changes.append(len(io_tags))
    new_return_tags = []
    for i in range(len(changes) - 1):
        new_return_tags.extend(
            _get_detailed_tags_for_span(
                scheme=scheme,
                cur_tags=io_tags[changes[i] : changes[i + 1]],
            ),
        )
    return new_return_tags
