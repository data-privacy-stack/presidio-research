import re

from faker import Generator

from presidio_evaluator import InputSample, Span

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")


class SpanGenerator(Generator):
    """Faker Generator which also returns the indices of fake values.

    :example:
    >>>from faker import Faker
    >>>from presidio_evaluator.data_generator.faker_extensions import SpanGenerator

    >>>generator = SpanGenerator()
    >>>faker = Faker(generator=generator)
    >>>res = faker.parse("My child's name is {{name}}", add_spans=True)

    >>>res.spans
        [{"value": "Daniel Gallagher", "start": 19, "end": 35, "type": "name"}]
    >>>res.fake
        "My child's name is Daniel Gallagher"
    >>>str(res)
        "My child's name is Daniel Gallagher"
    """

    def parse(
        self,
        text: str,
        add_spans: bool = False,
        template_id: int | None = None,
        sample_id: int | None = None,
        lower_case: bool = False,
    ) -> str | InputSample:
        """Parses a Faker template.

        This replaces the original parse method to introduce spans.
        :param text: Text holding the faker template, e.g. "My name is {{name}}".
        :param add_spans: Whether to return the spans of each fake value in the output string
        :param template_id: Template ID to be returned with the output
        :param sample_id: unique sample id
        :param lower_case: Whether to lower-case the template text and the fake values.
        Each piece is lowered on its own, which can differ from lowering the whole
        sentence where lower() depends on context (a Greek capital sigma at the end
        of a value becomes a final sigma even when a letter follows it).
        The {{placeholders}} in `masked` keep their case.
        """

        # Create Span objects for original placeholders
        spans = self._match_to_span(text)

        # Build the text from left to right, taking each offset from the text
        # built so far: a fake value, or a piece of template text once lowered
        # ("İ" -> "i̇"), can have a different length than what it replaces
        fake_pieces: list[str] = []
        masked_pieces: list[str] = []
        length = 0
        cursor = 0
        for span in sorted(spans, key=lambda x: x.start_position):
            literal = text[cursor : span.start_position]
            placeholder = text[span.start_position : span.end_position]
            value = str(span.entity_value)
            if lower_case:
                literal = literal.lower()
                value = value.lower()
            cursor = span.end_position

            fake_pieces += [literal, value]
            masked_pieces += [literal, placeholder]
            span.start_position = length + len(literal)
            span.end_position = span.start_position + len(value)
            span.entity_value = value
            span.entity_type = span.entity_type.strip()
            length = span.end_position

        rest = text[cursor:].lower() if lower_case else text[cursor:]
        fake_pieces.append(rest)
        masked_pieces.append(rest)
        fake_text = "".join(fake_pieces)

        if not add_spans:
            return fake_text

        return InputSample(
            full_text=fake_text,
            spans=sorted(spans, reverse=True, key=lambda x: x.start_position),
            masked="".join(masked_pieces),
            template_id=template_id,
            sample_id=sample_id,
        )

    def _match_to_span(self, text: str, **kwargs) -> list[Span]:
        matches = _re_token.finditer(text)

        results: list[Span] = []
        for match in matches:
            formatter = match.group()[2:-2].lower()
            results.append(
                Span(
                    entity_type=formatter,
                    start_position=match.start(),
                    end_position=match.end(),
                    entity_value=str(self.format(formatter.strip(), **kwargs)),
                ),
            )

        return results
