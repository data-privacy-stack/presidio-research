import collections
import json
import logging
import os
from pathlib import Path

import xmltodict
from tqdm import tqdm

from presidio_evaluator import InputSample
from presidio_evaluator.data_objects import Span
from presidio_evaluator.dataset_formatters import DatasetFormatter

logger = logging.getLogger("presidio-research")


class I2B22014Formatter(DatasetFormatter):
    def __init__(
        self,
        files_path="../../data/i2b2/2014/testing-PHI-Gold-fixed",
    ) -> None:
        self.files_path = files_path

    @staticmethod
    def _create_span(item):
        span = Span(
            entity_type=item["@TYPE"],
            entity_value=item["@text"],
            start_position=int(item["@start"]),
            end_position=int(item["@end"]),
        )
        return span

    def to_input_samples(self, folder: str | None = None) -> list[InputSample]:
        input_samples = []
        if folder:
            self.files_path = folder
        logger.info(f"Parsing files in {self.files_path}")

        for root, _dirs, files in os.walk(self.files_path):
            for file in tqdm(files, desc="Reading files..."):
                spans = []
                filename = os.path.join(root, file)
                xml_content = open(filename).read()

                ordered_dict = xmltodict.parse(
                    xml_input=xml_content,
                    strip_whitespace=False,
                )
                data = dict(ordered_dict["deIdi2b2"])
                text = data["TEXT"]
                tags = data["TAGS"]
                for key, value in tags.items():
                    if key == "#text":
                        continue
                    if type(value) is collections.OrderedDict:
                        span = self._create_span(value)
                        if self._span_and_text_are_identical(span, text):
                            raise ValueError(f"Mismatch on file {file}, span {span}")

                        spans.append(span)
                    else:
                        for sub in value:
                            span = self._create_span(sub)
                            if self._span_and_text_are_identical(span, text):
                                raise ValueError(
                                    f"Mismatch on file {file}, span {span}",
                                )
                            spans.append(span)
                input_samples.append(
                    InputSample(
                        full_text=text, spans=spans, create_tags_from_span=True
                    ),
                )

            return input_samples

    @staticmethod
    def dataset_to_json(input_path, output_path) -> None:
        formatter = I2B22014Formatter(files_path=input_path)
        train_samples = formatter.to_input_samples()
        json_dataset = [example.to_dict() for example in train_samples]

        logger.info(f"Saving dataset to json. Path: {output_path}")
        with open(f"{output_path}", "w+", encoding="utf-8") as f:
            json.dump(json_dataset, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _span_and_text_are_identical(span: Span, text: str) -> bool:
        """
        There are mismatches between annotations and text.

        To verify that the annotation matches the text,
        there is a need to ignore cases where the annotations are different
        due to whitespaces, newline or other cases.
        """

        # fmt: off
        return span.entity_value != text[
            span.start_position : span.end_position
        ].replace("Michael \n\nOgrady", "Michael   Ogrady")\
            .replace("Yehuda \n\nLanders", "Yehuda   Landers")\
            .replace("February\n\n2086", "February  2086")\
            .replace("31,\n\n", "31,  ")\
            .replace("pa\nPic", "paPic")\
            .replace("est\nBra", "estBra")\
            .replace("er \n30", "er 30")\
            .replace("pr\n29", "pr29")\
            .replace(" \n ", "  ")\
            .replace("\n\n", "")\
            .replace("\n", " ")\
            .replace("&", "and")
        # fmt: on


if __name__ == "__main__":
    # Data is assumed to be on the data folder (repo root) under i2b2/2014
    # train 1
    input_path1 = Path("../../data/i2b2/2014/training-PHI-Gold-Set1")
    output_path1 = Path("../../data/i2b2/2014/training-PHI-Gold-Set1.json")
    I2B22014Formatter.dataset_to_json(input_path1, output_path1)

    # train 2
    input_path2 = Path("../../data/i2b2/2014/training-PHI-Gold-Set2")
    output_path2 = Path("../../data/i2b2/2014/training-PHI-Gold-Set2.json")
    I2B22014Formatter.dataset_to_json(input_path2, output_path2)

    # test
    input_path3 = Path("../../data/i2b2/2014/testing-PHI-Gold-fixed")
    output_path3 = Path("../../data/i2b2/2014/testing-PHI-Gold-fixed.json")
    I2B22014Formatter.dataset_to_json(input_path3, output_path3)
