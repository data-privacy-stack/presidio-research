from pathlib import Path

import requests
from spacy.training.converters import conll_ner_to_docs
from tqdm import tqdm

from presidio_evaluator import InputSample
from presidio_evaluator.dataset_formatters import DatasetFormatter


class CONLL2003Formatter(DatasetFormatter):
    def __init__(
        self,
        files_path=Path("../../data/conll2003").resolve(),  # noqa: B008
        glob_pattern: str = "*.*",
    ) -> None:
        self.files_path = files_path
        self.glob_pattern = glob_pattern

    @staticmethod
    def download(
        local_data_path=Path("../../data/conll2003").resolve(),  # noqa: B008
        conll_gh_path="https://raw.githubusercontent.com/glample/tagger/master/dataset/",
    ) -> None:
        for fold in ("eng.train", "eng.testa", "eng.testb"):
            fold_path = conll_gh_path + fold
            if not local_data_path.exists():
                local_data_path.mkdir(parents=True)

            dataset_file = Path(local_data_path, fold)
            if dataset_file.exists():
                print("Dataset already exists, skipping download")
                return

            response = requests.get(fold_path, timeout=30)
            dataset_raw = response.text
            with open(dataset_file, "w") as f:
                f.write(dataset_raw)
            print(f"Finished writing fold {fold} to {local_data_path}")

        print("Finished downloading CoNNL2003")

    def to_input_samples(self, fold: str | None = None) -> list[InputSample]:
        files_found = False
        input_samples = []
        for _i, file_path in enumerate(self.files_path.glob(self.glob_pattern)):
            if fold and fold not in file_path.name:
                continue

            files_found = True
            with open(file_path, encoding="utf-8") as file:
                text = file.readlines()

            text = "".join(text)

            output_docs = conll_ner_to_docs(
                input_data=text,
                n_sents=None,
                no_print=True,
            )
            for doc in tqdm(output_docs, f"Processing doc for file {file_path.name}"):
                input_samples.append(InputSample.from_spacy_doc(doc=doc))

        if not files_found:
            raise FileNotFoundError(
                f"No files found for pattern {self.glob_pattern} and fold {fold}",
            )

        return input_samples


if __name__ == "__main__":
    conll_formatter = CONLL2003Formatter()
    train_samples = conll_formatter.to_input_samples(fold="train")
    print(train_samples[:5])
