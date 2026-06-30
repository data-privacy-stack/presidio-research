from abc import ABC, abstractmethod

from presidio_evaluator import InputSample


class DatasetFormatter(ABC):
    @abstractmethod
    def to_input_samples(self) -> list[InputSample]:
        """
        Translate a dataset structure into a list of documents, to be used by models and for evaluation
        :return:
        """
        pass
