import json
import os
import time
from pathlib import Path


class ExperimentTracker:
    def __init__(self) -> None:
        self.parameters = {}
        self.metrics = {}
        self.dataset_info = None
        self.confusion_matrix = None
        self.labels = None
        self.output_dir = os.getcwd()

    def log_parameter(self, key: str, value: object) -> None:
        self.parameters[key] = value

    def log_parameters(self, parameters: dict) -> None:
        for k, v in parameters.items():
            self.log_parameter(k, v)

    def log_metric(self, key: str, value: object) -> None:
        self.metrics[key] = value

    def log_metrics(self, metrics: dict) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_dataset_hash(self, data: str) -> None:
        pass

    def log_dataset_info(self, name: str) -> None:
        self.dataset_info = name

    def __str__(self) -> str:
        return json.dumps(self.__dict__)

    def log_confusion_matrix(
        self,
        matrix: list[list[int]],
        labels=list[str],
    ) -> None:
        self.confusion_matrix = matrix
        self.labels = labels

    def start(self) -> None:
        pass

    def end(self) -> None:
        datetime_val = time.strftime("%Y%m%d-%H%M%S")
        filename = f"experiment_{datetime_val}.json"
        output_dir = Path(self.output_dir)
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create full file path using proper path joining
        output_path = output_dir / filename
        print(f"saving experiment data to {output_path}")

        with open(output_path, "w") as json_file:
            json.dump(self.__dict__, json_file)
