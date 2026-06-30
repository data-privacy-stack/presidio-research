import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from presidio_evaluator.evaluation.model_error import ModelError
from presidio_evaluator.experiment_tracking.experiment_tracker import ExperimentTracker


class LocalExperimentTracker(ExperimentTracker):
    def __init__(self, directory: str, experiment_name: str) -> None:
        """`LocalExperimentTracker` is intended to store ModelError results from evaluation to a local directory.
            Each new experiment generate an experiment directory and stores the following files inside the directory:
            1. confusion_matrix.csv - The generated confusion matrix for each entity
            2. experiment_params.json - Dictionary containing any logged information needed to the user
            3. experiment_errors.pkl - Pickles representation of a `List[ModelError]` instances.

        Args:
            directory (str): Main location of experimentation folder
            experiment_name (str): Name of experiment to store information
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.dir = directory
        self.start()

    def start(self) -> None:
        """Generates a new folder within the experiments directory"""
        full_path = Path(Path.cwd(), self.dir, self.experiment_name)
        full_path.mkdir(parents=True, exist_ok=True)
        self.log_start_time()

    def log_start_time(self) -> None:
        self.experiment_start_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    def log_errors(self, errors: list[ModelError]) -> None:
        """Generate a serialized version of all model errors and store as `.pkl` file
        Args:
            errors (List[ModelError]): List containing generated errors by the NER model
        """
        errors_str = self._serialize_errors(errors)
        file_path = Path(
            Path.cwd(),
            self.dir,
            self.experiment_name,
            "experiment_errors.pkl",
        )
        file = open(file_path, "wb")
        pickle.dump(errors_str, file)
        file.close()

    def _serialize_errors(self, errors: list) -> list[dict]:
        errors_str = []
        for error in errors:
            error.vector = error.token.vector
            error.token = str(error.token)
            errors_str.append(error)
        return errors_str

    def log_confusion_matrix_table(
        self,
        matrix: list[list[int]],
        labels=list[str],
    ) -> None:
        """Store confusion matrix as a .csv file

        Args:
            matrix (List[List[int]]): Error counts
            labels (_type_, optional): Names of entities. Defaults to List[str].
        """
        self.confusion_matrix = matrix
        self.labels = labels
        file_path = Path(
            Path.cwd(),
            self.dir,
            self.experiment_name,
            "confusion_matrix.csv",
        )
        # store confusion matrix
        pd.DataFrame(matrix, columns=labels, index=labels).to_csv(file_path)

    def end(self) -> None:
        """Store all logged information into a JSON file"""
        file_path = Path(
            Path.cwd(),
            self.dir,
            self.experiment_name,
            "experiment_params.json",
        )
        print(f"saving experiment data to directory {self.experiment_name}")
        with file_path.open("w") as json_file:
            json.dump(self.__dict__, json_file)


if __name__ == "__main__":
    directory = Path(Path.cwd(), "experiments")
    experiment_name = "my_first_experiment"

    local_tracker = LocalExperimentTracker(
        str(directory),
        experiment_name=experiment_name,
    )

    # log a single parameter
    local_tracker.log_parameter("my_param", 3)

    # log confusion matrix
    matrix = [[1, 2], [4, 5]]
    labels = ["A", "B"]
    local_tracker.log_confusion_matrix_table(matrix, labels)

    local_tracker.end()

    # view stored files
    print("experiment generated the following files:")
    experiment_path = Path(directory, experiment_name)
    for f in experiment_path.iterdir():
        print(f.as_posix())
