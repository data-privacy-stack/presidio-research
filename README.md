# Presidio-research

This package provides evaluation and data-science capabilities for 
[Presidio](https://github.com/microsoft/presidio) and PII detection models in general.

It also includes a fake data generator that creates synthetic sentences based on templates and fake PII.

## Who should use it?

- Anyone interested in **developing or evaluating PII detection models**, an existing Presidio instance or a Presidio PII recognizer.
- Anyone interested in **generating new data based on previous datasets or sentence templates** (e.g., to increase the coverage of entity values) for Named Entity Recognition models.

## Getting started


### Using notebooks
The easiest way to get started is by reviewing the notebooks. 
- [Notebook 1](notebooks/1_Generate_data.ipynb): Shows how to use the PII data generator.
- [Notebook 2](notebooks/2_PII_EDA.ipynb): Shows a simple analysis of the PII dataset.
- [Notebook 3](notebooks/3_Split_by_pattern_number.ipynb): Provides tools to split the dataset into train/test/validation sets while avoiding leakage due to the same pattern appearing in multiple folds (only applicable for synthetically generated data).
- [Notebook 4](notebooks/4_Evaluate_Presidio_Analyzer.ipynb): Shows how to use the evaluation tools to evaluate how well Presidio detects PII. Note that this is using the vanilla Presidio, and the results aren't very accurate.
- [Notebook 5](notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb): Shows how one can configure Presidio to detect PII much more accurately, and boost the f score in ~30%.

### Installation

>Note: Presidio evaluator requires Python version 3.9 or higher.

#### From PyPI

``` sh
conda create --name presidio python=3.12
conda activate presidio
pip install presidio-evaluator
python -m spacy download en_core_web_sm # for tokenization
python -m spacy download en_core_web_lg # for NER

```

#### From source

To install the package:
1. Clone the repo
2. Install all dependencies:

``` sh
# Install package+dependencies
pip install poetry
poetry install --with=dev

# Download tge spaCy pipeline used for tokenization
poetry run python -m spacy download en_core_web_sm

# To install with all additional NER dependencies (e.g. Flair, Stanza), run:
# poetry install --with='ner,dev'

# To use the default Presidio configuration, a spaCy model is required:
poetry run python -m spacy download en_core_web_lg

# Verify installation
pytest
```

Note that some dependencies (such as Flair and Stanza) are not automatically installed to reduce installation complexity.

## What's in this package?

1. **Fake data generator** for PII recognizers and NER models
2. **Data representation layer** for data generation, modeling and analysis
3. Multiple **Model/Recognizer evaluation** files (e.g. for Presidio, Spacy, Flair, Azure AI Language)
4. **Training and modeling code** for multiple models
5. Helper functions for **results analysis**

## 1. Data generation

See [Data Generator README](presidio_evaluator/data_generator/README.md) for more details.

The data generation process takes a file with templates, e.g. `My name is {{name}}`. 
Then, it creates new synthetic sentences by sampling templates and PII values. 
Furthermore, it tokenizes the data, creates tags (either IO/BIO/BILUO) and spans for the newly created samples.

- For information on data generation/augmentation, see the data generator [README](presidio_evaluator/data_generator/README.md).
- For an example for running the generation process, see [this notebook](notebooks/1_Generate_data.ipynb).
- For an understanding of the underlying fake PII data used, see this [exploratory data analysis notebook](notebooks/2_PII_EDA.ipynb).

Once data is generated, it could be split into train/test/validation sets 
while ensuring that each template only exists in one set. 
See [this notebook for more details](notebooks/3_Split_by_pattern_number.ipynb).

## 2. Data representation

In order to standardize the process, 
we use specific data objects that hold all the information needed for generating, 
analyzing, modeling and evaluating data and models. Specifically, 
see [data_objects.py](presidio_evaluator/data_objects.py).

The standardized structure, `List[InputSample]`, can be translated into different formats:
- CoNLL
  - To CoNLL:
    ```python
    from presidio_evaluator import InputSample
    dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
    conll = InputSample.create_conll_dataset(dataset)
    conll.to_csv("dataset.csv", sep="\t")
    ```

  - From CoNLL
    ```python
    from pathlib import Path
    from presidio_evaluator.dataset_formatters import CONLL2003Formatter
    # Read from a folder containing ConLL2003 files
    conll_formatter = CONLL2003Formatter(files_path=Path("data/conll2003").resolve())
    train_samples = conll_formatter.to_input_samples(fold="train")
    ```  


- spaCy v3
  ```python
  from presidio_evaluator import InputSample
  dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
  InputSample.create_spacy_dataset(dataset, output_path="dataset.spacy")
  ```

- Flair
  ```python
  from presidio_evaluator import InputSample
  dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
  flair = InputSample.create_flair_dataset(dataset)
  ```

- json
  ```python
  from presidio_evaluator import InputSample
  dataset = InputSample.read_dataset_json("data/synth_dataset_v2.json")
  InputSample.to_json(dataset, output_file="dataset_json")
  ```

## 3. PII models evaluation

The presidio-evaluator framework allows you to evaluate Presidio as a system, a NER model, or a specific PII recognizer for precision, recall, and error analysis. See [Notebook 5](notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb) for an example.

## For more information

- [Blog post on NLP approaches to data anonymization](https://towardsdatascience.com/nlp-approaches-to-data-anonymization-1fb5bde6b929)
- [How to evaluate PII Detection output with Presidio Evaluator](https://tranguyen221.medium.com/how-to-evaluate-pii-detection-output-with-presidio-evaluator-3f2684ba3091)
- [Conference talk about leveraging Presidio and utilizing NLP approaches for data anonymization](https://youtu.be/Tl773LANRwY)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Copyright notice:

Fake Name Generator identities by the [Fake Name Generator](https://www.fakenamegenerator.com/)
are licensed under a [Creative Commons Attribution-Share Alike 3.0 United States License](http://creativecommons.org/licenses/by-sa/3.0/us/).
Fake Name Generator and the Fake Name Generator logo are trademarks of Corban Works, LLC.
