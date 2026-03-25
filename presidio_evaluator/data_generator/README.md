# Data Generation

The `PresidioSentenceFaker` generates sentences from templates (e.g. `my name is {{person}}`) where the placeholders
are replaced with fake PII entities, along with metadata about the spans (the start and end of each entity) for model training and evaluation.

## Scenarios

There are two main scenarios for using the `PresidioSentenceFaker`:

1. Create a fake dataset for evaluation or training purposes, given a list of predefined templates 
(uses [this file](raw_data/templates.txt) by default)
2. Augment an existing labeled dataset with additional fake values.

In both scenarios the process is similar. In scenario 2, the existing dataset is first translated into templates, 
and then scenario 1 is applied.

## Process

This generator heavily relies on the [Faker package](https://www.github.com/joke2k/faker) with a few differences:

1. `PresidioSentenceFaker` returns not only fake text, but also the spans in which fake entities appear in the text.
2. `Faker` samples each value independently.
In many cases, we would want to keep the semantic dependency between two values. 
For example, for the template `My name is {{name}} and my email is {{email}}`, 
we would prefer a result which has the name within the email address, 
such as `My name is Mike and my email is mike1243@gmail.com`. 
For this functionality, a new `RecordGenerator` (based on Faker's `Generator` class) is implemented. 
It accepts a dictionary / pandas DataFrame, and favors returning objects from the same record (if possible).

## Example

For a full example, see the [Generate Data Notebook](../../notebooks/1_Generate_data.ipynb).

`PresidioSentenceFaker` provides a high-level interface for using the full power of the `presidio_evaluator`
package. Its results use the presidio PII entities, not the `Faker` entities.
It is loaded by default with template strings, and the additional Presidio Entity Providers.

```python
from presidio_evaluator.data_generator import PresidioSentenceFaker

record_generator = PresidioSentenceFaker(locale='en', lower_case_ratio=0.05)
fake_records = record_generator.generate_new_fake_sentences(1500)

# Print the spans of the first sample
print(fake_records[0].fake)
print(fake_records[0].spans)
```

The process at a high level is the following:

1. Translate a NER dataset (e.g. CONLL or OntoNotes) into a list of
templates: `My name is John` -> `My name is [PERSON]`
2. Construct a `PresidioSentenceFaker` instance by:
   - Choosing your appropriate locale, e.g. `en_US`
   - Choosing the lower case ratio
   - Passing in your list of templates (or default to those provided)
     - Optionally extend with provided templates accessible via `from presidio_evaluator.data_generator import presidio_templates_file_path`
   - Passing in any custom entity providers (or default to those provided)
     - Optionally extend with inbuilt presidio entity providers accessible via `from presidio_evaluator.data_generator import presidio_additional_entity_providers`
     - Adding a mapping from the output provider entity type to a Presidio recognized entity type where appropriate
       - e.g. For a `TownProvider` which outputs entity type of `town`, execute `PresidioSentenceFaker.ENTITY_TYPE_MAPPING['town'] = 'GPE'`)
   - Passing in a DataFrame representing your underlying PII records (or default to those provided)
     - Optionally extend with inbuilt presidio entity providers accessible via `from presidio_evaluator.data_generator.faker_extensions.datasets import load_fake_person_df`
   - Adding any additional aliases required by your dataset by adding to `PresidioSentenceFaker.PROVIDER_ALIASES`
     - e.g. if the entity providers support "name" but your dataset templates contain "person", you can add this alias
     with `PresidioSentenceFaker.PROVIDER_ALIASES['name'] = 'person'`)
3. Generate sentences
4. Split the generated dataset into train/test/validation while making sure
that samples from the same template would only appear in one set
5. Adapt datasets for the various models (Spacy, Flair, CRF, sklearn)
6. Train models
7. Evaluate using one of the [evaluation notebooks](../../notebooks/models)

Notes:

- For steps 5, 6, 7 see the main [README](../../README.md).


*Copyright notice:*

Fake Name Generator identities by the Fake Name Generator are licensed under a
Creative Commons Attribution-Share Alike 3.0 United States License.
Fake Name Generator and the Fake Name Generator logo
are trademarks of Corban Works, LLC.
