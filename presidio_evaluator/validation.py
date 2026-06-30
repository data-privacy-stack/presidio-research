import json
import random
from collections import defaultdict

import numpy as np

from presidio_evaluator import InputSample


def split_dataset(dataset: list[InputSample], ratios):
    """
    Splits a provided dataset into n groups, by the template_id attribute
    :param dataset: List of InputSamples to be splitted
    :param ratios:  list of percentages. The len of the list would be the len of the splits returned,
    e.g. [0.7,0.2,0.1] for train, test, validation
    """
    splits = []
    remaining_dataset = dataset
    remaining_ratio = 1.0

    if sum(ratios) > 1 or sum(ratios) < 0.999:
        raise ValueError("Ratios should sum to 1 and be in (0,1]")

    for ratio in ratios:
        if 1 >= ratio > 0:
            first_templates, second_templates = split_by_template(
                remaining_dataset,
                ratio / remaining_ratio,
            )
            first_split = get_samples_by_pattern(remaining_dataset, first_templates)
            second_split = get_samples_by_pattern(remaining_dataset, second_templates)
            splits.append(first_split)
            remaining_dataset = second_split
            remaining_ratio -= ratio
        else:
            raise ValueError("Ratio needs to be in (0,1]")

    return tuple(splits)


def group_by_template(dataset: list[InputSample]) -> dict[str, list[InputSample]]:
    """
    Creates a dict of key = template ID and value = List[InputSamples] for this template id
    """
    samples_pattern_tup = [(sample.template_id, sample) for sample in dataset]

    group_by_template = defaultdict(list)
    for sample in samples_pattern_tup:
        group_by_template[sample[0]].append(sample[1])

    return group_by_template


def split_by_template(input_samples: list[InputSample], train_pct: float = 0.7):
    """
    Splits a daset of type List[InputSample] into a tuple of train template IDs and test template IDs
    """
    samples_grpd = group_by_template(input_samples)

    templates = np.array(list(samples_grpd.keys()))
    train_ind = set(
        random.sample(range(len(templates)), round(train_pct * len(templates))),
    )

    test_ind = set(range(len(templates))) - train_ind

    return templates[list(train_ind)], templates[list(test_ind)]


def get_samples_by_pattern(input_samples, patterns_list):
    samples_grpd = group_by_template(input_samples)
    dataset = []
    for pattern in patterns_list:
        dataset.extend(samples_grpd[pattern])
    random.shuffle(dataset)

    return dataset


def save_to_json(samples, output_file) -> None:
    examples_dict = [example.to_dict() for example in samples]

    with open(f"{output_file}", "w+", encoding="utf-8") as f:
        json.dump(examples_dict, f, ensure_ascii=False, indent=4)
