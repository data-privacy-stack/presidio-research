import random
import re

import pandas as pd
from pandas import DataFrame

from presidio_evaluator.data_generator import raw_data_dir


def _camel_to_snake(name):
    # Borrowed from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _full_name(row) -> str:
    if random.random() > 0.2:
        return f"{row.first_name} {row.last_name}"
    else:
        space_after_initials = " " if random.random() > 0.5 else ". "
        return f"{row.first_name} {row.middle_initial}{space_after_initials}{row.last_name}"


def _name_gendered(row):
    first_name_female, prefix_female, last_name_female = (
        (row.first_name, row.prefix, row.last_name)
        if row.gender == "female"
        else ("", "", "")
    )
    first_name_male, prefix_male, last_name_male = (
        (row.first_name, row.prefix, row.last_name)
        if row.gender == "male"
        else ("", "", "")
    )
    return (
        first_name_female,
        first_name_male,
        prefix_female,
        prefix_male,
        last_name_female,
        last_name_male,
    )


def load_fake_person_df() -> DataFrame:
    """
    :return: A DataFrame loaded with data from FakeNameGenerator.com, and cleaned to match faker conventions
    """
    fake_person_data_path = raw_data_dir / "FakeNameGenerator.com_3000.csv"
    fake_person_df = pd.read_csv(fake_person_data_path)
    fake_person_df.columns = [_camel_to_snake(col) for col in fake_person_df.columns]
    # Update some column names to fit Faker
    fake_person_df.rename(
        columns={"country": "country_code", "state": "state_abbr"},
        inplace=True,
    )
    fake_person_df.rename(
        columns={
            "country_full": "country",
            "name_set": "nationality",
            "street_address": "street_name",
            "state_full": "state",
            "given_name": "first_name",
            "surname": "last_name",
            "title": "prefix",
            "email_address": "email",
            "telephone_number": "phone_number",
            "telephone_country_code": "country_calling_code",
            "birthday": "date_of_birth",
            "cc_number": "credit_card_number",
            "cc_type": "credit_card_provider",
            "cc_expires": "credit_card_expire",
            "occupation": "job",
            "domain": "domain_name",
            "username": "user_name",
            "zip_code": "zipcode",
        },
        inplace=True,
    )
    fake_person_df["person"] = fake_person_df.apply(_full_name, axis=1)
    fake_person_df["name"] = fake_person_df["person"]
    genderized = fake_person_df.apply(
        lambda x: pd.Series(
            _name_gendered(x),
            index=[
                "first_name_female",
                "first_name_male",
                "prefix_female",
                "prefix_male",
                "last_name_female",
                "last_name_male",
            ],
        ),
        axis=1,
        result_type="expand",
    )
    # Remove credit card data, rely on Faker's as it is more realistic
    del fake_person_df["credit_card_number"]
    fake_person_df = pd.concat([fake_person_df, genderized], axis="columns")
    return fake_person_df
