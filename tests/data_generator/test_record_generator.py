import pytest
from faker import Faker
from faker.providers import BaseProvider

from presidio_evaluator.data_generator.faker_extensions import RecordGenerator


@pytest.fixture(scope="session")
def foo_provider():
    class FooProvider(BaseProvider):
        def foo(self):
            return "bar"

    return FooProvider


@pytest.mark.parametrize(
    "template, expected",
    [
        (
            "My name is {{name}} and my email is {{email}}",
            "My name is 1 and my email is a",
        ),
        ("My name is {{name}}", "My name is 1"),
        ("My name is {{name}} {{foo}}.", "My name is 1 bar."),
        ("Foo {{foo}}", "Foo bar"),
        ("pytesting stuff", "pytesting stuff"),
    ],
)
def test_record_generator(foo_provider, template, expected):
    records = [{"name": "1", "email": "a"}]

    generator = RecordGenerator(records=records)

    faker = Faker(generator=generator)
    faker.add_provider(foo_provider)

    res = faker.parse(template, add_spans=True)
    assert res.full_text == expected


@pytest.mark.parametrize("add_spans", [(True, False)])
def test_multiple_generations(foo_provider, add_spans):
    template1 = "My name is {{name}}, {{email}}"
    template2 = "My {{name}} {{foo}}, {{email}}"
    template3 = "My name is {{name}} or {{foo}}, {{email}}"

    records = [
        {"name": "a_name", "email": "a@a"},
        {"name": "b_name", "email": "b@b"},
        {"name": "c_name", "email": "c@c"},
    ]

    generator = RecordGenerator(records=records)

    Faker.seed(42)
    faker = Faker(generator=generator)
    faker.add_provider(foo_provider)

    res1 = faker.parse(template1, add_spans=add_spans)  # my name is c
    res2 = faker.parse(template2, add_spans=add_spans)  # my name is bar
    res3 = faker.parse(template3, add_spans=add_spans)
    if add_spans:
        responses = [res1.full_text, res2.full_text, res3.full_text]
    else:
        responses = [res1, res2, res3]

    for response in responses:
        for record in records:
            if record["name"] in response:
                assert record["email"] in response


@pytest.mark.parametrize("add_spans", [(True, False)])
def test_template_contains_multiple_of_same_entity(add_spans):
    template = "{{name}} or {{name}}"

    records = [
        {"name": "a_name", "email": "a@a"},
    ]
    generator = RecordGenerator(records=records)
    Faker.seed(42)

    faker = Faker(generator=generator)

    res = faker.parse(template, add_spans=add_spans)
    if add_spans:
        assert res.full_text.count("a_name") == 1
    else:
        assert res.count("a_name") == 1
