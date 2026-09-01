from unittest.mock import MagicMock, patch

from faker import Faker

from presidio_evaluator.data_generator.faker_extensions import (
    HospitalProvider,
    NationalityProvider,
    OrganizationProvider,
)


def test_nationality_provider():
    faker = Faker()
    faker.add_provider(NationalityProvider)
    element = faker.nation_man()
    assert element


def test_organization_provider():
    faker = Faker()
    faker.add_provider(OrganizationProvider)
    element = faker.organization()
    assert element


def test_hospital_provider():
    faker = Faker()
    faker.add_provider(HospitalProvider)
    element = faker.hospital_name()
    assert element


@patch("presidio_evaluator.data_generator.faker_extensions.providers.requests.get")
def test_hospital_provider_sends_user_agent_header(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": {"bindings": []}}
    mock_get.return_value = mock_response

    faker = Faker()
    faker.add_provider(HospitalProvider)

    mock_get.assert_called_once()
    _, call_kwargs = mock_get.call_args
    assert "User-Agent" in call_kwargs["headers"]


@patch("presidio_evaluator.data_generator.faker_extensions.providers.requests.get")
def test_hospital_provider_falls_back_on_non_200_response(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_get.return_value = mock_response

    faker = Faker()
    provider = HospitalProvider(faker)

    assert provider.hospitals == provider.default_list
