import random
import warnings
from collections import OrderedDict
from functools import reduce
from pathlib import Path

import pandas as pd
import requests
import yaml
from faker.providers import BaseProvider
from faker.providers.address.en_US import Provider as AddressProvider
from faker.providers.phone_number.en_US import Provider as PhoneNumberProvider

from presidio_evaluator.data_generator import raw_data_dir


class NationalityProvider(BaseProvider):
    def __init__(self, generator, nationality_file: str | Path = None) -> None:
        super().__init__(generator=generator)
        if not nationality_file:
            nationality_file = (raw_data_dir / "nationalities.csv").resolve()

        self.nationality_file = nationality_file
        self.nationalities = self.load_nationalities()

    def load_nationalities(self):
        return pd.read_csv(self.nationality_file)

    def country(self) -> None:
        self.random_element(self.nationalities["country"].tolist())

    def nationality(self):
        return self.random_element(self.nationalities["nationality"].tolist())

    def nation_man(self):
        return self.random_element(self.nationalities["man"].tolist())

    def nation_woman(self):
        return self.random_element(self.nationalities["woman"].tolist())

    def nation_plural(self):
        return self.random_element(self.nationalities["plural"].tolist())


class OrganizationProvider(BaseProvider):
    def __init__(
        self,
        generator,
        organizations_file: str | Path = None,
    ) -> None:
        super().__init__(generator=generator)
        if not organizations_file:
            # company names assembled from stock exchange listings (aex, bse, cnq, ger, lse, nasdaq, nse, nyse, par, tyo),
            # US government websites like https://www.sec.gov/rules/other/4-460list.htm, and other sources
            organizations_file = Path(
                Path(__file__).parent.parent,
                "raw_data",
                "companies_and_organizations.csv",
            ).resolve()
        self.organizations_file = organizations_file
        self.organizations = self.load_organizations()

    def load_organizations(self):
        return pd.read_csv(self.organizations_file, delimiter="\t")

    def organization(self):
        return self.random_element(self.organizations["organization"].tolist())

    def company(self):
        return self.organization()


class UsDriverLicenseProvider(BaseProvider):
    def __init__(self, generator) -> None:
        super().__init__(generator=generator)
        us_driver_license_file = Path(
            Path(__file__).parent.parent,
            "raw_data",
            "us_driver_license_format.yaml",
        ).resolve()
        formats = yaml.safe_load(open(us_driver_license_file))
        self.formats = formats["en"]["faker"]["driving_license"]["usa"]

    def us_driver_license(self) -> str:
        # US driver's licenses patterns vary by state. Here we sample a random state and format
        us_state = random.choice(list(self.formats))
        us_state_format = random.choice(self.formats[us_state])
        return self.bothify(text=us_state_format)


class ReligionProvider(BaseProvider):
    def __init__(
        self,
        generator,
        religions_file: str | Path = None,
    ) -> None:
        super().__init__(generator=generator)
        if not religions_file:
            religions_file = Path(
                Path(__file__).parent.parent,
                "raw_data",
                "religions.csv",
            ).resolve()
        self.religions_file = religions_file
        self.religions = self.load_religions()

    def load_religions(self):
        return pd.read_csv(self.religions_file, delimiter="\t")

    def religion(self) -> str:
        """Return a random (major) religion."""
        return self.random_element(self.religions["Religions"].tolist())


class IpAddressProvider(BaseProvider):
    """Generating both v4 and v6 IP addresses."""

    def ip_address(self):
        if random.random() < 0.8:
            return self.generator.ipv4()
        else:
            return self.generator.ipv6()


class AgeProvider(BaseProvider):
    formats = OrderedDict(
        [
            ("%#", 0.8),
            ("%", 0.1),
            ("1.%", 0.02),
            ("2.%", 0.02),
            ("100", 0.02),
            ("101", 0.01),
            ("104", 0.01),
            ("0.%", 0.02),
        ],
    )

    def age(self):
        return self.numerify(
            self.random_elements(elements=self.formats, length=1, use_weighting=True)[
                0
            ],
        )


class AddressProviderNew(AddressProvider):
    """
    Extending the Faker AddressProvider with additional templates
    """

    address_formats = OrderedDict(
        (
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{state}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{state_abbr}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}} {{secondary_address}} {{city}} {{country}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}}",
                5.0,
            ),
            (
                "{{building_number}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}} {{postcode}}",
                5.0,
            ),
            (
                "{{street_name}} {{street_name}}\n {{secondary_address}}\n {{city}}\n {{country}} {{postcode}}",
                5.0,
            ),
            ("the corner of {{street_name}} and {{street_name}}", 3.0),
            ("{{first_name}} and {{street_name}}", 3.0),
            ("{{street_address}}, {{city}}, {{country}}", 5.0),
            (
                "{{street_address}} {{secondary_address}}, {{city}}, {{country}} {{postcode}}",
                5.0,
            ),
            ("{{street_address}}\n{{city}}, {{state_abbr}} {{postcode}}", 25.0),
            ("{{street_address}}\n{{city}}\n, {{state_abbr}}\n {{postcode}}", 25.0),
            (
                "{{street_address}}\n{{city}}\n, {{state_abbr}}\n {{country}} {{postcode}}",
                25.0,
            ),
            #  military address formatting.
            ("{{military_apo}}\nAPO {{military_state}} {{postcode}}", 1.0),
            (
                "{{military_ship}} {{last_name}}\nFPO {{military_state}} {{postcode}}",
                1.0,
            ),
            ("{{military_dpo}}\nDPO {{military_state}} {{postcode}}", 1.0),
        ),
    )


class PhoneNumberProviderNew(PhoneNumberProvider):
    """
    Similar to the default PhoneNumberProvider, with different formats
    """

    formats = (
        # US
        "##########",
        "##########",
        "###-###-####",
        "###-###-####",
        "###-#######",
        # UK
        "07700 ### ###",
        "07700 ######",
        "07700######",
        "(07700) ### ###",
        "(07700) ######",
        "(07700)######",
        "+447700 ### ###",
        "+447700 ######",
        "+447700######",
        # India
        "+91##########",
        "0##########",
        "##########",
        # Switzerland
        "+41 2# ### ## ##",
        "+41 3# ### ## ##",
        "+41 4# ### ## ##",
        "+41 5# ### ## ##",
        "+41 6# ### ## ##",
        "+41 7# ### ## ##",
        "+41 8# ### ## ##",
        "+41 9# ### ## ##",
        "+41 (0)2# ### ## ##",
        "+41 (0)3# ### ## ##",
        "+41 (0)4# ### ## ##",
        "+41 (0)5# ### ## ##",
        "+41 (0)6# ### ## ##",
        "+41 (0)7# ### ## ##",
        "+41 (0)8# ### ## ##",
        "+41 (0)9# ### ## ##",
        "+46 (0)8 ### ### ##",
        "+46 (0)## ## ## ##",
        "+46 (0)### ### ##",
        # Optional 10-digit local phone number format
        "(###)###-####",
        "(###)###-####",
        "(###)###-####",
        "(###)###-####",
        # Non-standard 10-digit phone number format
        "###.###.####",
        "###.###.####",
        # Standard 10-digit phone number format with extensions
        "###-###-####x###",
        "###-###-####x####",
        # Optional 10-digit local phone number format with extensions
        "(###)###-####x###",
        "(###)###-####x####",
        # Non-standard 10-digit phone number format with extensions
        "###.###.####x###",
        "###.###.####x####",
        # Standard 11-digit phone number format
        "+1-###-###-####",
        "001-###-###-####",
        # Standard 11-digit phone number format with extensions
        "+1-###-###-####x###",
    )


class HospitalProvider(BaseProvider):
    def __init__(self, generator, hospital_file: str = None) -> None:
        """Load hospital data from file or wiki.

        :param hospital_file: Path to static file containing hospital names
        """

        super().__init__(generator=generator)

        self.default_list = [
            "Apollo Hospital",
            "St. Peter",
            "Mount Sinai",
            "Providence",
        ]
        self.hospitals = self.load_hospitals(hospital_file)

    def load_hospitals(self, hospital_file: str) -> list[str]:
        """Loads a list of hospital names based in the US.
        If a static file with hospital names is provided,
        the hospital names should be under a column named "name".
        If no file is provided, the information will be retrieved from WikiData.

        :param hospital_file: Path to static file containing hospital names
        """

        if hospital_file:
            hospitals = pd.read_csv(hospital_file)
            if "name" not in self.hospitals:
                print(
                    "Unable to retrieve hospital names, "
                    "file is missing column named 'name'",
                )
                return self.default_list
            return hospitals["name"].to_list()
        else:
            return self.load_wiki_hospitals()

    def hospital_name(self):
        return self.random_element(self.hospitals)

    def load_wiki_hospitals(
        self,
    ):
        """Executes a query on WikiData and extracts a list of US based hospitals"""
        url = "https://query.wikidata.org/sparql"
        query = """
        SELECT DISTINCT ?label_en
        WHERE 
        { ?item wdt:P31/wdt:P279* wd:Q16917; wdt:P17 wd:Q30
        OPTIONAL { ?item p:P31/ps:P31 wd:Q64578911 . BIND(wd:Q64578911 as ?status1) } BIND(COALESCE(?status1,wd:Q64624840) as ?status)
        OPTIONAL { ?item wdt:P131/wdt:P131* ?ac . ?ac wdt:P5087 [] }
        optional { ?item rdfs:label ?label_en FILTER((LANG(?label_en)) = "en") }   
        }

        """
        try:
            r = requests.get(url, params={"format": "json", "query": query}, timeout=10)
            if r.status_code != 200:
                print("Unable to read hospitals from WikiData, returning an empty list")
                return self.default_list
            data = r.json()
            bindings = data["results"].get("bindings", [])
            hospitals = [self.deep_get(x, ["label_en", "value"]) for x in bindings]
            hospitals = [x for x in hospitals if "no key" not in x]
            return hospitals
        except OSError:
            warnings.warn(
                "Can't download hospitals data. Returning default list", stacklevel=2
            )
            return self.default_list

    def deep_get(self, dictionary: dict, keys: list[str]):
        """Retrieve values from a nested dictionary for specific nested keys
        > example:
        > d = {"key_a":1, "key_b":{"key_c":2}}
        > deep_get(d, ["key_b","key_c"])
        > ... 2

        > deep_get(d, ["key_z"])
        > ... "no key key_z"
        :param dictionary: Nested dictionary to search for keys
        :type dictionary: dict
        :param keys: list of keys, each value should represent the next level of nesting
        :type keys: List
        :return: The value of the nested keys
        """

        return reduce(
            lambda dictionary, key: (
                dictionary.get(key, f"no key {key}")
                if isinstance(dictionary, dict)
                else f"no key {key}"
            ),
            keys,
            dictionary,
        )
