"""
Below is the definition of the settings for the splink linker object

These settings control the way that spink trains its model in order to
find duplicates.

The blocking rules determine the preliminary records to match to each other in 
order to train the model, for this to work the data needs a couple records with the
same first and last name in the input in order to find confirmed matches 

The comparisons list defines the way the model will compare the data in order to find
the probability that records are duplicates of each other. 
"""
import json
import os
import uuid
import xml.etree.cElementTree as ET
from typing import Any, Generator, List

import pandas as pd
from fhirclient.models.address import Address
from fhirclient.models.patient import Patient
from splink import SettingsCreator, block_on
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.comparison_creator import ComparisonCreator
from splink.internals.comparison_library import (
    DateOfBirthComparison,
    ExactMatch,
    NameComparison,
    PostcodeComparison,
)

from cli.deduplifhirLib.normalization import (
    normalize_addr_text,
    normalize_date_text,
    normalize_name_text,
)
from cli.deduplifhirLib.types import load_splink_settings

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'splink_settings.json')

try:
    settings = load_splink_settings(file_path)
except Exception as e:
    raise e

# apply blocking function to translate into sql rules
BLOCKING_RULE_STRINGS: list[str|list[str]] = settings.blocking_rules_to_generate_predictions


def get_additional_comparison_rules(parsed_data_df: pd.DataFrame) -> Generator[ComparisonCreator, Any, None]:
    """
    This function generates appropriate comparison rules based on pandas column names

    Arguments:
        parsed_data_df: The dataframe that was parsed from the user that we want to
        find duplicates in
    
    Returns:
        A generator collection object full of splink comparison objects
    """

    parsed_data_columns = parsed_data_df.columns

    for col in parsed_data_columns:
        if 'street_address' in col:
            yield ExactMatch(col)
        elif 'postal_code' in col:
            yield PostcodeComparison(col)


def create_blocking_rules() -> List[BlockingRuleCreator|dict[str, Any]]:
    blocking_rules: List[BlockingRuleCreator|dict[str, Any]] = []
    for rule in BLOCKING_RULE_STRINGS:
        if isinstance(rule, list):
            blocking_rules.append(block_on(*rule))
        else:
            blocking_rules.append(block_on(rule))

    return blocking_rules


def create_settings(parsed_data_df: pd.DataFrame):
    """
    This function generates a Splink SettingsCreator object based on the parsed
    input data's columns and the blocking settings in splink_settings.json

    Arguments:
        parsed_data_df: The dataframe that was parsed from the user that we want to
        find duplicates in
    
    Returns:
        A splink SettingsCreator object to be used with a splink linker object
    """

    blocking_rules = create_blocking_rules()

    comparison_rules: List[ComparisonCreator|dict[str, Any]] = [item for item in get_additional_comparison_rules(parsed_data_df)]
    comparison_rules.extend([
        ExactMatch("phone").configure(term_frequency_adjustments=True),
        NameComparison("given_name").configure(term_frequency_adjustments=True),
        NameComparison("family_name").configure(term_frequency_adjustments=True),
        DateOfBirthComparison("birth_date", input_is_string=True)
    ])

    return SettingsCreator(link_type=settings.link_type,
                           blocking_rules_to_generate_predictions=blocking_rules,
                           comparisons=comparison_rules,
                           max_iterations=settings.max_iterations,
                           em_convergence=settings.em_convergence)


def parse_fhir_dates(fhir_json_obj: Patient):
    """
    A generator function that parses the address portion of a FHIR file
    into a dictionary object that can be added to the overall patient record

    Arguments:
        fhir_json_obj: The object that has been parsed from the FHIR data
    
    Returns:
        A generator containing dictionaries of address data.
    """
    addresses: list[Address] = fhir_json_obj.address

    for n, addr in enumerate(sorted(addresses)):
        yield {
            f"street_address{n}": [normalize_addr_text(''.join(addr.line))],
            f"city{n}": [normalize_addr_text(addr.city)],
            f"state{n}": [normalize_addr_text(addr.state)],
            f"postal_code{n}": [normalize_addr_text(addr.postalCode)]
        }


# NOTE: The only reason this function is defined outside utils.py is because of a known bug with
# python multiprocessing: https://github.com/python/cpython/issues/69240
def read_fhir_data(patient_record_path: str) -> pd.DataFrame:
    """
    This function reads fhir data for a single patient and expresses the record as a dataframe
    with one record.

    Arguments:
        patient_record_path: The path to a single FHIR patient record, a JSON file.
    
    Returns:
        A dataframe holding FHIR data for a single patient.
    """
    try:
        with open(patient_record_path, "r", encoding="utf-8") as fdesc:
            patient_json_record_temp = json.load(fdesc)

        patient = Patient(patient_json_record_temp, strict=False)
    except Exception as e:
        print(e)
        print(f"File: {patient_record_path}")
        raise e

    patient_dict = {
        "unique_id": uuid.uuid4().int,
        "family_name": [normalize_name_text(patient.name[0].family)],
        "given_name": [normalize_name_text(patient.name[0].given[0])],
        "gender": [patient.gender],
        "birth_date": normalize_date_text(patient.birthDate.as_json()),
        "phone": [patient.telecom.value],
        # See CodeSystem codes here: https://terminology.hl7.org/6.1.0/CodeSystem-v2-0203.html
        "ssn": [next((id.value for id in patient.identifier if next((coding for coding in id.type.coding if coding.code == "SS"), None)), None)],
        "path": patient_record_path
    }

    try:
        patient_dict["middle_name"] = [normalize_name_text(patient.name[0].given[1])]
    except IndexError:
        patient_dict["middle_name"] = [""]
        print("no middle name found!")

    for date in parse_fhir_dates(patient):
        patient_dict.update(date)

    return pd.DataFrame(patient_dict)


def read_qrda_data(patient_record_path: str) -> pd.DataFrame:
    """
    This function reads QRDA XML data for a single patient and expresses the record as a dataframe
    with one record.

    Arguments:
        patient_record_path: The path to a single QRDA patient record, an XML file.
    
    Returns:
        A dataframe holding QRDA data for a single patient.
    """

    def safe_find(element: ET.Element, xpath: str, namespaces: dict[str, str]) -> ET.Element|None:
        """
        Safely find an XML element using an XPath.
        Returns None if the element or any part of the call chain is missing.
        """
        try:
            found = element.find(xpath, namespaces)
            return found
        except AttributeError:
            return None

    try:
        tree = ET.parse(patient_record_path)
        root = tree.getroot()
    except Exception as e:
        print(e)
        print(f"File: {patient_record_path}")
        raise e

    namespaces = {"qrda": "urn:hl7-org:v3"}  # Adjust namespace as per QRDA standard

    # Use safe_find to avoid AttributeError
    patient_id = safe_find(root, ".//qrda:id", namespaces)
    family_name = safe_find(root, ".//qrda:family", namespaces)
    given_name = safe_find(root, ".//qrda:given", namespaces)
    gender = safe_find(root, ".//qrda:administrativeGenderCode", namespaces)
    birth_date = safe_find(root, ".//qrda:birthTime", namespaces)
    phone = safe_find(root, ".//qrda:telecom", namespaces)
    ssn = safe_find(root, ".//qrda:id[@root='2.16.840.1.113883.4.1']", namespaces)
    middle_name_element = safe_find(root, ".//qrda:middle", namespaces)

    patient_dict = {
        "unique_id": uuid.uuid4().int,
        "family_name": [normalize_name_text(family_name.text if family_name is not None else "")],
        "given_name": [normalize_name_text(given_name.text if given_name is not None else "")],
        "middle_name": [normalize_name_text(middle_name_element.text if middle_name_element is not None else "")],
        "gender": [gender.get("code") if gender is not None else ""],
        "birth_date": normalize_date_text(birth_date.get("value") if birth_date is not None else ""),
        "phone": [phone.get("value") if phone is not None else ""],
        "ssn": [ssn.get("extension") if ssn is not None else ""],
        "path": patient_record_path
    }

    return pd.DataFrame(patient_dict)
