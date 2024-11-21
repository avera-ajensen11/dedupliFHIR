"""
Below are the definition of the utils used by the dedupliFHR tool.

The most important of these data structures is the context manager for linker 
generation for Splink. 

"""
import csv
import os
import time
import uuid
from functools import wraps
from multiprocessing import Pool
from typing import Any, Callable, Optional, Protocol

import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator
from splink.internals.blocking_analysis import (
    cumulative_comparisons_to_be_scored_from_blocking_rules_data, )
from splink.internals.linker_components.clustering import LinkerClustering
from splink.internals.linker_components.evaluation import LinkerEvalution
from splink.internals.linker_components.inference import LinkerInference
from splink.internals.linker_components.misc import LinkerMisc
from splink.internals.linker_components.table_management import LinkerTableManagement
from splink.internals.linker_components.training import LinkerTraining
from splink.internals.linker_components.visualisations import LinkerVisualisations

from cli.deduplifhirLib.enums import InFormat
from cli.deduplifhirLib.normalization import (
    normalize_addr_text,
    normalize_date_text,
    normalize_name_text,
)
from cli.deduplifhirLib.settings import (
    BLOCKING_RULE_STRINGS,
    create_blocking_rules,
    create_settings,
    read_fhir_data,
    read_qrda_data,
)


class DuckDBLinker(Linker):
    """
    A specialized Linker class for DuckDBAPI.

    This class encapsulates the creation of a Linker instance using the DuckDBAPI
    and provides an intuitive interface for using DuckDB as the backend.
    """

    # Exposing these members from Linker, since they are not declared within the class explicitly, but just instantiated within __init__
    clustering: LinkerClustering
    evaluation: LinkerEvalution
    inference: LinkerInference
    misc: LinkerMisc
    table_management: LinkerTableManagement
    training: LinkerTraining
    visualisations: LinkerVisualisations

    def __init__(self, train_frame: pd.DataFrame, settings: SettingsCreator|None = None):
        """
        Initialize the DuckDBLinker.

        Args:
            train_frame: The input data frame for training the linkage model.
            settings: The settings dictionary for Splink. If not provided, 
                      it will be generated using `create_settings`.
        """
        # Use a provided settings object or generate it dynamically
        settings = settings or create_settings(train_frame)

        # Call the parent Linker constructor with DuckDBAPI
        super().__init__(train_frame, settings, db_api=DuckDBAPI())


# Define a protocol for the Linker wrapper function signature
class LinkerFunction(Protocol):

    def __call__(self, fmt: InFormat, bad_data_path: str, output_path: str, linker: Optional[DuckDBLinker] = None) -> None:
        ...


base_dir = os.path.abspath(os.path.dirname(__file__))


def check_blocking_uniques(check_df: pd.DataFrame, blocking_field: str, required_uniques: int = 5):
    """
    Function that takes in a dataframe and asserts the required blocking values
    are present for splink to use. Throws an assertion error if it can't.

    Arguments:
        check_df: Pandas Dataframe to check
        blocking_field: Column of the frame to check uniques of
        required_uniques: Unique values to require for blocking rules
    """
    uniques = getattr(check_df, blocking_field).nunique(dropna=True)
    assert uniques >= required_uniques


def parse_qrda_data(path: str, cpu_cores: int = 4, parse_function: Callable[[str], pd.DataFrame] = read_qrda_data):
    all_patient_records = [os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(path) for f in filenames if f.split(".")[-1] == "xml"]

    print(f"Reading files with {cpu_cores} cores...")
    df_list = []
    start = time.time()
    with Pool(cpu_cores) as pool:
        df_list = pool.map(parse_function, all_patient_records)

    print(f"Read qrda data in {time.time() - start} seconds")
    print("Done parsing fhir data.")

    return pd.concat(df_list, axis=0, ignore_index=True)


#Fhir stores patient data in directories of json
def parse_fhir_data(path: str, cpu_cores: int = 4, parse_function: Callable[[str], pd.DataFrame] = read_fhir_data):
    """
    This function parses all json files in a given path structure as FHIR data. It walks
    through the given path and parses each json file it finds into a pandas Dataframe.

    The process of parsing the files is done through a number of processes that each read the
    JSON and output the Dataframe in parallel. The master process then returns the result of 
    concatenating each dataframe into a full record of FHIR data.

    Arguments:
        path: Directory path to walk through to look for JSON FHIR data
        cpu_cores: Number of processes to use at once to parse the JSON FHIR data
    
    Returns:
        Dataframe containing all patient FHIR data
    """
    #Get all files in path with fhir data.
    all_patient_records = [os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(path) for f in filenames if f.split(".")[-1] == "json"]

    print(len(all_patient_records))

    #Load files concurrently via multiprocessing
    print(f"Reading files with {cpu_cores} cores...")
    df_list = []
    start = time.time()
    with Pool(cpu_cores) as pool:
        df_list = pool.map(parse_function, all_patient_records)

    print(f"Read fhir data in {time.time() - start} seconds")
    print("Done parsing fhir data.")

    return pd.concat(df_list, axis=0, ignore_index=True)


def parse_csv_dict_row_addresses(row: dict[str|Any, str|Any]) -> dict[str|Any, str|Any]:
    """
    This function parses a row of patient data and normalizes any
    address data that is found

    Arguments:
        row: The row that is being parsed taken as a dictionary
    
    Returns:
        The row object with address data normalized and returned as a dict
    """
    parsed = row

    address_keys = ["address", "city", "state", "postal_code"]

    for k, v in row.items():
        if any(match in k.lower() for match in address_keys):
            parsed[k] = normalize_addr_text(v)

    return parsed


def parse_csv_dict_row_names(row: dict[str|Any, str|Any]) -> dict[str|Any, str|Any]:
    """
    This function parses a row of patient data and normalizes any
    name data that is found

    Arguments:
        row: The row that is being parsed taken as a dictionary
    
    Returns:
        The row object with name data normalized and returned as a dict
    """
    parsed = row

    for k, v in row.items():
        if '_name' in k.lower():
            parsed[k] = normalize_name_text(v)

    return parsed


def parse_test_data(path: str, marked: bool = False) -> pd.DataFrame:
    """
    This function parses a csv file in a given path structure as patient data. It
    parses through the csv and creates a dataframe from it. 

    Arguments:
        path: Path of CSV file
    Returns:
        Dataframe containing all patient data
    """

    df_list: list[pd.DataFrame] = []
    # reading csv file
    with open(path, 'r', encoding="utf-8") as csvfile:

        for row in csv.DictReader(csvfile, skipinitialspace=True):
            #print(row[2])
            try:
                #dob = datetime.datetime.strptime(row[5], '%m/%d/%Y').strftime('%Y-%m-%d')
                patient_dict = {"unique_id": uuid.uuid4().int, "path": ["TRAINING" if marked else ""]}

                normal_row = parse_csv_dict_row_addresses(row)
                normal_row = parse_csv_dict_row_names(normal_row)
                normal_row["birth_date"] = normalize_date_text(normal_row["birth_date"])

                patient_dict.update({k.lower(): [v] for k, v in normal_row.items()})
                #print(len(row))

                #print(patient_dict)
                df_list.append(pd.DataFrame(patient_dict))
            except IndexError:
                print("could not read row")

    return pd.concat(df_list)


def use_linker(func: LinkerFunction) -> LinkerFunction:
    """
    A contextmanager that is used to obtain a linker object with which to dedupe patient 
    records with. Automatically reads in the FHIR data for the requested dataset marked 
    by a slug.

    Arguments:
        slug: ID for the dataset to dedupe
    
    Returns:
        linker: the linker object to use for deduplication. 
    """

    @wraps(func)
    def wrapper(
        fmt: InFormat,
        bad_data_path: str,
        output_path: str,
        linker: DuckDBLinker|None = None,
    ) -> None:
        data_dir = bad_data_path

        print(f"Format is {fmt.value}")
        print(f"Data dir is {data_dir}")
        print(os.getcwd())

        dir_path = os.path.dirname(os.path.realpath(__file__))

        training_df = parse_test_data(os.path.join(dir_path, 'tests', 'test_data.csv'), marked=True)

        match fmt:
            case InFormat.FHIR:
                train_frame = pd.concat([parse_fhir_data(data_dir), training_df], axis=0, ignore_index=True)
            case InFormat.QRDA:
                train_frame = pd.concat([parse_qrda_data(data_dir), training_df], axis=0, ignore_index=True)
            case InFormat.CSV:
                train_frame = pd.concat([parse_test_data(data_dir), training_df], axis=0, ignore_index=True)
            case InFormat.TEST:
                train_frame = training_df
            case InFormat.DF:
                # NOTE: Removed the DF flag for now, since an exception will be thrown in "check_blocking_uniques" when passing it a string into the first param
                #train_frame = data_dir
                raise NotImplementedError("Removing the DF flag for now")

        #check blocking values
        for rule in BLOCKING_RULE_STRINGS:
            try:
                if isinstance(rule, list):
                    for sub_rule in rule:
                        check_blocking_uniques(train_frame, sub_rule)
                else:
                    check_blocking_uniques(train_frame, rule)
            except AssertionError as e:
                print(f"Could not assert the proper number of unique records for rule {rule}")
                raise e

        #lnkr = DuckDBLinker(train_frame, SPLINK_LINKER_SETTINGS_PATIENT_DEDUPE)

        preprocessing_metadata = cumulative_comparisons_to_be_scored_from_blocking_rules_data(table_or_tables=[train_frame],
                                                                                              blocking_rules=create_blocking_rules(),
                                                                                              link_type="dedupe_only",
                                                                                              db_api=DuckDBAPI())

        print("Stats for nerds:")
        print(preprocessing_metadata.to_string())

        lnkr = DuckDBLinker(train_frame, create_settings(train_frame))
        lnkr.training.estimate_u_using_random_sampling(max_pairs=5e6)

        return func(fmt, bad_data_path, output_path, lnkr)

    return wrapper
