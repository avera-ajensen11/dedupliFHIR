"""
Module to define cli for ecqm-deduplifhir library.
"""
import os
import tempfile

import click
import pandas as pd
from splink import block_on
from splink.internals.duckdb.dataframe import DuckDBDataFrame

from cli.deduplifhirLib.enums import EnumType, FileExtension, InFormat
from cli.deduplifhirLib.utils import DuckDBLinker, use_linker

CACHE_DIR = tempfile.gettempdir()


class DeduplicationCLI:
    """
    CLI handler for the ecqm-deduplifhir library.
    """

    def __init__(self):
        pass

    @staticmethod
    @click.command()
    @click.option('--fmt', default="FHIR", type=EnumType(InFormat), help='Format of patient data')
    @click.argument('bad_data_path')
    @click.argument('output_path')
    @use_linker
    def dedupe_data(fmt: InFormat, bad_data_path: str, output_path: str, linker: DuckDBLinker|None = None) -> None:
        """Program to dedupe patient data in many formats namely FHIR and QRDA"""

        if not linker:
            raise RuntimeError("No Linker instance found after parsing files")

        print(os.getcwd())
        #linker is created by use_linker decorator
        blocking_rule_for_training = block_on("ssn")
        linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

        blocking_rule_for_training = block_on("birth_date")  # block on year
        linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

        blocking_rule_for_training = block_on("street_address0", "postal_code0")
        linker.training.estimate_parameters_using_expectation_maximisation(blocking_rule_for_training)

        pairwise_predictions = linker.inference.predict()

        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(pairwise_predictions, 0.95)
        if not isinstance(clusters, DuckDBDataFrame):
            raise TypeError(f"Expected 'clusters' to be of type 'DuckDBDataFrame', "
                            f"but got '{type(clusters).__name__}' instead. Ensure that the Linker "
                            f"is properly configured to use DuckDBAPI and that the clustering method "
                            f"is returning the correct type.")

        deduped_record_mapping = clusters.as_pandas_dataframe()

        if fmt != InFormat.TEST:
            deduped_record_mapping = deduped_record_mapping.drop(deduped_record_mapping[deduped_record_mapping.path == "TRAINING"].index)  # type: ignore

        #Calculate only uniques
        unique_records = deduped_record_mapping.drop_duplicates(subset=['cluster_id'])
        #cache results
        deduped_record_mapping.to_csv(os.path.join(CACHE_DIR, "dedupe-cache.csv"))
        unique_records.to_csv(os.path.join(CACHE_DIR, "unique-records-cache.csv"))

        _, extension = os.path.splitext(output_path)
        extension = extension.lower()
        file_type = FileExtension.get_extension_type(extension)

        match file_type:
            case FileExtension.EXCEL:
                deduped_record_mapping.to_excel(output_path)
            case FileExtension.CSV:
                deduped_record_mapping.to_csv(output_path)
            case FileExtension.JSON:
                deduped_record_mapping.to_json(output_path)
            case FileExtension.HTML:
                deduped_record_mapping.to_html(output_path)
            case FileExtension.XML:
                deduped_record_mapping.to_xml(output_path)
            case FileExtension.LATEX:
                deduped_record_mapping.to_latex(output_path)
            case FileExtension.FEATHER:
                deduped_record_mapping.to_feather(output_path)
            case _:
                raise ValueError(f"File format '{extension}' not supported!")

    @staticmethod
    @click.command()
    def clear_cache() -> None:
        """Clear cache of dedupliFHIED patient data"""
        os.remove(os.path.join(CACHE_DIR, "unique-records-cache.csv"))
        os.remove(os.path.join(CACHE_DIR, "dedupe-cache.csv"))
        print("Cache cleared.")

    @staticmethod
    @click.command()
    def status() -> None:
        """Output status of cache as well as result and stats of last run"""
        try:
            #Print amount of duplicates found in cache if found
            cache_df = pd.read_csv(os.path.join(CACHE_DIR, "dedupe-cache.csv"))
        except FileNotFoundError:
            print("Cache is empty")
            return

        print("Cache contains data")
        number_patients = cache_df["cluster_id"].nunique(dropna=True)

        number_total = cache_df["unique_id"].nunique(dropna=True)

        print(f"There were {number_total - number_patients} duplicates found last run.")

        print(f"There are {number_patients} unique patients among " + f"{number_total} records among the data.")

    def register_commands(self):
        """
        Automatically register all commands to the CLI group.
        """
        cli = click.Group(help="CLI for ecqm-deduplifhir\n\nAdd the '--help' option to any Command below to view detailed help.")

        # Iterate through all class attributes
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            # Check if the attribute is a click command
            if callable(attribute) and hasattr(attribute, "callback") and isinstance(attribute, click.Command):
                cli.add_command(attribute)

        return cli


if __name__ == "__main__":
    cli_handler = DeduplicationCLI()
    cli = cli_handler.register_commands()
    cli()
