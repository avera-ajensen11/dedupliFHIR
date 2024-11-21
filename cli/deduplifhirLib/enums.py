from enum import Enum, unique

import click


class EnumType(click.Choice):
    """This class allows us to use Enums within 'click's option type wrapper

    For example, the below allows us to state that we want the 'InFormat' type to define the valid inputs for '--fmt'

    @click.option('--fmt', default="FHIR", type=EnumType(InFormat), help='Format of patient data')


    To follow the progress of the enhancement to have 'click' use Enums natively within the type param, see the below thread:
    https://github.com/pallets/click/issues/605
    """

    def __init__(self, enum: type[Enum], case_sensitive=False):
        self.__enum = enum
        super().__init__(choices=[item.value for item in enum], case_sensitive=case_sensitive)

    def convert(self, value, param, ctx):
        if value is None or isinstance(value, Enum):
            return value

        converted_str = super().convert(value, param, ctx)
        return self.__enum(converted_str)


@unique
class InFormat(Enum):
    FHIR = "FHIR"
    QRDA = "QRDA"
    CSV = "CSV"
    TEST = "TEST"
    DF = "DF"


class FileExtension(Enum):
    EXCEL = ('.xlsx', '.xls')
    CSV = ('.csv', )
    JSON = ('.json', )
    HTML = ('.html', '.htm')
    XML = ('.xml', )
    LATEX = ('.tex', )
    FEATHER = ('.feather', )

    @classmethod
    def get_extension_type(cls, extension: str):
        for ext_type in cls:
            if extension in ext_type.value:
                return ext_type
        return None
