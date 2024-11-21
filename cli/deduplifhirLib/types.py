import json
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class SplinkSettings(BaseModel):
    link_type: Literal["link_only", "link_and_dedupe", "dedupe_only"]
    blocking_rules_to_generate_predictions: list[str|list[str]]
    max_iterations: Annotated[int, Field(ge=1)]  # Must be an integer ≥ 1
    em_convergence: Annotated[float, Field(gt=0)]  # Must be a float > 0


def load_splink_settings(file_path: str) -> SplinkSettings:
    """
    Load the splink settings JSON file and deserialize it into a SplinkSettings.

    :param file_path: Path to the JSON settings file
    :return: ConfigModel instance
    """
    with open(file_path, "r", encoding="utf-8") as f:
        splink_settings_dict = json.load(f)

    try:
        return SplinkSettings(**splink_settings_dict)
    except Exception as e:
        raise ValueError(f"Error deserializing the settings file: {e}")
