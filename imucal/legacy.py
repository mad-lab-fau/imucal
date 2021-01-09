"""Helper functions to import calibration info export from older imucal versions."""

import json
import warnings
from distutils.version import StrictVersion
from pathlib import Path
from typing import Union, Type

from imucal import CalibrationInfo


def load_v1_json_files(
    path: Union[Path, str],
    base_class: Type[CalibrationInfo] = CalibrationInfo,
):
    """Load a exported json file that was created using imucal <= 2.0."""
    with open(path) as f:
        json_str = f.read()
    return load_v1_json(json_str, base_class=base_class)


def load_v1_json(
    json_str: str,
    base_class: Type[CalibrationInfo] = CalibrationInfo,
):
    """Load a json string that was created using imucal <= 2.0."""
    warnings.warn(
        "Importing a legacy calibration file, will use default values for all parameters that were newly"
        "introduced. "
        "These default values might not be correct for your calibration. "
        "Double check the resulting calibration info object and adapt the parameters manually. "
        "\n"
        "If you made any changes, make sure to save the modified calibration and load it with the normal "
        "loading function in the future."
    )
    json_dict = json.loads(json_str)

    # Check that the provided json is indeed a v1 calibration
    if "_format_version" in json_dict:
        raise ValueError(
            "The provided json does not seem to be a v1 json export, but has the format version {}.".format(
                json_dict["format_version"]
            )
        )

    # Apply the required modifications:
    json_dict["gyr_unit"] = json_dict.pop("gyro_unit")
    json_dict["_format_version"] = str(StrictVersion("2.0.0"))

    json_str = json.dumps(json_dict)

    return base_class.from_json(json_str)
