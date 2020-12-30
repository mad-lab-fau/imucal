"""Some general helper functions."""

from pathlib import Path
from typing import Union, Optional, Type

from typing_extensions import Literal

from imucal.calibration_info import CalibrationInfo


def load_calibration_info(
    path: Union[Path, str],
    file_type: Optional[Literal["hdf", "json"]] = None,
    base_class: Type[CalibrationInfo] = CalibrationInfo,
) -> CalibrationInfo:
    """Load any calibration info object from file.

    Parameters
    ----------
    path
        Path name to the file (can be .json or .hdf)
    file_type
        Format of the file (either `hdf` or `json`).
        If None, we try to figure out the correct format based on the file suffix.
    base_class
        This method finds the correct calibration info type by inspecting all subclasses of `base_class`.
        Usually that should be kept at the default value.

    Notes
    -----
    This function determines the correct calibration info class to use based on the `cal_type` parameter stored in the
    file.
    For this to work, the correct calibration info class must be accessible.
    This means, if you created a new calibration info class, you need to make sure that it is imported (or at least
    the file it is defined in), before using this function.

    """
    format_options = {"json": "from_json_file", "hdf": "from_hdf5"}
    path = Path(path)
    if file_type is None:
        # Determine format from file ending:
        suffix = path.suffix
        if suffix[1:] == "json":
            file_type = "json"
        elif suffix[1:] in ["hdf", "h5"]:
            file_type = "hdf"
        else:
            raise ValueError(
                "The loader format could not be determined from the file suffix." "Please specify `format` explicitly."
            )
    if file_type not in format_options:
        raise ValueError("`format` must be one of {}".format(list(format_options.keys())))

    return getattr(base_class, format_options[file_type])(path)
