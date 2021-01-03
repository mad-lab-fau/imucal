"""A set of highly opinionated helper functions to store and load calibration files for a medium number of sensors."""
import datetime
import re
import warnings
from pathlib import Path
from typing import Optional, List, Callable, TypeVar, Union, Type

import numpy as np
from typing_extensions import Literal

from imucal import CalibrationInfo

path_t = TypeVar("path_t", str, Path)  # noqa: invalid-name


class CalibrationWarning(Warning):
    """Indicate potential issues with a calibration."""


def save_calibration_info(
    cal_info: CalibrationInfo,
    sensor_id: str,
    cal_time: datetime.datetime,
    folder: path_t,
    folder_structure="{sensor_id}/{cal_info.CAL_TYPE}",
    **kwargs,
) -> Path:
    """Save a calibration info object in the correct format and file name for NilsPods.

    By default the files will be saved in the format:
    `folder/{sensor_id}/{cal_info.CAL_TYPE}/{sensor_id}_%Y-%m-%d_%H-%M.json`

    The naming schema and format is of course just a suggestion, and any structure can be used as long as it can be
    converted back into a CalibrationInfo object.
    However, following the naming convention will allow to use other calibration utils to search for suitable
    calibration files.

    .. note:: If the folder does not exist it will be created.

    Parameters
    ----------
    cal_info :
        The CalibrationInfo object ot be saved
    sensor_id :
        A unique id to identify the calibrated sensor.
        Note that this will converted to all lower-case!
    cal_time :
        The date and time (min precision) when the calibration was performed.
    folder :
        Basepath of the folder, where the file will be stored.
    folder_structure :
        A valid formatted Python string using the `{}` syntax.
        `sensor_id`, `cal_info` and kwargs will be passed to the `str.format` as keyword arguments and can be used
        in the string.

    Returns
    -------
    output_file_name
        The name under which the calibration file was saved

    Notes
    -----
    Yes, this way of storing files doubles information at various places, which is usually discouraged.
    However, in this case it ensures that you still have all information just from the file name and the file content,
    and also easy "categories" if you search through the file tree manually.

    """
    if not sensor_id.isalnum():
        raise ValueError(
            "Sensor ids must be alphanumerical characters to not interfere with pattern matching in the " "file name."
        )
    folder = Path(folder) / folder_structure.format(sensor_id=sensor_id, cal_info=cal_info, **kwargs)
    folder.mkdir(parents=True, exist_ok=True)
    f_name = folder / "{}_{}.json".format(sensor_id.lower(), cal_time.strftime("%Y-%m-%d_%H-%M"))
    cal_info.to_json_file(f_name)
    return f_name


def find_calibration_info_for_sensor(
    sensor_id: str,
    folder: path_t,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    custom_validator: Optional[Callable[[CalibrationInfo], bool]] = None,
    ignore_file_not_found: Optional[bool] = False,
) -> List[Path]:
    """Find possible calibration files based on the filename.

    As this only checks the filenames, this might return false positives depending on your folder structure and naming.

    Parameters
    ----------
    sensor_id :
        A unique id to identify the calibrated sensor
    folder :
        Basepath of the folder to search.
    recursive :
        If the folder should be searched recursive or not.
    filter_cal_type :
        Whether only files obtain with a certain calibration type should be found.
        This will look for the `CalType` inside the json file and hence cause performance problems.
        If None, all found files (over all potential subfolders) will be returned.
    custom_validator :
        A custom function that will be called with the CalibrationInfo object of each potential match.
        This needs load the json file of each match and could cause performance issues with many calibration files.
    ignore_file_not_found :
        If True this function will not raise an error, but rather return an empty list, if no calibration files were
        found for the specific sensor_type.

    Returns
    -------
    list_of_cals
        List of paths pointing to available calibration objects.

    """
    method = "glob"
    if recursive is True:
        method = "rglob"

    r = sensor_id.lower() + r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}"

    matches = [f for f in getattr(Path(folder), method)("{}_*.json".format(sensor_id)) if re.fullmatch(r, f.stem)]

    final_matches = []
    for m in matches:
        cal = load_calibration_info(m, file_type="json")
        if (filter_cal_type is None or cal.CAL_TYPE.lower() == filter_cal_type.lower()) and (
            custom_validator is None or custom_validator(cal)
        ):
            final_matches.append(m)

    if not final_matches and ignore_file_not_found is not True:
        raise ValueError("No Calibration for the sensor_type with the id {} could be found".format(sensor_id))
    return final_matches


def find_closest_calibration_info_to_date(
    sensor_id: str,
    cal_time: datetime.datetime,
    folder: Optional[path_t] = None,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    custom_validator: Optional[Callable[[CalibrationInfo], bool]] = None,
    before_after: Optional[str] = None,
    warn_thres: datetime.timedelta = datetime.timedelta(days=30),  # noqa E252
    ignore_file_not_found: Optional[bool] = False,
) -> Optional[Path]:
    """Find the calibration file for a sensor_type, that is closes to a given date.

    As this only checks the filenames, this might return a false positive depending on your folder structure and naming.

    Parameters
    ----------
    sensor_id :
        A unique id to identify the calibrated sensor
    cal_time :
        time and date to look for
    folder :
        Basepath of the folder to search. If None, tries to find a default calibration
    recursive :
        If the folder should be searched recursive or not.
    filter_cal_type :
        Whether only files obtain with a certain calibration type should be found.
        This will look for the `CalType` inside the json file and hence cause performance problems.
        If None, all found files (over all potential subfolders) will be returned.
    custom_validator :
        A custom function that will be called with the CalibrationInfo object of each potential match.
        This needs load the json file of each match and could cause performance issues with many calibration files.
    before_after :
        Can either be 'before' or 'after', if the search should be limited to calibrations that were
        either before or after the specified date.
        If None the closest value will be returned, ignoring if it was before or after the measurement.
    warn_thres :
        If the distance to the closest calibration is larger than this threshold, a warning is emitted
    ignore_file_not_found :
        If True this function will not raise an error, but rather return `None`, if no calibration files were found for
        the specific sensor_type.

    Notes
    -----
    If there are multiple calibrations that have the same date/hour/minute distance form the measurement,
    the calibration before the measurement will be chosen. This can be overwritten using the `before_after` para.

    See Also
    --------
    nilspodlib.calibration_utils.find_calibrations_for_sensor

    Returns
    -------
    cal_file_path or None
        The path to a suitable calibration file, or `None`, if no suitable file could be found.

    """
    if before_after not in ("before", "after", None):
        raise ValueError('Invalid value for `before_after`. Only "before", "after" or None are allowed')

    potential_list = find_calibration_info_for_sensor(
        sensor_id=sensor_id,
        folder=folder,
        recursive=recursive,
        filter_cal_type=filter_cal_type,
        custom_validator=custom_validator,
        ignore_file_not_found=ignore_file_not_found,
    )
    if not potential_list:
        if ignore_file_not_found is True:
            return None
        raise ValueError("No Calibration for the sensor with the id {} could be found".format(sensor_id))

    dates = [datetime.datetime.strptime("_".join(d.stem.split("_")[1:]), "%Y-%m-%d_%H-%M") for d in potential_list]

    dates = np.array(dates, dtype="datetime64[s]")
    potential_list, _ = zip(*sorted(zip(potential_list, dates), key=lambda x: x[1]))
    dates.sort()

    diffs = (dates - np.datetime64(cal_time, "s")).astype(float)

    if before_after == "after":
        diffs[diffs < 0] = np.nan
    elif before_after == "before":
        diffs[diffs > 0] = np.nan

    if np.all(diffs) == np.nan:
        raise ValueError(
            "No calibrations between {} and {} were found for sensor {}.".format(before_after, cal_time, sensor_id)
        )

    min_dist = float(np.nanmin(np.abs(diffs)))
    if warn_thres < datetime.timedelta(seconds=min_dist):
        warnings.warn(
            "For the sensor {} no calibration could be located that was in {} of the {}."
            "The closest calibration is {} away.".format(
                sensor_id, warn_thres, cal_time, datetime.timedelta(seconds=min_dist)
            ),
            CalibrationWarning,
        )

    return potential_list[int(np.nanargmin(np.abs(diffs)))]


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
