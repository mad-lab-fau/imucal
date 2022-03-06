"""Base Class for all CalibrationInfo objects."""
import json
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Tuple, Union, TypeVar, ClassVar, Optional, Type, Iterable

import numpy as np
import pandas as pd
from packaging.version import Version

CalInfo = TypeVar("CalInfo", bound="CalibrationInfo")

_CAL_FORMAT_VERSION = Version("2.0.0")


@dataclass(eq=False)
class CalibrationInfo:
    """Abstract BaseClass for all Calibration Info objects.

    .. note ::
        All `CalibrationInfo` subclasses implement a `CAL_TYPE` attribute.
        If the calibration is exported into any format, this information is stored as well.
        If imported, all constructor methods intelligently infer the correct CalibrationInfo subclass based on
        this parameter.

        >>> json_string = "{cal_type: 'Ferraris', ...}"
        >>> CalibrationInfo.from_json(json_string)
        <FerrarisCalibrationInfo ...>

    """

    CAL_FORMAT_VERSION: ClassVar[Version] = _CAL_FORMAT_VERSION  # noqa: invalid-name
    CAL_TYPE: ClassVar[str] = None  # noqa: invalid-name
    acc_unit: Optional[str] = None
    gyr_unit: Optional[str] = None
    from_acc_unit: Optional[str] = None
    from_gyr_unit: Optional[str] = None
    comment: Optional[str] = None

    _cal_paras: ClassVar[Tuple[str, ...]]

    def calibrate(
        self, acc: np.ndarray, gyr: np.ndarray, acc_unit: Optional[str], gyr_unit: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method to perform a calibration on both acc and gyro.

        This absolutely needs to implement by any daughter class.
        It is further recommended to call `self._validate_units` in the overwritten calibrate method, to check if the
        input units are as expected.

        Parameters
        ----------
        acc :
            3D acceleration
        gyr :
            3D gyroscope values
        acc_unit
            The unit of the acceleration data
        gyr_unit
            The unit of the gyroscope data

        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def calibrate_df(
        self,
        df: pd.DataFrame,
        acc_unit: Optional[str],
        gyr_unit: Optional[str],
        acc_cols: Iterable[str] = ("acc_x", "acc_y", "acc_z"),
        gyr_cols: Iterable[str] = ("gyr_x", "gyr_y", "gyr_z"),
    ) -> pd.DataFrame:
        """Apply the calibration to data stored in a dataframe.

        This calls `calibrate` for the respective columns and returns a copy of the df with the respective columns
        replaced by their calibrated counter-part.

        See the `calibrate` method for more information.

        Parameters
        ----------
        df :
            6 column dataframe (3 acc, 3 gyro)
        acc_cols :
            The name of the 3 acceleration columns in order x,y,z.
        gyr_cols :
            The name of the 3 acceleration columns in order x,y,z.
        acc_unit
            The unit of the acceleration data
        gyr_unit
            The unit of the gyroscope data

        Returns
        -------
        cal_df
            A copy of `df` with the calibrated data.

        """
        acc_cols = list(acc_cols)
        gyr_cols = list(gyr_cols)
        acc = df[acc_cols].to_numpy()
        gyr = df[gyr_cols].to_numpy()
        cal_acc, cal_gyr = self.calibrate(acc=acc, gyr=gyr, acc_unit=acc_unit, gyr_unit=gyr_unit)

        cal_df = df.copy()
        cal_df[acc_cols] = cal_acc
        cal_df[gyr_cols] = cal_gyr

        return cal_df

    def _validate_units(self, other_acc, other_gyr):
        check_pairs = {"acc": (other_acc, self.from_acc_unit), "gyr": (other_gyr, self.from_gyr_unit)}
        for name, (other, this) in check_pairs.items():
            if other != this:
                if this is None:
                    raise ValueError(
                        "This calibration does not provide any information about the expected input "
                        "units for {0}. "
                        "Set `{0}_unit` explicitly to `None` to ignore this error. "
                        "However, we recommend to recreate your calibration with proper information "
                        "about the input unit.".format(name)
                    )
                raise ValueError(
                    "The provided {} data has a unit of {}. "
                    "However, the calibration is created to calibrate data with a unit of {}.".format(name, other, this)
                )

    def __eq__(self, other):
        """Check if two calibrations are identical.

        Note, we use a custom function for that, as we need to compare the numpy arrays.
        """
        # Check type:
        if not isinstance(other, self.__class__):
            raise ValueError("Comparison is only defined between two {} object!".format(self.__class__.__name__))

        # Test keys equal:
        if fields(self) != fields(other):
            return False

        # Test method equal
        if not self.CAL_TYPE == other.CAL_TYPE:
            return False

        # Test all values
        for f in fields(self):
            a1 = getattr(self, f.name)
            a2 = getattr(other, f.name)
            if isinstance(a1, np.ndarray):
                equal = np.array_equal(a1, a2)
            else:
                equal = a1 == a2
            if not equal:
                return False
        return True

    def _to_list_dict(self):
        d = asdict(self)
        d["cal_type"] = self.CAL_TYPE
        d["_format_version"] = str(self.CAL_FORMAT_VERSION)
        return d

    @classmethod
    def _from_list_dict(cls, list_dict):
        for k in cls._cal_paras:
            list_dict[k] = np.array(list_dict[k])
        return cls(**list_dict)

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def find_subclass_from_cal_type(cls, cal_type):
        """Get a SensorCalibration subclass that handles the specified calibration type."""
        if cls.CAL_TYPE == cal_type:
            return cls
        try:
            out_cls = next(x for x in cls._get_subclasses() if x.CAL_TYPE == cal_type)
        except StopIteration as e:
            raise ValueError(
                "No suitable calibration info class could be found for caltype `{}`. "
                "The following classes were checked: {}. "
                "If your CalibrationInfo class is missing, make sure it is imported before loading a "
                "file.".format(cal_type, (cls.__name__, *(x.__name__ for x in cls._get_subclasses())))
            ) from e
        return out_cls

    def to_json(self) -> str:
        """Convert all calibration matrices into a json string."""
        data_dict = self._to_list_dict()
        return json.dumps(data_dict, indent=4, cls=NumpyEncoder)

    @classmethod
    def from_json(cls: Type[CalInfo], json_str: str) -> CalInfo:
        """Create a calibration object from a json string (created by `CalibrationInfo.to_json`).

        Parameters
        ----------
        json_str :
            valid json string object

        Returns
        -------
        cal_info
            A CalibrationInfo object.
            The exact child class is determined by the `cal_type` key in the json string.

        """
        raw_json = json.loads(json_str)
        check_cal_format_version(Version(raw_json.pop("_format_version", None)), cls.CAL_FORMAT_VERSION)
        subclass = cls.find_subclass_from_cal_type(raw_json.pop("cal_type"))
        return subclass._from_list_dict(raw_json)

    def to_json_file(self, path: Union[str, Path]):
        """Dump acc calibration matrices into a file in json format.

        Parameters
        ----------
        path :
            path to the json file

        """
        data_dict = self._to_list_dict()
        return json.dump(data_dict, open(path, "w", encoding="utf8"), cls=NumpyEncoder, indent=4)

    @classmethod
    def from_json_file(cls: Type[CalInfo], path: Union[str, Path]) -> CalInfo:
        """Create a calibration object from a valid json file (created by `CalibrationInfo.to_json_file`).

        Parameters
        ----------
        path :
            Path to the json file

        Returns
        -------
        cal_info
            A CalibrationInfo object.
            The exact child class is determined by the `cal_type` key in the json string.

        """
        with open(path, "r", encoding="utf8") as f:
            raw_json = json.load(f)
        check_cal_format_version(raw_json.pop("_format_version", None), cls.CAL_FORMAT_VERSION)
        subclass = cls.find_subclass_from_cal_type(raw_json.pop("cal_type"))
        return subclass._from_list_dict(raw_json)

    def to_hdf5(self, path: Union[str, Path]):
        """Save calibration matrices to hdf5 file format.

        Parameters
        ----------
        path :
            Path to the hdf5 file

        """
        import h5py  # noqa: import-outside-toplevel

        with h5py.File(path, "w") as hdf:
            for k, v in self._to_list_dict().items():
                if k in self._cal_paras:
                    hdf.create_dataset(k, data=v.tolist())
                elif v:
                    hdf[k] = v

    @classmethod
    def from_hdf5(cls: Type[CalInfo], path: Union[str, Path]):
        """Read calibration data stored in hdf5 fileformat (created by `CalibrationInfo.save_to_hdf5`).

        Parameters
        ----------
        path :
            Path to the hdf5 file

        Returns
        -------
        cal_info
            A CalibrationInfo object.
            The exact child class is determined by the `cal_type` key in the json string.

        """
        import h5py  # noqa: import-outside-toplevel

        with h5py.File(path, "r") as hdf:
            format_version = hdf.get("_format_version")
            if format_version:
                format_version = format_version[()]
            check_cal_format_version(format_version, cls.CAL_FORMAT_VERSION)
            subcls = cls.find_subclass_from_cal_type(hdf["cal_type"][()])
            data = {k.name: hdf.get(k.name)[()] for k in fields(subcls)}

        return subcls._from_list_dict(data)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy array."""

    def default(self, obj):  # noqa: arguments-differ
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_cal_format_version(version: Optional[Version] = None, current_version: Version = _CAL_FORMAT_VERSION):
    """Check if a calibration can be loaded with the current loader."""
    # No version means, the old 1.0 format is used that does not provide a version string
    if not version:
        version = Version("1.0.0")
    if isinstance(version, str):
        version = Version(version)

    if version == current_version:
        return
    if version > current_version:
        raise ValueError("The provided version, is larger than the currently supported version.")
    if version < current_version:
        raise ValueError(
            "The provided calibration format is no longer supported. "
            "Check `imucal.legacy` if conversion helper exist."
        )
