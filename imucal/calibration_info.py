"""Base Class for all CalibrationInfo objects."""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import ClassVar, Optional, TypeVar, Union

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

    CAL_FORMAT_VERSION: ClassVar[Version] = _CAL_FORMAT_VERSION
    CAL_TYPE: ClassVar[str] = None
    acc_unit: Optional[str] = None
    gyr_unit: Optional[str] = None
    from_acc_unit: Optional[str] = None
    from_gyr_unit: Optional[str] = None
    comment: Optional[str] = None

    _cal_paras: ClassVar[tuple[str, ...]]

    def calibrate(
        self, acc: np.ndarray, gyr: np.ndarray, acc_unit: Optional[str], gyr_unit: Optional[str]
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _validate_units(self, other_acc, other_gyr) -> None:
        check_pairs = {"acc": (other_acc, self.from_acc_unit), "gyr": (other_gyr, self.from_gyr_unit)}
        for name, (other, this) in check_pairs.items():
            if other != this:
                if this is None:
                    raise ValueError(
                        "This calibration does not provide any information about the expected input "
                        f"units for {name}. "
                        f"Set `{name}_unit` explicitly to `None` to ignore this error. "
                        "However, we recommend to recreate your calibration with proper information "
                        "about the input unit."
                    )
                raise ValueError(
                    f"The provided {name} data has a unit of {other}. "
                    f"However, the calibration is created to calibrate data with a unit of {this}."
                )

    def __eq__(self, other):
        """Check if two calibrations are identical.

        Note, we use a custom function for that, as we need to compare the numpy arrays.
        """
        # Check type:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Comparison is only defined between two {self.__class__.__name__} object!")

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
            equal = np.array_equal(a1, a2) if isinstance(a1, np.ndarray) else a1 == a2
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
        if cal_type == cls.CAL_TYPE:
            return cls
        try:
            out_cls = next(x for x in cls._get_subclasses() if cal_type == x.CAL_TYPE)
        except StopIteration as e:
            raise ValueError(
                f"No suitable calibration info class could be found for caltype `{cal_type}`. "
                f"The following classes were checked: {(cls.__name__, *(x.__name__ for x in cls._get_subclasses()))}. "
                "If your CalibrationInfo class is missing, make sure it is imported before loading a "
                "file."
            ) from e
        return out_cls

    def to_json(self) -> str:
        """Convert all calibration matrices into a json string."""
        data_dict = self._to_list_dict()
        return json.dumps(data_dict, indent=4, cls=NumpyEncoder)

    @classmethod
    def from_json(cls: type[CalInfo], json_str: str) -> CalInfo:
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
        with Path(path).open("w", encoding="utf8") as f:
            json.dump(data_dict, f, cls=NumpyEncoder, indent=4)

    @classmethod
    def from_json_file(cls: type[CalInfo], path: Union[str, Path]) -> CalInfo:
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
        with Path(path).open(encoding="utf8") as f:
            raw_json = json.load(f)
        check_cal_format_version(raw_json.pop("_format_version", None), cls.CAL_FORMAT_VERSION)
        subclass = cls.find_subclass_from_cal_type(raw_json.pop("cal_type"))
        return subclass._from_list_dict(raw_json)

    def to_hdf5(self, path: Union[str, Path]) -> None:
        """Save calibration matrices to hdf5 file format.

        Parameters
        ----------
        path :
            Path to the hdf5 file

        """
        import h5py

        with h5py.File(path, "w") as hdf:
            for k, v in self._to_list_dict().items():
                if k in self._cal_paras:
                    hdf.create_dataset(k, data=v.tolist())
                elif v:
                    hdf[k] = v

    @classmethod
    def from_hdf5(cls: type[CalInfo], path: Union[str, Path]):
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
        import h5py

        with h5py.File(path, "r") as hdf:
            format_version = hdf.get("_format_version")
            if format_version:
                format_version = format_version.asstr()[()]
            check_cal_format_version(format_version, cls.CAL_FORMAT_VERSION)
            subcls = cls.find_subclass_from_cal_type(hdf["cal_type"][()].decode("utf-8"))
            data = {}
            for k in fields(subcls):
                dp = hdf.get(k.name)
                if h5py.check_string_dtype(dp.dtype):
                    # String data
                    data[k.name] = dp.asstr()[()]
                else:
                    # No string data
                    data[k.name] = dp[()]
        return subcls._from_list_dict(data)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy array."""

    def default(self, obj):
        """Allow encoding of numpy objects by converting them to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_cal_format_version(version: Optional[Version] = None, current_version: Version = _CAL_FORMAT_VERSION) -> None:
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
