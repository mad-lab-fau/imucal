"""Base Class for all CalibrationInfo objects."""
import json
from pathlib import Path
from typing import Iterable, Tuple, Union, TypeVar

import numpy as np

CalInfo = TypeVar("CalInfo", bound="CalibrationInfo")


class CalibrationInfo:
    """Abstract BaseClass for all Calibration Info objects."""

    CAL_TYPE = None
    acc_unit = None
    gyro_unit = None

    _cal_type_explanation = """
    Note:
        All `CalibrationInfo` subclasses implement a `CAL_TYPE` attribute.
        If the calibration is exported into any format, this information is stored as well.
        If imported, all constructor methods intelligently infer the correct CalibrationInfo subclass based on
        this parameter.

        Example:
            >>> json_string = "{cal_type: 'Ferraris', ...}"
            >>> CalibrationInfo.from_json(json_string)
            <FerrarisCalibrationInfo ...>

    """

    __doc__ += _cal_type_explanation

    @property
    def _fields(self) -> Iterable[str]:
        """List of Calibration parameters that are required."""
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def calibrate(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method to perform a calibration on both acc and gyro.

        This absolutely needs to implement by any daughter class

        Parameter
        ---------
        acc :
            3D acceleration
        gyro :
            3D gyroscope values

        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def calibrate_gyro(self, gyro: np.ndarray) -> np.ndarray:
        """Abstract method to perform a calibration on both acc and gyro.

        This can implement by any daughter class, if separte calibration of acc makes sense for the calibration type.
        If not, an explicit error should be thrown

        Parameter
        ---------
        gro :
            3D gyroscope values

        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def calibrate_acc(self, acc: np.ndarray) -> np.ndarray:
        """Abstract method to perform a calibration on the gyro.

        This can implement by any daughter class, if separte calibration of acc makes sense for the calibration type.
        If not, an explicit error should be thrown

        Parameter
        ---------
        acc :
            3D acceleration

        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def __init__(self, **kwargs):
        """Create new CalibrationInfo instance.

        Parameter
        ---------
        kwargs :
            Matrices for all fields specified in `self._fields`

        """
        for field in self._fields:
            setattr(self, field, kwargs.get(field, None))

    def __repr__(self):
        """Show a proper string representation of a CalibrationInfo object."""
        out = self.__class__.__name__ + "("
        for val in self._fields:
            out += "\n" + val + " =\n" + getattr(self, val).__repr__() + ",\n"
        out += "\n)"
        return out

    def __eq__(self, other):
        """Check if two calibrations are identical."""
        # Check type:
        if not isinstance(other, self.__class__):
            raise ValueError("Comparison is only defined between two {} object!".format(self.__class__.__name__))

        # Test keys equal:
        if not self._fields == other._fields:
            return False

        # Test method equal
        if not self.CAL_TYPE == other.CAL_TYPE:
            return False

        # Test units
        if (not self.acc_unit == other.acc_unit) or (not self.gyro_unit == other.gyro_unit):
            return False

        # Test Calibration values
        for v1, v2 in zip(self.__dict__.values(), other.__dict__.values()):
            if not np.array_equal(v1, v2):
                return False
        return True

    def _to_list_dict(self):
        d = {key: getattr(self, key).tolist() for key in self._fields}
        d["cal_type"] = self.CAL_TYPE
        return d

    @classmethod
    def _from_list_dict(cls, list_dict):
        raw_json = {k: np.array(v) for k, v in list_dict.items()}
        return cls(**raw_json)

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def find_subclass_from_cal_type(cls, cal_type):
        """Get a SensorCalibration subclass that handles the specified calibration type."""
        return next(x for x in CalibrationInfo._get_subclasses() if x.CAL_TYPE == cal_type)

    def to_json(self) -> str:
        """Convert all calibration matrices into a json string."""
        data_dict = self._to_list_dict()
        return json.dumps(data_dict, indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> CalInfo:
        """Create a calibration object from a json string (created by `CalibrationInfo.to_json`).

        Parameter
        ---------
        json_str: valid json string object

        Returns
        -------
        cal_info
            A CalibrationInfo object.
            The exact child class is determined by the `cal_type` key in the json string.

        """
        raw_json = json.loads(json_str)
        subclass = cls.find_subclass_from_cal_type(raw_json["cal_type"])
        return subclass._from_list_dict(raw_json)

    def to_json_file(self, path: Union[str, Path]):
        """Dump acc calibration matrices into a file in json format.

        Parameter
        ---------
        path :
            path to the json file

        """
        data_dict = self._to_list_dict()
        return json.dump(data_dict, open(path, "w"))

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> CalInfo:
        """Create a calibration object from a valid json file (created by `CalibrationInfo.to_json_file`).

        Parameter
        ---------
        path :
            Path to the json file

        Returns
        -------
        cal_info
            A CalibrationInfo object.
            The exact child class is determined by the `cal_type` key in the json string.

        """
        raw_json = json.load(open(path, "r"))
        subclass = cls.find_subclass_from_cal_type(raw_json["cal_type"])
        return subclass._from_list_dict(raw_json)

    def to_hdf5(self, path: Union[str, Path]):
        """Save calibration matrices to hdf5 file format.

        Parameter
        ---------
        path :
            Path to the hdf5 file

        """
        import h5py  # noqa: import-outside-toplevel

        with h5py.File(path, "w") as hdf:
            d = {key: getattr(self, key).tolist() for key in self._fields}
            for k, v in d.items():
                hdf.create_dataset(k, data=v)
            hdf["cal_type"] = self.CAL_TYPE
            hdf["acc_unit"] = self.acc_unit
            hdf["gyro_unit"] = self.gyro_unit

    @classmethod
    def from_hdf5(cls, path: Union[str, Path]):
        """Read calibration data stored in hdf5 fileformat (created by `CalibrationInfo.save_to_hdf5`).

        Parameter
        ---------
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
            values = dict()
            subcls = cls.find_subclass_from_cal_type(hdf["cal_type"][...])
            for k in subcls._fields:
                values[k] = np.array(hdf.get(k))

        return subcls(**values)
