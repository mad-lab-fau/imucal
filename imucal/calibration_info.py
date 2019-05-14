import json
from typing import Iterable

import numpy as np


class CalibrationInfo:
    CAL_TYPE = None

    @property
    def _fields(self) -> Iterable[str]:
        """List of Calibration parameters that are required"""
        raise NotImplementedError('This method needs to be implemented by a subclass')

    def calibrate(self, acc, gyro):
        raise NotImplementedError('This method needs to be implemented by a subclass')

    def calibrate_gyro(self, gyro):
        raise NotImplementedError('This method needs to be implemented by a subclass')

    def calibrate_acc(self, acc):
        raise NotImplementedError('This method needs to be implemented by a subclass')

    def __init__(self, **kwargs):
        for field in self._fields:
            setattr(self, field, kwargs.get(field, None))

    def __repr__(self):
        out = self.__class__.__name__ + '('
        for val in self._fields:
            out += '\n' + val + ' =\n' + getattr(self, val).__repr__() + ',\n'
        out += '\n)'
        return out

    def __eq__(self, other):
        # Check type:
        if not isinstance(other, self.__class__):
            raise ValueError('Comparison is only defined between two {} object!'.format(self.__class__.__name__))

        # Test keys equal:
        if not self._fields == other._fields:
            return False

        # Test Calibration values
        for v1, v2 in zip(self.__dict__.values(), other.__dict__.values()):
            if not np.array_equal(v1, v2):
                return False
        return True

    def _to_list_dict(self):
        d = {key: getattr(self, key).tolist() for key in self._fields}
        d['cal_type'] = self.CAL_TYPE
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
    def _find_subclass_from_cal_type(cls, cal_type):
        return next(x for x in CalibrationInfo._get_subclasses() if x.CAL_TYPE == cal_type)

    def to_json(self):
        data_dict = self._to_list_dict()
        return json.dumps(data_dict, indent=4)

    @classmethod
    def from_json(cls, json_str):
        raw_json = json.loads(json_str)
        subclass = cls._find_subclass_from_cal_type(raw_json['cal_type'])
        return subclass._from_list_dict(raw_json)

    def to_json_file(self, path):
        data_dict = self._to_list_dict()
        return json.dump(data_dict, open(path, 'w'))

    @classmethod
    def from_json_file(cls, path):
        raw_json = json.load(open(path, 'r'))
        subclass = cls._find_subclass_from_cal_type(raw_json['cal_type'])
        return subclass._from_list_dict(raw_json)

    def to_hdf5(self, filename):
        """
        Saves calibration matrices to hdf5 fileformat
        :param filename: filename (including h5 at end)
        """
        import h5py

        with h5py.File(filename, 'w') as hdf:
            d = {key: getattr(self, key).tolist() for key in self._fields}
            for k, v in d.items():
                hdf.create_dataset(k, data=v)
            hdf['cal_type'] = self.CAL_TYPE

    @classmethod
    def from_hdf5(cls, path):
        """Reads calibration data stored in hdf5 fileformat (created by FerrarisCalibrationInfo save_to_hdf5).

        :param path: filename
        :return: FerrarisCalibrationInfo object
        """
        import h5py

        with h5py.File(path, 'r') as hdf:
            values = dict()
            subcls = cls._find_subclass_from_cal_type(hdf['cal_type'][...])
            for k in subcls._fields:
                values[k] = np.array(hdf.get(k))

        return subcls(**values)
