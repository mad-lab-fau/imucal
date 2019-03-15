import json
from collections import namedtuple

import h5py
import numpy as np

_calibration_info = namedtuple('CalibrationInfo', ['K_a', 'R_a', 'b_a', 'K_g', 'R_g', 'K_ga', 'b_g'])


class CalibrationInfo(_calibration_info):
    __slots__ = ()

    def __repr__(self):
        out = 'CalibrationInfo('
        for val in self._fields:
            out += '\n' + val + ' =\n' + getattr(self, val).__repr__() + ',\n'
        out += '\n)'
        return out

    def _to_list_dict(self):
        return {key: getattr(self, key).tolist() for key in self._fields}

    def to_hdf5(self, filename):
        """
        Saves calibration matrices to hdf5 fileformat
        :param filename: filename (including h5 at end)
        """

        with h5py.File(filename, 'w') as hdf:
            for k, v in self._asdict().iteritems():
                hdf.create_dataset(k, v)

    def to_json(self):
        data_dict = self._to_list_dict()
        return json.dumps(data_dict, indent=4)

    def to_json_file(self, path):
        data_dict = self._to_list_dict()
        return json.dump(data_dict, open(path, 'w'))

    @classmethod
    def from_hdf5(cls, path):
        """
        Reads calibration data stored in hdf5 fileformat (created by CalibrationInfo save_to_hdf5)
        :param filename: filename
        :return: CalibrationInfo object
        """

        with h5py.File(path, 'r') as hdf:
            values = dict()
            for k in cls._fields:
                values[k] = np.array(hdf.get(k))

        return cls(**values)

    @classmethod
    def _from_list_dict(cls, list_dict):
        raw_json = {k: np.array(v) for k, v in list_dict.items()}
        return cls(**raw_json)

    @classmethod
    def from_json(cls, json_str):
        raw_json = json.loads(json_str)
        return cls._from_list_dict(raw_json)

    @classmethod
    def from_json_file(cls, path):
        raw_json = json.load(open(path, 'r'))
        return cls._from_list_dict(raw_json)
