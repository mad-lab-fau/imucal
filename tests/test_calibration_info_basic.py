import tempfile
from copy import deepcopy

import numpy as np
import pytest

from imucal.ferraris_calibration_info import FerrarisCalibrationInfo


@pytest.fixture()
def sample_cal_dict():
    sample_data = {
        "K_a": np.array([[208.54567264, 0.0, 0.0], [0.0, 208.00113412, 0.0], [0.0, 0.0, 214.78455365]]),
        "R_a": np.array(
            [
                [0.99991252, 0.00712206, -0.01114566],
                [-0.00794738, 0.99968874, 0.0236489],
                [0.0213429, -0.01078188, 0.99971407],
            ]
        ),
        "b_a": np.array([-6.01886802, -48.28787402, -28.96636637]),
        "K_g": np.array([[16.67747318, 0.0, 0.0], [0.0, 16.18769383, 0.0], [0.0, 0.0, 16.25326253]]),
        "R_g": np.array(
            [
                [9.99918368e-01, 3.38399869e-04, -1.27727091e-02],
                [-5.19256254e-03, 9.99269158e-01, 3.78706515e-02],
                [1.28516088e-02, -3.63520887e-02, 9.99256404e-01],
            ]
        ),
        "K_ga": np.array(
            [
                [0.00229265, 0.01387371, -0.00925911],
                [-0.01613463, 0.00544361, 0.00850631],
                [0.01846544, -0.00881248, -0.00393538],
            ]
        ),
        "b_g": np.array([1.9693536, -4.46624421, -3.65097072]),
    }
    return sample_data


@pytest.fixture()
def sample_cal(sample_cal_dict):
    return FerrarisCalibrationInfo(**sample_cal_dict)


def test_equal(sample_cal):
    assert sample_cal == deepcopy(sample_cal)


def test_equal_wrong_type(sample_cal):
    with pytest.raises(ValueError):
        assert sample_cal == 3


def test_equal_data(sample_cal, sample_cal_dict):
    not_equal = sample_cal_dict
    not_equal["K_a"] = not_equal["K_a"] - 1
    assert not (sample_cal == FerrarisCalibrationInfo(**not_equal))


def test_json_roundtrip(sample_cal):
    assert sample_cal == FerrarisCalibrationInfo.from_json(sample_cal.to_json())


def test_json_file_roundtrip(sample_cal):
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_json_file(f.name)
        out = FerrarisCalibrationInfo.from_json_file(f.name)
    assert sample_cal == out


def test_hdf5_file_roundtrip(sample_cal):
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_hdf5(f.name)
        out = FerrarisCalibrationInfo.from_hdf5(f.name)
    assert sample_cal == out
