import tempfile
from copy import deepcopy

import pytest

from imucal import CalibrationInfo, load_calibration_info


def test_equal(sample_cal):
    assert sample_cal == deepcopy(sample_cal)


def test_equal_wrong_type(sample_cal):
    with pytest.raises(ValueError):
        assert sample_cal == 3


def test_equal_data(sample_cal, sample_cal_dict):
    not_equal = sample_cal_dict
    not_equal["K_a"] = not_equal["K_a"] - 1
    assert not (sample_cal == sample_cal.__class__(**not_equal))


def test_json_roundtrip(sample_cal):
    out = CalibrationInfo.from_json(sample_cal.to_json())
    assert sample_cal == out


def test_json_file_roundtrip(sample_cal):
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_json_file(f.name)
        out = load_calibration_info(f.name, file_type="json")
    assert sample_cal == out


def test_hdf5_file_roundtrip(sample_cal):
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_hdf5(f.name)
        out = load_calibration_info(f.name, file_type="hdf")
    assert sample_cal == out
