import tempfile
from copy import deepcopy

import pytest

from imucal import CalibrationInfo
from imucal.management import load_calibration_info


def test_equal(sample_cal) -> None:
    assert sample_cal == deepcopy(sample_cal)


def test_equal_wrong_type(sample_cal) -> None:
    with pytest.raises(TypeError):
        assert sample_cal == 3


def test_equal_data(sample_cal, sample_cal_dict) -> None:
    not_equal = sample_cal_dict
    not_equal["K_a"] = not_equal["K_a"] - 1
    assert sample_cal != sample_cal.__class__(**not_equal)


def test_json_roundtrip(sample_cal) -> None:
    out = CalibrationInfo.from_json(sample_cal.to_json())
    assert sample_cal == out


def test_json_file_roundtrip(sample_cal) -> None:
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_json_file(f.name)
        out = load_calibration_info(f.name, file_type="json")
    assert sample_cal == out


def test_hdf5_file_roundtrip(sample_cal) -> None:
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        sample_cal.to_hdf5(f.name)
        out = load_calibration_info(f.name, file_type="hdf")
    assert sample_cal == out


@pytest.mark.parametrize("unit", ["acc", "gyr"])
@pytest.mark.parametrize("is_none", ["some_value", None])
def test_error_on_wrong_calibration(unit, is_none, dummy_cal, dummy_data) -> None:
    # Test without error first:
    setattr(dummy_cal, f"from_{unit}_unit", is_none)
    units = {"acc_unit": dummy_cal.from_acc_unit, "gyr_unit": dummy_cal.from_gyr_unit}
    units[f"{unit}_unit"] = is_none
    dummy_cal.calibrate(*dummy_data, **units)

    # Now with error
    units[f"{unit}_unit"] = "wrong_value"
    with pytest.raises(ValueError) as e:
        dummy_cal.calibrate(*dummy_data, **units)

    if is_none is None:
        assert "explicitly to `None` to ignore this error" in str(e.value)
    else:
        assert is_none in str(e.value)
