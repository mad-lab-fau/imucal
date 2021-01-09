from pathlib import Path

import pytest

from imucal import FerrarisCalibrationInfo
from imucal.legacy import load_v1_json_files, load_v1_json
from imucal.management import load_calibration_info


@pytest.fixture()
def legacy_pre_2_0_cal():
    return Path(__file__).parent.parent / "example_data/legacy_calibration_pre_2.0.json"


def test_legacy_json_import(legacy_pre_2_0_cal):
    cal_info = load_v1_json_files(legacy_pre_2_0_cal)

    # Just check a couple of parameters to confirm that the file is loaded.
    assert isinstance(cal_info, FerrarisCalibrationInfo)
    assert cal_info.b_g[0] == -9.824970828471413


def test_legacy_json_str(legacy_pre_2_0_cal):
    with open(legacy_pre_2_0_cal) as f:
        json_str = f.read()
    cal_info = load_v1_json(json_str)

    # Just check a couple of parameters to confirm that the file is loaded.
    assert isinstance(cal_info, FerrarisCalibrationInfo)
    assert cal_info.b_g[0] == -9.824970828471413


def test_error_raised(legacy_pre_2_0_cal):
    with pytest.raises(ValueError) as e:
        load_calibration_info(legacy_pre_2_0_cal)

    assert "`imucal.legacy`" in str(e)
