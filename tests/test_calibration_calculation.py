from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from example_data import EXAMPLE_PATH
from imucal import FerrarisCalibrationInfo
from imucal.ferraris_calibration import (
    FerrarisCalibration,
    FerrarisSignalRegions,
    TurntableCalibration,
    ferraris_regions_from_df,
)


@pytest.fixture()
def example_calibration_data():
    calib = FerrarisCalibrationInfo.from_json_file(Path(__file__).parent / "snapshots/example_cal.json")
    data = pd.read_csv(EXAMPLE_PATH / "annotated_session.csv", index_col=[0, 1])
    sampling_rate = 204.8
    return data, sampling_rate, calib


def test_example_calibration(example_calibration_data):
    data, sampling_rate, calib = example_calibration_data

    cal = FerrarisCalibration()
    regions = ferraris_regions_from_df(data)
    cal_mat = cal.compute(regions, sampling_rate, from_acc_unit="a.u.", from_gyr_unit="a.u.")

    # # Uncomment if you want to save the new cal matrix to update the regression test
    # cal_mat.to_json_file(Path(__file__).parent / "snapshots/example_cal.json")

    assert cal_mat == calib


@pytest.fixture()
def default_expected():
    expected = {}
    expected["K_a"] = np.identity(3)
    expected["R_a"] = np.identity(3)
    expected["b_a"] = np.zeros(3)
    expected["K_g"] = np.identity(3)
    expected["R_g"] = np.identity(3)
    expected["b_g"] = np.zeros(3)
    expected["K_ga"] = np.zeros((3, 3))

    return expected


@pytest.fixture()
def default_data():
    data = {}

    data["acc_x_p"] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data["acc_x_a"] = -data["acc_x_p"]
    data["acc_y_p"] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data["acc_y_a"] = -data["acc_y_p"]
    data["acc_z_p"] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)
    data["acc_z_a"] = -data["acc_z_p"]

    data["gyr_x_p"] = np.zeros((100, 3))
    data["gyr_x_a"] = np.zeros((100, 3))
    data["gyr_y_p"] = np.zeros((100, 3))
    data["gyr_y_a"] = np.zeros((100, 3))
    data["gyr_z_p"] = np.zeros((100, 3))
    data["gyr_z_a"] = np.zeros((100, 3))

    data["acc_x_rot"] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data["acc_y_rot"] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data["acc_z_rot"] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)

    data["gyr_x_rot"] = -np.repeat(np.array([[360.0, 0, 0]]), 100, axis=0)
    data["gyr_y_rot"] = -np.repeat(np.array([[0, 360.0, 0]]), 100, axis=0)
    data["gyr_z_rot"] = -np.repeat(np.array([[0, 0, 360.0]]), 100, axis=0)

    out = {}
    out["signal_regions"] = FerrarisSignalRegions(**data)
    out["sampling_rate_hz"] = 100
    out["from_acc_unit"] = "a.u."
    out["from_gyr_unit"] = "a.u."
    return out


@pytest.fixture()
def k_ga_data(default_data, default_expected):
    # "Simulate" a large influence of acc on gyro.
    # Every gyro axis depends on acc_x to produce predictable off axis elements
    cal_regions = default_data["signal_regions"]._asdict()
    cal_regions["gyr_x_p"] += cal_regions["acc_x_p"]
    cal_regions["gyr_x_a"] += cal_regions["acc_x_a"]
    cal_regions["gyr_y_p"] += cal_regions["acc_x_p"]
    cal_regions["gyr_y_a"] += cal_regions["acc_x_a"]
    cal_regions["gyr_z_p"] += cal_regions["acc_x_p"]
    cal_regions["gyr_z_a"] += cal_regions["acc_x_a"]

    # add the influence artifact to the rotation as well
    cal_regions["gyr_x_rot"] += cal_regions["acc_x_p"]
    cal_regions["gyr_y_rot"] += cal_regions["acc_x_p"]
    cal_regions["gyr_z_rot"] += cal_regions["acc_x_p"]

    default_data["signal_regions"] = FerrarisSignalRegions(**cal_regions)

    # Only influence in K_ga expected
    # acc_x couples to 100% into all axis -> Therefore first row 1
    expected = np.zeros((3, 3))
    expected[0, :] = 1
    default_expected["K_ga"] = expected

    return default_data, default_expected


@pytest.fixture()
def bias_data(default_data, default_expected):
    cal_regions = default_data["signal_regions"]._asdict()
    # Add bias to acc
    acc_bias = np.array([1, 3, 5])
    cal_regions["acc_x_p"] += acc_bias
    cal_regions["acc_x_a"] += acc_bias
    cal_regions["acc_y_p"] += acc_bias
    cal_regions["acc_y_a"] += acc_bias
    cal_regions["acc_z_p"] += acc_bias
    cal_regions["acc_z_a"] += acc_bias

    default_expected["b_a"] = acc_bias

    # Add bias to gyro
    gyro_bias = np.array([2, 4, 6])
    cal_regions["gyr_x_p"] += gyro_bias
    cal_regions["gyr_x_a"] += gyro_bias
    cal_regions["gyr_y_p"] += gyro_bias
    cal_regions["gyr_y_a"] += gyro_bias
    cal_regions["gyr_z_p"] += gyro_bias
    cal_regions["gyr_z_a"] += gyro_bias

    cal_regions["gyr_x_rot"] += gyro_bias
    cal_regions["gyr_y_rot"] += gyro_bias
    cal_regions["gyr_z_rot"] += gyro_bias

    default_data["signal_regions"] = FerrarisSignalRegions(**cal_regions)

    default_expected["b_g"] = gyro_bias

    return default_data, default_expected


@pytest.fixture()
def scaling_data(default_data, default_expected):
    cal_regions = default_data["signal_regions"]._asdict()
    # Add scaling to acc
    acc_scaling = np.array([1, 3, 5])
    cal_regions["acc_x_p"] *= acc_scaling[0]
    cal_regions["acc_x_a"] *= acc_scaling[0]
    cal_regions["acc_y_p"] *= acc_scaling[1]
    cal_regions["acc_y_a"] *= acc_scaling[1]
    cal_regions["acc_z_p"] *= acc_scaling[2]
    cal_regions["acc_z_a"] *= acc_scaling[2]

    default_expected["K_a"] = np.diag(acc_scaling)

    # Add scaling to gyro
    gyro_scaling = np.array([2, 4, 6])
    cal_regions["gyr_x_rot"] *= gyro_scaling[0]
    cal_regions["gyr_y_rot"] *= gyro_scaling[1]
    cal_regions["gyr_z_rot"] *= gyro_scaling[2]

    default_data["signal_regions"] = FerrarisSignalRegions(**cal_regions)
    default_expected["K_g"] = np.diag(gyro_scaling)

    return default_data, default_expected


# TODO: Test for rotation is missing


@pytest.mark.parametrize("test_data", ["k_ga_data", "bias_data", "scaling_data"])
def test_simulations(test_data, request):
    test_data = request.getfixturevalue(test_data)
    cal = FerrarisCalibration()
    cal_mat = cal.compute(**test_data[0])

    for para, val in test_data[1].items():
        assert_array_almost_equal(getattr(cal_mat, para), val, err_msg=para)


def test_turntable_calibration(default_data, default_expected):
    cal = TurntableCalibration()
    cal_mat = cal.compute(**default_data)

    keys = set(default_expected.keys()) - {"K_g"}
    for para in keys:
        assert_array_almost_equal(getattr(cal_mat, para), default_expected[para], err_msg=para)

    assert_array_almost_equal(cal_mat.K_g, default_expected["K_g"] / 2, err_msg="K_g")

    assert cal.expected_angle == -720
