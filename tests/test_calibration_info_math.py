import numpy as np
import pandas as pd
import pytest

from imucal.ferraris_calibration_info import FerrarisCalibrationInfo, TurntableCalibrationInfo


@pytest.fixture(params=[FerrarisCalibrationInfo, TurntableCalibrationInfo])
def dummy_cal(request):
    sample_data = {
        "K_a": np.identity(3),
        "R_a": np.identity(3),
        "b_a": np.zeros(3),
        "K_g": np.identity(3),
        "R_g": np.identity(3),
        "K_ga": np.zeros((3, 3)),
        "b_g": np.zeros(3),
    }
    return request.param(**sample_data)


@pytest.fixture()
def dummy_data():
    sample_acc = np.repeat(np.array([[0, 0, 1.0]]), 100, axis=0)
    sample_gyro = np.repeat(np.array([[1, 1, 1.0]]), 100, axis=0)
    return sample_acc, sample_gyro


def test_dummy_cal(dummy_cal, dummy_data):
    acc, gyro = dummy_cal.calibrate(*dummy_data)
    assert np.array_equal(acc, dummy_data[0])
    assert np.array_equal(gyro, dummy_data[1])


def test_dummy_cal_df(dummy_cal, dummy_data):
    dummy_df = pd.DataFrame(np.column_stack(dummy_data))
    dummy_df.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    result_df = dummy_cal.calibrate_df(dummy_df)
    assert np.array_equal(dummy_data[0], result_df.filter(like="acc"))
    assert np.array_equal(dummy_data[1], result_df.filter(like="gyr"))


def test_ka_ba_ra(dummy_cal, dummy_data):
    dummy_cal.K_a *= 2
    dummy_cal.b_a += 2
    dummy_cal.R_a = np.flip(dummy_cal.R_a, axis=1)
    acc, _ = dummy_cal.calibrate(*dummy_data)
    assert np.all(acc == [-0.5, -1, -1])


def test_kg_rg_bg(dummy_cal, dummy_data):
    dummy_cal.K_g *= 2
    dummy_cal.b_g += 2
    dummy_cal.R_g = np.array([[0, 0, 1], [0, 1, 0], [0.5, 0, 0]])
    _, gyro = dummy_cal.calibrate(*dummy_data)
    assert np.all(gyro == [-1, -0.5, -0.5])


def test_kga(dummy_cal, dummy_data):
    dummy_cal.K_g *= 2
    dummy_cal.b_g += 2
    dummy_cal.R_g = np.array([[0, 0, 1], [0, 1, 0], [0.5, 0, 0]])
    dummy_cal.K_ga = np.identity(3)
    _, gyro = dummy_cal.calibrate(*dummy_data)
    assert np.all(gyro == [-2, -0.5, -0.5])
