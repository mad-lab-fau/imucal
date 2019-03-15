import pytest
import numpy as np

from imucal.calibration_info import CalibrationInfo


@pytest.fixture()
def dummy_cal():
    sample_data = {'K_a': np.identity(3),
                   'R_a': np.identity(3),
                   'b_a': np.zeros(3)[:, None],
                   'K_g': np.identity(3),
                   'R_g': np.identity(3),
                   'K_ga': np.identity(3),
                   'b_g': np.zeros(3)[:, None]}
    return CalibrationInfo(**sample_data)


@pytest.fixture()
def dummy_data():
    sample_acc = np.repeat(np.array([[0, 0, 1.]]), 100, axis=0)
    sample_gyro = np.repeat(np.array([[1, 1, 1.]]), 100, axis=0)
    return sample_acc, sample_gyro


def test_dummy_cal(dummy_cal, dummy_data):
    acc, gyro = dummy_cal.calibrate(*dummy_data)
    np.array_equal(acc, dummy_data[0])
    np.array_equal(gyro, dummy_data[1])
