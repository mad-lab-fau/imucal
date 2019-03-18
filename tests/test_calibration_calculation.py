import pytest

import pickle

from pathlib import Path
import numpy as np
from numpy.testing import assert_array_almost_equal

from imucal.calibration import Calibration


@pytest.fixture()
def example_calibration_data():
    tmp = pickle.load(open(Path(__file__).parent / '_test_data/example_cal.pk', 'rb'))
    data = tmp['data']
    sampling_rate = tmp['sampling_rate']
    calib = tmp['calib']
    return data, sampling_rate, calib


def test_example_calibration(example_calibration_data):
    data, sampling_rate, calib = example_calibration_data

    cal = Calibration.from_df(data, sampling_rate, acc_cols=('accX', 'accY', 'accZ'), gyro_cols=('gyroX', 'gyroY', 'gyroZ'))
    cal_mat = cal.compute_calibration_matrix()

    for val in cal_mat._fields:
        assert_array_almost_equal(getattr(cal_mat, val), getattr(calib, val), 5), val


@pytest.fixture()
def k_ga_data():
    data = dict()
    data['acc_x_p'] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data['acc_x_a'] = -data['acc_x_p']
    data['acc_y_p'] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data['acc_y_a'] = -data['acc_y_p']
    data['acc_z_p'] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)
    data['acc_z_a'] = -data['acc_z_p']

    # "Simulate" a large influence of acc on gyro.
    # Every gyro axis depends on acc_x to produce predictable off axis elements
    data['gyr_x_p'] = data['acc_x_p']
    data['gyr_x_a'] = data['acc_x_a']
    data['gyr_y_p'] = data['acc_x_p']
    data['gyr_y_a'] = data['acc_x_a']
    data['gyr_z_p'] = data['acc_x_p']
    data['gyr_z_a'] = data['acc_x_a']

    data['acc_x_rot'] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data['acc_y_rot'] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data['acc_z_rot'] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)

    # add the influence artifact to the rotation as well
    data['gyr_x_rot'] = np.repeat(np.array([[360, 0, 0]]), 100, axis=0) + data['acc_x_p']
    data['gyr_y_rot'] = np.repeat(np.array([[0, 360, 0]]), 100, axis=0) + data['acc_x_p']
    data['gyr_z_rot'] = np.repeat(np.array([[0, 0, 360]]), 100, axis=0) + data['acc_x_p']

    data['sampling_rate'] = 100

    return data


def test_a_g_influence(k_ga_data):
    cal = Calibration(**k_ga_data)
    cal_mat = cal.compute_calibration_matrix()

    # No influence on acc expected
    assert_array_almost_equal(cal_mat.K_a, np.identity(3))
    assert_array_almost_equal(cal_mat.R_a, np.identity(3))
    assert_array_almost_equal(cal_mat.b_a, np.zeros((3, 1)))

    # No influence in simple gyro expected
    assert_array_almost_equal(cal_mat.K_g, np.identity(3))
    assert_array_almost_equal(cal_mat.R_g, np.identity(3))
    assert_array_almost_equal(cal_mat.b_g, np.zeros((3, 1)))
    # Only influence in K_ga expected
    # acc_x couples to 100% into all axis -> Therefore first row 1
    expected = np.zeros((3, 3))
    expected[0, :] = 1
    assert_array_almost_equal(cal_mat.K_ga, expected)
