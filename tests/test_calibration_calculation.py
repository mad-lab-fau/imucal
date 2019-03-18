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
        assert_array_almost_equal(getattr(cal_mat, val), getattr(calib, val), 5, err_msg=val)


@pytest.fixture()
def default_expected():
    expected = dict()
    expected['K_a'] = np.identity(3)
    expected['R_a'] = np.identity(3)
    expected['b_a'] = np.zeros((3, 1))
    expected['K_g'] = np.identity(3)
    expected['R_g'] = np.identity(3)
    expected['b_g'] = np.zeros((3, 1))
    expected['K_ga'] = np.zeros((3, 3))

    return expected

@pytest.fixture()
def default_data():
    data = dict()

    data['sampling_rate'] = 100

    data['acc_x_p'] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data['acc_x_a'] = -data['acc_x_p']
    data['acc_y_p'] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data['acc_y_a'] = -data['acc_y_p']
    data['acc_z_p'] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)
    data['acc_z_a'] = -data['acc_z_p']

    data['gyr_x_p'] = np.zeros((100, 3))
    data['gyr_x_a'] = np.zeros((100, 3))
    data['gyr_y_p'] = np.zeros((100, 3))
    data['gyr_y_a'] = np.zeros((100, 3))
    data['gyr_z_p'] = np.zeros((100, 3))
    data['gyr_z_a'] = np.zeros((100, 3))

    data['acc_x_rot'] = np.repeat(np.array([[9.81, 0, 0]]), 100, axis=0)
    data['acc_y_rot'] = np.repeat(np.array([[0, 9.81, 0]]), 100, axis=0)
    data['acc_z_rot'] = np.repeat(np.array([[0, 0, 9.81]]), 100, axis=0)

    data['gyr_x_rot'] = np.repeat(np.array([[360., 0, 0]]), 100, axis=0)
    data['gyr_y_rot'] = np.repeat(np.array([[0, 360., 0]]), 100, axis=0)
    data['gyr_z_rot'] = np.repeat(np.array([[0, 0, 360.]]), 100, axis=0)

    return data


@pytest.fixture()
def k_ga_data(default_data, default_expected):
    # "Simulate" a large influence of acc on gyro.
    # Every gyro axis depends on acc_x to produce predictable off axis elements
    default_data['gyr_x_p'] += default_data['acc_x_p']
    default_data['gyr_x_a'] += default_data['acc_x_a']
    default_data['gyr_y_p'] += default_data['acc_x_p']
    default_data['gyr_y_a'] += default_data['acc_x_a']
    default_data['gyr_z_p'] += default_data['acc_x_p']
    default_data['gyr_z_a'] += default_data['acc_x_a']

    # add the influence artifact to the rotation as well
    default_data['gyr_x_rot'] += default_data['acc_x_p']
    default_data['gyr_y_rot'] += default_data['acc_x_p']
    default_data['gyr_z_rot'] += default_data['acc_x_p']

    # Only influence in K_ga expected
    # acc_x couples to 100% into all axis -> Therefore first row 1
    expected = np.zeros((3, 3))
    expected[0, :] = 1
    default_expected['K_ga'] = expected

    return default_data, default_expected


@pytest.mark.parametrize('test_data', ['k_ga_data'])
def test_simulations(test_data, request):
    test_data = request.getfixturevalue(test_data)
    cal = Calibration(**test_data[0])
    cal_mat = cal.compute_calibration_matrix()

    for para, val in test_data[1].items():
        assert_array_almost_equal(getattr(cal_mat, para), val, err_msg=para)
