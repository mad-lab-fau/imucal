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
