import pytest
import numpy as np

from imucal.ferraris_calibration_info import FerrarisCalibrationInfo, TurntableCalibrationInfo


@pytest.fixture(params=[
    FerrarisCalibrationInfo,
    TurntableCalibrationInfo
])
def dummy_cal(request):
    sample_data = {'K_a': np.identity(3),
                   'R_a': np.identity(3),
                   'b_a': np.zeros(3),
                   'K_g': np.identity(3),
                   'R_g': np.identity(3),
                   'K_ga': np.zeros((3, 3)),
                   'b_g': np.zeros(3)}
    return request.param(**sample_data)


@pytest.fixture()
def dummy_data():
    sample_acc = np.repeat(np.array([[0, 0, 1.]]), 100, axis=0)
    sample_gyro = np.repeat(np.array([[1, 1, 1.]]), 100, axis=0)
    return sample_acc, sample_gyro


def test_dummy_cal(dummy_cal, dummy_data):
    acc, gyro = dummy_cal.calibrate(*dummy_data)
    assert np.array_equal(acc, dummy_data[0])
    assert np.array_equal(gyro, dummy_data[1])


def test_dummy_cal_acc(dummy_cal, dummy_data):
    acc = dummy_cal.calibrate_acc(dummy_data[0])
    assert np.array_equal(acc, dummy_data[0])


def test_dummy_cal_gyro(dummy_cal, dummy_data):
    with pytest.warns(UserWarning) as rec:
        gyro = dummy_cal.calibrate_gyro(dummy_data[1])

    assert len(rec) == 1
    assert 'CalibrationInfo.calibrate' in str(rec[0])
    assert np.array_equal(gyro, dummy_data[1])
