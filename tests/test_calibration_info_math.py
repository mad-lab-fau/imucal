import numpy as np
import pandas as pd


def test_dummy_cal(dummy_cal, dummy_data):
    acc, gyro = dummy_cal.calibrate(*dummy_data, None, None)
    assert np.array_equal(acc, dummy_data[0])
    assert np.array_equal(gyro, dummy_data[1])


def test_dummy_cal_df(dummy_cal, dummy_data):
    dummy_df = pd.DataFrame(np.column_stack(dummy_data))
    dummy_df.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    result_df = dummy_cal.calibrate_df(dummy_df, None, None)
    assert np.array_equal(dummy_data[0], result_df.filter(like="acc"))
    assert np.array_equal(dummy_data[1], result_df.filter(like="gyr"))


def test_ka_ba_ra(dummy_cal, dummy_data):
    dummy_cal.K_a *= 2
    dummy_cal.b_a += 2
    dummy_cal.R_a = np.flip(dummy_cal.R_a, axis=1)
    acc, _ = dummy_cal.calibrate(*dummy_data, None, None)
    assert np.all(acc == [-0.5, -1, -1])


def test_kg_rg_bg(dummy_cal, dummy_data):
    dummy_cal.K_g *= 2
    dummy_cal.b_g += 2
    dummy_cal.R_g = np.array([[0, 0, 1], [0, 1, 0], [0.5, 0, 0]])
    _, gyro = dummy_cal.calibrate(*dummy_data, None, None)
    assert np.all(gyro == [-1, -0.5, -0.5])


def test_kga(dummy_cal, dummy_data):
    dummy_cal.K_g *= 2
    dummy_cal.b_g += 2
    dummy_cal.R_g = np.array([[0, 0, 1], [0, 1, 0], [0.5, 0, 0]])
    dummy_cal.K_ga = np.identity(3)
    _, gyro = dummy_cal.calibrate(*dummy_data, None, None)
    assert np.all(gyro == [-2, -0.5, -0.5])
