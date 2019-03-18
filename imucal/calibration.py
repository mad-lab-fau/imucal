from typing import Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
from imucal import calibration_info as cm
from imucal.calibration_info import CalibrationInfo


class Calibration:
    acc_x_p: np.ndarray
    acc_x_a: np.ndarray
    acc_y_p: np.ndarray
    acc_y_a: np.ndarray
    acc_z_p: np.ndarray
    acc_z_a: np.ndarray
    gyr_x_p: np.ndarray
    gyr_x_a: np.ndarray
    gyr_y_p: np.ndarray
    gyr_y_a: np.ndarray
    gyr_z_p: np.ndarray
    gyr_z_a: np.ndarray

    acc_x_rot: np.ndarray
    acc_y_rot: np.ndarray
    acc_z_rot: np.ndarray
    gyr_x_rot: np.ndarray
    gyr_y_rot: np.ndarray
    gyr_z_rot: np.ndarray

    _fields = (
        'acc_x_p', 'acc_x_a', 'acc_y_p', 'acc_y_a', 'acc_z_p', 'acc_z_a', 'gyr_x_p', 'gyr_x_a', 'gyr_y_p', 'gyr_y_a',
        'gyr_z_p', 'gyr_z_a', 'acc_x_rot', 'acc_y_rot', 'acc_z_rot', 'gyr_x_rot', 'gyr_y_rot', 'gyr_z_rot'
    )

    sampling_rate: float
    grav: float

    EXPECTED_ANGLE: float = 360.

    def __init__(self, sampling_rate: float, grav: Optional[float] = 9.81, **kwargs) -> None:
        for field in self._fields:
            setattr(self, field, kwargs.get(field, None))

        self.sampling_rate = sampling_rate
        self.grav = grav
        super().__init__()

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                sampling_rate: float,
                grav: Optional[float] = 9.81,
                acc_cols: Optional[Iterable[str]] = ('acc_x', 'acc_y', 'acc_z'),
                gyro_cols: Optional[Iterable[str]] = ('acc_x', 'acc_y', 'acc_z')
                ):
        # TODO: need proper documentation

        acc_df = df[list(acc_cols)]
        gyro_df = df[list(gyro_cols)]
        acc_dict = acc_df.groupby(level=0).apply(lambda x: x.values).to_dict()
        gyro_dict = gyro_df.groupby(level=0).apply(lambda x: x.values).to_dict()
        acc_dict = {'acc_' + k: v for k, v in acc_dict.items()}
        gyro_dict = {'gyr_' + k: v for k, v in gyro_dict.items()}

        return cls(sampling_rate, grav, **acc_dict, **gyro_dict)

    def compute_calibration_matrix(self) -> CalibrationInfo:
        cal_mat = CalibrationInfo()

        ###############################################################################################################
        # Compute Acceleration Matrix

        # Calculate means from all static phases and stack them into 3x3 matrices
        # Note: Each measurement should be a column
        U_a_p = np.vstack((
            np.mean(self.acc_x_p, axis=0),
            np.mean(self.acc_y_p, axis=0),
            np.mean(self.acc_z_p, axis=0),
        )).T
        U_a_n = np.vstack((
            np.mean(self.acc_x_a, axis=0),
            np.mean(self.acc_y_a, axis=0),
            np.mean(self.acc_z_a, axis=0),
        )).T

        # Eq. 19
        U_a_s = U_a_p + U_a_n

        # Bias Matrix
        # Eq. 20
        B_a = U_a_s / 2

        # Bias Vector
        b_a = np.diag(B_a)
        cal_mat.b_a = b_a

        # Compute Scaling and Rotation
        # No need for bias correction, since it cancels out!
        # Eq. 21
        U_a_d = U_a_p - U_a_n

        # Calculate Scaling matrix
        # Eq. 23
        k_a_sq = 1 / (4 * self.grav ** 2) * np.diag(U_a_d @ U_a_d.T)
        K_a = np.diag(np.sqrt(k_a_sq))
        cal_mat.K_a = K_a

        # Calculate Rotation matrix
        # Eq. 22
        R_a = inv(K_a) @ U_a_d / (2 * self.grav)
        cal_mat.R_a = R_a

        ###############################################################################################################
        # Calculate Gyroscope Matrix

        # Gyro Bias from the static phases of the acc calibration
        # One static phase would be sufficient, but why not use all of them if you have them.
        # Note that this calibration ignores any influences due to the earth rotation.

        b_g = np.mean(np.vstack((
            self.gyr_x_p,
            self.gyr_x_a,
            self.gyr_y_p,
            self.gyr_y_a,
            self.gyr_z_p,
            self.gyr_z_a,
        )), axis=0)

        cal_mat.b_g = b_g

        # Acceleration sensitivity

        # Note: Each measurement should be a column? or should it
        U_g_p = np.vstack((
            np.mean(self.gyr_x_p, axis=0),
            np.mean(self.gyr_y_p, axis=0),
            np.mean(self.gyr_z_p, axis=0),
        )).T
        U_g_a = np.vstack((
            np.mean(self.gyr_x_a, axis=0),
            np.mean(self.gyr_y_a, axis=0),
            np.mean(self.gyr_z_a, axis=0),
        )).T

        # Eq. 9
        K_ga = (U_g_p - U_g_a) / (2 * self.grav)
        cal_mat.K_ga = K_ga

        # Gyroscope Scaling and Rotation

        # First apply partial calibration to remove offset and acc influence
        acc_x_rot_cor = cal_mat.calibrate_acc(self.acc_x_rot)
        acc_y_rot_cor = cal_mat.calibrate_acc(self.acc_y_rot)
        acc_z_rot_cor = cal_mat.calibrate_acc(self.acc_z_rot)
        gyr_x_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_x_rot, acc_x_rot_cor)
        gyr_y_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_y_rot, acc_y_rot_cor)
        gyr_z_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_z_rot, acc_z_rot_cor)

        # Integrate gyro readings
        # Eg. 13/14
        W_s = np.zeros((3, 3))
        W_s[:, 0] = np.sum(gyr_x_rot_cor, axis=0) / self.sampling_rate
        W_s[:, 1] = np.sum(gyr_y_rot_cor, axis=0) / self.sampling_rate
        W_s[:, 2] = np.sum(gyr_z_rot_cor, axis=0) / self.sampling_rate

        # Eq.15
        expected_angles = self.EXPECTED_ANGLE * np.identity(3)
        multiplied = W_s @ inv(expected_angles)

        # Eq. 12
        k_g_sq = np.diag(multiplied @ multiplied.T)
        K_g = np.diag(np.sqrt(k_g_sq))
        cal_mat.K_g = K_g

        R_g = inv(K_g) @ multiplied
        cal_mat.R_g = R_g

        return cal_mat


# Plot the Calibration result uncalibrated vs calibrated
def plotCalibration(data, calib_mat, fs):
    """
    Plots the calibration
    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: calibration matrices
    :param fs: samplingrate for integration
    """

    data_calibrated = calibrate_array(data, calib_mat)

    acc = data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    acc_calibrated = data_calibrated.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro_calibrated = data_calibrated.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    # Compute angle measures
    angles_original = np.zeros(gyro_calibrated.shape)
    angles_calibrated = np.zeros(gyro_calibrated.shape)
    for i in np.arange(0, acc_calibrated.shape[0]):
        if i == 0:
            angles_original[i, :] = angles_original[i, :] * 1 / fs
            angles_calibrated[i, :] = angles_calibrated[i, :] * 1 / fs
        else:
            angles_original[i, :] = angles_original[i - 1, :] + gyro[i, :] * 1 / fs
            angles_calibrated[i, :] = angles_calibrated[i - 1, :] + gyro_calibrated[i, :] * 1 / fs

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(acc)
    plt.grid(True)
    plt.title("Accelerometer uncalibrated")
    plt.xlabel("idx")
    plt.ylabel("acceleration [m/s^2]")
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(acc_calibrated)
    plt.grid(True)
    plt.title("Accelerometer calibrated")
    plt.xlabel("idx")
    plt.ylabel("acceleration [m/s^2]")

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(gyro)
    plt.grid(True)
    plt.title("Gyroscope uncalibrated")
    plt.xlabel("idx")
    plt.ylabel("angular rate [deg/s]")
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(gyro_calibrated)
    plt.grid(True)
    plt.title("Gyroscope calibrated")
    plt.xlabel("idx")
    plt.ylabel("angular rate [deg/s]")

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(angles_original)
    plt.grid(True)
    plt.title("Integrated angles uncalibrated")
    plt.xlabel("idx")
    plt.ylabel("angle [deg/s]")
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(angles_calibrated)
    plt.grid(True)
    plt.title("Integrated angles calibrated")
    plt.xlabel("idx")
    plt.ylabel("angle [deg/s]")

    # plt.show()

    return


# Compute integration results and check if they make sense
def checkCalibration(data, calib_mat, points, fs):
    """
    Prints calibration relevant paramers (function may be deleted)
    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: object with calibration matrices (class CalibrationInfo)
    :param points: pandas array with start and end of all fields in calibration data
    :param fs: sampling rate
    """

    acc = data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    acc_calibrated, gyro_calibrated = calibrate_array(acc, gyro, calib_mat)

    # Compute angle measures
    angles_original = np.zeros(gyro_calibrated.shape)
    angles_calibrated = np.zeros(gyro_calibrated.shape)
    for i in np.arange(0, acc_calibrated.shape[0]):
        if i == 0:
            angles_original[i, :] = angles_original[i, :] * 1 / fs
            angles_calibrated[i, :] = angles_calibrated[i, :] * 1 / fs
        else:
            angles_original[i, :] = angles_original[i - 1, :] + gyro[i, :] * 1 / fs
            angles_calibrated[i, :] = angles_calibrated[i - 1, :] + gyro_calibrated[i, :] * 1 / fs

    acc_xp = np.mean(acc_calibrated[points.loc['accX+', 'Start']:points.loc['accX+', 'End'], :], axis=0)
    acc_xn = np.mean(acc_calibrated[points.loc['accX-', 'Start']:points.loc['accX-', 'End'], :], axis=0)
    acc_yp = np.mean(acc_calibrated[points.loc['accY+', 'Start']:points.loc['accY+', 'End'], :], axis=0)
    acc_yn = np.mean(acc_calibrated[points.loc['accY-', 'Start']:points.loc['accY-', 'End'], :], axis=0)
    acc_zp = np.mean(acc_calibrated[points.loc['accZ+', 'Start']:points.loc['accZ+', 'End'], :], axis=0)
    acc_zn = np.mean(acc_calibrated[points.loc['accZ-', 'Start']:points.loc['accZ-', 'End'], :], axis=0)

    angle_rot_x = angles_calibrated[points.loc['gyroX', 'End'], :] - angles_calibrated[points.loc['gyroX', 'Start'], :]
    angle_rot_y = angles_calibrated[points.loc['gyroY', 'End'], :] - angles_calibrated[points.loc['gyroY', 'Start'], :]
    angle_rot_z = angles_calibrated[points.loc['gyroZ', 'End'], :] - angles_calibrated[points.loc['gyroZ', 'Start'], :]

    print(acc_xp)
    print(acc_xn)
    print(acc_yp)
    print(acc_yn)
    print(acc_zp)
    print(acc_zn)

    print(angle_rot_x)
    print(angle_rot_y)
    print(angle_rot_z)


def reverseCalibration(acc, gyro, calib_mat):
    """
    Reverse calibration of calibrated input data arrays acc, gyro
    :param calib_mat:
    :param acc: Acceleration x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :param gyro: Gyroscope x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :return: calibrated acceleration, calibrated gyroscope (numpy ndarray)
    """

    # Precomputation of combined rotation/scaling matrix
    accel_mat = np.matmul(calib_mat.K_g, calib_mat.R_g)
    gyro_mat = np.matmul(calib_mat.K_a, calib_mat.R_a)

    # Initialize Calibrated arrays
    acc_reverse = np.zeros(acc.shape)
    gyro_reverse = np.zeros(gyro.shape)

    # Do reverse calibration calibration!
    for i in np.arange(0, acc.shape[0]):
        acc_reverse[i, :] = np.transpose(np.matmul(accel_mat, np.transpose(acc[i, :]))) + calib_mat.b_a
        gyro_reverse[i, :] = np.transpose(np.matmul(gyro_mat, np.transpose(gyro[i, :]))) + calib_mat.b_g + np.transpose(
            np.matmul(calib_mat.K_ga, np.transpose([acc[i, :]])))
        # acc_calib[i,:]=np.transpose(np.matmul(accel_mat,(np.transpose([acc[i,:]])-b_a)))
        # gyro_calib[i,:]=np.transpose(np.matmul(gyro_mat,(np.transpose([gyro[i,:]])-np.matmul(K_ga,np.transpose([acc_calib[i,:]]))-b_g)))

    return acc_reverse, gyro_reverse
