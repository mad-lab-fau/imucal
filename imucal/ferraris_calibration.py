from typing import Optional, Iterable

import numpy as np
import pandas as pd
from numpy.linalg import inv

from imucal.ferraris_calibration_info import FerrarisCalibrationInfo

FERRARIS_SECTIONS = (
    'acc_x_p', 'acc_x_a', 'acc_y_p', 'acc_y_a', 'acc_z_p', 'acc_z_a', 'gyr_x_p', 'gyr_x_a', 'gyr_y_p', 'gyr_y_a',
    'gyr_z_p', 'gyr_z_a', 'acc_x_rot', 'acc_y_rot', 'acc_z_rot', 'gyr_x_rot', 'gyr_y_rot', 'gyr_z_rot'
)


class FerrarisCalibration:
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

    _fields = FERRARIS_SECTIONS

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

    def compute_calibration_matrix(self) -> FerrarisCalibrationInfo:
        cal_mat = FerrarisCalibrationInfo()

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


def find_calibration_sections_interactive(acc: np.ndarray, gyro: np.ndarray, debug_plot: Optional[bool] = False):
    """
    Prepares the calibration data for the later calculation of calibration matrices.

    Operations manual:
    Use the move cursor and a double click to place a label at the plot.
    Accelerometer: Place the labels where the data is steady.
                   Two labels for each position +x,-x,+y,-y,+z,-z. That makes in total 12 labels.
    Gyroscope:     Place the labels where the sensor is rotated around a axis.
                   Two labels for each axis. Makes 6 in total.
    The space between the labels of a single position is kept, everything else is discarded.

    :param acc: numpy array with the shape (n, 3) where n is the number of samples
    :param gyro: numpy array with the shape (n, 3) where n is the number of samples
    :param debug_plot: set true to see, whether data cutting was successful
    """
    from matplotlib import pyplot as plt

    # remove the unnecessary data
    allData = np.concatenate((np.array(acc), np.array(gyro)), axis=1)

    # plot the data
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(211)
    plt.plot(acc)
    plt.grid(True)
    plt.title("Set a label at start/end of accelerometer placements (12 in total)")
    ticks = np.arange(0, allData.shape[0], 2000)
    plt.xticks(ticks, ticks // 200)
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [m/s^2]")
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(gyro)
    plt.grid(True)
    plt.title("Set a label at start/end of gyroscope rotation (6 in total)")
    plt.xlabel("time[s]")
    plt.ylabel("rotation [°/s]")

    acc_list_markers = []
    gyro_list_markers = []
    list_labels = []

    def onclick(event):
        # switch to the move cursor
        # set a marker with doubleclick left
        # remove the last marker with doubleclick right

        # with button 1 (double left click) you will set a marker
        if event.button == 1 and event.dblclick:
            x = int(event.xdata)
            list_labels.append(x)
            marker_acc = ax1.axvline(x)
            marker_gyro = ax2.axvline(x)
            acc_list_markers.append(marker_acc)
            gyro_list_markers.append(marker_gyro)

        # with button 3 (double right click) you will remove a marker
        elif event.button == 3 and event.dblclick:
            # position of the last marker
            x = list_labels[-1]
            a = acc_list_markers.pop()
            g = gyro_list_markers.pop()
            # remove the last marker
            a.remove(), g.remove()
            del a, g
            list_labels.remove(x)
        fig.canvas.draw()
        fig.canvas.flush_events()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    # sort the labels in ascending order
    list_labels.sort()

    # use the labels to cut out the unnecessary data and add a column with the part
    X_p = allData[list_labels[0]:list_labels[1], :]
    X_p = np.column_stack((X_p, np.array([1] * X_p.shape[0])))
    X_a = allData[list_labels[2]:list_labels[3], :]
    X_a = np.column_stack((X_a, np.array([2] * X_a.shape[0])))
    Y_p = allData[list_labels[4]:list_labels[5], :]
    Y_p = np.column_stack((Y_p, np.array([3] * Y_p.shape[0])))
    Y_a = allData[list_labels[6]:list_labels[7], :]
    Y_a = np.column_stack((Y_a, np.array([4] * Y_a.shape[0])))
    Z_p = allData[list_labels[8]:list_labels[9], :]
    Z_p = np.column_stack((Z_p, np.array([5] * Z_p.shape[0])))
    Z_a = allData[list_labels[10]:list_labels[11], :]
    Z_a = np.column_stack((Z_a, np.array([6] * Z_a.shape[0])))
    Rot_X = allData[list_labels[12]:list_labels[13], :]
    Rot_X = np.column_stack((Rot_X, np.array([7] * Rot_X.shape[0])))
    Rot_Y = allData[list_labels[14]:list_labels[15], :]
    Rot_Y = np.column_stack((Rot_Y, np.array([8] * Rot_Y.shape[0])))
    Rot_Z = allData[list_labels[16]:list_labels[17], :]
    Rot_Z = np.column_stack((Rot_Z, np.array([9] * Rot_Z.shape[0])))

    # put all of it together again
    list_parts = [X_p, X_a, Y_p, Y_a, Z_p, Z_a, Rot_X, Rot_Y, Rot_Z]
    pre_pro_data = np.concatenate(list_parts, axis=0)
    pre_pro_data = pd.DataFrame(pre_pro_data, columns=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'part'])

    if debug_plot:
        # plot the resulting data
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 2)
        ax3 = fig.add_subplot(gs[0, 0])
        ax3.plot(pre_pro_data.iloc[:, 0:3])
        ax3.set_title('Preprocessed Accelerometer Data')
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.plot(pre_pro_data.iloc[:, 3:6])
        ax4.set_title('Preprocessed Gyroscope Data')
        ax5 = fig.add_subplot(gs[1, :])
        ax5.plot(pre_pro_data.iloc[:, 0:6])
        ax5.set_title('Preprocessed Sensor Data')

    return pre_pro_data
