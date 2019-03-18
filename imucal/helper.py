import numpy as np

from imucal.ferraris_calibration_info import FerrarisCalibrationInfo


def plot_calibration(data, calib_mat: FerrarisCalibrationInfo, fs: float):
    """Plot the FerrarisCalibration result uncalibrated vs calibrated.

    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: calibration matrices
    :param fs: samplingrate for integration
    """
    from matplotlib import pyplot as plt

    acc = data.loc[:, ['accX', 'accY', 'accZ']].values
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].values

    acc_calibrated, gyro_calibrated = calib_mat.calibrate(acc, gyro)

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


def check_calibration(data, calib_mat: FerrarisCalibrationInfo, points, fs: float):
    """
    Prints calibration relevant paramers (function may be deleted)
    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: object with calibration matrices (class FerrarisCalibrationInfo)
    :param points: pandas array with start and end of all fields in calibration data
    :param fs: sampling rate
    """
    from matplotlib import pyplot as plt

    acc = data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    acc_calibrated, gyro_calibrated = calib_mat.calibrate(acc, gyro)

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


def reverse_calibration(acc, gyro, calib_mat):
    """
    Reverse calibration of calibrated input data arrays acc, gyro
    :param calib_mat:
    :param acc: Acceleration x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :param gyro: Gyroscope x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :return: calibrated acceleration, calibrated gyroscope (numpy ndarray)
    """
    # TODO: Reimplement
    raise NotImplemented('Reverse FerrarisCalibration is currently not implemented')

    # # Precomputation of combined rotation/scaling matrix
    # accel_mat = np.matmul(calib_mat.K_g, calib_mat.R_g)
    # gyro_mat = np.matmul(calib_mat.K_a, calib_mat.R_a)
    #
    # # Initialize Calibrated arrays
    # acc_reverse = np.zeros(acc.shape)
    # gyro_reverse = np.zeros(gyro.shape)
    #
    # # Do reverse calibration calibration!
    # for i in np.arange(0, acc.shape[0]):
    #     acc_reverse[i, :] = np.transpose(np.matmul(accel_mat, np.transpose(acc[i, :]))) + calib_mat.b_a
    #     gyro_reverse[i, :] = np.transpose(np.matmul(gyro_mat, np.transpose(gyro[i, :]))) + calib_mat.b_g + np.transpose(
    #         np.matmul(calib_mat.K_ga, np.transpose([acc[i, :]])))
    #     # acc_calib[i,:]=np.transpose(np.matmul(accel_mat,(np.transpose([acc[i,:]])-b_a)))
    #     # gyro_calib[i,:]=np.transpose(np.matmul(gyro_mat,(np.transpose([gyro[i,:]])-np.matmul(K_ga,np.transpose([acc_calib[i,:]]))-b_g)))
    #
    # return acc_reverse, gyro_reverse
