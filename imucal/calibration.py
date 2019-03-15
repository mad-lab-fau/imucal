import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
from imucal import calibration_info as cm


def calibrate_array(data, calib_mat):
    """Calibration of input data arrays acc, gyro

    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: calibration matrices
    :return: calibrated acceleration, calibrated gyroscope (numpy ndarray)
    """

    data_calib = data.copy()

    acc = data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    # Combine Scaling and rotation matrix to one matrix
    accel_mat = np.matmul(inv(calib_mat.R_a), inv(calib_mat.K_a))
    gyro_mat = np.matmul(inv(calib_mat.R_g), inv(calib_mat.K_g))

    # Initialize Calibrated arrays
    acc_calib = np.zeros(acc.shape)
    gyro_calib = np.zeros(gyro.shape)

    # Do calibration!
    for i in np.arange(0, acc.shape[0]):
        acc_calib[i, :] = np.transpose(np.matmul(accel_mat, (np.transpose([acc[i, :]]) - calib_mat.b_a)))
        gyro_calib[i, :] = np.transpose(np.matmul(gyro_mat, (np.transpose([gyro[i, :]]) - np.matmul(calib_mat.K_ga,
                                                                                                    np.transpose([
                                                                                                        acc_calib[
                                                                                                        i,
                                                                                                        :]])) - calib_mat.b_g)))

    data_calib.loc[:, ['accX', 'accY', 'accZ']] = acc_calib
    data_calib.loc[:, ['gyroX', 'gyroY', 'gyroZ']] = gyro_calib

    return data_calib


def calibrate_sample(data, calib_mat):
    """
    Calibration of single  acc, gyro
    :param data: pandas Dataframe with columns [accX, accY, accZ, gyroX. gyroY, gyroZ]
    :param calib_mat: object with calibration matrices
    :return: calibrated acceleration, calibrated gyroscope (numpy ndarray)
    """

    data_calib = data.copy()

    acc = data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

    # Combine Scaling and Rotation matrix
    accel_mat = np.matmul(inv(calib_mat.K_a), inv(calib_mat.R_a))
    gyro_mat = np.matmul(inv(calib_mat.K_g), inv(calib_mat.R_g))
    # Calibration step
    acc_calib = np.transpose(np.matmul(accel_mat, (np.transpose([acc]) - calib_mat.b_a)))
    gyro_calib = np.transpose(np.matmul(gyro_mat, (
            np.transpose([gyro]) - np.matmul(calib_mat.K_ga, np.transpose([acc_calib])) - calib_mat.b_g)))

    data_calib.loc[:, ['accX', 'accY', 'accZ']] = acc_calib
    data_calib.loc[:, ['gyroX', 'gyroY', 'gyroZ']] = gyro_calib

    return data_calib


# Compute calibration matrix
def compute_calibration_matrix(X_p, X_a, Y_p, Y_a, Z_p, Z_a, Rot_X, Rot_Y, Rot_Z, fs):
    # The calibration consists of mainly two parts
    # 1. Stationary phase: Turn sensor on each side and measure 1g
    # 2. Rotation phase: Rotate sensor around each axis (counterclockwise, positive axis pointing upwards)

    # Order data
    # Take mean of stationary phases
    # Save acc and gyro data of rotation phases
    accX_p = np.mean(X_p[:, :3], axis=0)
    accX_a = np.mean(X_a[:, :3], axis=0)
    accY_p = np.mean(Y_p[:, :3], axis=0)
    accY_a = np.mean(Y_a[:, :3], axis=0)
    accZ_p = np.mean(Z_p[:, :3], axis=0)
    accZ_a = np.mean(Z_a[:, :3], axis=0)
    gyroX_p = np.mean(X_p[:, 3:6], axis=0)
    gyroX_a = np.mean(X_a[:, 3:6], axis=0)
    gyroY_p = np.mean(Y_p[:, 3:6], axis=0)
    gyroY_a = np.mean(Y_a[:, 3:6], axis=0)
    gyroZ_p = np.mean(Z_p[:, 3:6], axis=0)
    gyroZ_a = np.mean(Z_a[:, 3:6], axis=0)
    gyroRot_X = Rot_X[:, 3:6]
    gyroRot_Y = Rot_Y[:, 3:6]
    gyroRot_Z = Rot_Z[:, 3:6]
    accRot_X = Rot_X[:, :3]
    accRot_Y = Rot_Y[:, :3]
    accRot_Z = Rot_Z[:, :3]

    # ACCELERATION MATRIX COMPUTATIONs ###############################################################################

    # COMPUTATION OF ACCELERATION BIAS
    # Build up measurement matrices
    Ua_plus = np.concatenate((np.transpose([accX_p]), np.transpose([accY_p]), np.transpose([accZ_p])), axis=1)
    Ua_minus = np.concatenate((np.transpose([accX_a]), np.transpose([accY_a]), np.transpose([accZ_a])), axis=1)

    # Compute Bias matrix
    B_a = (Ua_plus + Ua_minus) / 2

    # Bias vector is vector on diagonal
    b_a = np.transpose([np.diagonal(B_a, offset=0)])

    # COMPUTATION OF ACCELERATION SCALING AND ROTATION MATRIX
    # No need for bias correction, since it cancels out!

    one_g = 9.81
    # equation (21)
    # ToDo: Transpose
    U_aD = Ua_plus - Ua_minus

    # equation(23)
    K_a_square = 1 / (4 * one_g * one_g) * np.diagonal(np.matmul(U_aD, np.transpose(U_aD)))

    K_a = np.zeros((3, 3))
    for j in range(0, 3):
        K_a[j, j] = np.sqrt(K_a_square[j])

    # Rotation estimation
    R_a = np.matmul(inv(K_a), U_aD) / 2 / one_g

    # Added: newly implemented, for further computations, acceleration values need to be translated to calibrated values
    accel_mat = np.matmul(inv(R_a), inv(K_a))
    # Initialize Calibrated arrays
    accRot_X_calib = np.zeros(accRot_X.shape)
    accRot_Y_calib = np.zeros(accRot_Y.shape)
    accRot_Z_calib = np.zeros(accRot_Z.shape)

    for i in np.arange(0, accRot_X_calib.shape[0]):
        accRot_X_calib[i, :] = np.transpose(np.matmul(accel_mat, (np.transpose([accRot_X[i, :]]) - b_a)))
    for i in np.arange(0, accRot_Y_calib.shape[0]):
        accRot_Y_calib[i, :] = np.transpose(np.matmul(accel_mat, (np.transpose([accRot_Y[i, :]]) - b_a)))
    for i in np.arange(0, accRot_Z_calib.shape[0]):
        accRot_Z_calib[i, :] = np.transpose(np.matmul(accel_mat, (np.transpose([accRot_Z[i, :]]) - b_a)))

    # GYROSCOPE MATRIX COMPUTATIONs ###############################################################################

    # COMPUTATION OF GYRO BIAS
    gyro_stat = np.concatenate(([gyroX_p], [gyroX_a], [gyroY_p], [gyroY_a], [gyroZ_p], [gyroZ_a]), axis=0)

    b_0x = np.mean(gyro_stat[:, 0], axis=0)
    b_0y = np.mean(gyro_stat[:, 1], axis=0)
    b_0z = np.mean(gyro_stat[:, 2], axis=0)

    b_g = np.transpose([np.array([b_0x, b_0y, b_0z])])

    # COMPUTATION OF ACCELERATION SENSITIVITY MATRIX
    # Here I do no bias correction, because in the calibration step later on, the gyroscope is also not debiased

    k_g_ax = (gyroX_p - gyroX_a) / (2 * one_g)
    k_g_ay = (gyroY_p - gyroY_a) / (2 * one_g)
    k_g_az = (gyroZ_p - gyroZ_a) / (2 * one_g)

    K_ga = np.concatenate(([k_g_ax], [k_g_ay], [k_g_az]), axis=0)

    # COMPUTATION OF GYROSCOPE SCALING AND ROTATION MATRIX

    # desired output
    desired_angles = 360 * np.eye(3)

    # We need to correct the acceleration sensitivity and the bias before computing angles measured in the calibraton procedure
    gyroRot_x_corrected = np.zeros(gyroRot_X.shape)
    gyroRot_y_corrected = np.zeros(gyroRot_Y.shape)
    gyroRot_z_corrected = np.zeros(gyroRot_Z.shape)

    # get rid of offset and acc influence
    for i in np.arange(0, gyroRot_X.shape[0]):
        gyroRot_x_corrected[i, :] = np.transpose(
            np.transpose([gyroRot_X[i, :]]) - b_g - np.matmul(K_ga, np.transpose([accRot_X_calib[i, :]])))

    for i in np.arange(0, gyroRot_Y.shape[0]):
        gyroRot_y_corrected[i, :] = np.transpose(
            np.transpose([gyroRot_Y[i, :]]) - b_g - np.matmul(K_ga, np.transpose([accRot_Y_calib[i, :]])))

    for i in np.arange(0, gyroRot_Z.shape[0]):
        gyroRot_z_corrected[i, :] = np.transpose(
            np.transpose([gyroRot_Z[i, :]]) - b_g - np.matmul(K_ga, np.transpose([accRot_Z_calib[i, :]])))

    # real_angles: uncalibrated angles that result from integration of individual gyroscope axes
    real_angles = np.zeros((3, 3))
    for i in np.arange(0, 3):
        # x angles
        if i == 0:
            for j in np.arange(0, gyroRot_X.shape[0]):
                real_angles[0, i] = real_angles[0, i] + 1 / fs * gyroRot_x_corrected[j, 0]
                real_angles[1, i] = real_angles[1, i] + 1 / fs * gyroRot_x_corrected[j, 1]
                real_angles[2, i] = real_angles[2, i] + 1 / fs * gyroRot_x_corrected[j, 2]
        # y angles
        elif i == 1:
            for j in np.arange(0, gyroRot_Y.shape[0]):
                real_angles[0, i] = real_angles[0, i] + 1 / fs * gyroRot_y_corrected[j, 0]
                real_angles[1, i] = real_angles[1, i] + 1 / fs * gyroRot_y_corrected[j, 1]
                real_angles[2, i] = real_angles[2, i] + 1 / fs * gyroRot_y_corrected[j, 2]
        # z angles
        elif i == 2:
            for j in np.arange(0, gyroRot_Z.shape[0]):
                real_angles[0, i] = real_angles[0, i] + 1 / fs * gyroRot_z_corrected[j, 0]
                real_angles[1, i] = real_angles[1, i] + 1 / fs * gyroRot_z_corrected[j, 1]
                real_angles[2, i] = real_angles[2, i] + 1 / fs * gyroRot_z_corrected[j, 2]

    # equation (12)/(15)
    # equation(23) -> scaling matrix
    multiplied = np.matmul(real_angles, inv(desired_angles))
    K_g_square = np.diagonal(np.matmul(multiplied, np.transpose(multiplied)))

    K_g = np.zeros((3, 3))
    for j in range(0, 3):
        K_g[j, j] = np.sqrt(K_g_square[j])

    # -> rotation matrix
    R_g = np.matmul(inv(K_g), multiplied)

    calib_mat = cm.CalibrationInfo(K_a, R_a, b_a, K_g, R_g, K_ga, b_g)

    return calib_mat


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


def save_calibration_data(filename, acc, gyro, calib_mat):
    """
    Save both uncalibrated and calibrated data to filename. Saved files contain accx, accY, accZ, gyroX, gyroY, gyroZ
    :param filename to save, calibrated and uncalibrated will be added
    :param acc: Acceleration x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :param gyro: Gyroscope x,y,z (numpy ndarray, first dimension samples, second dimension different x,y,z)
    :param calib_mat: object with calibration matrices (CalibrationInfo    """

    acc_calib, gyro_calib = calibrate_array(acc, gyro, calib_mat)

    test = np.concatenate((acc_calib, gyro_calib), axis=1)
    test2 = pd.DataFrame(data=test, columns=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'])
    test2.to_csv(filename + '_calibratedData.csv', sep=';', encoding='utf-8', index=False)

    test = np.concatenate((acc, gyro), axis=1)
    test2 = pd.DataFrame(data=test, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    test2.to_csv(filename + '_uncalibratedData.csv', sep=';', encoding='utf-8', index=False)
