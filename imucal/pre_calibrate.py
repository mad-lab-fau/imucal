import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_prepro_data(imu_data, debug_plot=False):
    """
    Prepares the calibration data for the later calculation of calibration matrices.

    Operations manual:
    Use the move cursor and a double click to place a label at the plot.
    Accelerometer: Place the labels where the data is steady.
                   Two labels for each position +x,-x,+y,-y,+z,-z. That makes in total 12 labels.
    Gyroscope:     Place the labels where the sensor is rotated around a axis.
                   Two labels for each axis. Makes 6 in total.
    The space between the labels of a single position is kept, everything else is discarded.

    :param imu_data: pandas data frame with accX/Y/Z, gyroX/Y/Z
    :param debug_plot: set true to see, whether data cutting was successful
    """

    acc = imu_data.loc[:, ['accX', 'accY', 'accZ']].as_matrix()
    gyro = imu_data.loc[:, ['gyroX', 'gyroY', 'gyroZ']].as_matrix()

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
    test = 0

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
            plt.show()

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
            plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

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
