import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import datareader.datareader as dr
import imucal.ferraris_calibration
import imucal.helper
from imucal import ferraris_calibration as cal
from NilsPodLib import session as sensor

import optparse

parser = optparse.OptionParser()

parser.add_option('-f', '--file',
                  action="store", dest="filename",
                  help="-f [filename]", default="")

parser.add_option('-r', '--samplingrate',
                  action="store", dest="samplingrate",
                  help="-r [samplingrate Hz]", default=102.4)

options, args = parser.parse_args()

plt.close('all')

# Read the uncalibrated data
# subject = '11'
# path = 'D:/Data\Distance_Running_Data_Set/Subject' + subject + '/FerrarisCalibration/'
# filename = 'right.csv'   # ToDo S05, left
save = True

path = "./Calibration_Sessions/"

sensorType = "NilsPod"
file_name = str(options.filename)
samplingRate_Hz = float(options.samplingrate)

if sensorType is "NilsPod":
    dataset = sensor.dataset(path + file_name)
    imu_data = pd.DataFrame(np.concatenate((dataset.acc.data, dataset.gyro.data), axis=-1),
                            columns=["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"])

if sensorType is "miPodV3":
    data = np.loadtxt(open(path + file_name, "rb"), delimiter=",", skiprows=21 + 8)
    imu_data = pd.DataFrame(data[:, 2:9], columns=["accX", "accY", "accZ", "temp", "gyroX", "gyroY", "gyroZ"])

# imu_data = [dataset.acc.data, dataset.gyro.data]

# imu_data = dr.read_mipod(path+filename, only_6D_imu=True)

# get and read the preprocessed_data
allData = imucal.ferraris_calibration.find_calibration_sections_interactive(imu_data, debug_plot=True)

# Bring data in data structure for the calibration
X_p = np.array(allData[allData['part'] == 1])[:, :6]
X_a = np.array(allData[allData['part'] == 2])[:, :6]
Y_p = np.array(allData[allData['part'] == 3])[:, :6]
Y_a = np.array(allData[allData['part'] == 4])[:, :6]
Z_p = np.array(allData[allData['part'] == 5])[:, :6]
Z_a = np.array(allData[allData['part'] == 6])[:, :6]

Rot_X = np.array(allData[allData['part'] == 7])[:, :6]
Rot_Y = np.array(allData[allData['part'] == 8])[:, :6]
Rot_Z = np.array(allData[allData['part'] == 9])[:, :6]

# Compute calibration matrices
calib_mat = cal.compute_calibration_matrix(X_p, X_a, Y_p, Y_a, Z_p, Z_a, Rot_X, Rot_Y, Rot_Z, fs=samplingRate_Hz)

# save calibration matrices
# cal.save_calibration_data(path + 'data', acc, gyro, calib_mat)

print('My Matrices:')
calib_mat.print()

# Plot the calibration. In this step, the calibration working phase is called
imucal.helper.plot_calibration(allData, calib_mat, fs=samplingRate_Hz)

# check whether saving works
if save:
    calib_mat.save_to_hdf5(path + file_name + '_calib_mat.h5')

plt.show()
