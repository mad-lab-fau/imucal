# IMU Calibration based on Ferraris

Use Ferraris Calibration Script:

Calibration Sessions need to be located in the folder: "../Calibration_Sessions/"

python calibration_script.py -f "NilsPodX-922A_28_01_2019-11-00-29.bin"

Double-Click to set marker positions => 6 segments (12 marker for Acc)
									 => 3 segments (6 marker for Gyro)

Close Plot => Calibration Parameters are calculated automatically and printed in console + saved in hdf5