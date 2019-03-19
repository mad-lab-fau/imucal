# IMU Calibration

## Installation

HTTPS (this will ask you for your Gitlab username and pw):
```
pip install git+https://mad-srv.informatik.uni-erlangen.de/lo94zeny/sensorcalibration.git
```

SSH (this will ask you for your SSH-key pw, if set):
```
pip install git+ssh://git@mad-srv.informatik.uni-erlangen.de/lo94zeny/sensorcalibration.git
```

## Ferraris

This package implements the IMU-infield calibration based on Ferraris1995.
This calibration methods requires the IMU data from 6 static positions (3 axis parallel and antiparallel to gravitation vector) and 3 rotations around the 3 main axis.
In this implementation these parts are referred to as follows `{acc,gry}_{x,y,z}_{p,a}` for the static regions and `{acc,gry}_{x,y,z}_rot` for the rotations.
As example, `acc_y_a` would be the 3D-acceleration data measured during a static phase, where the **y-axis** was oriented **antiparallel** to the gravitation.

### Creating a new Calibration Object

If the data of all of these sections is already available separately as numpy arrays of the shape `(n x 3)`, where `n` is the number of samples in each section, they can be directly used to initialize a Calibration object:

```python
from imucal import FerrarisCalibration

sampling_rate = 100 #Hz 
cal = FerrarisCalibration(sampling_rate=sampling_rate,
                          acc_x_p=data_acc_x_p,
                          ...,
                          gyr_z_rot=data_gyr_x_rot)
```
or via class variables:
```python
from imucal import FerrarisCalibration

sampling_rate = 100 #Hz 
cal = FerrarisCalibration(sampling_rate=sampling_rate)
cal.gyr_z_a = data_gyr_z_a
...
```

If the data was recorded as a single continuous stream, we first need to identify the different regions in the data.
If the regions were recorded in the correct order (`'x_p', 'x_a', 'y_p', 'y_a', 'z_p', 'z_a', 'x_rot', 'y_rot', 'z_rot'`) the `from_interactive_plot` method can be used to extract them manually.
For this and for moth other methods, we expect the data to be a `pd.DataFrame` with the columns `'acc_x', 'acc_y', 'acc_z, 'gyr_x', 'gyr_y', 'gyr_z'`, where each column represents the datastream of one sensor axis.

Note: The expected column names can be overwritten

```python
from imucal import FerrarisCalibration

sampling_rate = 100 #Hz 
# This will open an interactive plot, where you can select the start and the stop sample of each region
cal, section_list = FerrarisCalibration.from_interactive_plot(data, sampling_rate=sampling_rate)

section_list.to_csv('./calibration_sections.csv')  # This is optional, but recommended
```

This method also returns the `section_list` in addition to the Calibration object.
It is advised to save this list, as it can be used recreate the Calibration without performing the manual selection of the regions again:

```python
# At some other day:
from imucal import FerrarisCalibration
import pandas as pd

section_list = pd.read_csv('./calibration_sections.csv', index_col=0)
sampling_rate = 100 #Hz 
# This will recreate the calibration
cal = FerrarisCalibration.from_section_list(data, section_list, sampling_rate=sampling_rate)
```

### Performing the Calibration

When the calibration object was successful initialized, you can obtain the calibration by simply calling `compute_calibration_matrix`:

```python
cal_mat = cal.compute_calibration_matrix()
print(cal_mat)
```

This will return an `FerrarisCalibrationInfo` object, which holds all required calibration information.
This object can be saved and loaded to and from `hdf5` and `json` using the respective methods:

```python
cal_mat.to_json_file('./calibration.json')
# some other day:
from imucal import FerrarisCalibrationInfo

cal_mat = FerrarisCalibrationInfo.from_json_file('./calibration.json') 
```

```python
cal_mat.to_hdf5('./calibration.h5')
# some other day:
from imucal import FerrarisCalibrationInfo

cal_mat = FerrarisCalibrationInfo.from_hdf5('./calibration.h5') 
```

### Applying the Calibration

The `FerrarisCalibrationInfo` object can be used to apply the Calibration to new data from the same sensor:

```python
calibrated_acc, calibrated_gyro = cal_mat.calibrate(acc, gyro)
```

`acc` and `gyro` are expected to be numpy arrays in the shape (n x 3)