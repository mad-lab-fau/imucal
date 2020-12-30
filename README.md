# IMU Calibration
[![PyPI](https://img.shields.io/pypi/v/imucal)](https://pypi.org/project/imucal/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/imucal)

This package provides methods to calculate and apply calibrations for IMUs based on multiple different methods.

So far supported are:

- Ferraris Calibration (Ferraris1995)
- Ferraris Calibration using a Turntable

## Installation

```
pip install imucal
```

To use the included calibration GUI you also need matplotlib (version >2.2).
You can install it using:

```
pip install imucal[calplot]
```

## Quickstart

To annotate a Ferraris calibration session that was recorded in a single go, you can use the following code snippet.
Note: This will open an interactive Tkinter plot.
Therefore, this will only work on your local PC and not on a server or remote hosted Jupyter instance.

### Ferraris

This package implements the IMU-infield calibration based on Ferraris1995.
This calibration methods requires the IMU data from 6 static positions (3 axis parallel and antiparallel to gravitation vector) and 3 rotations around the 3 main axis.
In this implementation these parts are referred to as follows `{acc,gry}_{x,y,z}_{p,a}` for the static regions and `{acc,gry}_{x,y,z}_rot` for the rotations.
As example, `acc_y_a` would be the 3D-acceleration data measured during a static phase, where the **y-axis** was oriented **antiparallel** to the gravitation.

#### Creating a new Calibration Object

If the data of all of these sections is already available separately as numpy arrays of the shape `(n x 3)`, where `n` is the number of samples in each section, they can be directly used to initialize a Calibration object:

```python
from imucal import ferraris_regions_from_interactive_plot

# Your data as a 6 column dataframe
data = ...

section_data, section_list = ferraris_regions_from_interactive_plot(
    data, acc_cols=["acc_x", "acc_y", "acc_z"], gyr_cols=["gyr_x", "gyr_y", "gyr_z"]
)
# Save the section list as reference for the future
section_list.to_csv('./calibration_sections.csv')  # This is optional, but recommended
```

Now you can perform the calibration
```python
from imucal import FerrarisCalibration

sampling_rate = 100 #Hz 
cal = FerrarisCalibration()
cal_mat = cal.compute(section_data, sampling_rate, from_acc_unit="m/s^2", from_gyr_unit="g")
# `cal_mat` is you final calibration matrix object, you can use to calibrate data
cal_mat.to_json_file('./calibration.json')
```

Applying a calibration:

```python
from imucal import load_calibration_info

cal_mat = load_calibration_info('./calibration.json')
new_data = ...
new_data_acc = new_data[["acc_x", "acc_y", "acc_z"]].to_numpy()
new_data_gyr = new_data[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
calibrated_acc, calibrated_gyr = cal_mat.calibrate(acc=new_data_acc, gyr=)
```
