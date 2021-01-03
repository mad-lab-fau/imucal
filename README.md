# IMU Calibration
![Test and Lint](https://github.com/mad-lab-fau/imucal/workflows/Test%20and%20Lint/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/imucal/badge/?version=latest)](https://imucal.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/imucal)](https://pypi.org/project/imucal/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/imucal)

This package provides methods to calculate and apply calibrations for IMUs based on multiple different methods.

So far supported are:

- Ferraris Calibration (Ferraris1995)
- Ferraris Calibration using a Turntable

## WARNING: VERSION UPDATE

Version 2.0 was recently released and contains multiple API breaking changes!
To learn more about that, check `Changelog.md`.

If you want to ensure that your old code still works, specify a correct version during install and in your
`requirement.txt` files

```
pip install "imucal<2.0"
```

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
This package implements the IMU-infield calibration based on Ferraris1995.
This calibration methods requires the IMU data from 6 static positions (3 axis parallel and antiparallel to gravitation
vector) and 3 rotations around the 3 main axis.
In this implementation these parts are referred to as follows `{acc,gry}_{x,y,z}_{p,a}` for the static regions and
`{acc,gry}_{x,y,z}_rot` for the rotations.
As example, `acc_y_a` would be the 3D-acceleration data measured during a static phase, where the **y-axis** was 
oriented **antiparallel** to the gravitation.

To annotate a Ferraris calibration session that was recorded in a single go, you can use the following code snippet.
Note: This will open an interactive Tkinter plot.
Therefore, this will only work on your local PC and not on a server or remote hosted Jupyter instance.

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
from imucal.management import load_calibration_info

cal_mat = load_calibration_info('./calibration.json')
new_data = pd.DataFrame(...)
calibrated_data = cal_mat.calibrate_df(new_data, acc_unit="m/s^2", gyr_unit="deg/s")
```

For further information on how to perform a calibration check the 
[User Guides](https://imucal.readthedocs.io/en/latest/guides/index.html) or the
[examples](https://imucal.readthedocs.io/en/latest/auto_examples/index.html)

## Further Calibration Methods

At the moment, this package only implements calibration methods based on Ferraris1994, because this is what we use to
calibrate our IMUs.
We are aware that various other methods exist and would love to add them to this package as well.
Unfortunately, at the moment we can not justify the time requirement.

Still, we think that this package provides a suitable framework to implement other calibration emthods with relative
easy.
If you would like to contribute such a method, let us know on the github-issue page and we will try to help you as good
as possible.

## Contributing

All project management and development happens through this Github project.
If you have any issues, ideas, or any comments at all, just open a new issue.
Please be polite and considerate of our time.
We appreciate everyone who is using our software or even wants to improve it, but sometime other things come in the way,
and it takes us a couple of days to get back to you.
