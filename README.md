# imucal - A Python library to calibrate 6 DOF IMUs
![Test and Lint](https://github.com/mad-lab-fau/imucal/workflows/Test%20and%20Lint/badge.svg)
[![codecov](https://codecov.io/gh/mad-lab-fau/imucal/branch/master/graph/badge.svg?token=0OPHTQDYIB)](https://codecov.io/gh/mad-lab-fau/imucal)
[![Documentation Status](https://readthedocs.org/projects/imucal/badge/?version=latest)](https://imucal.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/imucal)](https://pypi.org/project/imucal/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/imucal)
[![status](https://joss.theoj.org/papers/3dd1a7dd5ba06ce024326eee2e9be148/status.svg)](https://joss.theoj.org/papers/3dd1a7dd5ba06ce024326eee2e9be148)

This package provides methods to calculate and apply calibrations for 6 DOF IMUs based on multiple different methods.

So far supported are:

- Ferraris Calibration ([Ferraris1994](https://www.sciencedirect.com/science/article/pii/0924424794800316) / [Ferraris1995](https://www.researchgate.net/publication/245080041_Calibration_of_three-axial_rate_gyros_without_angular_velocity_standards))
- Ferraris Calibration using a Turntable

For more information check the quickstart guide below and the [documentation](https://imucal.readthedocs.io/en/latest/).

## Installation

```
pip install imucal
```

To use the included calibration GUI you also need [matplotlib](https://pypi.org/project/matplotlib/) (version >2.2).
You can install it using:

```
pip install imucal[calplot]
```

### Supported Python versions and Platforms

`imucal` is officially tested on Python 3.7, 3.8, 3.9 and 3.10.
It should further work with all major operating systems.

## Quickstart
This package implements the IMU-infield calibration based on [Ferraris1995](https://www.researchgate.net/publication/245080041_Calibration_of_three-axial_rate_gyros_without_angular_velocity_standards).
This calibration method requires the IMU data from 6 static positions (3 axes parallel and antiparallel to the gravitation
vector) for calibrating the accelerometer and 3 rotations around the 3 main axes for calibrating the gyroscope.
In this implementation, these parts are referred to as `{acc,gyr}_{x,y,z}_{p,a}` for the static regions and
`{acc,gyr}_{x,y,z}_rot` for the rotations.
As example, `acc_y_a` would be the 3D-acceleration data measured during a static phase, where the **y-axis** was 
oriented **antiparallel** to the gravitation vector.
For more information on how to perform the calibration check [our guide](https://imucal.readthedocs.io/en/latest/guides/ferraris_guide.html).

For the calibration, you need to separate your data into these individual sections.
If you already recorded them separately or know where each section starts and ends in a continuous recording, you can 
use [`ferraris_regions_from_df`](https://imucal.readthedocs.io/en/latest/modules/generated/imucal.ferraris_regions_from_df.html)
and [`ferraris_regions_from_section_list`](https://imucal.readthedocs.io/en/latest/modules/generated/imucal.ferraris_regions_from_section_list.html),
respectively to convert the data into the correct format for the calibration (`section_data` in the snippet below).

If you don't have that information yet, we recommend to use the included GUI to annotate the data.
To annotate a Ferraris calibration session that was recorded in a single go, you can use the following code snippet.  
**Note**: This will open an interactive Tkinter plot. Therefore, this will only work on your local PC and not on a server or remote hosted Jupyter instance.

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

Independent of how you obtained the `section_data` in the correct format, you can now calculate the calibration
parameters:

```python
from imucal import FerrarisCalibration

sampling_rate = 100 #Hz 
cal = FerrarisCalibration()
cal_mat = cal.compute(section_data, sampling_rate, from_acc_unit="m/s^2", from_gyr_unit="deg/s")
# `cal_mat` is your final calibration matrix object you can use to calibrate data
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
[Examples](https://imucal.readthedocs.io/en/latest/auto_examples/index.html).

## Further Calibration Methods

At the moment, this package only implements calibration methods based on Ferraris1994/95, because this is what we use to
calibrate our IMUs.
We are aware that various other methods exist and would love to add them to this package as well.
Unfortunately, at the moment we can not justify the time investment.

Still, we think that this package provides a suitable framework to implement other calibration methods with relative
ease.
If you would like to contribute such a method, let us know via [GitHub Issue](https://github.com/mad-lab-fau/imucal/issues), and we will try to help you as good
as possible.

## Citation

If you are using `imucal` in your scientific work, we would appreciate if you would cite our [JOSS paper](https://joss.theoj.org/papers/3dd1a7dd5ba06ce024326eee2e9be148) or link the project.

```
Küderle, Arne, Nils Roth, Robert Richer, and Bjoern M. Eskofier. 
“Imucal - A Python Library to Calibrate 6 DOF IMUs.”
Journal of Open Source Software 7, no. 73 (May 26, 2022): 4338. https://doi.org/10.21105/joss.04338.
```

## Contributing

All project management and development happens through [this GitHub project](https://github.com/mad-lab-fau/imucal).
If you have any issues, ideas, or any comments at all, just open a new issue.
We are always happy when people are interested to use our work and would like to support you in this process.
In particular, we want to welcome contributions of new calibration algorithms, to make this package even more useful for a wider audience.

## Dev Setup

We use [poetry](https://python-poetry.org) to manage our dependencies.
Therefore, you need to first install Poetry locally on you machine.

Then you can run the following command to install a local development version of this library in a dedicated venv.

```bash
git clone https://github.com/mad-lab-fau/imucal
cd imucal
poetry install --all-extras
```

To run tests/the linter/... we use [poethepoet](https://github.com/nat-n/poethepoet).
You can see all available commands by running:

```
poetry run poe list
```

This should show you all configured commands:

```
CONFIGURED TASKS
  format         
  lint           Lint all files with ruff.
  ci_check       Check all potential format and linting issues.
  test           Run Pytest with coverage.
  docs           Build the html docs using Sphinx.
  bump_version  
```

You execute any command by running

```
poetry run doit <command-name>
```

### Updating dependencies

If you update or add dependencies using (`poetry add` or `poetry update`) you will see that the `pyproject.toml` and the `poetry.lock` files are both updated.
Make sure you commit the changes to **both** files.
Otherwise, wrong versions of dependencies will be used in the CI and by other developers.

In case you update dependencies by directly editing the `pyproject.toml` file, you need to be very careful and make sure, you run `poetry lock [--no-update]` afterwards.
Otherwise, the lock file will be out of date.

In general, it is a good idea to just run `poetry update` from time to time.
This will install the latest version of all dependencies that are still allowed by the version constrains in the `pyproject.toml`.
This allows to check, if everything still works well with the newest versions of all libraries.
