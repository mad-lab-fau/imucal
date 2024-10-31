# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section), and 
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [2.5.0] - 31.10.2024

- Tooling overhaul to make the development process easier and fix doc building issues
- A bunch of quality of life improvements for the GUI.
  You can now use shortcuts (o, p, h, b, and f) to activate the matplib tools for zooming, panning, home, back, and forward.
  Once zoomed in, you can use the scrollwheel to scroll along the x-axis.
  The first section is also activated by default now, so you can start labeling right away.
- Updated minimal versions of dependencies. Notable: pandas >=2.0, python >=3.9

# [2.4.0] - 26.05.2023

- Tooling overhaul to make the development process easier and fix doc building issues
- Changed some ValueError to TypeError

# [2.3.1] - 17.10.2022

- Fixed import so that no tkinter or matplotlib are required when the GUI is not required

# [2.3.0] - 17.10.2022

- Removed upper version bounds to reduce the chance of version conflicts

# [2.2.1] - 02.05.2022

- Some minor updates to README.md
- Improved the doc building process: No additional requirements.txt file!
- JOSS paper accepted! :)

# [2.2.0] - 26.04.2022

- Dropped support for `h5py` < 3.0.0. This should only affect users using the hdf5 export feature.
  Dropping old versions allows for proper support of Python 3.10.

# [2.1.1] - 04.04.2022

- Looser version requirements for typing-extensions

# [2.1.0] - 08.03.2022

- switched from `distutils` to `packaging` for version handling.
- Added a paper describing the basics of imucal for submission in JOSS
- Fixed small issues in README and docs
- manual gui test is working again
- dependency updates

# [2.0.2] - 02.11.2021

### Changed

- Minor typos
- Zenodo Release
- Coverage badge

# [2.0.1] - 02.03.2021

### Changed

- Made sure that tkinter is only imported, if the GUI is really used.
  This is important if the library is used in a context were no graphical output is possible (e.g. a Docker container).
  Previous versions of imucal would result in an import error in these situations.

# [2.0.0] - 09.01.2021 

2.0 is a rewrite of a lot of the API and requires multiple changes to legacy code using this library.
Please refer to the migration guide for more information.
During this refactoring multiple bugs were fixed as well.
Therefore, it is highly suggested upgrading to the new version, even if it takes some work!

### Added

- A new `calibrate_df` method for the `CalibrationInfo` that can calibrate df directly.
- It is now possible to define which `CalibrationInfo` subclass should be used by the `FerrarisCalibration`
- A set of "management" functions to save, find, and load IMU calibrations of multiple sensors
- The ability to add a custom comment to a `CalibrationInfo`
- The user is now forced to provide the units of the input data to avoid applying calibrations that were meant for unit
  conversion.
- Applying a calibration now checks if the units of your data match with the input unit of the calibration.
- The export format of calibration-info objects is now versioned.
  This helps to make changes to the format in the future while still supporting old exports.
  See the migration guide for more information.
- Helper functions to load "legacy" calibration info objects. (`imucal.legacy`)

### Changed

- `FerrarisCalibration` has a new interface.
  Instead of providing all calibration and data related parameters in the `__init__`, the `__init__` is now only used
  to configure the calibration.
  The data and all data related parameter are now passed to the `compute` method (replaces `compute_calibration_matrix`)
- Using `from_...` constructors on a subclass of `CalibrationInfo` does not search all subclasses of `CalibrationInfo`
  anymore, but only the subclasses (and the class itself), it is called on.
  For example, `FerrarisCalibrationInfo.from_json` will only consider subclasses of `FerrarisCalibrationInfo`, but not
  other subclasses of `CalibrationInfo`.
- The short hand "gyro" is not replaced with "gyr" in all parameter and variable names.
  This might cause an issue when loading old calibration files.

### Deprecated

### Removed

- `FerrarisCalibration` does not have any `from_...` constructors anymore.
  The functionality of these constructors can now be accessed via the `ferraris_regions_from_...` helper functions.
- It is not possible anymore to calibrate the acc and gyro separately.
  No one was using this feature, and hence, was removed to simplify the API.

### Fixed

### Migration Guide

- The main change is how `FerrarisCalibration` and `TurntableCalibration` are used.
  Before you would do:
  ```python
  from imucal import FerrarisCalibration

  cal, section_list = FerrarisCalibration.from_interactive_plot(data, sampling_rate=sampling_rate)
  cal_info = cal.compute_calibration_matrix()
  ```

  Now you need to first create your Ferraris sections and then provide them as arguments for the `compute` method:

  ```python
  from imucal import FerrarisCalibration, ferraris_regions_from_interactive_plot

  sections, section_list = ferraris_regions_from_interactive_plot(data)
  cal = FerrarisCalibration()
  cal_info = cal.compute(sections, sampling_rate_hz=sampling_rate, from_acc_unit="m/s^2", from_gyr_unit="deg/s")
  ```
  
  Note, that you are also forced to provide the units of the input data.
  We always recommend to first turn your data into the same units you would expect after the calibration and then using
  the calibrations as refinement.
- `CalibrationInfot.calibrate` now requires you to specify the units of your data and validates that they match the 
  units expected by the calibration.
  You need to add the parameters `acc_unit` and `gyr_unit` to all calls to calibrate.
  Note, that older calibrations will not have `from` units and hence, can not perform this check.
  In this case you can set the units to `None` to avoid an error.
  However, it is recommended to recreate the calibrations with proper `from` units.
- If you were using `from_json_file` before, double check, if this still works for you, as the way the correct baseclass
  is selected have been chosen.
  In any case, you should consider to use `imucal.management.load_calibration_info` instead, as it is more flexible.
- If you were using any parameters or package variables, that contained the short hand `gyro`, replace it with `gyr`.
  Note that this also effects the exported calibration files.
  You will not be able to load them unless you replace `gyro_unit` with `gyr_unit` in all files.
- The `CalibrationInfo` objects now have more fields by default.
  To avoid issues with missing values, we highly recommend recreating all calibration files you have using the original
  session data and the most current version of `imucal`.
  Alternatively you can use the functions provided in `imucal.legacy` to load the old
  calibration.
  Then you can modify the loaded calibration info object and save it again to replace the old calibration.

