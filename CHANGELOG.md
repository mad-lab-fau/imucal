# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section), and 
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [2.0] - 

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
- If you were using `from_json_file` before, double check, if this still works for you, as the way the correct baseclass
  is selected have been chosen.
  In any case, you should consider to use `imucal.management.load_calibration_info` instead, as it is more flexible.
- If you were using any parameters or package variables, that contained the short hand `gyro`, replace it with `gyr`.
  Note that this also effects the exported calibration files.
  You will not be able to load them unless you replace `gyro_unit` with `gyr_unit` in all files.
- The `CalibrationInfo` objects now have more fields by default.
  To avoid issues with missing values, we highly recommend recreating all calibration files you have using the original
  session data and the most current version of `imucal`

