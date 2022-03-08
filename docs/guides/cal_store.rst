.. _cal_store_guide:

=====================================================
Storing and Managing Calibrations of Multiple Sensors
=====================================================

When you build or use multiple sensors it is advisable to store all calibrations at a central location and with a
certain structure to ensure that the correct calibration is used every time.
Keep in mind that it is usually required to store multiple calibration files per sensor.
Over time and when measuring in vastly different environmental conditions, the calibration coefficients of the sensor
might change.
This management system should fulfill the following requirements:

* It should be easy to find calibrations belonging to a specific sensor
* It should be easy to add new calibrations
* It should be possible to find calibrations with certain requirements
* All users should have easy access to the calibrations and all updates


Storing Meta-information for Calibrations
========================================

To store the actual calibration information, you can simply use the :meth:`~imucal.CalibrationInfo.to_json_file` and
:meth:`~imucal.CalibrationInfo.to_hdf5` methods of the :class:`~imucal.CalibrationInfo` objects.
However, `imucal` does explicitly not include meta information of a calibration (besides the generic `comments` field),
as we can simply not anticipate what you might want to do with your sensors or calibrations.
To store such meta-information, you basically have three options:

1. Store meta-information separately and not in the actual calibration json/hdf5 file.
   You could put that information into a second file or encode it in the file or folder name.
2. Create a custom writer that extracts the information from the calibration-info objects and store them in your own
   file format that includes the meta-information you require.
   You could use the :meth:`~imucal.CalibrationInfo.to_json` method to get a json string that could be embedded in a
   larger json file.
   For many sensors it might also be feasible to ditch files entirely and write a custom interface to store all the
   information in a database.
3. Create a subclass of :class:`~imucal.CalibrationInfo` and include the meta-information you need as additional fields.
   This will ensure that this information is included in all json and hdf5 exports.
   This is covered in this example (TODO!).


In the following, we will first discuss which meta-information you might typically require and then explain an
implementation of option 1 (that we use internally as well) and provide further information, when the other cases might
be useful.

Which Meta-information Should I Save?
------------------------------------

Primarily, the meta-information is there to ensure that you apply the correct calibration to the correct recordings.
This means you should at least store the following:

* Unique identifier for your sensor.
* Sensor settings that might change calibration-relevant parameters.
  This will depend on your sensor, but some units have a different bias when certain settings are chosen.
* The datetime at which the calibration was performed.
* Special circumstances that might have effected the outcome of the calibration.

Based on this meta information, you should then be able to select the correct calibration for any recording.

Where to Store Meta-information?
--------------------------------

For a small team of tech-savvy users with a medium amount of sensors (<100) we suggest to encode all information in
file and folder names and to store the final calibration files as json (and the raw calibration-session recordings)
at a shared network location everyone can access.
This way, the calibration files can be easily searched with a simple file browser or a small script that performs
regex checks against the filenames.

The exact names and the way how you encode the information relevant to you in the filenames, is of course up to you.
For us, the two important pieces of information are the sensor ID and the datetime.
Therefore, we store the files using the following format:

`base_folder/{sensor_id}/{cal_info.CAL_TYPE}/{sensor_id}_%Y-%m-%d_%H-%M.json`

Note that we include the calibration type in the folder structure as well.
This is because we usually have turntable and manual calibration for many sensors.
As these two methods might have different accuracy, we decided that this is an important piece of information that you
should get immediately when looking through the folder structure manually.
If you want to use the same structure as we, you can simply use the function
:func:`~imucal.management.save_calibration_info` from `imucal.management` .

Besides storing a json export of the actual calibration matrices, we recommend storing the raw data of the calibration
session and the session-list produced by :func:`~imucal.ferraris_regions_from_interactive_plot` (if used).
We store these files in a separate file tree with an identical naming scheme.
Further, we maintain a small script that can recreate all calibration files based on these raw data.

To give you some further inside into how we manage these files: We actually keep all calibration files under version
control.
We have two repositories. The first stores the raw calibration session and session list (using `git-lfs` for the raw
sessions).
The second repository contains just the final calibrations and is installable as a python package.
This makes it easy to install and keep it as a fixed version dependency for a data analysis project.
When we want to update the calibrations, we push a new raw session and its annotation to the first repository.
Through continuous integration this triggers the calculation of the actual calibration object, which is then
automatically exported as json and pushed to the second repository.

Note that our solution works for our specific use case and our specific work environment, which consists of tech-savvy
users that are part of our team, and use Python and git as art of their workflow anyway.
If you want to provide calibration files to end users, we would recommend the second option listed above and create a
database to store all calibrations.
Whatever type of end user interface you deploy for your customers can then access this database.

The third option, extending the :class:`~imucal.CalibrationInfo` class, is only recommended if you have pieces of
meta-information that fundamentally change how a calibration should be applied (i.e., similar to the expected units
of the input data) or are actually required as part of the calibration procedure.
If you do that, make sure that you provide a new `CAL_TYPE` value for the subclass and use it when calculating the
calibration.
Otherwise, loading the stored files with :func:`~imucal.management.load_calibration_info` is not possible.

Finding Calibration Files
=========================

If you followed our example and stored meta information encoded in file and folder names, you can use simple regex
searches to find calibrations that fulfill your specifications.
To make that even easier for you, we provide the functions :func:`~imucal.management.find_calibration_info_for_sensor`
and :func:`~imucal.management.find_closest_calibration_info_to_date` to simplify the two most typical queries, namely,
finding all calibrations for a sensor and finding the calibration that was performed closest (timewise)
to the actual recording.
Both functions further allow you to filter based on the calibration type and provide a custom validation to check
parameters inside the calibration file (e.g. the expected units of the input data).
Note that these functions expect you to store the calibrations using :func:`~imucal.management.save_calibration_info`.

In general, it is the best to use a calibration that was recorded as close as possible before the actual recording.
However, it will depend on your application and tolerances which criteria you should use.


Further Notes
=============

Our unique identifier for the sensors is based on the internal mac-addresses of the bluetooth chip.
While this sounds like a good choice initially, there are situations where we expect the calibration information to
change.
To name a few: A new IMU-chip is soldered onto the board, the sensor board is transferred into a new enclosure
(this does not change scaling factors, but might change the expected directions relative to the casing), the board was
damaged and resoldered in a reflow oven, etc.

With these things in mind, we would advise to maintain an additional version number that really uniquely identifies
a sensor unit/configuration and not just a sensor board.
