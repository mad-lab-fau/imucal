r"""
.. _basic_ferraris:

Annotate a session and perform a Ferraris Calibration
=====================================================

The following example demonstrates the main workflow we will use when working with `imucal`.

We assume that you recorded a Ferraris session with you IMU unit following our :ref:`tutorial <ferraris_guide>`.
We further assume that you recorded all calibration steps in a single recording and still need to annotate the various
sections.
If you have already separated them, you can create a instance of :class:`~imucal.FerrarisSignalRegions` or use
:func:`~imucal.ferraris_regions_from_df`.

But let's start with the tutorial
"""

# %%
# Loading the session
# -------------------
# First we need to load the recorded session.
# This will depend on you sensor unit.
# In this example we have exported the sensor data as csv first, so that we can easily import it.
# Note, that the sensor data already has the units `m/s^2` and `deg/s` and not some raw bit values, which you might
# encounter when just streaming raw sensor values from a custom IMU board.
# We highly recommend to perform this conversion using the equations provided in the IMU chip documentation before
# applying a Ferraris calibration.
# This will ensure that you calibration is independent of the selected sensor ranges and you do not need to record a
# new calibration, when you change these settings.
from pathlib import Path

import pandas as pd

from example_data import EXAMPLE_PATH

data = pd.read_csv(EXAMPLE_PATH / "example_ferraris_session.csv", header=0, index_col=0)

data.head()

# %%
# Annotating the data
# -------------------
# Now we need to annotate the different sections of the ferraris calibration in the interactive GUI.
# Note, that this will only work, if your Python process runs on your machine and not some kind of sever.
# Otherwise, the GUI will not open.
#
# To start the annotation, run:
#
#   >>> from imucal import ferraris_regions_from_interactive_plot
#   >>> regions, section_list = ferraris_regions_from_interactive_plot(data)
#
# You can see in the gif below, how you can annotate the session.
#
# Some Notes:
#
# * If your data has other column names than `acc_x, ..., gyr_z`, you can provide them in the function call
# * If you have recorded the section in the correct order, you can just jump to the next section by pressing Enter
# * If you need to annotate/correct a specific section, click on it in the sidebar
# * If you use the zoom or pan tool of matplotlib, you need to deselect it again, before you can make annotations
# * Once you annotated all sections, simply close the plot
# * In the video you can see that the gyro rotations are split up in 4 sections of 90 deg instead of one 360 deg
#   rotation.
#   This was simply how the operator performed the rotation in this case.
#   But usually you would expect a single large spike there.
#
# .. raw:: html
#
#     <video width="500" controls>
#       <source src="../_static/videos/gui_guide.webm" type="video/webm">
#     Your browser does not support the video tag.
#     </video>
#
# Instead of performing the annotation in this example, we will load the section list from a previous annotation of the
# data.
# In general it is advisable to save the annotated sections, so that you can rerun the calibration in the future.
from imucal import ferraris_regions_from_section_list

section_list = pd.read_json(EXAMPLE_PATH / "example_ferraris_session_list.json").T

section_list

# %%
# This section list can then be used to recreate the regions
regions = ferraris_regions_from_section_list(data, section_list)
regions

# %%
# Now we can calculate the actual calibration parameters.
# For this we will create a instance of `FerrarisCalibration` with the desired settings and then call `compute` with
# the regions we have extracted.
#
# Note that we need to specify the units of the input data.
# This information is stored with the calibration as documentation to check in the future that data is suitable for a
# given calibration.
# We can further add a comment if we want.
from imucal import FerrarisCalibration

cal = FerrarisCalibration()
cal_info = cal.compute(
    regions, sampling_rate_hz=204.8, from_acc_unit="m/s^2", from_gyr_unit="deg/s", comment="My comment"
)


print(cal_info.to_json())

# %%
# The final `cal_info` now contains all information to calibrate future data recordings from the same sensor.
# For now we will save it to disk and then see how to load it again.
#
# Note, that we will use a temporary folder here.
# In reality you would chose some folder where you can keep the calibration files save until eternity.
import tempfile

d = tempfile.TemporaryDirectory()
d.name

# %%
# You can either use the `to_json_file` or `to_hdf5` methods of :class:`~imucal.FerrarisCalibration` directly ...

cal_info.to_json_file(Path(d.name) / "my_sensor_cal.json")

# %%
# ... or use the provided management tools to save the file in a predefined folder structure.
# Read more about this in the our :ref:`guide on that topic <cal_store_guide>`.
from imucal.management import save_calibration_info
from datetime import datetime

file_path = save_calibration_info(
    cal_info, sensor_id="imu1", cal_time=datetime(2020, 8, 12, 13, 21), folder=Path(d.name)
)
file_path

# %%
# In the latter case, we can use the helper functions :func:`~imucal.management.find_calibration_info_for_sensor`
# and :func:`~imucal.management.find_closest_calibration_info_to_date` to find the calibration again.
from imucal.management import find_calibration_info_for_sensor

cals = find_calibration_info_for_sensor("imu1", Path(d.name))
cals

# %%
# In any case, we can use :func:`~imucal.management.load_calibration_info` to load the calibration if we know the file
# path.
from imucal.management import load_calibration_info

loaded_cal_info = load_calibration_info(cals[0])
print(loaded_cal_info.to_json())

# %%
# After loading the calibration file, we will apply it to a "new" recording (we will just use the calibration session
# as example here).

calibrated_data = loaded_cal_info.calibrate_df(data, "m/s^2", "deg/s")

# %%
# We can see the effect of the calibration, when we plot the acc norm in the beginning of the recording.
# The calibrated values are now much closer to 9.81 m/s^2 compared to before the calibration.
import matplotlib.pyplot as plt
from numpy.linalg import norm

plt.figure()
plt.plot(norm(data.filter(like="acc"), axis=1)[500:1000], label="before cal")
plt.plot(norm(calibrated_data.filter(like="acc")[500:1000], axis=1), label="after cal")
plt.legend()
plt.ylabel("acc norm [m/s^2]")
plt.show()

# %%
# Finally, remove temp directory.
d.cleanup()
