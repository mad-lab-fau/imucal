r"""
.. _custom_info_class:

Custom Calibration Info Subclass
================================

Whenever you want to implement your own version of a calibration or simply want to add some meta information to your
calibration info objects, you need to create a new subclass of :class:`~imucal.CalibrationInfo`.

When creating a entirely new calibration, you should subclass from :class:`~imucal.CalibrationInfo` directly.
If you just want to extend an existing object, subclass from the respective class.
"""

# %%
# In the following, we will see how to extend the :class:`~imucal.FerrarisCalibrationInfo`.
#
# Here are the important things to keep in mind:
#
# 1. Each subclass needs to overwrite `CAL_TYPE`. Otherwise it is not recognised as a separate class when loading
#    objects from file.
# 2. The new class needs to inherit from `dataclass`, if you add fields that should be serialized when using the export
#    methods.
# 3. All new attributes need Type annotation and a default value (if in doubt use None)
# 4. Note, that all new attributes need to be json serializable by default, if you want ot use json export.
from dataclasses import dataclass
from typing import Optional

from imucal import FerrarisCalibrationInfo, ferraris_regions_from_df


@dataclass
class ExtendedFerrarisCalibrationInfo(FerrarisCalibrationInfo):
    CAL_TYPE = "ExtendedFerraris"
    new_meta_info: Optional[str] = None


# %%
# With that we have a new subclass of the ferraris calibration info.
# As it is marked as dataclass, the `__init__` is created automatically and saving and loading from file is also taken
# care of.
# To use the new class, we need to provide it when initializing our `FerrarisCalibration`.
from imucal import FerrarisCalibration

cal = FerrarisCalibration(calibration_info_class=ExtendedFerrarisCalibrationInfo)

# %%
# Now, we can calculate the calibration as normal (here we just use some dummy data).
# To provide a value for our `new_meta_info` field, we can pass it directly to the `calculate` method.
import pandas as pd

cal_data = ferraris_regions_from_df(pd.read_csv("../example_data/annotated_session.csv", header=0, index_col=[0, 1]))

cal_info = cal.compute(
    cal_data, sampling_rate_hz=204.8, from_acc_unit="a.u.", from_gyr_unit="a.u.", new_meta_info="my value"
)

cal_info.new_meta_info

# %%
# And of course we can simply export and reimport the new calibration info to json or hdf5 (we will use a tempfile
# here to not clutter the example folder).
import tempfile
from pathlib import Path
from imucal.management import load_calibration_info

with tempfile.TemporaryDirectory() as d:
    file_name = Path(d) / "my_cal_info.json"
    cal_info.to_json_file(file_name)

    # An load it again
    loaded_cal_info = load_calibration_info(file_name)

loaded_cal_info.new_meta_info
