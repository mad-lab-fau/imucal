"""A library to calibrate 6 DOF IMUs."""

from imucal.calibration_info import CalibrationInfo
from imucal.ferraris_calibration import (
    FerrarisCalibration,
    FerrarisSignalRegions,
    TurntableCalibration,
    ferraris_regions_from_df,
    ferraris_regions_from_interactive_plot,
    ferraris_regions_from_section_list,
)
from imucal.ferraris_calibration_info import FerrarisCalibrationInfo, TurntableCalibrationInfo

__version__ = "2.6.0"

__all__ = [
    "CalibrationInfo",
    "FerrarisCalibration",
    "TurntableCalibration",
    "FerrarisSignalRegions",
    "FerrarisCalibrationInfo",
    "TurntableCalibrationInfo",
    "ferraris_regions_from_df",
    "ferraris_regions_from_interactive_plot",
    "ferraris_regions_from_section_list",
]
