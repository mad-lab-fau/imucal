from dataclasses import dataclass

import numpy as np
import pytest

from imucal import FerrarisCalibrationInfo, TurntableCalibrationInfo


@pytest.fixture()
def sample_cal_dict():
    sample_data = {
        "K_a": np.array([[208.54567264, 0.0, 0.0], [0.0, 208.00113412, 0.0], [0.0, 0.0, 214.78455365]]),
        "R_a": np.array(
            [
                [0.99991252, 0.00712206, -0.01114566],
                [-0.00794738, 0.99968874, 0.0236489],
                [0.0213429, -0.01078188, 0.99971407],
            ]
        ),
        "b_a": np.array([-6.01886802, -48.28787402, -28.96636637]),
        "K_g": np.array([[16.67747318, 0.0, 0.0], [0.0, 16.18769383, 0.0], [0.0, 0.0, 16.25326253]]),
        "R_g": np.array(
            [
                [9.99918368e-01, 3.38399869e-04, -1.27727091e-02],
                [-5.19256254e-03, 9.99269158e-01, 3.78706515e-02],
                [1.28516088e-02, -3.63520887e-02, 9.99256404e-01],
            ]
        ),
        "K_ga": np.array(
            [
                [0.00229265, 0.01387371, -0.00925911],
                [-0.01613463, 0.00544361, 0.00850631],
                [0.01846544, -0.00881248, -0.00393538],
            ]
        ),
        "b_g": np.array([1.9693536, -4.46624421, -3.65097072]),
        "acc_unit": "custom_acc_unit",
        "gyr_unit": "custom_gyro_unit",
        "from_acc_unit": "custom_from_acc_unit",
        "from_gyr_unit": "custom_from_gyr_unit",
        "comment": "my custom comment",
    }
    return sample_data


@dataclass(eq=False)
class CustomFerraris(FerrarisCalibrationInfo):
    CAL_TYPE = "Custom Ferraris"
    custom_field: str = "default_custom_value"


@pytest.fixture(params=(FerrarisCalibrationInfo, TurntableCalibrationInfo, CustomFerraris))
def sample_cal(sample_cal_dict, request):
    info_class = request.param
    if info_class == CustomFerraris:
        sample_cal_dict["custom_field"] = "custom_value"
    return info_class(**sample_cal_dict)
