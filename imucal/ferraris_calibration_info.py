"""Wrapper object to hold calibration matrices for a Ferraris Calibration."""
from dataclasses import dataclass
from typing import Tuple, ClassVar, Optional

import numpy as np

from imucal.calibration_info import CalibrationInfo


@dataclass(eq=False)
class FerrarisCalibrationInfo(CalibrationInfo):
    """Calibration object that represents all the required information to apply a Ferraris calibration to a dataset.

    Parameters
    ----------
    K_a :
        Scaling matrix for the acceleration
    R_a :
        Rotation matrix for the acceleration
    b_a :
        Acceleration bias
    K_g :
        Scaling matrix for the gyroscope
    R_g :
        Rotation matrix for the gyroscope
    K_ga :
        Influence of acceleration on gyroscope
    b_g :
        Gyroscope bias

    """

    CAL_TYPE: ClassVar[str] = "Ferraris"  # noqa: invalid-name

    acc_unit: str = "m/s^2"
    gyr_unit: str = "deg/s"
    K_a: Optional[np.ndarray] = None  # noqa: invalid-name
    R_a: Optional[np.ndarray] = None  # noqa: invalid-name
    b_a: Optional[np.ndarray] = None
    K_g: Optional[np.ndarray] = None  # noqa: invalid-name
    R_g: Optional[np.ndarray] = None  # noqa: invalid-name
    K_ga: Optional[np.ndarray] = None  # noqa: invalid-name
    b_g: Optional[np.ndarray] = None

    _cal_paras: ClassVar[Tuple[str, ...]] = ("K_a", "R_a", "b_a", "K_g", "R_g", "K_ga", "b_g")

    def calibrate(
        self, acc: np.ndarray, gyr: np.ndarray, acc_unit: Optional[str], gyr_unit: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate the accelerometer and the gyroscope.

        This corrects:
            acc: scaling, rotation, non-orthogonalities, and bias
            gyro: scaling, rotation, non-orthogonalities, bias, and acc influence on gyro

        Parameters
        ----------
        acc :
            3D acceleration
        gyr :
            3D gyroscope values
        acc_unit
            The unit of the acceleration data
        gyr_unit
            The unit of the gyroscope data

        Returns
        -------
        Corrected acceleration and gyroscope values

        """
        # Check if all required paras are initialized to throw appropriate error messages:
        for v in self._cal_paras:
            if getattr(self, v, None) is None:
                raise ValueError(
                    "{} need to initialised before an acc calibration can be performed. {} is missing.".format(
                        self._cal_paras, v
                    )
                )
        self._validate_units(acc_unit, gyr_unit)
        acc_out = self._calibrate_acc(acc)
        gyro_out = self._calibrate_gyr(gyr, acc_out)

        return acc_out, gyro_out

    def _calibrate_acc(self, acc: np.ndarray) -> np.ndarray:
        """Calibrate the accelerometer.

        This corrects scaling, rotation, non-orthogonalities and bias.

        Parameters
        ----------
        acc :
            3D acceleration

        Returns
        -------
        Calibrated acceleration

        """
        # Check if all required paras are initialized to throw appropriate error messages:
        paras = ("K_a", "R_a", "b_a")
        for v in paras:
            if getattr(self, v, None) is None:
                raise ValueError(
                    "{} need to initialised before an acc calibration can be performed. {} is missing".format(paras, v)
                )

        # Combine Scaling and rotation matrix to one matrix
        acc_mat = np.linalg.inv(self.R_a) @ np.linalg.inv(self.K_a)
        acc_out = acc_mat @ (acc - self.b_a).T

        return acc_out.T

    def _calibrate_gyr(self, gyr, calibrated_acc=None):
        # Check if all required paras are initialized to throw appropriate error messages:
        required = ["K_g", "R_g", "b_g"]
        if calibrated_acc is not None:
            required += ["K_ga"]
        for v in required:
            if getattr(self, v, None) is None:
                raise ValueError(
                    "{} need to initialised before an gyro calibration can be performed. {} is missing".format(
                        required, v
                    )
                )
        # Combine Scaling and rotation matrix to one matrix
        gyro_mat = np.matmul(np.linalg.inv(self.R_g), np.linalg.inv(self.K_g))
        tmp = self._calibrate_gyr_offsets(gyr, calibrated_acc)

        gyro_out = gyro_mat @ tmp.T
        return gyro_out.T

    def _calibrate_gyr_offsets(self, gyr, calibrated_acc=None):
        if calibrated_acc is None:
            d_ga = np.array(0)
        else:
            d_ga = self.K_ga @ calibrated_acc.T
        offsets = d_ga.T + self.b_g
        return gyr - offsets


class TurntableCalibrationInfo(FerrarisCalibrationInfo):
    """Calibration object that represents all the required information to apply a Turntable calibration to a dataset.

    A Turntable calibration is identical to a Ferraris calibration.
    However, because the parameters are calculated using a calibration table instead of hand rotations,
    higher precision is expected.

    Parameters
    ----------
    K_a :
        Scaling matrix for the acceleration
    R_a :
        Rotation matrix for the acceleration
    b_a :
        Acceleration bias
    K_g :
        Scaling matrix for the gyroscope
    R_g :
        Rotation matrix for the gyroscope
    K_ga :
        Influence of acceleration on gyroscope
    b_g :
        Gyroscope bias

    """

    CAL_TYPE = "Turntable"
