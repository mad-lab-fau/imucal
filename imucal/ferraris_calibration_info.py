"""Wrapper object to hold calibration matrices for a Ferraris Calibration."""
import warnings
from typing import Tuple

import numpy as np

from imucal.calibration_info import CalibrationInfo


# TODO: Add example to docstring
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

    CAL_TYPE = "Ferraris"
    acc_unit = "m/s^2"
    gyro_unit = "deg/s"
    K_a: np.ndarray
    R_a: np.ndarray
    b_a: np.ndarray
    K_g: np.ndarray
    R_g: np.ndarray
    K_ga: np.ndarray
    b_g: np.ndarray

    _fields = ("K_a", "R_a", "b_a", "K_g", "R_g", "K_ga", "b_g")

    __doc__ += CalibrationInfo._cal_type_explanation

    def calibrate_acc(self, acc: np.ndarray) -> np.ndarray:
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

    def calibrate_gyro(self, gyro: np.ndarray) -> np.ndarray:
        """Calibrate the gyroscope.

        .. warning ::
            This is not supported for the FerrarisCalibration, as it is not possible to fully calibrate the gyroscope
            without the acc values.
            Any acc interference on the gyroscope (`K_ga`) will not be taken into account.
            Use `FerrarisCalibrationInfo.calibrate` instead.

        Parameters
        ----------
        gyro :
            3D gyroscope values

        Warns
        -----
        UserWarning
            Always, informing about the missing `K_ga` calibration

        """
        warnings.warn(
            "Performing a calibration on the Gyro data alone, will not correct potential acc-gyro"
            " interferences. Use `{}CalibrationInfo.calibrate` to calibrate acc and gyro"
            " together.".format(self.CAL_TYPE)
        )
        return self._calibrate_gyro(gyro, calibrated_acc=None)

    def calibrate(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate the accelerometer and the gyroscope.

        This corrects:
            acc: scaling, rotation, non-orthogonalities, and bias
            gyro: scaling, rotation, non-orthogonalities, bias, and acc influence on gyro

        Parameters
        ----------
        acc :
            3D acceleration
        gyro :
            3D gyroscope values

        Returns
        -------
        Corrected acceleration and gyroscope values

        """
        # Check if all required paras are initialized to throw appropriate error messages:
        for v in self._fields:
            if getattr(self, v, None) is None:
                raise ValueError(
                    "{} need to initialised before an acc calibration can be performed. {} is missing".format(
                        self._fields, v
                    )
                )

        acc_out = self.calibrate_acc(acc)
        gyro_out = self._calibrate_gyro(gyro, acc_out)

        return acc_out, gyro_out

    def _calibrate_gyro(self, gyro, calibrated_acc=None):
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
        tmp = self._calibrate_gyro_offsets(gyro, calibrated_acc)

        gyro_out = gyro_mat @ tmp.T
        return gyro_out.T

    def _calibrate_gyro_offsets(self, gyro, calibrated_acc=None):
        if calibrated_acc is None:
            d_ga = np.array(0)
        else:
            d_ga = self.K_ga @ calibrated_acc.T
        offsets = d_ga.T + self.b_g
        return gyro - offsets


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
