"""Wrapper object to hold calibration matrices for a Ferraris Calibration."""
from typing import Tuple

import numpy as np

from imucal.calibration_info import CalibrationInfo


# TODO: Add example to docstring
class FerrarisCalibrationInfo(CalibrationInfo):
    """Calibration object that represents all the required information to apply a Ferraris calibration to a dataset.

    Attributes:
        K_a: Scaling matrix for the acceleration
        R_a: Rotation matrix for the acceleration
        b_a: Acceleration bias
        K_g: Scaling matrix for the gyroscope
        R_g: Rotation matrix for the gyroscope
        K_ga: Influence of acceleration on gyroscope
        b_g: gyroscope bias

    """

    CAL_TYPE = 'Ferraris'
    K_a: np.ndarray
    R_a: np.ndarray
    b_a: np.ndarray
    K_g: np.ndarray
    R_g: np.ndarray
    K_ga: np.ndarray
    b_g: np.ndarray

    _fields = ('K_a', 'R_a', 'b_a', 'K_g', 'R_g', 'K_ga', 'b_g')

    __doc__ += CalibrationInfo._cal_type_explanation

    def _calibrate_gyro_offsets(self, gyro, calibrated_acc):
        d_ga = (self.K_ga @ calibrated_acc.T)
        offsets = d_ga.T + self.b_g
        return gyro - offsets

    def calibrate_acc(self, acc: np.ndarray) -> np.ndarray:
        """Calibrate the accelerometer.

        This corrects scaling, rotation, non-orthogonalities and bias.

        Args:
          acc: 3D acceleration

        Returns:
            Calibrated acceleration

        """
        # Check if all required paras are initialized to throw appropriate error messages:
        paras = ('K_a', 'R_a', 'b_a')
        for v in paras:
            if getattr(self, v, None) is None:
                raise ValueError(
                    '{} need to initialised before an acc calibration can be performed. {} is missing'.format(paras, v))

        # Combine Scaling and rotation matrix to one matrix
        acc_mat = np.linalg.inv(self.R_a) @ np.linalg.inv(self.K_a)
        acc_out = acc_mat @ (acc - self.b_a).T

        return acc_out.T

    def calibrate_gyro(self, gyro: np.ndarray) -> np.ndarray:
        """Calibrate the gyroscope.

        Warning:
            This is not supported for the FerrarisCalibration, as it is not possible to calibrate the gyroscope alone.
            Use `FerrarisCalibrationInfo.calibrate` instead.

        Args:
          gyro: 3D gyroscope values

        Raises:
            NotImplementedError: Always

        """
        raise NotImplementedError('The Ferraris calibration does not provide a dedicated gyro calibration, because'
                                  'the accelerometer data is required for this step anyway. Use the general `calibrate`'
                                  'method to calibrate the gyro and the acc together.')

    def _calibrate_gyro(self, gyro, calibrated_acc):
        # Combine Scaling and rotation matrix to one matrix
        gyro_mat = np.matmul(np.linalg.inv(self.R_g), np.linalg.inv(self.K_g))
        tmp = self._calibrate_gyro_offsets(gyro, calibrated_acc)

        gyro_out = gyro_mat @ tmp.T
        return gyro_out.T

    def calibrate(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate the accelerometer and the gyroscope.

        Args:
            acc: 3D acceleration
            gyro: 3D gyroscope values

        This corrects:
            acc: scaling, rotation, non-orthogonalities, and bias
            gyro: scaling, rotation, non-orthogonalities, bias, and acc influence on gyro

        Returns:
            Corrected acceleration and gyroscope values

        """
        # Check if all required paras are initialized to throw appropriate error messages:
        for v in self._fields:
            if getattr(self, v, None) is None:
                raise ValueError(
                    '{} need to initialised before an acc calibration can be performed. {} is missing'.format(
                        self._fields, v))

        acc_out = self.calibrate_acc(acc)
        gyro_out = self._calibrate_gyro(gyro, acc_out)

        return acc_out, gyro_out


class TurntableCalibrationInfo(FerrarisCalibrationInfo):
    """Calibration object that represents all the required information to apply a Turntable calibration to a dataset.

    A Turntable calibration is identical to a Ferraris calibration.
    However, because the parameters are calculated using a calibration table instead of hand rotations,
    higher precision is expected.

    Attributes:
        K_a: Scaling matrix for the acceleration
        R_a: Rotation matrix for the acceleration
        b_a: Acceleration bias
        K_g: Scaling matrix for the gyroscope
        R_g: Rotation matrix for the gyroscope
        K_ga: Influence of acceleration on gyroscope
        b_g: gyroscope bias

    """

    CAL_TYPE = 'Turntable'

    def calibrate_gyro(self, gyro: np.ndarray) -> np.ndarray:
        """Calibrate the gyroscope.

        Warning:
            This is not supported for the TurntableCalibration, as it is not possible to calibrate the gyroscope alone.
            Use `TurntableCalibrationnInfo.calibrate` instead.

        Args:
          gyro: 3D gyroscope values

        Raises:
            NotImplementedError: Always

        """
        raise NotImplementedError('The Turntable calibration does not provide a dedicated gyro calibration, because'
                                  'the accelerometer data is required for this step anyway. Use the general `calibrate`'
                                  'method to calibrate the gyro and the acc together.')
