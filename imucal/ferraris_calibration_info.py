import numpy as np

from imucal.calibration_info import CalibrationInfo


class FerrarisCalibrationInfo(CalibrationInfo):
    # TODO: Add version for Turntable to make sure it is possible to differentiate the two
    CAL_TYPE = 'Ferraris'
    K_a: np.ndarray
    R_a: np.ndarray
    b_a: np.ndarray
    K_g: np.ndarray
    R_g: np.ndarray
    K_ga: np.ndarray
    b_g: np.ndarray

    _fields = ('K_a', 'R_a', 'b_a', 'K_g', 'R_g', 'K_ga', 'b_g')

    def _calibrate_gyro_offsets(self, gyro, calibrated_acc):
        d_ga = (self.K_ga @ calibrated_acc.T)
        offsets = d_ga.T + self.b_g
        return gyro - offsets

    def calibrate_acc(self, acc):
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

    def calibrate_gyro(self, gyro):
        raise NotImplementedError('The Ferraris calibration does not provide a dedicated gyro calibration, because'
                                  'the accelerometer data is required for this step anyway. Use the general `calibrate`'
                                  'method to calibrate the gyro and the acc together.')

    def _calibrate_gyro(self, gyro, calibrated_acc):
        # Combine Scaling and rotation matrix to one matrix
        gyro_mat = np.matmul(np.linalg.inv(self.R_g), np.linalg.inv(self.K_g))
        tmp = self._calibrate_gyro_offsets(gyro, calibrated_acc)

        gyro_out = gyro_mat @ tmp.T
        return gyro_out.T

    def calibrate(self, acc, gyro):
        # Check if all required paras are initialized to throw appropriate error messages:
        for v in self._fields:
            if getattr(self, v, None) is None:
                raise ValueError(
                    '{} need to initialised before an acc calibration can be performed. {} is missing'.format(
                        self._fields, v))

        acc_out = self.calibrate_acc(acc)
        gyro_out = self._calibrate_gyro(gyro, acc_out)

        return acc_out, gyro_out
