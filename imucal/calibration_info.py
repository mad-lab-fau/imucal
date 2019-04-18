class CalibrationInfo:
    CAL_TYPE = None

    def calibrate(self, acc, gyro):
        raise NotImplementedError('This method needs to be implemnted by a subclass')
