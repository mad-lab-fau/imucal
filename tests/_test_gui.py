from pathlib import Path

from imucal.calibration_gui import CalibrationGui
import pandas as pd

data = pd.read_csv(Path(__file__).parent / '_test_data/example_data.csv')

CalibrationGui(data[['accX', 'accY', 'accZ']], data[['gyroX', 'gyroY', 'gyroZ']],
               ['x_p', 'x_a', 'y_p', 'y_a', 'z_p', 'z_a', 'x_rot', 'y_rot', 'z_rot'])
