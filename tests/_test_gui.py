from pathlib import Path

from imucal import FerrarisCalibration
import pandas as pd

data = pd.read_csv(Path(__file__).parent / '_test_data/example_data.csv')

FerrarisCalibration.from_interactive_plot(data, 100, acc_cols=['accX', 'accY', 'accZ'], gyro_cols=['gyroX', 'gyroY', 'gyroZ'])
