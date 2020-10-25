from pathlib import Path

import pandas as pd

from imucal import FerrarisCalibration

data = pd.read_csv(Path(__file__).parent / "_test_data/example_data.csv")

FerrarisCalibration.from_interactive_plot(
    data, 100, acc_cols=["accX", "accY", "accZ"], gyro_cols=["gyroX", "gyroY", "gyroZ"]
)
