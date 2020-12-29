from pathlib import Path

import pandas as pd

from imucal import ferraris_regions_from_interactive_plot

data = pd.read_csv(Path(__file__).parent / "_test_data/example_data.csv").sort_values("samples")

section_data, section_list = ferraris_regions_from_interactive_plot(
    data, acc_cols=["accX", "accY", "accZ"], gyr_cols=["gyroX", "gyroY", "gyroZ"]
)

print(section_data, section_list)
