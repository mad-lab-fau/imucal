from pathlib import Path

import matplotlib
import pandas as pd

# This is needed to avoid plots to open
matplotlib.use("Agg")


def test_custom_calibration_info() -> None:
    from examples.custom_calibration_info import cal_info, loaded_cal_info

    assert cal_info.new_meta_info == "my value"
    assert loaded_cal_info.new_meta_info == "my value"


def test_basic_ferraris_calibration() -> None:
    from examples.basic_ferraris import calibrated_data

    # Uncomment the following lines to update the snapshot
    # calibrated_data.head(50).to_csv(Path(__file__).parent / "ferraris_example.csv")

    expected = pd.read_csv(Path(__file__).parent / "ferraris_example.csv", header=0, index_col=0)

    pd.testing.assert_frame_equal(expected, calibrated_data.head(50))
