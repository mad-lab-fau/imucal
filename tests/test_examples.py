from pathlib import Path

import pandas as pd


def test_custom_calibration_info():
    from examples.custom_calibration_info import cal_info, loaded_cal_info

    assert cal_info.new_meta_info == "my value"
    assert loaded_cal_info.new_meta_info == "my value"


def test_basic_ferraris_calibration():
    from examples.basic_ferraris import calibrated_data

    # Uncomment the following lines to update the snapshot
    # calibrated_data.head(50).to_csv(Path(__file__).parent / "ferraris_example.csv")

    expected = pd.read_csv(Path(__file__).parent / "ferraris_example.csv", header=0, index_col=0)

    pd.testing.assert_frame_equal(expected, calibrated_data.head(50))


