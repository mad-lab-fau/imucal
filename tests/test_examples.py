def test_custom_calibration_info():
    from examples.custom_calibration_info import cal_info, loaded_cal_info

    assert cal_info.new_meta_info == "my value"
    assert loaded_cal_info.new_meta_info == "my value"
