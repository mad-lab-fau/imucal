import copy
import datetime
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imucal import FerrarisCalibrationInfo, TurntableCalibrationInfo
from imucal.management import (
    save_calibration_info,
    find_calibration_info_for_sensor,
    find_closest_calibration_info_to_date, load_calibration_info,
)
from tests.conftest import CustomFerraris


@pytest.fixture()
def sample_cal_folder(sample_cal):
    with tempfile.TemporaryDirectory() as f:
        for sid in ["test1", "test2", "test3"]:
            for min in range(10, 30, 2):
                date = datetime.datetime(2000, 10, 3, 13, min)
                save_calibration_info(sample_cal, sid, date, f, folder_structure="")
        yield f


@pytest.fixture()
def sample_cal_folder_recursive(sample_cal_dict):
    with tempfile.TemporaryDirectory() as f:
        for s in [FerrarisCalibrationInfo, TurntableCalibrationInfo, CustomFerraris]:
            for sid in ["test1", "test2", "test3"]:
                new_cal = s(**sample_cal_dict)
                for min in range(10, 30, 2):
                    date = datetime.datetime(2000, 10, 3, 13, min)
                    save_calibration_info(new_cal, sid, date, Path(f))
        yield f


class TestSaveCalibration:
    temp_folder = Path

    @pytest.fixture(autouse=True)
    def temp_folder(self):
        with tempfile.TemporaryDirectory() as f:
            self.temp_folder = Path(f)
            yield

    def test_default_filename(self, sample_cal):
        out = save_calibration_info(sample_cal, "test", datetime.datetime(2000, 10, 3, 13, 22), self.temp_folder)

        expected_out = next((self.temp_folder / "test" / sample_cal.CAL_TYPE).glob("*"))

        assert out == expected_out
        assert expected_out.name == "test_2000-10-03_13-22.json"

    @pytest.mark.parametrize("sensor_id", ["a_b", "tes*", ""])
    def test_valid_s_id(self, sample_cal, sensor_id):
        with pytest.raises(ValueError):
            save_calibration_info(sample_cal, sensor_id, datetime.datetime(2000, 10, 3, 13, 22), self.temp_folder)

    @pytest.mark.parametrize(
        "str_in, folder_path",
        (
            ("simple", ("simple",)),
            ("{sensor_id}/{cal_info.from_gyr_unit}_custom", ("test", "custom_from_gyr_unit_custom")),
        ),
    )
    def test_custom_folder_path(self, sample_cal, str_in, folder_path):
        out = save_calibration_info(
            sample_cal, "test", datetime.datetime(2000, 10, 3, 13, 22), self.temp_folder, folder_structure=str_in
        )

        match = out.parts[-len(folder_path) - 1 : -1]
        for e, o in zip(match, folder_path):
            assert e == o

    def test_empty_folder_structure(self, sample_cal):
        out = save_calibration_info(
            sample_cal, "test", datetime.datetime(2000, 10, 3, 13, 22), self.temp_folder, folder_structure=""
        )

        assert out.parent == self.temp_folder

    def test_kwargs(self, sample_cal):
        out = save_calibration_info(
            sample_cal,
            "test",
            datetime.datetime(2000, 10, 3, 13, 22),
            self.temp_folder,
            folder_structure="{sensor_id}/{my_custom}",
            my_custom="my_custom_val",
        )

        assert out.parts[-2] == "my_custom_val"
        assert out.parts[-3] == "test"


class TestFindCalibration:
    def test_simple(self, sample_cal_folder):
        cals = find_calibration_info_for_sensor("test1", sample_cal_folder)

        assert len(cals) == 10
        assert all(["test1" in str(x) for x in cals])

    def test_find_calibration_non_existent(self, sample_cal_folder):
        with pytest.raises(ValueError):
            find_calibration_info_for_sensor("wrong_sensor", sample_cal_folder)

        cals = find_calibration_info_for_sensor("wrong_sensor", sample_cal_folder, ignore_file_not_found=True)

        assert len(cals) == 0

    def test_find_calibration_recursive(self, sample_cal_folder_recursive):
        with pytest.raises(ValueError):
            find_calibration_info_for_sensor("test1", sample_cal_folder_recursive, recursive=False)

        cals = find_calibration_info_for_sensor("test1", sample_cal_folder_recursive, recursive=True)

        assert len(cals) == 30
        assert all(["test1" in str(x) for x in cals])

    def test_find_calibration_type_filter(self, sample_cal_folder_recursive):
        cals = find_calibration_info_for_sensor(
            "test1", sample_cal_folder_recursive, recursive=True, filter_cal_type="ferraris"
        )

        assert len(cals) == 10
        assert all(["test1" in str(x) for x in cals])
        assert all([load_calibration_info(c).CAL_TYPE.lower() == "ferraris" for c in cals])

    @pytest.mark.parametrize("string", ("Ferraris", "ferraris", "FERRARIS"))
    def test_find_calibration_type_filter_case_sensitive(self, sample_cal_folder_recursive, string):
        cals = find_calibration_info_for_sensor(
            "test1", sample_cal_folder_recursive, recursive=True, filter_cal_type=string
        )

        assert len(cals) == 10
        assert all(["test1" in str(x) for x in cals])
        assert all([load_calibration_info(c).CAL_TYPE.lower() == "ferraris" for c in cals])

    def test_custom_validator(self, sample_cal_folder_recursive):
        # We simulate the caltype filter with a custom validator
        validator = lambda x: x.CAL_TYPE.lower() == "ferraris"

        cals = find_calibration_info_for_sensor(
            "test1", sample_cal_folder_recursive, recursive=True, filter_cal_type=None, custom_validator=validator
        )

        assert len(cals) == 10
        assert all(["test1" in str(x) for x in cals])
        assert all([load_calibration_info(c).CAL_TYPE.lower() == "ferraris" for c in cals])


class TestFindClosestCalibration:
    @pytest.mark.parametrize("relative", ("before", "after", None))
    def test_find_closest(self, sample_cal_folder, relative):
        # Test that before and after still return the correct one if there is an exact match

        cal = find_closest_calibration_info_to_date(
            "test1", datetime.datetime(2000, 10, 3, 13, 14), sample_cal_folder, before_after=relative
        )

        assert cal.name == "test1_2000-10-03_13-14.json"

    def test_find_closest_non_existend(self, sample_cal_folder):
        with pytest.raises(ValueError):
            find_closest_calibration_info_to_date(
                "wrong_sensor", datetime.datetime(2000, 10, 3, 13, 14), sample_cal_folder
            )

        cal = find_closest_calibration_info_to_date(
            "wrong_sensor", datetime.datetime(2000, 10, 3, 13, 14), sample_cal_folder, ignore_file_not_found=True
        )

        assert cal is None

    @pytest.mark.parametrize(
        "relative, expected",
        (
            ("before", "test1_2000-10-03_13-14.json"),
            ("after", "test1_2000-10-03_13-16.json"),
            (None, "test1_2000-10-03_13-14.json"),
        ),
    )
    def test_find_closest_before_after(self, sample_cal_folder, relative, expected):
        # Default to earlier if same distance before and after.
        cal = find_closest_calibration_info_to_date(
            "test1", datetime.datetime(2000, 10, 3, 13, 15), sample_cal_folder, before_after=relative
        )

        assert cal.name == expected

    @pytest.mark.parametrize("warn_type, day", ((UserWarning, 15), (None, 14)))
    def test_find_closest_warning(self, sample_cal_folder, warn_type, day):
        with pytest.warns(warn_type) as rec:
            find_closest_calibration_info_to_date(
                "test1",
                datetime.datetime(2000, 10, 3, 13, day),
                sample_cal_folder,
                warn_thres=datetime.timedelta(seconds=30),
            )
        expected_n = 1
        if warn_type is None:
            expected_n = 0
        assert len(rec) == expected_n


class TestLoadCalFiles:
    def test_invalid_file(self):
        with pytest.raises(ValueError) as e:
            load_calibration_info("invalid_file.txt")

        assert "The loader format could not be determined" in str(e)

    def test_finds_subclass(self, sample_cal):
        """If the wrong subclass is used it can not find the correct calibration."""
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            sample_cal.to_json_file(f.name)
            if isinstance(sample_cal, CustomFerraris):
                assert load_calibration_info(f.name, file_type="json", base_class=CustomFerraris) == sample_cal
            else:
                with pytest.raises(ValueError) as e:
                    load_calibration_info(f.name, file_type="json", base_class=CustomFerraris)
                assert sample_cal.CAL_TYPE in str(e)

    @pytest.mark.parametrize("file_type", ("json", "hdf"))
    def test_fixed_loader(self, file_type, sample_cal):
        method = dict(json="to_json_file", hdf="to_hdf5")
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            getattr(sample_cal, method[file_type])(f.name)
            out = load_calibration_info(f.name, file_type=file_type)
        assert sample_cal == out

    @pytest.mark.parametrize("file_type", ("json", "hdf"))
    def test_auto_loader(self, file_type, sample_cal):
        method = dict(json="to_json_file", hdf="to_hdf5")
        with tempfile.NamedTemporaryFile(mode="w+", suffix="." + file_type) as f:
            getattr(sample_cal, method[file_type])(f.name)
            out = load_calibration_info(f.name)
        assert sample_cal == out

    def test_invalid_loader(self):
        with pytest.raises(ValueError) as e:
            load_calibration_info("invalid_file.txt", file_type="invalid")

        assert "`format` must be one of" in str(e)
