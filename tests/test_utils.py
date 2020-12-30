import tempfile

import pytest

from imucal import load_calibration_info
from tests.conftest import CustomFerraris


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
