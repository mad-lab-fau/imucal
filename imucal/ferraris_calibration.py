"""Calculate a Ferraris calibration from sensor data."""
from itertools import product
from typing import Optional, Iterable, TypeVar, Type, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv

from imucal.calibration_gui import _convert_data_from_section_list_to_df, CalibrationGui
from imucal.calibration_info import CalibrationInfo
from imucal.ferraris_calibration_info import FerrarisCalibrationInfo, TurntableCalibrationInfo

T = TypeVar("T", bound="FerrarisCalibration")


class FerrarisCalibration:
    """Calculate a Ferraris calibration matrices based on a set of calibration movements.

    The Ferraris calibration is derived based on a well defined series of data recordings:

    `x_p`: positive x-axis of sensor is aligned with gravity (x-acc measures +1g)
    `x_a`: negative x-axis of sensor is aligned with gravity (x-acc measures -1g)
    `y_p`: positive y-axis of sensor is aligned with gravity (y-acc measures +1g)
    `y_a`: negative y-axis of sensor is aligned with gravity (y-acc measures -1g)
    `z_p`: positive z-axis of sensor is aligned with gravity (z-acc measures +1g)
    `z_a`: negative z-axis of sensor is aligned with gravity (z-acc measures -1g)

    `x_rot`: sensor is rotated clockwise in the `x_p` position around the x-axis (x-gyro shows negative values) for a
        well known angle (typically 360 deg)
    `y_rot`: sensor is rotated clockwise in the `y_p` position around the y-axis (y-gyro shows negative values) for a
        well known angle (typically 360 deg)
    `z_rot`: sensor is rotated clockwise in the `z_p` position around the z-axis (z-gyro shows negative values) for a
        well known angle (typically 360 deg)

    All sections need to be recorded for a sensor and then annotated.
    In particular for the rotation, it is important to annotate the data directly at the end and the beginning of the
    rotation to avoid noise and artifact degrading the integration results.
    This class offers various helper constructors to support the annotation process.


    Notes
    -----
    Depending on how the axis of your respective sensor coordinate system are defined and how you perform the
    calibration, you might need to change the `grav` and `expected_angle` parameter.

    Typical situations are:
        - If you define the positive axis direction as the direction, where the acc measures -g, change `grav` to
            -9.81 m/s^2
        - If you perform a counter-clockwise rotation during the calibration, set `expected_angle` to +360
        - For combinations of both, both parameter might need to be adapted


    Examples
    --------
    >>> from imucal import FerrarisCalibration
    >>> sampling_rate = 100 #Hz
    >>> data = ... # my data as 6 col pandas dataframe
    >>> # This will open an interactive plot, where you can select the start and the stop sample of each region
    >>> cal, section_list = FerrarisCalibration.from_interactive_plot(data, sampling_rate=sampling_rate)
    >>> section_list.to_csv('./calibration_sections.csv')  # This is optional, but recommended
    >>> calibration_info = cal.compute_calibration_matrix()  # Calculate the actual matrizes.
    >>> calibration_info
    < FerrarisCalibration object at ... >

    """

    acc_x_p: np.ndarray
    acc_x_a: np.ndarray
    acc_y_p: np.ndarray
    acc_y_a: np.ndarray
    acc_z_p: np.ndarray
    acc_z_a: np.ndarray
    gyr_x_p: np.ndarray
    gyr_x_a: np.ndarray
    gyr_y_p: np.ndarray
    gyr_y_a: np.ndarray
    gyr_z_p: np.ndarray
    gyr_z_a: np.ndarray

    acc_x_rot: np.ndarray
    acc_y_rot: np.ndarray
    acc_z_rot: np.ndarray
    gyr_x_rot: np.ndarray
    gyr_y_rot: np.ndarray
    gyr_z_rot: np.ndarray

    sampling_rate: float
    grav: float
    expected_angle: float

    EXPECTED_ANGLE: float = -360.0  # +360 for anti-clockwise rotation, -360 for clockwise rotation
    DEFAULT_GRAV: float = 9.81
    ACC_COLS = ("acc_x", "acc_y", "acc_z")
    GYRO_COLS = ("gyr_x", "gyr_y", "gyr_z")
    FERRARIS_SECTIONS = ("x_p", "x_a", "y_p", "y_a", "z_p", "z_a", "x_rot", "y_rot", "z_rot")

    _CALIBRATION_INFO = FerrarisCalibrationInfo

    _fields = tuple("{}_{}".format(x, y) for x, y in product(("acc", "gyr"), FERRARIS_SECTIONS))

    def __init__(
        self, sampling_rate: float, grav: Optional[float] = None, expected_angle: Optional[float] = None, **kwargs
    ) -> None:
        """Create a Calibration object.

        Usually you would want to use one of the provided "from" constructors instead of directly using the init.

        Parameter
        ---------
        sampling_rate :
            Sampling rate of the data
        expected_angle :
            expected rotation angle for the gyroscope rotation.
            If None defaults to `FerrarisCalibration.EXPECTED_ANGLE`
        grav :
            The expected value of the gravitational acceleration.
            Defaults to `FerrarisCalibration.DEFAULT_GRAV`
        kwargs :
            A 3D numpy array for each section (acc and gyro separately) required by the ferraris calibration.
            The arguments need to be named acc/gyr_section (e.g. acc_x_p)

        """
        for field in self._fields:
            setattr(self, field, kwargs.get(field, None))

        self.sampling_rate = sampling_rate
        self.grav = grav or self.DEFAULT_GRAV
        self.expected_angle = expected_angle or self.EXPECTED_ANGLE
        super().__init__()

    @classmethod
    def from_df(
        cls: Type[T],
        df: pd.DataFrame,
        sampling_rate: float,
        grav: Optional[float] = None,
        expected_angle: Optional[float] = None,
        acc_cols: Optional[Iterable[str]] = None,
        gyro_cols: Optional[Iterable[str]] = None,
    ) -> T:
        """Create a Calibration object based on a dataframe which has all required sections labeled.

        The expected Dataframe has the section label as index and has at least the 6 required data columns.
        The index must contain all sections as specified by `FerrarisCalibration.FERRARIS_SECTIONS`.

        >>> print(example_df)
                acc_x acc_y   acc_z  gyr_x  gyr_y  gyr_z
        part
        x_a   -2052.0 -28.0   -73.0    1.0    0.0   -5.0
        x_a   -2059.0 -29.0   -77.0    2.0   -3.0   -5.0
        x_a   -2054.0 -25.0   -71.0    3.0   -2.0   -3.0
        ...       ...   ...     ...    ...    ...    ...
        z_rot   -36.0  35.0  2079.0    2.0   -5.0   -2.0
        z_rot   -28.0  36.0  2092.0    6.0   -5.0   -4.0
        z_rot   -36.0  21.0  2085.0    5.0   -4.0   -4.0

        Examples
        --------
        >>> from imucal import FerrarisCalibration
        >>> import pandas as pd
        >>> sampling_rate = 100 #Hz
        >>> df = ... # A valid DataFrame with all sections in the index
        >>> cal = FerrarisCalibration.from_dict(df, sampling_rate=sampling_rate)
        < FerrarisCalibration object at ... >

        Parameter
        ---------
        df :
            6 column dataframe (3 acc, 3 gyro)
        sampling_rate :
            Sampling rate of the data
        expected_angle :
            expected rotation angle for the gyroscope rotation.
            If None defaults to `FerrarisCalibration.EXPECTED_ANGLE`
        grav :
            The expected value of the gravitational acceleration.
            Defaults to `FerrarisCalibration.DEFAULT_GRAV`
        acc_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.ACC_COLS`
        gyro_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.GYRO_COLS`

        Returns
        -------
        ferraris_cal_obj : FerrarisCalibration

        """
        if acc_cols is None:
            acc_cols = list(cls.ACC_COLS)
        if gyro_cols is None:
            gyro_cols = list(cls.GYRO_COLS)

        acc_df = df[list(acc_cols)]
        gyro_df = df[list(gyro_cols)]
        acc_dict = acc_df.groupby(level=0).apply(lambda x: x.values).to_dict()
        gyro_dict = gyro_df.groupby(level=0).apply(lambda x: x.values).to_dict()
        acc_dict = {"acc_" + k: v for k, v in acc_dict.items()}
        gyro_dict = {"gyr_" + k: v for k, v in gyro_dict.items()}

        return cls(sampling_rate, grav, expected_angle, **acc_dict, **gyro_dict)

    @classmethod
    def from_section_list(
        cls: Type[T],
        data: pd.DataFrame,
        section_list: pd.DataFrame,
        sampling_rate: float,
        grav: Optional[float] = None,
        expected_angle: Optional[float] = None,
        acc_cols: Optional[Iterable[str]] = None,
        gyro_cols: Optional[Iterable[str]] = None,
    ) -> T:
        """Create a Calibration object based on a valid section list.

        A section list marks the start and the endpoints of each required section in the data object.
        A valid section list is usually created using `FerrarisCalibration.from_interactive_plot()`.
        This section list can be stored on disk and this method can be used to turn it back into a valid calibration
        object.

        Parameter
        ---------
        df :
            6 column dataframe (3 acc, 3 gyro)
        section_list :
            A pandas dataframe representing a section list
        sampling_rate :
            Sampling rate of the data
        expected_angle :
            expected rotation angle for the gyroscope rotation.
            If None defaults to `FerrarisCalibration.EXPECTED_ANGLE`
        grav :
            The expected value of the gravitational acceleration.
            Defaults to `FerrarisCalibration.DEFAULT_GRAV`
        acc_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.ACC_COLS`
        gyro_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.GYRO_COLS`

        Returns
        -------
        ferraris_cal_obj : FerrarisCalibration


        Examples
        --------
        >>> from imucal import FerrarisCalibration
        >>> import pandas as pd
        >>> # Load a valid section list from disk. Note the `index_col=0` to preserve correct format!
        >>> section_list = pd.read_csv('./calibration_sections.csv', index_col=0)
        >>> sampling_rate = 100 #Hz
        >>> data = ... # my data as 6 col pandas dataframe
        >>> cal = FerrarisCalibration.from_section_list(data, section_list, sampling_rate=sampling_rate)
        < FerrarisCalibration object at ... >

        """
        df = _convert_data_from_section_list_to_df(data, section_list)
        return cls.from_df(
            df, sampling_rate, expected_angle=expected_angle, grav=grav, acc_cols=acc_cols, gyro_cols=gyro_cols
        )

    @classmethod
    def from_interactive_plot(
        cls: Type[T],
        data: pd.DataFrame,
        sampling_rate: float,
        expected_angle: Optional[float] = None,
        grav: Optional[float] = None,
        acc_cols: Optional[Iterable[str]] = None,
        gyro_cols: Optional[Iterable[str]] = None,
    ) -> Tuple[T, pd.DataFrame]:
        """Create a Calibration object by selecting the individual signal sections manually in an interactive GUI.

        This will open a Tkinter Window that allows you to label the start and the end all required sections for a
        Ferraris Calibration.
        See the class docstring for more detailed explanations of these sections.

        Examples
        --------
        >>> from imucal import FerrarisCalibration
        >>> sampling_rate = 100 #Hz
        >>> data = ... # my data as 6 col pandas dataframe
        >>> # This will open an interactive plot, where you can select the start and the stop sample of each region
        >>> cal, section_list = FerrarisCalibration.from_interactive_plot(data, sampling_rate=sampling_rate)
        >>> section_list.to_csv('./calibration_sections.csv')  # This is optional, but recommended
        >>> cal
        < FerrarisCalibration object at ... >

        Parameter
        ---------
        df :
            6 column dataframe (3 acc, 3 gyro)
        sampling_rate :
            Sampling rate of the data
        expected_angle :
            expected rotation angle for the gyroscope rotation.
            If None defaults to `FerrarisCalibration.EXPECTED_ANGLE`
        grav :
            The expected value of the gravitational acceleration.
            Defaults to `FerrarisCalibration.DEFAULT_GRAV`
        acc_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.ACC_COLS`
        gyro_cols :
            The name of the 3 acceleration columns in order x,y,z.
            Defaults to `FerrarisCalibration.GYRO_COLS`

        Returns
        -------
        ferraris_cal_obj : FerrarisCalibration
        section_list : pd.DataFrame
            Section list representing the start and stop of each section.
            It is advised to save this to disk to avoid repeated manual labeling.
            `FerrarisCalibration.from_section_list()` can be used to recreate the calibration object

        """
        if acc_cols is None:
            acc_cols = list(cls.ACC_COLS)
        if gyro_cols is None:
            gyro_cols = list(cls.GYRO_COLS)

        acc = data[acc_cols].values
        gyro = data[gyro_cols].values

        section_list = _find_calibration_sections_interactive(acc, gyro)
        return (
            cls.from_section_list(
                data,
                section_list,
                sampling_rate,
                expected_angle=expected_angle,
                grav=grav,
                acc_cols=acc_cols,
                gyro_cols=gyro_cols,
            ),
            section_list,
        )

    def compute_calibration_matrix(self) -> CalibrationInfo:
        """Compute the calibration Information.

        This actually performs the Ferraris calibration following the original publication equation by equation.
        """
        cal_mat = self._CALIBRATION_INFO()

        ###############################################################################################################
        # Compute Acceleration Matrix

        # Calculate means from all static phases and stack them into 3x3 matrices
        # Note: Each measurement should be a column
        U_a_p = np.vstack(  # noqa: invalid-name
            (np.mean(self.acc_x_p, axis=0), np.mean(self.acc_y_p, axis=0), np.mean(self.acc_z_p, axis=0),)
        ).T
        U_a_n = np.vstack(  # noqa: invalid-name
            (np.mean(self.acc_x_a, axis=0), np.mean(self.acc_y_a, axis=0), np.mean(self.acc_z_a, axis=0),)
        ).T

        # Eq. 19
        U_a_s = U_a_p + U_a_n  # noqa: invalid_name

        # Bias Matrix
        # Eq. 20
        B_a = U_a_s / 2  # noqa: invalid_name

        # Bias Vector
        b_a = np.diag(B_a)
        cal_mat.b_a = b_a

        # Compute Scaling and Rotation
        # No need for bias correction, since it cancels out!
        # Eq. 21
        U_a_d = U_a_p - U_a_n  # noqa: invalid_name

        # Calculate Scaling matrix
        # Eq. 23
        k_a_sq = 1 / (4 * self.grav ** 2) * np.diag(U_a_d @ U_a_d.T)
        K_a = np.diag(np.sqrt(k_a_sq))  # noqa: invalid_name
        cal_mat.K_a = K_a  # noqa: invalid_name

        # Calculate Rotation matrix
        # Eq. 22
        R_a = inv(K_a) @ U_a_d / (2 * self.grav)  # noqa: invalid_name
        cal_mat.R_a = R_a  # noqa: invalid_name

        ###############################################################################################################
        # Calculate Gyroscope Matrix

        # Gyro Bias from the static phases of the acc calibration
        # One static phase would be sufficient, but why not use all of them if you have them.
        # Note that this calibration ignores any influences due to the earth rotation.

        b_g = np.mean(
            np.vstack((self.gyr_x_p, self.gyr_x_a, self.gyr_y_p, self.gyr_y_a, self.gyr_z_p, self.gyr_z_a,)), axis=0,
        )

        cal_mat.b_g = b_g

        # Acceleration sensitivity

        # Note: Each measurement should be a column
        U_g_p = np.vstack(  # noqa: invalid_name
            (np.mean(self.gyr_x_p, axis=0), np.mean(self.gyr_y_p, axis=0), np.mean(self.gyr_z_p, axis=0),)
        ).T
        U_g_a = np.vstack(  # noqa: invalid_name
            (np.mean(self.gyr_x_a, axis=0), np.mean(self.gyr_y_a, axis=0), np.mean(self.gyr_z_a, axis=0),)
        ).T

        # Eq. 9
        K_ga = (U_g_p - U_g_a) / (2 * self.grav)  # noqa: invalid_name
        cal_mat.K_ga = K_ga  # noqa: invalid_name

        # Gyroscope Scaling and Rotation

        # First apply partial calibration to remove offset and acc influence
        acc_x_rot_cor = cal_mat.calibrate_acc(self.acc_x_rot)
        acc_y_rot_cor = cal_mat.calibrate_acc(self.acc_y_rot)
        acc_z_rot_cor = cal_mat.calibrate_acc(self.acc_z_rot)
        gyr_x_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_x_rot, acc_x_rot_cor)
        gyr_y_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_y_rot, acc_y_rot_cor)
        gyr_z_rot_cor = cal_mat._calibrate_gyro_offsets(self.gyr_z_rot, acc_z_rot_cor)

        # Integrate gyro readings
        # Eg. 13/14
        W_s = np.zeros((3, 3))  # noqa: invalid_name
        W_s[:, 0] = np.sum(gyr_x_rot_cor, axis=0) / self.sampling_rate
        W_s[:, 1] = np.sum(gyr_y_rot_cor, axis=0) / self.sampling_rate
        W_s[:, 2] = np.sum(gyr_z_rot_cor, axis=0) / self.sampling_rate

        # Eq.15
        expected_angles = self.expected_angle * np.identity(3)
        multiplied = W_s @ inv(expected_angles)

        # Eq. 12
        k_g_sq = np.diag(multiplied @ multiplied.T)
        K_g = np.diag(np.sqrt(k_g_sq))  # noqa: invalid_name
        cal_mat.K_g = K_g  # noqa: invalid_name

        R_g = inv(K_g) @ multiplied  # noqa: invalid_name
        cal_mat.R_g = R_g  # noqa: invalid_name

        return cal_mat


class TurntableCalibration(FerrarisCalibration):
    """Calculate a Ferraris calibration matrices based on a turntable measurement.

    This calibration is basically identical to the FerrarisCalibration.
    However, the calibrate method will return a `TurntableCalibrationInfo` to indicate the expected higher precision
    of this calibration method.

    Further this Calibration expects rotations of 720 deg by default, as this is common for many turntables.
    For further information on the sign of the expected rotation angle see the `FerrarisCalibration`.
    """

    _CALIBRATION_INFO = TurntableCalibrationInfo

    EXPECTED_ANGLE: float = -720.0


def _find_calibration_sections_interactive(acc: np.ndarray, gyro: np.ndarray, title: Optional[str] = None):
    """Prepare the calibration data for the later calculation of calibration matrices.

    Parameters
    ----------
    acc : (n, 3) array
        Acceleration data
    gyro : (n, 3) array
        Gyroscope data
    title :
        Optional title for the Calibration GUI

    """
    plot = CalibrationGui(acc, gyro, FerrarisCalibration.FERRARIS_SECTIONS, title=title)

    section_list = plot.section_list

    check_all = (all(v) for v in section_list.values())
    if not all(check_all):
        raise ValueError("Some regions are missing in the section list. Label all regions before closing the plot")

    section_list = pd.DataFrame(section_list, index=("start", "end")).T

    return section_list
