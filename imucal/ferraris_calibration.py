"""Calculate a Ferraris calibration from sensor data."""
from typing import Optional, TypeVar, Type, NamedTuple, Iterable, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv
from typing_extensions import ClassVar

from imucal.calibration_info import CalibrationInfo
from imucal.ferraris_calibration_info import FerrarisCalibrationInfo, TurntableCalibrationInfo

T = TypeVar("T", bound="FerrarisCalibration")


class FerrarisSignalRegions(NamedTuple):
    """NamedTuple containing all signal regions required for a Ferraris Calibration."""

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

    def validate(self):
        """Validate that all fields are populated with numpy arrays."""
        for k in self._fields:
            if not isinstance(getattr(self, k), np.ndarray) or len(getattr(self, k)) == 0:
                raise ValueError("The the signal region {} is no valid numpy array.")


class FerrarisCalibration:
    """Calculate a Ferraris calibration matrices based on a set of calibration movements.

    The Ferraris calibration is derived based on a well defined series of data recordings:

    =====  ======================================================================
    Static Holds
    -----------------------------------------------------------------------------
    Name   Explanation
    =====  ======================================================================
    `x_p`  positive x-axis of sensor is aligned with gravity (x-acc measures +1g)
    `x_a`  negative x-axis of sensor is aligned with gravity (x-acc measures -1g)
    `y_p`  positive y-axis of sensor is aligned with gravity (y-acc measures +1g)
    `y_a`  negative y-axis of sensor is aligned with gravity (y-acc measures -1g)
    `z_p`  positive z-axis of sensor is aligned with gravity (z-acc measures +1g)
    `z_a`  negative z-axis of sensor is aligned with gravity (z-acc measures -1g)
    =====  ======================================================================

    =======  ========================================================================================================
    Rotations
    -----------------------------------------------------------------------------------------------------------------
    Name     Explanation
    =======  ========================================================================================================
    `x_rot`  sensor is rotated clockwise in the `x_p` position around the x-axis (x-gyro shows negative values) for a
             well known angle (typically 360 deg)
    `y_rot`  sensor is rotated clockwise in the `y_p` position around the y-axis (y-gyro shows negative values) for a
             well known angle (typically 360 deg)
    `z_rot`  sensor is rotated clockwise in the `z_p` position around the z-axis (z-gyro shows negative values) for a
             well known angle (typically 360 deg)
    =======  ========================================================================================================

    All sections need to be recorded for a sensor and then annotated.
    This class then takes the data of each section (represented as a
    :class:`~imucal.FerrarisSignalRegions` object) and calculates the calibration matrizes for
    the gyroscope and accelerometer.

    As it is quite tedious to obtain the data of each section in a seperate array, you should make use of the available
    helper functions to turn a continous recording into annotated sections (See the See also section)

    Parameters
    ----------
    sampling_rate :
        Sampling rate of the data
    expected_angle :
        expected rotation angle for the gyroscope rotation.
    grav :
        The expected value of the gravitational acceleration.
    calibration_info_class :
        The calibration Info class to use to store the final calibration information.
        This should be a FerrarisCalibrationInfo or a custom subclass.

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
    >>> from imucal import FerrarisCalibration, ferraris_regions_from_interactive_plot
    >>> sampling_rate = 100 #Hz
    >>> data = ... # my data as 6 col pandas dataframe
    >>> # This will open an interactive plot, where you can select the start and the stop sample of each region
    >>> section_data, section_list = ferraris_regions_from_interactive_plot(data, sampling_rate=sampling_rate)
    >>> section_list.to_csv('./calibration_sections.csv')  # Store the annotated section list as reference for the
    ...                                                    # future
    >>> cal = FerrarisCalibration()  # Create new calibration object
    >>> calibration_info = cal.compute(  # Calculate the actual matrizes.
    ...     section_data,
    ...     sampling_rate_hz=sampling_rate,
    ...     from_acc_unit="a.u.",
    ...     from_gyr_unit="a.u.",
    ...     comment="my custom comment."
    ...)
    >>> calibration_info_class
    < FerrarisCalibration object at ... >

    See Also
    --------
    imucal.ferraris_regions_from_df: Generate valid sections from preannotated dataframe.
    imucal.ferraris_regions_from_interactive_plot: Generate valid sections via manual annotation in an interactive
        GUI.
    imucal.ferraris_regions_from_section_list: Generate valid sections based on raw data and start-end labels for the
        individual sections.

    """

    grav: float
    expected_angle: float
    calibration_info_class: Type[FerrarisCalibrationInfo]

    OUT_ACC_UNIT: ClassVar[str] = "m/s^2"
    OUT_GYR_UNIT: ClassVar[str] = "deg/s"

    FERRARIS_SECTIONS: ClassVar[Tuple[str, ...]] = (
        "x_p",
        "x_a",
        "y_p",
        "y_a",
        "z_p",
        "z_a",
        "x_rot",
        "y_rot",
        "z_rot",
    )

    def __init__(
        self,
        grav: float = 9.81,
        expected_angle: float = -360,
        calibration_info_class: Type[FerrarisCalibrationInfo] = FerrarisCalibrationInfo,
    ):
        self.grav = grav
        self.expected_angle = expected_angle
        self.calibration_info_class = calibration_info_class

    def compute(
        self,
        signal_regions: FerrarisSignalRegions,
        sampling_rate_hz: float,
        from_acc_unit: str,
        from_gyr_unit: str,
        **kwargs,
    ) -> CalibrationInfo:
        """Compute the calibration Information.

        This actually performs the Ferraris calibration following the original publication equation by equation.
        """
        signal_regions.validate()

        # Initialize the cal info with all the meta data
        cal_mat = self.calibration_info_class(
            from_acc_unit=from_acc_unit,
            from_gyr_unit=from_gyr_unit,
            acc_unit=self.OUT_ACC_UNIT,
            gyr_unit=self.OUT_GYR_UNIT,
            **kwargs,
        )

        ###############################################################################################################
        # Compute Acceleration Matrix

        # Calculate means from all static phases and stack them into 3x3 matrices
        # Note: Each measurement should be a column
        U_a_p = np.vstack(  # noqa: invalid-name
            (
                np.mean(signal_regions.acc_x_p, axis=0),
                np.mean(signal_regions.acc_y_p, axis=0),
                np.mean(signal_regions.acc_z_p, axis=0),
            )
        ).T
        U_a_n = np.vstack(  # noqa: invalid-name
            (
                np.mean(signal_regions.acc_x_a, axis=0),
                np.mean(signal_regions.acc_y_a, axis=0),
                np.mean(signal_regions.acc_z_a, axis=0),
            )
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
            np.vstack(
                (
                    signal_regions.gyr_x_p,
                    signal_regions.gyr_x_a,
                    signal_regions.gyr_y_p,
                    signal_regions.gyr_y_a,
                    signal_regions.gyr_z_p,
                    signal_regions.gyr_z_a,
                )
            ),
            axis=0,
        )

        cal_mat.b_g = b_g

        # Acceleration sensitivity

        # Note: Each measurement should be a column
        U_g_p = np.vstack(  # noqa: invalid_name
            (
                np.mean(signal_regions.gyr_x_p, axis=0),
                np.mean(signal_regions.gyr_y_p, axis=0),
                np.mean(signal_regions.gyr_z_p, axis=0),
            )
        ).T
        U_g_a = np.vstack(  # noqa: invalid_name
            (
                np.mean(signal_regions.gyr_x_a, axis=0),
                np.mean(signal_regions.gyr_y_a, axis=0),
                np.mean(signal_regions.gyr_z_a, axis=0),
            )
        ).T

        # Eq. 9
        K_ga = (U_g_p - U_g_a) / (2 * self.grav)  # noqa: invalid_name
        cal_mat.K_ga = K_ga  # noqa: invalid_name

        # Gyroscope Scaling and Rotation

        # First apply partial calibration to remove offset and acc influence
        acc_x_rot_cor = cal_mat._calibrate_acc(signal_regions.acc_x_rot)
        acc_y_rot_cor = cal_mat._calibrate_acc(signal_regions.acc_y_rot)
        acc_z_rot_cor = cal_mat._calibrate_acc(signal_regions.acc_z_rot)
        gyr_x_rot_cor = cal_mat._calibrate_gyr_offsets(signal_regions.gyr_x_rot, acc_x_rot_cor)
        gyr_y_rot_cor = cal_mat._calibrate_gyr_offsets(signal_regions.gyr_y_rot, acc_y_rot_cor)
        gyr_z_rot_cor = cal_mat._calibrate_gyr_offsets(signal_regions.gyr_z_rot, acc_z_rot_cor)

        # Integrate gyro readings
        # Eg. 13/14
        W_s = np.zeros((3, 3))  # noqa: invalid_name
        W_s[:, 0] = np.sum(gyr_x_rot_cor, axis=0) / sampling_rate_hz
        W_s[:, 1] = np.sum(gyr_y_rot_cor, axis=0) / sampling_rate_hz
        W_s[:, 2] = np.sum(gyr_z_rot_cor, axis=0) / sampling_rate_hz

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
    However, the calibrate method will return a :class:`~imucal.TurntableCalibrationInfo` to indicate the expected
    higher precision of this calibration method.

    Further this Calibration expects rotations of 720 deg by default, as this is common for many turntables.
    For further information on the sign of the expected rotation angle see the :class:`~imucal.FerrarisCalibration`.

    See Also
    --------
    imucal.FerrarisCalibration

    """

    def __init__(
        self,
        grav: float = 9.81,
        expected_angle: float = -720,
        calibration_info_class: Type[TurntableCalibrationInfo] = TurntableCalibrationInfo,
    ):
        super().__init__(grav=grav, expected_angle=expected_angle, calibration_info_class=calibration_info_class)


def ferraris_regions_from_df(
    df: pd.DataFrame,
    acc_cols: Optional[Iterable[str]] = ("acc_x", "acc_y", "acc_z"),
    gyr_cols: Optional[Iterable[str]] = ("gyr_x", "gyr_y", "gyr_z"),
) -> FerrarisSignalRegions:
    """Create a Calibration object based on a dataframe which has all required sections labeled.

    The expected Dataframe has the section label as index and has at least the 6 required data columns.
    The index must contain all the sections listed in :class:`~imucal.FerrarisCalibration``.FERRARIS_SECTIONS`.

    Examples
    --------
    >>> import pandas as pd
    >>> sampling_rate = 100 #Hz
    >>> df = ... # A valid DataFrame with all sections in the index
    >>> print(df)
            acc_x acc_y   acc_z  gyr_x  gyr_y  gyr_z
    part
    x_a   -2052.0 -28.0   -73.0    1.0    0.0   -5.0
    x_a   -2059.0 -29.0   -77.0    2.0   -3.0   -5.0
    x_a   -2054.0 -25.0   -71.0    3.0   -2.0   -3.0
    ...       ...   ...     ...    ...    ...    ...
    z_rot   -36.0  35.0  2079.0    2.0   -5.0   -2.0
    z_rot   -28.0  36.0  2092.0    6.0   -5.0   -4.0
    z_rot   -36.0  21.0  2085.0    5.0   -4.0   -4.0
    >>> regions = ferraris_regions_from_df(df)
    >>> regions
    FerrarisSignalRegions(x_a=array([...]), ..., z_rot=array([...]))

    Parameters
    ----------
    df :
        6 column dataframe (3 acc, 3 gyro)
    acc_cols :
        The name of the 3 acceleration columns in order x,y,z.
    gyr_cols :
        The name of the 3 acceleration columns in order x,y,z.

    Returns
    -------
    ferraris_cal_obj : FerrarisSignalRegions

    See Also
    --------
    ferraris_regions_from_interactive_plot
    ferraris_regions_from_section_list

    """
    acc_df = df[list(acc_cols)]
    gyro_df = df[list(gyr_cols)]
    acc_dict = acc_df.groupby(level=0).apply(lambda x: x.values).to_dict()
    gyro_dict = gyro_df.groupby(level=0).apply(lambda x: x.values).to_dict()
    acc_dict = {"acc_" + k: v for k, v in acc_dict.items()}
    gyro_dict = {"gyr_" + k: v for k, v in gyro_dict.items()}

    return FerrarisSignalRegions(**acc_dict, **gyro_dict)


def ferraris_regions_from_section_list(
    data: pd.DataFrame,
    section_list: pd.DataFrame,
    acc_cols: Optional[Iterable[str]] = ("acc_x", "acc_y", "acc_z"),
    gyr_cols: Optional[Iterable[str]] = ("gyr_x", "gyr_y", "gyr_z"),
) -> FerrarisSignalRegions:
    """Create a Calibration object based on a valid section list.

    A section list marks the start and the endpoints of each required section in the data object.
    A valid section list is usually created using `FerrarisCalibration.from_interactive_plot()`.
    This section list can be stored on disk and this method can be used to turn it back into a valid calibration
    object.

    Parameters
    ----------
    data :
        6 column dataframe (3 acc, 3 gyro)
    section_list :
        A pandas dataframe representing a section list
    acc_cols :
        The name of the 3 acceleration columns in order x,y,z.
        Defaults to `FerrarisCalibration.ACC_COLS`
    gyr_cols :
        The name of the 3 acceleration columns in order x,y,z.
        Defaults to `FerrarisCalibration.GYRO_COLS`

    Returns
    -------
    ferraris_cal_obj : FerrarisSignalRegions


    Examples
    --------
    >>> import pandas as pd
    >>> # Load a valid section list from disk. Note the `index_col=0` to preserve correct format!
    >>> section_list = pd.read_csv('./calibration_sections.csv', index_col=0)
    >>> sampling_rate = 100 #Hz
    >>> df = ... # my data as 6 col pandas dataframe
    >>> regions = ferraris_regions_from_section_list(df)
    >>> regions
    FerrarisSignalRegions(x_a=array([...]), ..., z_rot=array([...]))

    See Also
    --------
    ferraris_regions_from_interactive_plot
    ferraris_regions_from_df

    """
    from imucal.calibration_gui import _convert_data_from_section_list_to_df  # noqa: import-outside-toplevel

    df = _convert_data_from_section_list_to_df(data, section_list)
    return ferraris_regions_from_df(df, acc_cols=acc_cols, gyr_cols=gyr_cols)


def ferraris_regions_from_interactive_plot(
    data: pd.DataFrame,
    acc_cols: Iterable[str] = ("acc_x", "acc_y", "acc_z"),
    gyr_cols: Iterable[str] = ("gyr_x", "gyr_y", "gyr_z"),
    title: Optional[str] = None,
) -> Tuple[FerrarisSignalRegions, pd.DataFrame]:
    """Create a Calibration object by selecting the individual signal sections manually in an interactive GUI.

    This will open a Tkinter Window that allows you to label the start and the end all required sections for a
    Ferraris Calibration.
    See the class docstring for more detailed explanations of these sections.

    Examples
    --------
    >>> sampling_rate = 100 #Hz
    >>> data = ... # my data as 6 col pandas dataframe
    >>> # This will open an interactive plot, where you can select the start and the stop sample of each region
    >>> regions, section_list = ferraris_regions_from_interactive_plot(data, sampling_rate=sampling_rate)
    >>> section_list.to_csv('./calibration_sections.csv')  # This is optional, but recommended
    >>> regions
    FerrarisSignalRegions(x_a=array([...]), ..., z_rot=array([...]))

    Parameters
    ----------
    data :
        6 column dataframe (3 acc, 3 gyro)
    acc_cols :
        The name of the 3 acceleration columns in order x,y,z.
    gyr_cols :
        The name of the 3 acceleration columns in order x,y,z.
    title :
        Optional title of the plot window

    Returns
    -------
    ferraris_cal_obj : FerrarisSignalRegions
    section_list : pd.DataFrame
        Section list representing the start and stop of each section.
        It is advised to save this to disk to avoid repeated manual labeling.
        :py:func:`~imucal.ferraris_regions_from_section_list` can be used to recreate the regions object

    See Also
    --------
    ferraris_regions_from_section_list
    ferraris_regions_from_df

    """
    acc = data[list(acc_cols)].to_numpy()
    gyr = data[list(gyr_cols)].to_numpy()

    section_list = _find_ferraris_regions_interactive(acc, gyr, title=title)
    return (
        ferraris_regions_from_section_list(data, section_list, gyr_cols=gyr_cols, acc_cols=acc_cols),
        section_list,
    )


def _find_ferraris_regions_interactive(acc: np.ndarray, gyro: np.ndarray, title: Optional[str] = None):
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
    from imucal.calibration_gui import CalibrationGui  # noqa: import-outside-toplevel

    plot = CalibrationGui(acc, gyro, FerrarisCalibration.FERRARIS_SECTIONS, title=title)

    section_list = plot.section_list

    check_all = (all(v) for v in section_list.values())
    if not all(check_all):
        raise ValueError("Some regions are missing in the section list. Label all regions before closing the plot")

    section_list = pd.DataFrame(section_list, index=("start", "end")).T

    return section_list
