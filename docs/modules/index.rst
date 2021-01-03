=============
API Reference
=============

Ferraris(based) Calibrations
============================

Calibration Classes
-------------------
Classes to actually calculate calibrations based on the Ferraris method.

.. currentmodule:: imucal
.. autosummary::
   :toctree: generated
   :template: class_with_private.rst

    FerrarisCalibration
    TurntableCalibration

Calibration Info Classes
------------------------

Class objects representing the calibration results and have methods to apply these calibrations to data, save them to
disk and load them again.

.. currentmodule:: imucal
.. autosummary::
   :toctree: generated
   :template: class.rst

    FerrarisCalibrationInfo
    TurntableCalibrationInfo

Data Preparation Helper
-----------------------

Helper Functions to generate valid input data for Ferraris like calibrations.

.. currentmodule:: imucal
.. autosummary::
   :toctree: generated
   :template: function.rst

    ferraris_regions_from_interactive_plot
    ferraris_regions_from_df
    ferraris_regions_from_section_list

.. autosummary::
   :toctree: generated
   :template: class.rst

    FerrarisSignalRegions


Calibration File Management
===========================

.. automodule:: imucal.management
    :no-members:
    :no-inherited-members:

.. currentmodule:: imucal.management
.. autosummary::
   :toctree: generated
   :template: function.rst

    load_calibration_info
    save_calibration_info
    find_calibration_info_for_sensor
    find_closest_calibration_info_to_date

.. autosummary::
   :toctree: generated
   :template: class.rst

    CalibrationWarning

Label GUI
=========
The gui label class.
Normally you do not need to interact with these directly.

.. currentmodule:: imucal.calibration_gui

Classes
-------
.. autosummary::
   :toctree: generated
   :template: class_with_private.rst

    CalibrationGui

Constants
=========



Base Classes
============
This is only interesting for developers!

.. currentmodule:: imucal

.. autosummary::
   :toctree: generated
   :template: class_with_private.rst

    CalibrationInfo