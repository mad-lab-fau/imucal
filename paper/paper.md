---
title: 'imucal - A Python library to calibrate 6 DOF IMUs'  
tags:
  - Python
  - Machine Learning
  - Data Analysis
authors:
  - name: Arne Küderle^[corresponding author]   
    orcid: 0000-0002-5686-281X  
    affiliation: 1
  - name: Nils Roth
    orcid: 0000-0002-9166-3920
    affiliation: 1
  - name: Robert Richer  
    orcid: 0000-0003-0272-5403  
    affiliation: 1
  - name: Bjoern M. Eskofier  
    orcid: 0000-0002-0417-0336  
    affiliation: 1
affiliations:
  - name: Machine Learning and Data Analytics Lab (MaD Lab), Department Artificial Intelligence in Biomedical Engineering (AIBE), Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
    index: 1
date: 06 March 2022
bibliography: imucal.bib
---

# Summary

Inertial measurement units (IMUs) have wide application areas from human movement analysis to commercial drone navigation.
However, to use modern micro-electromechanical systems (MEMS)-based IMUs a calibration is required to transform the raw output of the sensor into physically meaningful units.
To obtain such calibrations one needs to perform sets of predefined motions with the sensor unit and then apply a calibration algorithm to obtain the required transformation and correction factors for this unit.
The `imucal` library implements the calibration algorithm described by Ferraris et al. [@Ferraris1994; @Ferraris1995] and provides functionality to calculate calibration parameters and apply them to new measurements.
As typically multiple calibrations are recorded per sensor over time, `imucal` further provides a set of opinionated tools to save, organize, and retrieve recorded calibrations.
This helps to make sure that always the best possible calibration is applied for each recording even when dealing with multiple sensors and dozens of measurements.

# Statement of Need

When working with MEMS-based IMUs, calibrations are required to correct sensor errors like bias, scaling, or non-orthogonality of the included gyroscope and accelerometer as well as to transform the raw sensor output into physical units.
While out-of-the-box factory calibrations and self-calibration procedures have become better over the past years, high precision applications still benefit from manual calibrations temporally close to the measurement itself.
This is because the parameters of the sensor can change because of the soldering process, change in humidity, or temperature. Also, it could simply change over time as the silicon ages.
Various algorithms and protocols exist to tackle this issue. To calibrate the accelerometer, most of them require the sensor unit to be placed in multiple well-defined orientations relative to gravity. To calibrate the gyroscope, the sensor is required to be rotated either with known angular rate or by a known degree.
From the data recorded in the different phases of the calibrations, the correction and transformation parameters for a specific sensor can be calculated.

While multiple of these procedures have been published in literature, for example [@Skog2006; @Tedaldi2014a; @Ferraris1994; @Kozlov2014; @Qureshi2017; @Scapellato2005; @Zhang2009], no high-quality code implementations are available for most of them.
Existing implementations that can be found on the internet are usually "one-off" scripts that would require adaptation and tinkering to make them usable for custom use cases.
Further, many practical aspects of calibrating IMUs, like which information needs to be stored to make sure that a calibration can be correctly applied to a new recording, are usually not discussed in research papers or are not easily available online.

Hence, well-maintained reference implementations of algorithms, clear guidelines, and informal guides are needed to make the procedure of creating and using calibrations easier.

# Provided Functionality

With `imucal` and its documentation, we address all the above needs and hope to even further expand on that in the future based on community feedback.
The library provides a sensor-agnostic object-oriented implementation of the calibration algorithm by Ferraris et al. [@Ferraris1994,@Ferraris1995] and functionality to apply it to new data.
Further, we provide a simple GUI interface to annotate recorded calibration sessions (\autoref{fig:ferraris_gui}).

\begin{figure}[!h]
\includegraphics[width=0.9\textwidth]{img/imucal_ferraris_gui.png}
\caption{Screenshot of the GUI to annotate recorded Ferraris sessions.
Each region corresponds to one of the required static positions or rotations.
The annotation is performed using the mouse with support for keyboard shortcuts to speed up some interactions.}
\label{fig:ferraris_gui}
\end{figure}

When working with sensors and multiple calibrations, storing and managing them can become complicated.
Therefore, `imucal` also implements a set of opinionated helpers to store the calibrations and required metadata as _.json_ files and functions to retrieve them based on sensor ID, date, type of calibration, or custom metadata.

While `imucal` itself only implements a single calibration algorithm so far, all tools in the library have been designed with the idea of having multiple algorithms in mind.
Therefore, the provided structure and base classes should provide a solid basis to implement further algorithms, either as part of the library itself or as part of custom software packages.

To ensure that all the provided tools are usable, the documentation contains full guides on how to practically perform the calibration and then use `imucal` to process the recorded sessions.
As much as possible, these guides include informal tips to avoid common pitfalls.

The `imucal` library has been used extensively in the background of all movement research at the Machine Learning and Data Analytics Lab (MaD Lab) to calibrate our over 100 custom and commercial IMU sensors.
Therefore, we hope this library can bring similar value to research groups working on IMU-related topics.  

# Availability

The software is available as a pip installable package (`pip install imucal`) and via [GitHub](https://github.com/mad-lab-fau/imucal).
Documentation can be found on [Read the Docs](https://imucal.readthedocs.io/).

# Acknowledgments

`imucal` was developed to solve the chaos of random calibration scripts, old calibrations in unknown formats on shared folders, and general uncertainty when it came to calibrating or finding calibrations for one of the hundreds of self-build or off-the-shelf IMU units at the MaD Lab.
Therefore, we would like to thank all members of the team and our students for their feedback and suggestions when working with the library.
