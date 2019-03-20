# -*- coding: utf8 -*-
import os

from setuptools import setup, find_packages

# Meta information
dirname = os.path.dirname(__file__)

setup(
    # Basic info
    name='imucal',
    description='Convenience methods to calibrate IMUs.',
    long_description=open('README.md').read(),
    classifiers=[
    ],

    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'pandas',
        'numpy'
    ],
)
