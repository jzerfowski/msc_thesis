#!/usr/bin/env python

"""
Default settings for the data_organizer can be defined here
"""

import pathlib

default_experiment_basepath = pathlib.Path(r"D:\Thesis\data\thesis")
default_experiment_basepaths = [default_experiment_basepath]
default_participant_meta_filepath = default_experiment_basepath / 'participant_meta.xlsx'
ERD_FREQ_DEFAULT = 12