#!/usr/bin/env python

"""
Utility functions
Module to prepare subject data for analyses
"""

import pandas

from settings import default_participant_meta_filepath


def get_subjects_df(participants_meta_filepath=None) -> pandas.DataFrame:
    if participants_meta_filepath is None:
        participants_meta_filepath = default_participant_meta_filepath
    return pandas.read_excel(participants_meta_filepath)


def get_experiment_names_for_analysis(analysis_name='all', participants_meta_filepath=None):
    subjects_df = get_subjects_df(participants_meta_filepath=participants_meta_filepath)
    return subjects_df["EID"][subjects_df[analysis_name] == 'X'].tolist()


def get_info_for_experiment_name(experiment_name, participants_meta_filepath=None) -> dict:
    subjects_df = get_subjects_df(participants_meta_filepath=participants_meta_filepath)
    subject_df = subjects_df[subjects_df['EID'] == experiment_name][0:1]
    d = subject_df.T.to_dict()
    subject_dict = d[list(d.keys())[0]]
    return subject_dict