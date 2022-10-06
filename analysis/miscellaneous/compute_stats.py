#!/usr/bin/env python

"""
Compute miscellaneous statistics referenced in the text
"""
import mne
import numpy as np

from data_organizer import ExperimentOrganizer
from utils_participant_meta import get_subjects_df

import pandas as pd


## Analyze start and stop timing of the experiments
organizer = ExperimentOrganizer()
experiments = organizer.get_experiments_for_analysis('participant_included')

durations_block = []
durations_experiment = []
durations_break = []

for experiment in experiments[:]:
    raw = mne.io.read_raw_fif(experiment.basepath / experiment.get_raw_original_filenames()[0])
    block_starts = raw.annotations[raw.annotations.description == 'BLOCK_START/EVENT/ON'].onset
    block_stops = raw.annotations[raw.annotations.description == 'BLOCK_STOP/EVENT/OFF'].onset
    durations_block.append(block_stops - block_starts)
    durations_experiment.append(block_stops[-1] - block_starts[0])
    durations_break.append(block_starts[1:] - block_stops[:-1])

durations_block = np.array(durations_block)
durations_experiment = np.array(durations_experiment)

avg_block = np.mean(durations_block)
avg_experiment = np.mean(durations_experiment)
print(f"The average block length is {avg_block} seconds ({avg_block/60} min) and average experiment duration is {avg_experiment} s ({avg_experiment/60} min)")
print(f"The average break time between blocks is {np.mean(durations_break)} s")

## Analyze age and sex distribution
df = get_subjects_df()

df = df.dropna(subset=['PID'])
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

for key in ['excluded_crit', 'excluded_other', 'participant_included', 'analysis_erd_freq', 'analysis_erd_timecourse']:
    df_key = df[df[key] == 'X']

    sex = df_key['Sex'].value_counts()
    age = df_key['Age'].mean()
    age_std = df_key['Age'].std()
    _str_sex = ', '.join([f"{s}={sex[s]}" for s in sex.index])
    print(f"For {key=}, {len(df_key)} datasets are marked. Age {age:0.1f}±{age_std:0.2f}. {_str_sex}")


n_measurements = df['Date'].count()
n_unique_dates = len(df['Date'].unique())
n_days_total = np.ptp(df['Date'])

print(f"In total {n_measurements} participants were recorded within {n_days_total.days}, on {n_unique_dates} days of measurement. "
      f"This averages to {n_measurements/n_unique_dates} measurements per measurement day")