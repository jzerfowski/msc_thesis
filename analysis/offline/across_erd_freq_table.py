#!/usr/bin/env python

"""
Offline Analysis
Module to build Table 3.1
"""

import numpy as np
import pandas as pd

from data_organizer import ExperimentOrganizer

##
organizer = ExperimentOrganizer()
experiments = organizer.get_experiments_for_analysis('analysis_erd_timecourse')

freqs = {'PID': [], 'Freq': []}

for experiment in experiments[:]:
    participant_id = experiment.get_participant_id()
    erd_freq = experiment.get_erd_freq()
    freqs['PID'].append(participant_id)
    freqs['Freq'].append(erd_freq)

freqs['PID'].append('Avg.')
freqs['Freq'].append(f"{np.nanmean(freqs['Freq']):.1f}±{np.nanstd(freqs['Freq']):.1f}")

df = pd.DataFrame.from_dict(data=freqs)
df = df.rename(columns={'PID': 'Participant', 'Freq': "Frequency (Hz)"})
df = df.set_index('Participant')

print(df.T.to_latex())