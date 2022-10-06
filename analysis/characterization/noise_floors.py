#!/usr/bin/env python

"""
FieldLine OPM Sensor characterization
Module to compute the noise floors of the open and closed loop mode
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.signal

import mne

from utils_analysis import array_to_data_frame
from utils_plotting import save_fig, palette, set_context, mark_range_y
from data_organizer import ExperimentOrganizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()

##
experiment_open = organizer['20210709_empty_ak3b_ms2_shielding_all_sensors_test']
experiment_closed = organizer['20210709_empty_ak3b_ms2_shielding_all_sensors_test']
raw_open = mne.io.read_raw_fif(experiment_closed.basepath / 'empty_ak3b_ms2_shielding_all_sensors_test_open_raw.fif')
raw_closed = mne.io.read_raw_fif(experiment_open.basepath / 'empty_ak3b_ms2_shielding_all_sensors_test_closed_raw.fif')
filename = 'noise_floor_ak3b_measurement'

raw_open.crop(tmin=10, tmax=280)
raw_closed.crop(tmin=10, tmax=280)


##
picks_regex = '00:(0[4-9]|1.).*'  # select channels which were outside of the MS-2 shielded tube

picks_closed = mne.pick_channels_regexp(raw_closed.ch_names,picks_regex)
print(f"{len(picks_closed)} channels selected")
print([raw_closed.ch_names[i] for i in picks_closed])
picks_open = mne.pick_channels_regexp(raw_open.ch_names, picks_regex)

freqs, psd_open = scipy.signal.welch(raw_open.get_data(picks=picks_open), fs=raw_open.info['sfreq'], nperseg=2048, noverlap=1024, window='hann')
freqs, psd_closed, = scipy.signal.welch(raw_closed.get_data(picks=picks_closed), fs=raw_closed.info['sfreq'], nperseg=2048, noverlap=1024, window='hann')

df_open = array_to_data_frame(psd_open, axes=[list(range(len(picks_open))), freqs], axes_names=['Sensor', 'Frequency']).reset_index()
df_open['Mode'] = 'Open'
df_closed = array_to_data_frame(psd_closed, axes=[list(range(len(picks_closed))), freqs], axes_names=['Sensor', 'Frequency']).reset_index()
df_closed['Mode'] = 'Closed'

df = pd.concat([df_closed, df_open], ignore_index=True)
df['noise_floor'] = np.sqrt(df.value)*1e15

##
for foi in [1, 2, 5, 10, 100, 200, 300, 450]:
    print(f"Computing noise floors at frequency {foi}+/-1 Hz")
    df_noise_floor = df[(df['Frequency'] >= foi - 1) & (df['Frequency'] <= foi + 1)].groupby(['Mode', 'Sensor']).mean().groupby(['Mode']).mean()
    df_noise_floor['Frequency'] = foi
    print(df_noise_floor)

print(f"Measurement duration: Open loop: {np.ptp(raw_open.times)/60} min, Closed loop: {np.ptp(raw_closed.times)/60} min")

##
set_context('paper')

fmin, fmax = 0, 70
fig, [ax0, ax1] = plt.subplots(ncols=2)
sns.lineplot(data=df[df['Frequency'] <= 490], x='Frequency', y='noise_floor', style='Mode', hue='Mode', ci=95, palette=palette, ax=ax0)
sns.lineplot(data=df[(df['Frequency'] <= fmax) & (df['Frequency'] > fmin)], x='Frequency', y='noise_floor', style='Mode', hue='Mode', ci=95, palette=palette, ax=ax1)
mark_range_y(fmin, fmax, ax=ax0, alpha=0.1)

ax0.set(ylabel=r'Noise floor ($\textrm{fT}/\sqrt{\textrm{Hz}}$)', xlabel='Frequency (Hz)')
ax0.set(ylim=(0, 85))
ax0.legend().remove()
ax1.set(ylabel=r'', xlabel='Frequency (Hz)')
ax1.set(ylim=(15, 45))

save_fig(fig, filename=filename, subdir='characterization', figsize_pgf=(6.5, 2.5))