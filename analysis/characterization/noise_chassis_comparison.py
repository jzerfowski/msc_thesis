#!/usr/bin/env python
import scipy.signal

import xdf2mne

"""
FieldLine OPM Sensor characterization
Module to compute the noise floor with a distinction per chassis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal

import mne


from data_organizer import ExperimentOrganizer
from utils_plotting import save_fig, palette, set_context, mark_range_y
from utils_analysis import array_to_data_frame

from positioning_utils import add_channel_coords_from_file

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()

# Load experiment data
experiment = organizer['20211222_empty_room_sensors_in_grid']
raw = mne.io.read_raw_fif(experiment.basepath / '20211222_130546_PXX_empty_room_run2_raw.fif')
add_channel_coords_from_file(raw.info, fname=experiment.basepath / 'grid_assignment.json')

# Compute Spectra
freqs, psd = scipy.signal.welch(raw.get_data(picks='all'), fs=raw.info['sfreq'], nperseg=2048, noverlap=1024, window='hann')
df = array_to_data_frame(psd, axes=[raw.ch_names, freqs], axes_names=['Sensor', 'Frequency']).reset_index()
df['NoiseFloor'] = np.sqrt(df.value)*1e15
df['Chassis'] = df['Sensor'].map(lambda s: 0 if s.startswith('00:') else 1)
df['ChannelIdx'] = df['Sensor'].map(lambda s: raw.ch_names.index(s))

##
noise_foi = 300
df_noise_floor = df.loc[df['Frequency'] == freqs[np.argmin(np.abs((freqs - noise_foi)))]]
noise_floor = df_noise_floor['NoiseFloor']
noise_floor_zscored = scipy.stats.zscore(noise_floor)
noise_floor_outliers = noise_floor_zscored > 3
sensor_outliers = [ch_name for ch_name, is_outlier in zip(raw.ch_names, noise_floor_outliers) if is_outlier]


##
set_context('paper')
fig, axes = plt.subplots(ncols=2)
ax_sensor, ax_avg = axes[0], axes[1]
sns.lineplot(data=df.loc[(df["Frequency"]>=4) & (df['Frequency'] < 500) & (df['Sensor'].map(lambda s: s not in sensor_outliers))], x='Frequency', y='NoiseFloor', hue='Chassis', ci=95, legend='auto', ax=ax_avg, palette=palette)
sns.lineplot(data=df.loc[(df["Frequency"]>=4) & (df['Frequency'] < 500)], x='Frequency', y='NoiseFloor', hue='Chassis', style='Sensor', legend=False, ax=ax_sensor, palette=palette)
axes[0].set(xlabel='Frequency (Hz)', ylabel=r'Noise floor ($\textrm{fT}/\sqrt{\textrm{Hz}}$)', )
axes[1].set(xlabel='Frequency (Hz)', ylabel=None)

ax_avg.set(ylim = (15, 60), title='Chassis average (w/o outliers)')
ax_sensor.set(ylabel=None, title='Single sensors')

dx, dy = 10, -10
for noise_floor_value in noise_floor[noise_floor_outliers].values:
    ax_sensor.annotate('Outlier', (noise_foi, noise_floor_value), (noise_foi-230, noise_floor_value), arrowprops=dict(arrowstyle="->", color='black'))
    # plt.arrow(noise_foi-dx, noise_floor_value-dy, dx, dy, facecolor='red', length_includes_head=True, arrowprops=dict(arrowstyle="->"))

save_fig(fig, filename='app_noise_floor_chassis_comparison', subdir='appendix')
