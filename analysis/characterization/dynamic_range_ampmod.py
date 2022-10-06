#!/usr/bin/env python

"""
FieldLine OPM Sensor characterization
Module to compute the measured amplitude dependent on the applied amplitude for dynamic range estimation
"""

import mne.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils_plotting import save_fig, mark_range_y, palette
from utils_analysis import array_to_data_frame

sns.set_style('darkgrid')
sns.set_context('paper')

from characterization.utils_characterization import fit_sine_data_known_freq, get_float_annotations
from data_organizer import ExperimentOrganizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()

experiment = organizer['20210702_bandwidth_dynamic_range_ak3b']
filename_closed = 'dynamic_range_ampmod_closed_40nTpp_12Hz_annotated_raw.fif'
filename_open = 'dynamic_range_ampmod_open_10nTpp_12Hz_annotated_raw.fif'

frequency_applied = 12

raw_closed = mne.io.read_raw_fif(experiment.basepath / filename_closed)
raw_open = mne.io.read_raw_fif(experiment.basepath / filename_open)

##
def get_dynamic_range_df(raw):
    sfreq = raw.info['sfreq']
    fits = pd.DataFrame()

    for amplitude, annotation in get_float_annotations(raw.annotations):
        signal = raw.get_data(tmin=annotation['onset'], tmax=annotation['onset']+annotation['duration'])
        fit = fit_sine_data_known_freq(signal, sfreq, frequency_applied=frequency_applied)
        fit['amplitude_applied'] = amplitude
        fits = pd.concat([fits, fit])

    fits = fits.reset_index()
    return fits

fits_closed = get_dynamic_range_df(raw_closed)
fits_open = get_dynamic_range_df(raw_open)

fits_closed['Mode'] = 'Closed'
fits_open['Mode'] = 'Open'

fits_closed = fits_closed[fits_closed['amplitude_applied'] >= 1e-10]
fits_open = fits_open[fits_open['amplitude_applied'] >= 1e-10]


##
fits = pd.concat([fits_closed, fits_open], ignore_index=True)

fits[['amplitude_measured_nT', 'amplitude_applied_nT']] = fits[['amplitude_measured', 'amplitude_applied']] * 1e9 # in nT
fits['amplitude_applied_nT_p2p'] = fits['amplitude_applied_nT'] * 2

##
fits = fits.set_index(keys=['Sensor', 'Mode'])
amplitude_measured_ref = fits[fits['amplitude_applied_nT_p2p'] == 0.4]['amplitude_measured']
# amplitude_measured_ref = fits[fits['frequency_applied'] == frequency_applied_ref]['amplitude_measured']
fits['amplitude_ref'] = amplitude_measured_ref
fits = fits.reset_index()

##
fits['amplitude_ratio_ref'] = fits['amplitude_measured'] / fits['amplitude_ref']
fits['amplitude_ratio_applied'] = fits['amplitude_measured'] / fits['amplitude_applied']


## Plot for the results section
amin, amax = 0, 5
fig, (ax0, ax1) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [0.65, 0.35]})
sns.lineplot(data=fits, x='amplitude_applied_nT', y='amplitude_ratio_applied', hue='Mode', palette=palette, style='Sensor', ax=ax0)
sns.lineplot(data=fits, x='amplitude_applied_nT', y='amplitude_ratio_applied', hue='Mode', palette=palette, style='Sensor', ax=ax1)
mark_range_y(amin, amax, ax=ax0, alpha=0.1)
ax0.set(xlabel='Amplitude applied $A_0$ (nT)', ylabel='Amplitude ratio $A_m$/$A_0$', xlim=[-0.2, 16], ylim=[0.96, 1.00])
ax1.set(xlabel='Amplitude applied $A_0$ (nT)', ylabel='', xlim=[-0.2, 5.5], ylim=[0.8, 1.])
ax1.set_xticks([0, 1, 2, 3, 4, 5])

ax0.legend(loc='lower right', ncol=2)
ax1.legend().remove()
fig.show()

save_fig(fig, 'dynamic_range_amplitude_modulation_both', 'characterization', figsize_pgf=(6.5, 2.5))


## Plot of the full amplitude range for appendix
fig, ax = plt.subplots()
sns.lineplot(data=fits, x='amplitude_applied_nT', y='amplitude_ratio_applied', hue='Mode', palette=palette, style='Sensor', ax=ax)
ax.set(xlabel='Amplitude applied $A_0$ (nT)', ylabel='Amplitude ratio $A_m$/$A_0$')
ax.legend(loc='upper left', ncol=2)
fig.show()

save_fig(fig, 'app_dynamic_range_amplitude_modulation_full', 'appendix', figsize_pgf=(6.5, 2.5))

## Plot of signal railing for appendix
for raw, mode, annotations in zip([raw_open, raw_closed], ['Open', 'Closed'], [[20, 50, 70, 90], [40, 70, 75, 80, 90]]):

    annotations_example = get_float_annotations(raw.annotations[annotations])

    sfreq = raw.info['sfreq']

    fig, axes = plt.subplots(ncols=len(annotations_example), sharey=True, sharex=True)

    for (amplitude, annotation), ax in zip(annotations_example, axes.flat):
        signal = raw.get_data(tmin=annotation['onset'], tmax=annotation['onset'] + annotation['duration'])
        n_sensors, n_times = signal.shape

        data = array_to_data_frame(signal * 1e9, axes=[range(n_sensors), np.arange(n_times) / sfreq], axes_names=['Sensor', 't'])
        sns.lineplot(data=data, x='t', y='value', style='Sensor', palette=palette,  ax=ax, color=palette[mode])
        ax.axhline(amplitude*1e9, color='grey', linewidth=2, zorder=1, alpha=0.7)
        ax.axhline(-amplitude*1e9, color='grey', linewidth=2, zorder=1, alpha=0.7)
        if mode == 'Closed':
            title = f"$A_0={amplitude * 1e9:.0f}$ nT"
        else:
            title = f"$A_0={amplitude * 1e9:.1f}$ nT"
        ax.set(xlabel='Time (s)', ylabel='Measured field (nT)', xlim=(0, 0.4), title=title)
        if not ax.get_subplotspec().is_first_col():
            ax.legend().set_visible(False)
    axes[0].set_xticks([0, 0.2])

    save_fig(fig, f'app_dynamic_range_amplitude_modulation_signal_{mode.lower()}', 'appendix')
