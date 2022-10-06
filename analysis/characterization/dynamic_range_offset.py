#!/usr/bin/env python

"""
FieldLine OPM Sensor characterization
Module to compute the measured amplitude dependent on the applied offset field for dynamic range estimation
"""

import mne.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils_plotting import palette, save_fig, set_context

from characterization.utils_characterization import fit_sine_data_known_freq, get_float_annotations
from utils_analysis import array_to_data_frame
from data_organizer import ExperimentOrganizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()

experiment = organizer['20210702_bandwidth_dynamic_range_ak3b']
filename_closed = 'dynamic_range_offset_closed_10nTpp_12Hz_annotated_raw.fif'
filename_open = 'dynamic_range_offset_open_10nTpp_12Hz_annotated_raw.fif'

frequency_applied = 12

# all amplitudes are peak amplitudes, not peak to peak!
amplitude_applied = 50e-12

raw_closed = mne.io.read_raw_fif(experiment.basepath / filename_closed)
raw_open = mne.io.read_raw_fif(experiment.basepath / filename_open)

##
def get_dynamic_range_df(raw):
    sfreq = raw.info['sfreq']
    fits = pd.DataFrame()

    for offset, annotation in get_float_annotations(raw.annotations):
        signal = raw.get_data(tmin=annotation['onset'], tmax=annotation['onset']+annotation['duration'])
        fit = fit_sine_data_known_freq(signal, sfreq, frequency_applied=frequency_applied)
        fit['offset_applied'] = offset
        fits = pd.concat([fits, fit])

    fits = fits.reset_index()
    return fits

fits_closed = get_dynamic_range_df(raw_closed)
fits_closed['Mode'] = 'Closed'

fits_open = get_dynamic_range_df(raw_open)
fits_open['Mode'] = 'Open'

fits = pd.concat([fits_closed, fits_open], ignore_index=True)

fits['amplitude_applied'] = amplitude_applied

##
fits = fits.set_index(keys=['Sensor', 'Mode'])
fits_ref = fits[fits['offset_applied'] == fits['offset_applied'].abs().min()]['amplitude_measured']

fits['amplitude_ref'] = fits_ref.groupby(fits_ref.index).mean()

fits = fits.reset_index()
fits['amplitude_measured_dB'] = 20 * np.log10(fits['amplitude_measured'] / fits['amplitude_ref'])
fits['attenuation_dB'] = 20 * np.log10(fits['amplitude_ref']/ fits['amplitude_measured'])
fits['magnitude_dB']  = -20 * np.log10(fits['amplitude_ref']/ fits['amplitude_measured'])
fits[['offset_applied_nT', 'offset_measured_nT']] = fits[['offset_applied', 'offset_measured']] * 1e9

##
set_context('paper')

fig_amplitude, ax = plt.subplots()
sns.lineplot(data=fits, x='offset_applied_nT', y='magnitude_dB', hue='Mode', palette=palette, style='Sensor', ax=ax)
ylims = ax.get_ylim()
ax.axhline(-3, color='grey', linewidth=2, zorder=1, alpha=0.7)
ax.set_yticks(list(ax.get_yticks()) + [-3])
ax.set_ylim(*ylims)

ax.legend(loc='lower center')
ax.set(xlabel='Offset applied (nT)', ylabel='Magnitude (dB)')
fig_amplitude.show()

save_fig(fig_amplitude, 'dynamic_range_offset_modulation_amplitude', 'characterization', figsize_pgf=(6.5, 2.5))

##
fig_offset, ax = plt.subplots()
sns.lineplot(data=fits, x='offset_applied_nT', y='offset_measured_nT', hue='Mode', palette=palette, style='Sensor', ax=ax)
ax.set(xlabel=r'Offset applied (nT)', ylabel='Offset measured (nT)')
ax.legend()
fig_offset.show()
##
save_fig(fig_offset, 'app_dynamic_range_offset_modulation_offset', 'appendix', figsize_pgf=(6.5, 2.5))

##
for raw, mode, annotations in zip([raw_open, raw_closed], ['open', 'closed'], [[0, 20, 40, 70], [0, 20, 40, 70]]):

    annotations_example = get_float_annotations(raw.annotations[annotations])

    sfreq = raw.info['sfreq']

    fig, axes = plt.subplots(ncols=len(annotations_example))

    for (amplitude, annotation), ax in zip(annotations_example, axes.flat):
        signal = raw.get_data(tmin=annotation['onset'], tmax=annotation['onset'] + annotation['duration'])
        n_sensors, n_times = signal.shape

        data = array_to_data_frame(signal * 1e9, axes=[range(n_sensors), np.arange(n_times) / sfreq], axes_names=['Sensor', 't'])
        sns.lineplot(data=data, x='t', y='value', style='Sensor', palette=palette,  ax=ax)

        scale_pos = np.mean(signal[0])*1e9
        y1, y2 = scale_pos - 50e-3, scale_pos + 50e-3
        ax.fill_between(x=[0, 1], y1=y1, y2=y2, transform=ax.get_yaxis_transform(), alpha=0.1)

        # ax.axhline(-amplitude*1e9, color='grey', linewidth=2, zorder=1, alpha=0.7)
        ax.set(xlabel='Time (s)', ylabel='', xlim=(0, 0.4), title=r"\textrm{Offset}"+f"$={amplitude*1e9:.0f}\,nT$")

        ylim = ax.get_ylim()
        ax.axhline(amplitude*1e9, color='grey', linewidth=2, zorder=1, alpha=0.7)
        ax.set(ylim=ylim)

        if not ax.get_subplotspec().is_last_col():
            ax.legend().set_visible(False)

    axes.flat[0].set(ylabel='Measured field (nT)')
    save_fig(fig, f'app_dynamic_range_amplitude_modulation_signal_{mode}', 'appendix')
