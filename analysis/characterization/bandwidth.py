#!/usr/bin/env python

"""
FieldLine OPM Sensor characterization
Module to analyze the open and closed loop bandwidth
"""

import mne.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from characterization.utils_characterization import fit_sine_data_known_freq, get_float_annotations
from utils_plotting import save_fig, set_context, palette
from data_organizer import ExperimentOrganizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()

experiment = organizer['20210702_bandwidth_dynamic_range_ak3b']
##
raw_closed = mne.io.read_raw_fif(experiment.basepath / 'bandwidth_closed_01_annotated_raw.fif')
raw_open = mne.io.read_raw_fif(experiment.basepath / 'bandwidth_open_01_annotated_raw.fif')

# all amplitudes are peak amplitudes, not peak to peak!
amplitude_applied = 50e-12


def get_bandwidth_df(raw):
    sfreq = raw.info['sfreq']

    fits = pd.DataFrame()

    for frequency, annotation in get_float_annotations(raw.annotations):
        signal = raw.get_data(tmin=annotation['onset'], tmax=annotation['onset']+annotation['duration'])
        fit = fit_sine_data_known_freq(signal, sfreq, frequency_applied=frequency)
        fits = pd.concat([fits, fit])

    fits = fits.reset_index()
    return fits

##
fits_closed = get_bandwidth_df(raw_closed)
fits_closed['Mode'] = 'Closed'

fits_open = get_bandwidth_df(raw_open)
fits_open['Mode'] = 'Open'

##
fits = pd.concat([fits_closed, fits_open], ignore_index=True)

fits = fits[fits['frequency_applied'] < 500]
fits = fits[fits['frequency_applied'] != 50]  # Skip the 50 Hz frequency bin to avoid interference with line noise

fits['amplitude_applied'] = amplitude_applied

# Determine the amplitude for the lowest frequency as 'reference'
# fits['amplitude_ref'] = fits['amplitude_applied']
fits = fits.set_index(keys=['Sensor', 'Mode'])
frequency_applied_ref = fits['frequency_applied'].min()
amplitude_measured_ref = fits[fits['frequency_applied'] == frequency_applied_ref]['amplitude_measured']
fits['amplitude_ref'] = amplitude_measured_ref
fits = fits.reset_index()

fits['amplitude_measured_dB'] = 20 * np.log10(fits['amplitude_measured'] / fits['amplitude_ref'])


##
print(f"The reference frequency is {frequency_applied_ref}.")
print(f"The applied amplitude (p2p) was {amplitude_applied*2e12}pT")
print(f"The measured reference amplitudes (peak-to-peak in pT) are \n{amplitude_measured_ref*2e12}")

##
set_context('paper')

fig_dB, ax = plt.subplots()
sns.lineplot(data=fits, x='frequency_applied', y='amplitude_measured_dB', style='Sensor', palette=palette, hue='Mode', dashes=[(1, 0), (2, 3)], ax=ax)
ax.set(xlabel="Frequency applied (Hz)", ylabel="Magnitude (dB)")
ax.axhline(-3, color='grey', linewidth=2, zorder=1, alpha=0.7)
ax.set_yticks(list(ax.get_yticks()) + [-3])
ax.set_ylim((-20, None))
ax.legend(ncol=2)
fig_dB.show()

save_fig(fig_dB, 'bandwidth_dB', 'characterization')


