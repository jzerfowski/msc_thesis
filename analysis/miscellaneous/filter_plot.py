#!/usr/bin/env python

"""
Appendix
Module to plot the spectral filters used in offline and online analysis
"""

import mne
import matplotlib.pyplot as plt
import numpy as np

from utils_plotting import save_fig, set_context

from offline.utils_analysis_offline import get_filter_params

sfreq = 1000

## Filter Offline
set_context(None)
center_freq = 10
params_filter = get_filter_params(erd_freq=center_freq, mode='erd')
f = mne.filter.create_filter(**params_filter, data=None, sfreq=sfreq)

print(f"Filter params: {params_filter}")

fig = mne.viz.plot_filter(f, sfreq=sfreq)

save_fig(fig, f"app_filter_response_offline", 'appendix', figsize_pgf=(6.5, 4.5))

## Filter Online
set_context(None)
from BandpassFilterNode import BandpassFilterNode
sfreq = 1000
node_bandpass = BandpassFilterNode(in_channel_labels=['In-1'], out_channel_labels=None, filter_length=499, f_highpass=8.5, f_lowpass=11.5, sfreq=sfreq)

fig = mne.viz.plot_filter(node_bandpass.filter_b, sfreq=sfreq, compensate=True)

save_fig(fig, f"app_filter_response_online", 'appendix', figsize_pgf=(6.5, 4.5))


## Filter single-pole
# See https://tomroelandts.com/articles/low-pass-single-pole-iir-filter

set_context('paper')

sfreq = 10
time_const = 0.5
t_block = time_const * sfreq
decay_factor = 1-np.exp(-1/t_block)

freqs = np.linspace(0, np.pi, 300, endpoint=False, )

fr = np.abs(decay_factor/(1-(1-decay_factor)*np.exp(-1j*freqs)))
fr_dB = 20*np.log10(fr)
threedB = freqs[np.argmin(np.abs(fr_dB + 3))]/(2*np.pi)*sfreq

fig, ax = plt.subplots()
ax.plot(freqs/np.pi*sfreq/2, fr_dB)
ax.axvline(threedB, color='black')
ax.axhline(-3, color='black', linestyle = '--')
ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (dB)')

save_fig(fig, f"app_filter_response_single_pole", 'appendix', figsize_pgf=(6.5, 4.5))


