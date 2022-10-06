#!/usr/bin/env python

"""Module documentation goes here"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import scipy.signal


bandpass_center = 12
bandpass_width = 3

f_highpass = max([0.01, bandpass_center-bandpass_width/2])
f_lowpass = min([499, bandpass_center+bandpass_width/2])

filter_b = scipy.signal.firwin(499, [f_highpass, f_lowpass], pass_zero=False,
                                    fs=1000)
filter_a = 1

w, h = scipy.signal.freqz(filter_b, filter_a, fs=1000)
fig, ax1 = plt.subplots()
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.axhline(-3)
ax1.axvline(f_highpass)
ax1.axvline(f_lowpass)
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
