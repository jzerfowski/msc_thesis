#!/usr/bin/env python

"""
FieldLine OPM Sensor characterization
Utility functions
Module with utility functions for the sensor characterization
"""

import numpy as np
import pandas as pd
import mne
from scipy.optimize import curve_fit


import logging
logger = logging.getLogger(__name__)

def my_sine(t, frequency, amplitude, phase, offset):
    return np.sin(t * 2 * np.pi * frequency + phase) * amplitude + offset


def fit_sine_data_known_freq(signal, sfreq, frequency_applied):
    n_sensors, n_times = signal.shape

    # For the amplitude we assume that the mean of the signal is 0.
    # Then RMS (Root mean square) is = std(signal)
    # https://en.wikipedia.org/wiki/Root_mean_square
    # For a zero-mean sine wave, the relationship between RMS and peak-to-peak amplitude is:
    # peak2peak = 2*sqrt(2)*rms
    # Thus our guessed amplitude should be sqrt(2)*rms(x) = sqrt(2) * std(x)
    amplitude_guess = np.sqrt(2) * np.std((signal - np.mean(signal, axis=1)[..., np.newaxis]), axis=1)
    offset_guess = np.mean(signal, axis=1)

    t = np.arange(0, n_times) / sfreq

    fits = []
    for i in range(n_sensors):
        phase_guess = 0
        p0 = [amplitude_guess[i], phase_guess, offset_guess[i]]
        # p_bounds = tuple(zip([0, np.inf], [-np.pi, np.pi], [-np.inf, np.inf]))
        def p_sine(t, p_amplitude, p_phase, p_offset):
            return my_sine(t, frequency_applied, amplitude=p_amplitude, phase=p_phase, offset=p_offset)
        popt, pcov = curve_fit(p_sine, t, signal[i, :], p0=p0, method='lm')
        fits.append(popt)
        logger.info(f"Fitted sine (a*sin(t*2*pi*{frequency_applied}+phase)+offset for sensor {i}\n" \
                    f"Guessed parameters:\n" \
                    f"f=a(p2p)={p0[0] * 2 * 1e12:.3f}pT, phase={p0[1] / (2 * np.pi) * 360:.2f}°, offset={p0[2] * 1e12:.2f}pT\n" \
                    f"Fitted parameters:\n"
                    f"f=a(p2p)={popt[0] * 2 * 1e12:.3f}pT, phase={popt[1] / (2 * np.pi) * 360:.2f}°, offset={popt[2] * 1e12:.2f}pT")

    index = pd.Index(name='Sensor', data=list(range(n_sensors)))
    fit = pd.DataFrame(data=fits, columns=['amplitude_measured', 'phase_measured', 'offset_measured'], index=index)

    fit['amplitude_measured'] = fit['amplitude_measured'].apply(np.abs)
    fit['frequency_applied'] = frequency_applied
    return fit


def get_float_annotations(annotations: mne.Annotations):
    """
    Obtain all annotations in raw that can be casted into float
    Returns a list with tuples where the first value is the description casted into float
    and the second value is an OrderedDict (single annotation)
    :return: List[tuple(float, OrderedDict)]
    """
    float_annotations = []
    for annotation in annotations:
        try:
            value = float(annotation['description'])
            float_annotations.append((value, annotation))
        except ValueError:
            pass
    return float_annotations
