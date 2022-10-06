#!/usr/bin/env python

"""
Offline Analysis
Utility functions
Module with utility functions for the offline analysis
"""

from typing import Tuple

import mne
import numpy as np

from utils_analysis import epochs_get_labels
from CSPNode import CSPNode

erd_band_half_width = 1
trans_bandwidth_filter = 4


def get_filter_params(erd_freq, mode='erd'):
    _params = dict(l_trans_bandwidth=trans_bandwidth_filter, h_trans_bandwidth=trans_bandwidth_filter,
                         phase='minimum', fir_window="hamming", fir_design='firwin', method='fir')
    if mode == 'generic' or erd_freq is None:
        _params.update(l_freq=8, h_freq=16)
    elif mode == 'erd':
        _params.update(l_freq=erd_freq-erd_band_half_width, h_freq=erd_freq+erd_band_half_width)

    return _params


def train_csp_on_epochs(epochs, csp, tmin, tmax, compute_patterns=True):
    labels = epochs_get_labels(epochs)
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    csp.train(epochs_cropped.get_data(picks='all'), labels=labels, timestamps=epochs_cropped.times, compute_patterns=compute_patterns)
    return csp


def apply_csp_on_epochs(epochs, csp, component_idxs=None):
    epochs = epochs.copy()
    data_epochs = epochs.get_data(picks='all')
    csp_data = CSPNode.apply_csp(data_epochs, csp.W)

    if component_idxs is None:
        component_idxs = np.arange(csp.n_components)

    epochs_csp_info = mne.create_info(ch_names=[csp.out_channel_labels[i] for i in component_idxs], sfreq=epochs.info['sfreq'], ch_types='misc')
    epochs_csp = mne.EpochsArray(csp_data[:, component_idxs, ...], info=epochs_csp_info, events=epochs.events, event_id=epochs.event_id, tmin=epochs.tmin, on_missing='warn')
    return epochs_csp


def compute_erd_timecourse_csp_causal(epochs: mne.Epochs, erd_freq, tmin_train=1, tmax_train=5, var_window_length=0.4, csp_settings=None, train_csp=True) -> Tuple[mne.Epochs, CSPNode]:
    params_filter = get_filter_params(erd_freq=erd_freq)

    epochs = epochs.copy().filter(**params_filter)

    _csp_settings = dict(in_channel_labels=epochs.ch_names, n_components=1, method='physiological')

    if csp_settings is not None:
        _csp_settings.update(csp_settings)

    csp = CSPNode(**_csp_settings)

    if train_csp:
        train_csp_on_epochs(epochs, csp, tmin=tmin_train, tmax=tmax_train)

    epochs_csp = apply_csp_on_epochs(epochs, csp)

    window_length = int(var_window_length * epochs_csp.info['sfreq'])

    def apply_func(data):
        return np.var(data, axis=-1)

    def func(data, window_length=window_length, apply_func=apply_func):
        # shift the data by the window length to ensure causality
        data = np.concatenate([np.zeros(data.shape[:-1] + (window_length-1,)), data], axis=-1)
        return apply_func(np.lib.stride_tricks.sliding_window_view(data, window_shape=window_length, axis=-1))

    epochs_erd_timecourse = epochs_csp.copy().apply_function(func, picks='all')

    return epochs_erd_timecourse, csp