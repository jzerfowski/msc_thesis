#!/usr/bin/env python

"""
Utility
Utility functions to manipulate the data in different ways needed for analysis
"""

import re
from typing import Optional, List, Union
import itertools
import mne
import logging
import pandas as pd

logger = logging.getLogger(__name__)
import numpy as np

from scipy.stats import kurtosis

ALL_CONDITIONS = ["RELAX", "CLOSE"]
ALL_BLOCKS = list(range(1, 6))
ALL_FEEDBACKS = [True, False]


def epochs_selector(epochs: mne.Epochs, conditions: Optional[Union[List[str], str]] = None, blocks: Optional[Union[List[int], int, str]] = None, feedback: Optional[bool] = None) -> mne.Epochs:
    def generate_selectors(selectors_pre, selectors_add):
        selectors_all = []
        if len(selectors_pre):
            if len(selectors_add):
                selectors_all = [f"{pre}/{add}" for pre, add in list(itertools.product(selectors_pre, selectors_add))]
            else:
                # No new selectors
                selectors_all = selectors_pre
        else:
            if selectors_add:
                selectors_all = selectors_add
            else:
                # No old and no new selectors
                pass
        return selectors_all

    selector_list = []

    if conditions is None or (isinstance(conditions, str) and conditions == 'all'):
        pass
    elif isinstance(conditions, str):
        selector_list = generate_selectors(selector_list, [conditions])
    elif isinstance(conditions, list):
        selector_list = generate_selectors(selector_list, conditions)
    else:
        raise Exception(f"{conditions=} is invalid")

    if blocks is None or (isinstance(blocks, str) and blocks == 'all'):
        pass
    elif isinstance(blocks, int):
        selector_list = generate_selectors(selector_list, [f"BLOCK_{blocks}"])
    elif isinstance(blocks, list):
        selector_list = generate_selectors(selector_list, [f"BLOCK_{block}" for block in blocks])
    else:
        raise Exception(f"{blocks=} is invalid")

    if feedback is None or (isinstance(feedback, str) and feedback == 'all'):
        pass
    elif isinstance(feedback, bool):
        selector_list = generate_selectors(selector_list, [f"FEEDBACK_{'ON' if feedback else 'OFF'}"])
    else:
        raise Exception(f"{feedback=} is invalid")

    logger.info(f"Selecting Epochs with list of selectors: {selector_list}")

    if len(selector_list):
        epochs_selected = mne.concatenate_epochs([epochs[selector] for selector in selector_list])
    else:
        epochs_selected = epochs.copy()

    return epochs_selected


def shift_over_time(data, window_length):
    n_times = data.shape[-1]
    x_shifted = []
    for i in range(0, n_times-window_length+1, window_length):
        x_shifted.append(data[..., i:i+window_length])
    return np.array(x_shifted)


def compute_kurtosis_per_channel(data, window_length):
    return np.mean(kurtosis(shift_over_time(data, window_length), axis=-1), axis=0)


def epochs_data_frame_fix_condition(df):
    # df should have column 'Condition'
    df = df.copy()
    df[['Condition', 'Feedback', 'Block']] = df['Condition'].str.extract(r'(CLOSE|RELAX)/FEEDBACK_(ON|OFF)/BLOCK_(.)')
    df['Condition'] = df['Condition'].map({'RELAX': "Relax", "CLOSE": "Close"})
    if 'classification' in df:
        df['classification'] = df['classification'].map({'RELAX': "Relax", "CLOSE": "Close"})
    df['Feedback'] = df['Feedback'] == 'ON'
    df['Block'] = df['Block'].astype(int)

    return df


def epochs_to_data_frame(epochs, time_format=None, scalings=None) -> pd.DataFrame:
    df = epochs.to_data_frame(time_format=time_format, scalings=scalings)
    df = df.rename(columns={'condition': "Condition"})
    df = epochs_data_frame_fix_condition(df)
    df = df.melt(id_vars=['time', 'epoch', 'Condition', 'Feedback', 'Block'], var_name='channel',
                 value_name='value')
    return df


def epochs_get_conditions(epochs: mne.Epochs) -> List[str]:
    # The event_id dict is in the wrong order, so we reverse it manually to extract the labels
    event_id_rev = {event_id: label for label, event_id in epochs.event_id.items()}
    # Map the event ids with their labels
    events = list(map(event_id_rev.get, epochs.events[:, 2]))
    return events


def epochs_get_labels(epochs, asarray=False) -> List[str]:
    events = epochs_get_conditions(epochs)
    # Extract the first group from match (i.e., RELAX or CLOSE)
    labels = [re.match("(RELAX|CLOSE)/.*", event)[1] for event in events]
    if asarray:
        labels = np.array(labels)
    return labels


def epochs_permutation_test(epochs: mne.Epochs, channel_adjacency=True, conditions=['RELAX', 'CLOSE']):

    data = [epochs[condition].get_data() for condition in conditions]

    # data_relax = epochs['RELAX'].get_data()  # n_epochs, n_channels, n_times
    # data_close = epochs['CLOSE'].get_data()

    if channel_adjacency:
        channel_adjacency = mne.channels.find_ch_adjacency(epochs.info)
    else:
        n_channels = len(epochs.ch_names)
        channel_adjacency = np.zeros((n_channels, n_channels))
    adjacency = mne.stats.combine_adjacency(channel_adjacency, len(epochs.times))

    F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(data, threshold=None,
                                                                         n_permutations=1024, out_type='mask',
                                                                         adjacency=adjacency, n_jobs=6)

    return F_obs, clusters, cluster_pv, H0


def draw_clusters(clusters, cluster_pv, ax, xdim, y=0.025, color='black'):
    # xdim corresponds to times or freqs depending on the application
    p_levels = np.array([0.05, 0.01, 0.001])

    xmin, xmax = np.nan, np.nan
    x_total = 0

    cluster_stats = None
    for cluster, pvalue in zip(clusters, cluster_pv):
        cp_level = np.count_nonzero(pvalue < p_levels)

        n_clusters = 0
        if cp_level:
            mask = np.any(cluster, axis=tuple(range(len(np.array(cluster).shape) - 1)))
            cluster_slice = np.ma.clump_masked(np.ma.masked_array(mask, mask))[0]
            xstart, xstop = xdim[cluster_slice.start], xdim[cluster_slice.stop - 1]

            # Compute stats
            n_clusters += 1
            xmin = np.nanmin([xmin, xstart])
            xmax = np.nanmax([xmax, xstop])
            x_total += xstop - xstart

            p_string = f"{'*' * cp_level}"  # + f"(p < {p_levels[cp_level-1]})"


            print(f"Drawing cluster ({p_string}, <= {p_levels[cp_level-1]}) from {xstart}-{xstop} s")

            if ax is not None:
                ax.plot([xstart, xstop], [0.025]*2, transform=ax.get_xaxis_transform(), color=color, linewidth=4)
                ax.text(x=(xstart + xstop) / 2, y=y, s=p_string, horizontalalignment='center',
                        transform=ax.get_xaxis_transform())
        cluster_stats = dict(n_clusters=n_clusters, xmin=xmin, xmax=xmax, x_total=x_total)
    return cluster_stats


def array_to_data_frame(arr: np.ndarray, axes: List[np.ndarray], axes_names: List[str], value_name='value'):
    axes_shape = tuple([len(ax) for ax in axes])
    assert all([arr.shape == axes_shape]), f"{arr.shape=} must be equivalent to the shape of the axes ({axes_shape})"
    index = pd.MultiIndex.from_product(axes, names=axes_names)
    df = pd.DataFrame(index=index)
    df[value_name] = arr.flatten()
    return df


