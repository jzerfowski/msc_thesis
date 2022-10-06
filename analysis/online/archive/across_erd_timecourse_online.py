#!/usr/bin/env python

"""Module documentation goes here"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import mne
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.utils_analysis import epochs_to_data_frame, \
    epochs_permutation_test, draw_clusters, epochs_selector, array_to_data_frame
from data_organizer import ExperimentOrganizer
from utils_analysis_offline import train_csp_on_epochs, apply_csp_on_epochs, compute_erd_timecourse_csp_causal
from analysis.utils_plotting import save_fig, palette, set_context, plot_csp_pattern, mark_range_y


##
blocks = [2, 3, 4, 5]
feedback = False

##
tmin_baseline, tmax_baseline = -2, 0
tmin_crop, tmax_crop = -2.5, 7.0

##
organizer = ExperimentOrganizer()
experiments_all = organizer.get_experiments_for_analysis('analysis_online')



def plot_across_erd_timecourse(experiments, title, ax=None):
    participants = [experiment.get_participant_id() for experiment in experiments]
    data_relax = np.array([dict_relax[participant] for participant in participants])
    data_close = np.array([dict_close[participant] for participant in participants])

    df = array_to_data_frame(np.array([data_relax, data_close]), axes=[['Relax', 'Close'], participants, times], axes_names=['Condition', 'subject', 'time']).reset_index()

    if ax is None:
        fig, ax = plt.subplots()

    sns.lineplot(data=df, x='time', y='value', hue='Condition', palette=palette, ci=95, n_boot=1000, ax=ax)
    ylim = np.array([np.max(np.abs(ax.get_ylim()))]*2) * (-1, 1)
    ax.set(xlabel='Time (s)', ylabel='Classification value', title=title, xlim=(tmin_crop, tmax_crop), ylim=ylim)

    ax.legend(loc='upper left')

    # F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test([data_relax, data_close], threshold=None,
    #                                                                      n_permutations=1024, out_type='mask', tail=0,
    #                                                                      adjacency=None, n_jobs=6)
    # draw_clusters(clusters, cluster_pv, ax, times)
    return ax


dict_relax = {}
dict_close = {}

for experiment in experiments_all[:]:
    participant_id = experiment.get_participant_id()
    epochs = experiment.get_epochs_online()

    epochs = epochs_selector(epochs, conditions=None, blocks=[1, 2, 3, 4], feedback=True)

    dict_relax[participant_id] = np.mean(epochs['RELAX'], axis=0).squeeze()
    dict_close[participant_id] = np.mean(epochs['CLOSE'], axis=0).squeeze()
    times = epochs.times

set_context('paper')
ax = plot_across_erd_timecourse(organizer.get_experiments_for_analysis('analysis_online'),
                                 title='Across-participants classification')

mark_range_y(x1=0, x2=5, ax=ax, color='grey', alpha=0.2, zorder=0)
ax.axhline(0, color='k')

ax.text(0.5, 0.02, 'RELAX', horizontalalignment='center', verticalalignment='center', transform=ax.get_yaxis_transform())
ax.text(0.5, -0.02, 'CLOSE', horizontalalignment='center', verticalalignment='center', transform=ax.get_yaxis_transform())

