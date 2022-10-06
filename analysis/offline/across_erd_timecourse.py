#!/usr/bin/env python

"""
Offline Analysis
Compute the ERD timecourse averaged over all participants
"""

import json

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from utils_analysis import draw_clusters, epochs_selector, array_to_data_frame
from data_organizer import ExperimentOrganizer
from utils_analysis_offline import compute_erd_timecourse_csp_causal
from utils_plotting import save_fig, palette, set_context, mark_range_y

##
blocks = [2, 3, 4, 5]
feedback = False

##
tmin_crop, tmax_crop = -2.0, 6.5
tmin_baseline, tmax_baseline = -2, 0

##
organizer = ExperimentOrganizer()
experiments_all = organizer.get_experiments_for_analysis('analysis_erd_timecourse')

def get_erd_timecourse_data(experiment):
    epochs: mne.Epochs = experiment.get_epochs()
    epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)

    erd_freq = experiment.get_erd_freq()

    with open(experiment.basepath / 'erd_freq_csp_settings_individual.json', 'r') as fp:
        csp_settings = json.load(fp)

    epochs_erd_timecourse, csp = compute_erd_timecourse_csp_causal(epochs, erd_freq=erd_freq, csp_settings=csp_settings, train_csp=False)

    epochs_erd_timecourse.resample(sfreq=100, npad='auto', window='boxcar', pad='edge')
    mne.baseline.rescale(epochs_erd_timecourse._data, epochs_erd_timecourse.times, copy=False, mode='percent', baseline=(tmin_baseline, tmax_baseline))
    epochs_erd_timecourse.crop(tmin=tmin_crop, tmax=tmax_crop)
    #

    participant_close = np.mean(epochs_erd_timecourse['CLOSE'].get_data(), axis=0)
    participant_relax = np.mean(epochs_erd_timecourse['RELAX'].get_data(), axis=0)
    times = epochs_erd_timecourse.times
    channels = epochs_erd_timecourse.ch_names

    return participant_relax, participant_close, channels, times



def plot_across_erd_timecourse(experiments, title):
    participants = [experiment.get_participant_id() for experiment in experiments]
    data_relax = np.array([dict_relax[participant] for participant in participants])
    data_close = np.array([dict_close[participant] for participant in participants])

    data = np.array([data_relax, data_close])

    df = array_to_data_frame(data, axes=[['Relax', 'Close'], participants, channels, times], axes_names=['Condition', 'subject', 'channel', 'time']).reset_index()
    df['ERD'] = df['value'] * 100

    F_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test([data_relax, data_close], threshold=None,
                                                                         n_permutations=1024, out_type='mask',
                                                                         adjacency=None, n_jobs=6, seed=42)

    _, clusters_0, cluster_pv_0, H0_0 = mne.stats.permutation_cluster_test([data_close, np.zeros_like(data_close)], threshold=None,
                                                                         n_permutations=1024, out_type='mask',
                                                                         adjacency=None, n_jobs=6, seed=42)
    print("Clusters against 0:")
    draw_clusters(clusters_0, cluster_pv_0, None, times)
    print("End clusters against 0")

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='time', y='ERD', hue='Condition', palette=palette, ci=95, n_boot=1000, ax=ax)
    draw_clusters(clusters, cluster_pv, ax, times)
    mark_range_y(x1=0, x2=5, ax=ax, color='grey', alpha=0.2, zorder=0)
    if mpl.rcParams['text.usetex']:
        ylabel = 'ERD (\%)'
    else:
        ylabel = 'ERD (%)'
    ax.set(xlabel='Time (s)', ylabel=ylabel, title=title, xlim=(tmin_crop, tmax_crop))
    ax.legend(loc='upper left')
    return fig, ax

##
dict_relax = {}
dict_close = {}

for experiment in experiments_all[:]:
    participant_id = experiment.get_participant_id()
    participant_relax, participant_close, channels, times = get_erd_timecourse_data(experiment)
    dict_relax[participant_id] = participant_relax
    dict_close[participant_id] = participant_close

## Plot Across-participants average
set_context('paper')
fig, ax = plot_across_erd_timecourse(organizer.get_experiments_for_analysis('analysis_erd_timecourse'), title='')
ax.set(ylim=(-100, 200))
save_fig(fig, 'across_erd_timecourse', 'erd_timecourse')


