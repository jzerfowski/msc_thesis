#!/usr/bin/env python

"""Module documentation goes here"""
import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.online.utils_analysis_online import get_classification, compute_confusion_matrix, ax_timecourse_enhance, \
    get_classification_df
from data_organizer import ExperimentOrganizer
from analysis.utils_analysis import epochs_selector, epochs_to_data_frame, epochs_permutation_test


organizer = ExperimentOrganizer()

experiments = organizer.get_experiments_for_analysis('analysis_online')

experiment = experiments[2]

##
epochs = experiment.get_epochs_online()

##
##
blocks = [2, 3, 4, 5]  # In block 1 the classifier has been untrained, so the data is unusable
epochs = epochs_selector(epochs, feedback=None, blocks=blocks)
epochs.crop(tmin=-2, tmax=6.5)

##
df_confusion = compute_confusion_matrix(epochs)
print(df_confusion)


##

def plot_classification_timecourse(epochs: mne.Epochs, ax=None, ptest=True, pval=0.05, title=None):
    p_levels = np.array([0.05, 0.01, 0.001])
    df = epochs_to_data_frame(epochs)
    df = df.rename(columns={'value': 'class'})
    ax = sns.lineplot(data=df, x='time', y='class', hue='Condition', ax=ax)
    ax_timecourse_enhance(ax, ylabel='Classification value', title=title)

    if ptest:
        times = epochs.times
        F_obs, clusters, cluster_pv, H0 = epochs_permutation_test(epochs)

        for cluster, pvalue in zip(clusters, cluster_pv):
            cluster = cluster[0]
            cp_level = np.count_nonzero(pvalue < p_levels)

            if cp_level:
                tstart, tstop = times[cluster.start], times[cluster.stop-1]
                ax.hlines(y=0.02, xmin=tstart, xmax=tstop, transform=ax.get_xaxis_transform(), color='black', linewidth=4)
                p_string = f"{'*'*cp_level}" # + f"(p < {p_levels[cp_level-1]})"
                ax.text(x=(tstart + tstop)/2, y=0.025, s=p_string, horizontalalignment='center', transform=ax.get_xaxis_transform())
    return ax


fig, (ax1) = plt.subplots(ncols=1)
plot_classification_timecourse(epochs, ax=ax1, title=f"Online Classification (Subject {experiment.get_participant_id()})")

##

##
