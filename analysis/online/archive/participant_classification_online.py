#!/usr/bin/env python

"""Module documentation goes here"""
import itertools

import mne
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from analysis.online.utils_analysis_online import get_classification, compute_confusion_matrix, ax_timecourse_enhance, \
    get_classification_df
from analysis.utils_plotting import save_fig
from data_organizer import ExperimentOrganizer
from analysis.utils_analysis import epochs_selector, epochs_to_data_frame, epochs_permutation_test, epochs_get_labels


organizer = ExperimentOrganizer()

experiments = organizer.get_experiments_for_analysis('all')
# experiments_erd = organizer.get_experiments_for_analysis('analysis_erd')

blocks = [2, 3, 4, 5]  # In block 1 the classifier has been untrained, so the data is unusable

##

def classify_epochs_online(epochs, tmin, tmax, **kwargs):
    data = epochs.get_data(tmin=tmin, tmax=tmax)
    classify = np.vectorize(lambda x: "CLOSE" if x < 0 else "RELAX")
    y_array = classify(data).squeeze()

    y_pred = []
    y_certainty = []
    for epoch in y_array:
        labels, counts = np.unique(epoch, return_counts=True)
        label_idx = np.argmax(counts)
        label = labels[label_idx]
        certainty = counts[label_idx] / np.sum(counts)
        y_pred.append(label)
        y_certainty.append(certainty)

    y_pred = np.array(y_pred)
    y_certainty = np.array(y_certainty)

    return y_pred, y_certainty

def accuracy_epochs_online_accuracy(epochs, tmin, tmax, y_true, **kwargs):
    epochs_data = epochs.get_data(tmin=tmin, tmax=tmax)
    *_, n_samples = epochs_data.shape
    classify = np.vectorize(lambda x: "CLOSE" if x < 0 else "RELAX")

    y_pred = classify(epochs_data).squeeze()

    accuracy = []
    for true_label, y_pred_epoch in zip(y_true, y_pred):
        accuracy.append(np.mean(y_pred_epoch == true_label))
    return accuracy


def classify_epochs_online_duration(epochs, tmin, tmax, min_duration=3, **kwargs):
    epochs_data = epochs.get_data(tmin=tmin, tmax=tmax)
    sfreq = epochs.info['sfreq']
    min_samples = sfreq * min_duration
    *_, n_samples = epochs_data.shape

    classify = np.vectorize(lambda x: "CLOSE" if x < 0 else "RELAX")
    y_array = classify(epochs_data).squeeze()

    y_pred = []
    y_certainty = []
    for epoch in y_array:
        labels, counts = np.unique(epoch, return_counts=True)
        label_counted_max = np.max(counts)
        if label_counted_max >= min_samples:
            epoch_label = labels[np.argmax(counts)]
        else:
            epoch_label = None
        y_pred.append(epoch_label)
        y_certainty.append(label_counted_max / n_samples)

    return np.array(y_pred), np.array(y_certainty)


df_accuracy = pd.DataFrame()
for experiment in experiments:
    participant_id = experiment.get_participant_id()
    epochs = experiment.get_epochs_online()

    # for conditions, blocks, feedback in itertools.product(['RELAX', 'CLOSE', 'all'], [1, 2, 3, 4, 5, 'all'], [True, False, 'all']):
    # for conditions, blocks, feedback in zip(['all', 'all'])
    for conditions, blocks, feedback in [('all', [2, 3, 4, 5], True), ('all', [2, 3, 4, 5], False)]:
        epochs_ = epochs_selector(epochs, conditions=conditions, blocks=blocks, feedback=feedback)
        # epochs.crop(tmin=-2, tmax=6.5)

        y_true = epochs_get_labels(epochs_, asarray=True)
        y_pred, y_certainty = classify_epochs_online(epochs_, tmin=1, tmax=5, y_true=y_true)
        y_pred_duration, y_certainty_duration = classify_epochs_online_duration(epochs_, tmin=1, tmax=5, min_duration=3, y_true=y_true)

        score_test = np.mean(y_true == y_pred)
        score_certainty = np.mean(y_certainty)
        score_test_duration = np.mean(y_true == y_pred_duration)
        score_certainty_duration = np.mean(y_certainty_duration)
        score_accuracy_method = np.mean(accuracy_epochs_online_accuracy(epochs_, tmin=1, tmax=5, y_true=y_true))

        df_participant = pd.DataFrame([dict(Participant=participant_id,
                                            subset_erd = experiment.get_info()['analysis_erd'] == 'X',
                                            ch_name='online', Conditions=conditions, Blocks=blocks, Feedback=feedback,
                                            score_test=score_test, score_certainty=score_certainty,
                                            score_test_duration=score_test_duration, score_certainty_duration=score_certainty_duration,
                                            score_accuracy_method=score_accuracy_method,
        )])
        df_accuracy = pd.concat([df_accuracy, df_participant], ignore_index=True)

df_accuracy = df_accuracy.rename(columns={'score_test': 'Accuracy', 'score_test_duration': 'Accuracy_duration'})

##
# fig, ax = plt.subplots()
# sns.scatterplot(data=df_accuracy, x='blocks', y='score_test', hue='feedback', ax=ax)
# sns.lineplot(data=df_accuracy[df_accuracy['Blocks'] != 'all'], x='Blocks', y='Accuracy', hue='Feedback', ax=ax, hue_order=[False, True, 'all'], ci=None)
# sns.boxplot(data=df_accuracy, x='Blocks', y='Accuracy', hue='Feedback', ax=ax, hue_order=[False, True, 'all'])
y_vars = ['Accuracy', 'Accuracy_duration', 'score_accuracy_method']
fig, axes = plt.subplots(nrows=len(y_vars), ncols=2)

df_accuracy['ERD'] = 'ERD'

for ax_i, y_var in enumerate(y_vars):
    ax_all, ax_erd = axes[ax_i, :]
    sns.boxplot(data=df_accuracy, x='Feedback', y=y_var, ax=ax_all)
    ax_all.set(title='All')

    sns.boxplot(data=df_accuracy[df_accuracy['subset_erd']], x='ERD', y=y_var, ax=ax_erd)
    ax_erd.set(title='ERD')



    # g = sns.catplot(x="Blocks", y="Accuracy",
    #                 hue="Conditions", col="Feedback",
    #                 data=df_accuracy, kind="box",
    #                 height=4, aspect=.7)
    # ax = sns.boxplot()
    # g = sns.catplot(x="Feedback", y=y_var,
    #                 hue="Conditions",
    #                 data=df_accuracy, kind="box",
    #                 height=4, aspect=.7)