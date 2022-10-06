#!/usr/bin/env python

"""Module documentation goes here"""

import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from matplotlib import pyplot as plt

from analysis.utils_analysis import epochs_get_labels, epochs_selector
from data_organizer import ExperimentOrganizer

organizer = ExperimentOrganizer()

experiments = organizer.get_experiments_for_analysis('analysis_online')
blocks = [2, 3, 4, 5]
# blocks = [1]
feedback = True


def get_classifier_sample(margin=0.1):
    def _classify(x):
        if x < -margin:
            return 'CLOSE'
        elif x > margin:
            return 'RELAX'
        else:
            return None

    return _classify

def classify(epochs, epoch_classifier, tmin, tmax, epoch_classifier_kwargs={}):
    data = epochs.get_data(tmin=tmin, tmax=tmax).squeeze()

    y = np.apply_along_axis(epoch_classifier, axis=1, arr=data, **epoch_classifier_kwargs)

    return y

def get_epoch_vote_classifier(sample_classifier):
    def epoch_vote_classifer(epoch_data, **kwargs):
        y_epoch = sample_classifier(epoch_data)
        n_close_samples = np.sum(y_epoch == 'CLOSE')
        n_relax_samples = np.sum(y_epoch == 'RELAX')
        n_samples = len(y_epoch)

        if n_close_samples > n_relax_samples:
            return 'CLOSE', n_close_samples / n_samples
        elif n_relax_samples > n_close_samples:
            return 'RELAX', n_relax_samples / n_samples
        else:
            return None, (n_relax_samples + n_close_samples) * 0.5 / n_samples

    return epoch_vote_classifer

def get_epoch_duration_classifier(sample_classifier, duration, sfreq):
    def epoch_duration_classifier(epoch_data, **kwargs):
        y_epoch = sample_classifier(epoch_data)
        n_samples_pos = sfreq * duration

        n_close_samples = np.sum(y_epoch == 'CLOSE')
        n_relax_samples = np.sum(y_epoch == 'RELAX')
        n_samples = len(y_epoch)
        if n_close_samples >= n_samples_pos and n_relax_samples < n_samples_pos:
            return 'CLOSE', n_close_samples / n_samples
        elif n_relax_samples >= n_samples_pos and n_close_samples < n_samples_pos:
            return 'RELAX', n_relax_samples / n_samples
        else:
            return None, (n_close_samples + n_relax_samples) * 0.5 / n_samples

    return epoch_duration_classifier


##
accuracies = []

margin = 0.1
duration = 2
tmin = 1
tmax = 5


for experiment in experiments[:]:
    epochs = experiment.get_epochs_online()
    epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)
    # for margin, duration in itertools.product([0, 0.1, 0.2], [1, 2, 2.5, 3]):
    for margin, duration in itertools.product([0, 0.1], [2]):

        classifier_sample = np.vectorize(get_classifier_sample(margin=margin))
        epoch_classifier = get_epoch_duration_classifier(classifier_sample, duration=duration,
                                                         sfreq=epochs.info['sfreq'])
        epoch_classifier = get_epoch_vote_classifier(classifier_sample)

        participant_id = experiment.get_participant_id()
        y_true = epochs_get_labels(epochs, asarray=True)
        y_pred, y_certainty = classify(epochs, epoch_classifier, tmin=tmin, tmax=tmax).T
        y_certainty = y_certainty.astype(float)
        accuracies.append(dict(participant_id=participant_id,
                               accuracy=np.sum(y_true == y_pred) / len(y_true),
                               nanvals=np.sum(y_pred == None),
                               margin=margin,
                               duration=duration,
                               tmin=tmin,
                               tmax=tmax,
                               y_certainty=np.mean(y_certainty)
                               ))

##
df = pd.DataFrame(accuracies)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='participant_id', y='accuracy', hue='duration', style='margin', ax=ax)

ax.axhline(0.5)