#!/usr/bin/env python

"""Module documentation goes here"""

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from analysis.online.utils_analysis_online import compute_confusion_matrix, compute_error_matrix

from data_organizer import ExperimentOrganizer
from analysis.utils_analysis import epochs_selector

organizer = ExperimentOrganizer()

##

# accuracies = []
accuracies = pd.DataFrame()


for experiment in organizer.get_experiments_for_analysis('all')[:]:
    subject_id = experiment.get_participant_id()

    epochs = experiment.get_epochs_online()

    epochs_offline = experiment.get_epochs_annotated()
    if epochs_offline:
        epochs_cleaned = epochs.copy().drop(np.argwhere(epochs_offline.drop_log).squeeze())
        epochs_cleaned = epochs_selector(epochs_cleaned, blocks=[2, 3, 4, 5])
        error_matrix_cleaned = compute_error_matrix(epochs_cleaned, tmin=1, tmax=5, tmin_sign_duration=3)

    epochs = epochs_selector(epochs, blocks=[2, 3, 4, 5])

    confusion_matrix = compute_confusion_matrix(epochs, tmin=1, tmax=5, tmin_sign_duration=3)
    # confusion_matrix_cleaned = compute_confusion_matrix(epochs_cleaned, tmin=1, tmax=5, tmin_sign_duration=3)
    error_matrix = compute_error_matrix(epochs, tmin=1, tmax=5, tmin_sign_duration=3)


    print(f'Subject {subject_id}:')
    print(confusion_matrix)

    acc = (error_matrix["TPR"]+error_matrix["TNR"])/2
    if epochs_offline:
        # balanced accuracy
        acc_cleaned = (error_matrix_cleaned["TPR"]+error_matrix_cleaned["TNR"])/2
    else:
        acc_cleaned = None
    accuracies = accuracies.append(dict(subject=subject_id, accuracy=acc, accuracy_cleaned=acc_cleaned), ignore_index=True)
