#!/usr/bin/env python

"""Module documentation goes here"""
from typing import Callable

import numpy as np
import mne
import re
import matplotlib.pyplot as plt

from data_organizer import Experiment, ExperimentOrganizer
from xdf2mne import streams2raw

organizer = ExperimentOrganizer()
current_subject_regex = '.*_opm_motor_imagery_p002'

from analysis.utils_analysis import epochs_selector


experiment = organizer.match(current_subject_regex)[0]

# Dejitter timestamps because the classification values were directly corresponding to the samples
experiment.load_xdf_default_args.update(dejitter_timestamps=True)

logger = experiment.getLogger_analysis_online()

streams = experiment.get_xdf_streams()
##
raw = streams2raw(streams['ClassifierOutput'], marker_streams=[streams['TaskOutput']])

## Enhance annotations to indicate condition, block and feedback and have the proper duration
annotations: mne.annotations.Annotations = raw.annotations
logger.info(f"Found {len(annotations)} annotations")

logger.info(f"Enhancing annotations to indicate condition, block and feedback and have the proper duration ")

descriptions = annotations.description.astype('<U30')  # Obtain all descriptions. Fix dtype to accomodate longer strings
onsets = annotations.onset
# Obtain the index of the block of all annotations
block_idxs = np.digitize(annotations.onset, annotations.onset[annotations.description == 'BLOCK_START/EVENT/ON'], right=False)

descriptions_trials = []
onsets_trials = []
durations_trials = []
annotations_delete_idx = []

for i, (description, onset, block_idx) in enumerate(zip(descriptions, onsets, block_idxs)):

    # Create a new set of properly named conditions
    match_start = re.match("(RELAX|CLOSE)/START/(ON|OFF)", description)
    if match_start:
        annotations_delete_idx.append(i)  # Delete the 'old' type of annotations
        # This annotation is a trial annotation
        condition, feedback = match_start.groups()
        descriptions_trials.append(f'{condition}/FEEDBACK_{feedback}/BLOCK_{block_idx}')
        onsets_trials.append(onset)
        durations_trials.append(5)

    if re.match("(RELAX|CLOSE)/STOP/(ON|OFF)", description):
        annotations_delete_idx.append(i)

annotations.delete(annotations_delete_idx)

# annotations.description = descriptions
annotations += mne.Annotations(onset=onsets_trials, duration=durations_trials, description=descriptions_trials)
logger.info(f"Added {len(onsets_trials)} trial annotations to raw")

##
tmin, tmax = -4, 8

events, event_id = mne.events_from_annotations(raw, regexp='(CLOSE|RELAX)/FEEDBACK_(ON|OFF)/BLOCK_.', )
logger.info(f"Found {len(events)} events in raw")
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True,
                    reject_by_annotation=True)


##

blocks = [2, 3, 4, 5]  # In block 1 the classifier has been untrained, so the data is unusable

fig, ax = plt.subplots()
for condition in ["CLOSE", "RELAX"]:
    epochs_condition = epochs_selector(epochs, conditions=condition, blocks=blocks)

    evoked_condition: mne.Evoked = epochs_condition.average(picks='class')
    # evoked_condition.plot()
    plt.plot(evoked_condition.times, evoked_condition.get_data()[0, :], label=f"{condition} trials")

##
plt.axhline(y=0, color='black', linestyle=':')
plt.axvline(x=0, color='black', linestyle='--')
plt.axvline(x=5, color='black', linestyle='--')


plt.title("Online Classification Value")
plt.xlabel("Time [s]")
plt.ylabel("Classification value [AU]")
plt.legend()

##



# def find_successful_epochs(epochs: mne.Epochs, success_criterion: Callable):
#     successful = [success_criterion(epoch) for epoch in epochs]
#     return successful

