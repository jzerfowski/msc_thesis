#!/usr/bin/env python

"""
Preprocessing
Epoch the classifier output stream from raw .xdf file for the online analysis
"""

import mne
import re
import numpy as np

from data_organizer import ExperimentOrganizer, _suffix_mne_epochs_online
from xdf2mne import streams2raw


organizer = ExperimentOrganizer()

epoching_tmin, epoching_tmax = -4, 8

def epoch_experiment_online(experiment):

    # Dejitter timestamps because the classification values were directly corresponding to the samples
    experiment.load_xdf_default_args.update(dejitter_timestamps=True)

    logger = experiment.getLogger_epoching_online()
    logger.info(f"Beginning online epoching for experiment {experiment.name}")


    ##
    streams = experiment.get_xdf_streams()

    raw = streams2raw(streams['ClassifierOutput'], marker_streams=[streams['TaskOutput']])
    logger.info(f"Extracted {raw.times.max()}s of raw data from ClassifierOutput stream (sfreq={raw.info['sfreq']})")

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

    # In case needed, the raw classifications could be saved here. Take care that the (MNE) timestamps are not necessarily
    # in sync with the offline raw's timestamps


    ##

    events, event_id = mne.events_from_annotations(raw, regexp='(CLOSE|RELAX)/FEEDBACK_(ON|OFF)/BLOCK_.', )
    logger.info(f"Found {len(events)} events in raw")
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=epoching_tmin, tmax=epoching_tmax, baseline=(None, 0), preload=True,
                        reject_by_annotation=True)

    ## Write epochs file
    filepath_fif_epo_online = experiment.basepath / f"{experiment.get_basename()}_{_suffix_mne_epochs_online}"
    # filepath_fif_epo = experiment.basepath / f"{filename_xdf_raw}_{_suffix_mne_epochs}"
    logger.info(f"Saving epoched online classification data to {filepath_fif_epo_online}")
    epochs.save(filepath_fif_epo_online, overwrite=True)

if __name__ == '__main__':
    experiments = organizer.get_experiments_for_analysis('all')
    for experiment in experiments:
        epoch_experiment_online(experiment)

