#!/usr/bin/env python

"""
Preprocessing
Epoch the annotated data for the offline analysis
"""

import mne.io

from data_organizer import Experiment, ExperimentOrganizer, _suffix_mne_epochs_offline_annotated

import logging

logging.basicConfig(level=logging.INFO)

tmin, tmax = -3.5, 8.5

organizer = ExperimentOrganizer()


def get_event_count_dict(epochs, events_raw):
    def condition_count_dict(description, n_events, n_expected):
        return {
            f"{description}": n_events,
            f"{description}_expected": n_expected,
        }


    conditions = {
        'CLOSE/FEEDBACK_OFF': 45,
        'RELAX/FEEDBACK_OFF': 45,
        'CLOSE/FEEDBACK_ON': 30,
        'RELAX/FEEDBACK_ON': 30,
        **{f"BLOCK_{i}": 30 for i in range(1, 6)},
    }

    epochs_count_dict = {
        **condition_count_dict("*", len(epochs.events), 150),
    }

    for condition, n_expected in conditions.items():
        epochs_count_dict.update(**condition_count_dict(condition, len(epochs[condition]), n_expected))

    epochs_count_dict.update(** condition_count_dict("raw_*", len(events_raw), 150))

    return epochs_count_dict


def epoch_experiment(experiment: Experiment):
    ## Prepare logger
    logger = experiment.getLogger_epoching(mode='w')
    logger.info(f"Beginning epoching of {experiment}")

    ## Read raw .fif file
    raw_filepath = experiment.basepath / experiment.get_raw_annotated_filenames()[0]
    logger.info(f"Reading raw .fif file from {raw_filepath}")
    raw: mne.io.Raw = mne.io.read_raw_fif(raw_filepath)

    ##
    logger.info("Beginning epoching")

    events, event_id = mne.events_from_annotations(raw, regexp='(CLOSE|RELAX)/FEEDBACK_(ON|OFF)/BLOCK_.', )
    logger.info(f"Found {len(events)} events in raw")
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True,
                        reject_by_annotation=True)

    logger.info(f"After rejecting 'BAD' Epochs, {len(epochs.events)} events remain")

    ## Write epochs file
    filepath_fif_epo = experiment.basepath / f"{experiment.get_basename()}_{_suffix_mne_epochs_offline_annotated}"

    logger.info(f"Saving epoched data to {filepath_fif_epo}")
    epochs.save(filepath_fif_epo, overwrite=True)

    epochs_count_dict = get_event_count_dict(epochs, events)

    details = dict(
        subject_id=experiment.get_participant_id(),
        **epochs_count_dict,
        experiment_name=experiment.name,
        filepath_epo=filepath_fif_epo.__str__(),
        filepath_raw=raw_filepath.__str__(),
        duration_raw=raw.times.max(),
    )

    return epochs, details


if __name__ == '__main__':
    ## Load experiment
    experiments = organizer.get_experiments_for_analysis('annotated')

    for experiment in experiments[:None]:
        epochs, details = epoch_experiment(experiment)

