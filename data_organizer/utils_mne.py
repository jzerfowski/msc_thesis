#!/usr/bin/env python

"""
Utility function for MNE
Utility function to obtain events following a regular expression from an mne.RawArray
"""

import logging
import mne.io
import xdf2mne
logger = logging.getLogger(__name__)


def get_events_from_raw(raw: mne.io.RawArray, training_classes_regexp: bool = None):
    if training_classes_regexp is None:
        training_classes_regexp = {"CLOSE": "CLOSE/START/.*", "RELAX": "RELAX/START/.*"}

    regexp_all = "|".join([f"({class_regexp})" for class_regexp in training_classes_regexp.values()])

    # Get the events matching the regular expression of all classes
    # event_id will be sorted by condition due to events_from_annotations()
    try:
        events, event_id = mne.events_from_annotations(raw, regexp=regexp_all)
        if len(events) == 0:
            logger.error("No events found, returning None")
            return None, None
        logger.info(
            f"Get training epochs from receiver: Extracted {len(events)} events with {len(event_id)} different conditions: { {event_name: sum(events[:, 2] == event) for event_name, event in event_id.items()} }")
    except ValueError as e:
        logger.error(
            f"get_training_epochs() (/events_from_annotations) did not find a single epoch in the data. Returning None")
        return None, None

    # Merge all the conditions into one
    for condition_new, regexp in training_classes_regexp.items():
        events, event_id = xdf2mne.merge_event_ids(events, event_id, regexp, condition_new)

    return events, event_id
