#!/usr/bin/env python
import scipy.signal

"""
Online Analysis
Load the CSP filter eigenvalues of all participants and compare them in a table
"""

import numpy as np
import json
from data_organizer import ExperimentOrganizer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

organizer = ExperimentOrganizer()


def find_node(nodes_list, node_type):
    return [node for node in nodes_list if node['type'] == node_type][0]


experiments = organizer.get_experiments_for_analysis('analysis_online')
experiments_streams = {}

for experiment in experiments:
    streams = experiment.get_xdf_streams()
    experiments_streams[experiment.name] = streams

##
for experiment in experiments:
    streams = experiments_streams[experiment.name]

    stream_classification_pipeline_settings = streams['ClassificationPipelineSettings']['time_series']

    csp_settings_updates = [find_node(json.loads(settings[1])['nodes'], 'CSPNode') for settings in stream_classification_pipeline_settings][:5]

    ds = np.sort([update['d'] for update in csp_settings_updates], axis=1)
    ds_both_above_05 = np.argwhere([all(d>0.5) for d in ds])
    if any(ds_both_above_05):
        print(f"For participant {experiment.get_participant_id()}, both eigenvalues are >0.5 in blocks {ds_both_above_05.flatten()}\n\t(participant does {'' if experiment.get_info()['analysis_online_with_erd'] == 'X' else 'NOT '}have an ERD)")
    else:
        print(f"For participant {experiment.get_participant_id()}, the average eigenvalue is {np.mean(ds):0.3} {'' if experiment.get_info()['analysis_online_with_erd'] == 'X' else '(participant does NOT have an ERD)'}")
