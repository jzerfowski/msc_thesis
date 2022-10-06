#!/usr/bin/env python

"""
Create a new participant in the data folder and create all configuration files and metadata automatically from a template
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
import time

from utils_create_participant import exp_id, exp_date, participant_prefix, experiments_basepath, organizer, task, \
    grid_assignment_template_path, experiment_template_path, classifier_settings_template_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Enter participant details
participant_number = int(input(f"Participant Number ({participant_prefix}XXX), type only digits, not prefix letter, e.g., 000: "))
input_grid_query = input("Did you update the template for a grid_assignment file in ./templates? y/n: ")

exp_participant = f'{participant_prefix}{participant_number:03}'

# and create full experiment name
exp_suffix = f'{exp_id}_{exp_participant}'
exp_regex = f'.*_{exp_suffix}'
exp_name = f'{exp_date}_{exp_suffix}'

## Check if the folder exists already
logger.info(f"Looking for experiment {exp_regex} in {organizer.datafolders}")
experiments_matched = organizer.match(exp_regex, attrs=['name'])
if experiments_matched:
    logger.warning(f'Matching experiment exists already: {experiments_matched}')
    exit()
logger.info(f"Creating new experiment {exp_date}...")

## Create the folder if it doesn't exist
experiment_folder = experiments_basepath / exp_name
experiment_folder.mkdir(exist_ok=False)
organizer.update_experiments()

## Create configuration file to start experiment with
# Create beamBCI config containing paths to generic preprocessing and classification pipeline
with open(experiment_template_path, 'r') as fp:
    experiment_config = json.load(fp)

## Create a grid_assignment file if it has been provided
time.sleep(0.2)
path_grid_assignment = None
if input_grid_query.lower() in ["yes", "y"]:
    path_grid_assignment = experiment_folder / f'grid_assignment.json'
    with open(grid_assignment_template_path, 'r') as fp:
        grid_assignment = json.load(fp)
    grid_assignment['comment'] += f". Copied on {datetime.now().strftime('%d.%m.%Y at %H:%M')} as part of {__file__} to {path_grid_assignment}"
    if not path_grid_assignment.exists():
        logger.info(f"Writing grid assignment into {path_grid_assignment}")
        with open(path_grid_assignment, 'w') as fp:
            json.dump(grid_assignment, fp, indent='\t')
    else:
        logger.warning(f"grid_assignment exists in {path_grid_assignment}. Not writing")
else:
    logger.info(f"Not writing grid_assignment. Please create manually as {experiment_folder / f'grid_assignment.json'}")

## Put the path to the grid_assignment into the experiment configuration
if path_grid_assignment is not None:
    experiment_config['classification']['parameters'].update(grid_assignment_file=str(path_grid_assignment))

## Copy the pipeline settings into the participant folder
classifier_settings_path = experiment_folder / 'classifier_settings.json'
shutil.copy(classifier_settings_template_path, classifier_settings_path)

experiment_config['classification']['parameters'].update(settings_file=str(classifier_settings_path))
experiment_config['recording']['parameters'].update(Location=str(experiment_folder), Study=exp_id, Subject=exp_participant, Task=task)

## Write the experiment configuration file into the participant's folder
path_experiment_config = experiment_folder / f'experiment_configuration.json'
if path_experiment_config.exists():
    logger.error("The experiment config exists already")
    sys.exit()

logger.info(f"Writing experiment configuration into {path_experiment_config}")
with open(path_experiment_config, 'w') as fp:
    json.dump(experiment_config, fp, indent='\t')

with open('./.last_created_participant', 'w') as fp:
    fp.write(exp_name)

## And open in explorer
os.startfile(experiment_folder, 'explore')

logger.info(f"Experiment '{exp_name}' created")

logger.info(f'Use args: --experiment_config="{path_experiment_config}"')