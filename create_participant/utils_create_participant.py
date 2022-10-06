import pathlib
from datetime import datetime
from data_organizer import ExperimentOrganizer

organizer = ExperimentOrganizer()
experiments_basepath = organizer.experiment_basepaths[0]

## Create experiment details
exp_date = datetime.now().strftime('%Y%m%d')
exp_id = 'opm_motor_imagery'
participant_prefix = 'p'  # Iterated from p (participant/pilot, q, ...)
task = 'full_experiment'
grid_assignment_template_path = r'./templates/grid_assignment.json'
experiment_template_path = r'templates/experiment_template.json'
classifier_settings_template_path = r'templates/classifier_settings.json'

##
def get_last_created_experiment():
    with open('./.last_created_subject', 'r') as fp:
        exp_name = fp.readline()
    return exp_name