#!/usr/bin/env python

"""
Module for organization of experiment data
ExperimentOrganizer can be called with multiple paths where experiment files are stored
Can be called with a list of basepaths
The settings file contains all the personal settings (default location of experiment data) for convenience
"""

import functools
import os
import re
from typing import Union, List, Optional

import pathlib

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyxdf import pyxdf
import logging
import xdf2mne

import utils_mne
import positioning_utils

import settings
from utils_participant_meta import get_info_for_experiment_name, get_experiment_names_for_analysis

logger = logging.getLogger(__name__)

_suffix_mne_epochs_offline_postica = 'offline_postica_epo.fif'
_suffix_mne_epochs_offline_annotated = 'annotated_epo.fif'
_suffix_mne_epochs_online = 'online_epo.fif'
_suffix_mne_raw_original = 'raw.fif'
_suffix_mne_raw_annotated = 'annotated_raw.fif'
_suffix_mne_raw_postica = 'preprocessed_postica_raw.fif'
_suffix_xdf_raw = '.xdf'
_suffix_log_preprocessing = 'preprocessing.log'
_suffix_log_preprocessing_ica = 'preprocessing_ica.log'
_suffix_log_epoching = 'epoching.log'
_suffix_log_epoching_online = 'epoching_online.log'
_suffix_log_tfr_compute = 'tfr_compute.log'
_suffix_log_analysis_offline = 'analysis_offline.log'
_suffix_log_analysis_online = 'analysis_online.log'
_suffix_log_epoching_details = 'epoching_details.json'

_suffix_tfr_morlet = 'morlet_tfr.h5'
_suffix_tfr_morlet_full = 'full_morlet_tfr.h5'
_suffix_tfr_morlet_alpha = 'alpha_morlet_tfr.h5'


class Experiment:
    load_xdf_default_args = dict(dejitter_timestamps=False)

    def __init__(self, basepath: pathlib.Path):
        self.basepath = basepath
        self.name = self.basepath.name

        self.filenames = [filepath.name for filepath in self.basepath.iterdir()]

        logger.debug(f"Indexed Experiment {self.name} in {self.basepath}")

        self._xdf_filenames = None
        self._xdf_raw_filepath = None
        self._xdf_raw = None

        self._grid_assignment_filepath = None

        self._raw_filepath = None
        self._raw_cleaned_filepath = None
        self._epochs_filenames = None

    def __str__(self):
        return f"Experiment '{self.name}' at {self.basepath}"

    def __repr__(self):
        return self.__str__()

    def get_grid_assignment_filenames(self):
        regexp = '.*grid_assignment.*.json'

        grid_assignment_files = [filename for filename in self.filenames if re.match(regexp, filename)]
        if len(grid_assignment_files) == 0:
            logger.warning(f"No grid assignment files found in experiment")

        return grid_assignment_files

    def plot_grid_assignment(self) -> Optional[plt.Figure]:
        if not self.get_grid_assignment_filenames():
            logger.warning("No grid_assignment found, nothing to plot")
            return None
        else:
            fig = positioning_utils.plot_grid_assignment(self.basepath / self.get_grid_assignment_filenames()[0])
            return fig

    @property
    def xdf_files(self):
        if self._xdf_filenames is not None:
            return self._xdf_filenames

        self._xdf_filenames = [filename for filename in self.filenames if pathlib.Path(filename).suffix == '.xdf']

        if len(self._xdf_filenames) == 0:
            logger.warning("No xdf-files in experiment found")
        else:
            logger.debug(f"Found {len(self._xdf_filenames)} xdf-files in Experiment")

        return self._xdf_filenames

    @property
    def xdf_raw_filepath(self):
        if self._xdf_raw_filepath is not None:
            return self._xdf_raw_filepath

        if len(self.xdf_files) == 0:
            logger.warning("Raw .xdf-file not found")
            self._xdf_raw_filepath = None
        elif len(self.xdf_files) == 1:
            self._xdf_raw_filepath = pathlib.Path(self.basepath, self.xdf_files[0])
        else:
            logger.info(f"{len(self.xdf_files)} xdf-files found, defaulting to first in list")
            self._xdf_raw_filepath = pathlib.Path(self.basepath, self.xdf_files[0])

        return self._xdf_raw_filepath

    def get_xdf_streams(self, filename=None):
        if filename is None:
            filename = self.xdf_raw_filepath
        else:
            filename = self.basepath / filename

        if filename is None:
            logger.warning("Could not load streams from non-existent .xdf file")

        streams, header = self.load_xdf(filename=filename, **self.load_xdf_default_args)
        return streams

    def get_basename(self):
        return pathlib.Path(self.get_xdf_raw_filenames()[0]).stem

    def getLogger(self, name__, suffix, basepath, mode='a') -> logging.Logger:
        logger = logging.getLogger(f"{self.name}.{name__}.")
        logger.addHandler(self.get_logging_filehandler(suffix, basepath=basepath, mode=mode))
        return logger

    def getLogger_epoching(self, mode='w'):
        return self.getLogger('epoching', _suffix_log_epoching, basepath=self.basepath, mode=mode)

    def getLogger_preprocessing(self, mode='a'):
        return self.getLogger('preprocessing', _suffix_log_preprocessing, basepath=self.basepath, mode=mode)

    def getLogger_preprocessing_ica(self, mode='a'):
        return self.getLogger('preprocessing_ica', _suffix_log_preprocessing_ica, basepath=self.basepath, mode=mode)

    def getLogger_epoching_online(self, mode='w'):
        return self.getLogger('epoching.online', _suffix_log_epoching_online, basepath=self.basepath, mode=mode)

    def get_logging_filehandler(self, suffix, basepath=None, mode='a'):
        if basepath is None:
            basepath = self.basepath
        fh = logging.FileHandler(basepath / f"{self.name}_{suffix}", mode=mode)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        return fh

    def get_xdf_header(self):
        if self.xdf_raw_filepath is None:
            logger.warning("Could not load header from non-existent .xdf file")

        streams, header = self.load_xdf(filename=self.xdf_raw_filepath, **self.load_xdf_default_args)
        return header

    def load_xdf(self, filename=None, **kwargs):
        if filename is None:
            filename = self.xdf_raw_filepath
            if filename is None:
                logger.warning(f"Could not load xdf-file because no filename was provided")
                return None, None

        logger.info(f"Loading streams from {filename}")
        streams_, header = pyxdf.load_xdf(filename=filename, **kwargs)
        streams = {stream['info']['name'][0]: stream for stream in streams_}
        logger.info(f"Found {len(streams)} streams in {filename}: {list(streams.keys())}")

        return streams, header

    def get_filenames(self, basepath: Optional[pathlib.Path] = None):
        if basepath is None:
            basepath = self.basepath
        return [filepath.name for filepath in basepath.iterdir()]

    def get_xdf_raw_filenames(self):
        return self.get_suffixed_filenames(_suffix_xdf_raw)

    def get_epochs_postica_filenames(self):
        return self.get_suffixed_filenames(_suffix_mne_epochs_offline_postica)

    def get_epochs_annotated_filenames(self):
        return self.get_suffixed_filenames(_suffix_mne_epochs_offline_annotated)

    def get_epochs_online_filenames(self):
        return self.get_suffixed_filenames(_suffix_mne_epochs_online)

    def get_erd_df_filename(self):
        return f"{self.get_basename()}_erd_df.h5"

    def write_erd_df(self, df_erd):
        df_erd.to_hdf(self.basepath / self.get_erd_df_filename(), key='erd_df')

    def get_erd_df(self) -> pd.DataFrame:
        return pd.read_hdf(self.basepath / self.get_erd_df_filename(), key='erd_df')

    @functools.lru_cache(maxsize=100, typed=False)
    def get_epochs_deprecated(self) -> Optional[mne.epochs.Epochs]:
        if self.get_epochs_postica_filenames():
            return mne.epochs.read_epochs(self.basepath / self.get_epochs_postica_filenames()[0])
        else:
            return None

    @functools.lru_cache(maxsize=5, typed=False)
    def get_epochs(self, stage='annotated') -> Optional[mne.epochs.Epochs]:
        if stage == 'annotated':
            return self.get_epochs_annotated()
        else:
            logger.warning("Only 'annotated' is implement as stage argument for get_epochs()!")
            return None

    @functools.lru_cache(maxsize=1, typed=False)
    def get_epochs_annotated(self) -> Optional[mne.epochs.Epochs]:
        if self.get_epochs_annotated_filenames():
            return mne.epochs.read_epochs(self.basepath / self.get_epochs_annotated_filenames()[0])
        else:
            return None

    @functools.lru_cache(maxsize=100, typed=False)
    def get_epochs_online(self) -> Optional[mne.epochs.Epochs]:
        if self.get_epochs_online_filenames():
            return mne.epochs.read_epochs(self.basepath / self.get_epochs_online_filenames()[0])
        else:
            return None

    def get_raw_filenames(self):
        return self.get_suffixed_filenames(_suffix_mne_raw_original)

    def get_raw_original_filenames(self):
        return [filename for filename in self.get_raw_filenames() if filename not in self.get_raw_annotated_filenames()]

    def get_raw_annotated_filenames(self):
        return self.get_suffixed_filenames(_suffix_mne_raw_annotated)

    def get_log_preprocessing_filenames(self):
        return self.get_suffixed_filenames(_suffix_log_preprocessing)

    def get_suffixed_filenames(self, suffix, basepath: Optional[pathlib.Path] = None):
        return [filename for filename in self.get_filenames(basepath=basepath) if filename.endswith(suffix)]

    def get_info(self):
        return get_info_for_experiment_name(self.name)

    def get_erd_freq(self, default=None, key='erd_freq'):
        erd_freq = self.get_info()[key]
        if erd_freq is None or np.isnan(erd_freq):
            erd_freq = default
        return erd_freq

    def get_participant_id(self):
        return f"{int(self.get_info()['PID']):03d}"


    def build_raw_from_xdf(self, data_stream: str = 'FieldLineOPM', marker_stream: Optional[str] = 'TaskOutput', filename = None):
        streams = self.get_xdf_streams(filename=filename)
        if not streams:
            logger.error(f"No xdf-streams found to build Epochs from {self}")
            return None

        if marker_stream is None or marker_stream == '':
            marker_streams = []
        elif marker_stream in streams:
            marker_streams = [streams[marker_stream]]
        else:
            logger.error(f"Marker stream {marker_stream} not found in streams {streams.keys()}")
            marker_streams = []

        raw = xdf2mne.streams2raw(streams[data_stream], marker_streams=marker_streams,
                                  ch_type_t=lambda x=None: xdf2mne.ch_type_transform_default(x, 'mag'))

        if self.get_grid_assignment_filenames():
            logger.info(f"Adding grid_assignment to raw")
            try:
                positioning_utils.add_channel_coords_from_file(raw.info, fname=self.basepath /
                                                                               self.get_grid_assignment_filenames()[0])
            except RuntimeError as e:
                logger.warning(
                    f"Could not add grid assignment file {self.get_grid_assignment_filenames()[0]} because an exception was raised: {e}")

        return raw

    def build_epochs_from_xdf(self, tmin=0, tmax=5, baseline=None, training_classes_regexp=None,
                              data_stream='FieldLineOPM', marker_stream='TaskOutput'):
        raw = self.build_raw_from_xdf(data_stream=data_stream, marker_stream=marker_stream)

        if raw is None:
            logger.error(f"No xdf-strems found to build raw from {self}")
            return None

        events, event_id = utils_mne.get_events_from_raw(raw, training_classes_regexp=training_classes_regexp)
        if events is None:
            logger.error(f"No events could be extracted from raw. Returning None")
            return None

        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
        return epochs

    def match(self, regexp: str, attrs: Optional[List[str]] = None):
        """
        Evaluates an arbitrary regular expression on attributes of the experiment. Default: Match only on name
        :param regexp:
        :param attrs:
        :return:
        """
        if attrs is None:
            attrs = ['name']
        matched = False
        for attr in attrs:
            try:
                if re.match(regexp, str(getattr(self, attr))):
                    logger.debug(f"Regex matched in experiment {self} in attribute {attr}")
                    matched = True
            except TypeError as e:
                logger.warning(f"Could not match attribute {attr} in experiment {self} because of TypeError {e}")
        return matched

    @classmethod
    def create_experiment(cls, basepath, verbose=True):
        if not cls._valid_experiment_path(basepath):
            if verbose:
                logger.warning(f"{basepath} is not a valid path for an experiment")
            return None
        else:
            return cls(basepath)

    @classmethod
    def _valid_experiment_path(cls, basepath):
        is_valid = True
        basepath = pathlib.Path(basepath)
        if basepath.name.startswith("."):
            is_valid = False
        if not basepath.is_dir():
            is_valid = False

        return is_valid


class DataFolder:
    def __init__(self, basepath: Union[str, pathlib.Path]):
        basepath = pathlib.Path(basepath)
        if not basepath.exists():
            raise RuntimeError(f"The path {basepath} does not exist")
        self.basepath: pathlib.Path = basepath

        self._experiments = None
        self.get_experiments(update=True)

    def __str__(self):
        return f"DataFolder containing {len(self.experiments)} experiments"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, experiment_id: Union[str, int]):
        if isinstance(experiment_id, str):
            return self.get_experiment_by_name(experiment_name=experiment_id)
        elif isinstance(experiment_id, int):
            return self.get_experiment_by_index(experiment_idx=experiment_id)

    def get_experiment_by_index(self, experiment_idx):
        if self._experiments is None:
            self.update_experiments()

        try:
            return self.experiments[experiment_idx]
        except IndexError:
            raise IndexError(
                f"{experiment_idx} not a valid index for DataFolder with {len(self.experiments)} experiments")

    def get_experiment_by_name(self, experiment_name):
        if self._experiments is None:
            self.update_experiments()

        if experiment_name in self._experiments:
            return self._experiments[experiment_name]
        else:
            raise KeyError(f"Experiment with name {experiment_name} not found")

    @property
    def experiments(self) -> List[Experiment]:
        if self._experiments is None:
            self.update_experiments()
        return list(self._experiments.values())

    def update_experiments(self) -> List[Experiment]:
        return self.get_experiments(update=True)

    def get_experiments(self, update=False) -> List[Experiment]:
        if update is False and self._experiments is None:
            return []
        elif update is False and self._experiments is not None:
            return self._experiments

        self._experiments = {}

        # In all other cases update would be true and we want to iterate over all folders here
        for experiment_dir in self.basepath.iterdir():
            experiment_name = experiment_dir.name
            if experiment_name.startswith("."):
                # Don't index folders starting with "."
                continue
            if experiment_name in self._experiments:
                # The experiment has been indexed already, skip.
                continue
            else:
                experiment = Experiment.create_experiment(experiment_dir, verbose=False)
                if experiment is not None:
                    self._experiments[experiment_name] = experiment

        return list(self._experiments.values())


class ExperimentOrganizer:
    def __init__(self, experiment_basepaths: Union[str, pathlib.Path, List[str], List[pathlib.Path]] = None):
        if experiment_basepaths is None:
            logger.info(f"No experiment basepaths given, fallback to default: {settings.default_experiment_basepaths}")
            experiment_basepaths = settings.default_experiment_basepaths
        elif isinstance(experiment_basepaths, (str, pathlib.Path)):
            experiment_basepaths = [pathlib.Path(experiment_basepaths)]
        else:
            pass

        self.experiment_basepaths = experiment_basepaths

        self.datafolders: List[DataFolder] = [DataFolder(pathlib.Path(basepath)) for basepath in experiment_basepaths]

        self._experiments: Optional[List[Experiment]] = None

        self.update_experiments()
        logger.info(
            f"ExperimentOrganizer initialized with {len(self.datafolders)} data folders and total {len(self.experiments)} experiments")

    def __str__(self):
        return f"ExperimentOrganizer holding {len(self.datafolders)} data folders with total {len(self.experiments)} experiments"

    def __repr__(self):
        return self.__str__()

    @property
    def experiments(self) -> List[Experiment]:
        if self._experiments is None:
            self.update_experiments()

        return self._experiments

    def match(self, regexp, attrs=None) -> List[Experiment]:
        experiments_matched = [exp for exp in self.experiments if exp.match(regexp, attrs=attrs)]
        logger.debug(f'Found {len(experiments_matched)} matching experiments for regular expression {regexp}')
        return experiments_matched

    def update_experiments(self) -> List[Experiment]:
        experiments = []
        for datafolder in self.datafolders:
            experiments.extend(datafolder.update_experiments())

        self._experiments = experiments
        return self._experiments

    def __getitem__(self, experiment_id: Union[str, int]) -> Optional[Experiment]:
        for datafolder in self.datafolders:
            try:
                return datafolder[experiment_id]
            except (KeyError, IndexError):
                pass
        logger.error(f"Experiment {experiment_id} not found in {self}")
        return None

    def get_experiments_for_analysis(self, analysis_name='all', participants_meta_filepath=None):
        experiment_names = get_experiment_names_for_analysis(analysis_name,
                                                             participants_meta_filepath=participants_meta_filepath)
        return [self[name] for name in experiment_names]
