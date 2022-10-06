#!/usr/bin/env python

"""
Preprocess and annotate the data for one participant
This script involves heavy manual work and `current_participant_regex` is used to select the participant

Script that was used to preprocess all subject datasets. Contains manual steps for data rejection
"""

import re
import mne
import numpy as np
from scipy.stats import zscore

from xdf2mne import get_nonlinear_annotations
from data_organizer import ExperimentOrganizer, _suffix_mne_raw_original, _suffix_mne_raw_annotated
from utils_analysis import compute_kurtosis_per_channel

import logging
logging.basicConfig(level=logging.INFO)

organizer = ExperimentOrganizer()

current_participant_regex = '.*_opm_motor_imagery_p014'

# Load experiment and prepare logger
experiment = organizer.match(current_participant_regex)[0]
experiment.load_xdf_default_args.update(dejitter_timestamps=False)

logger = experiment.getLogger_preprocessing()

logger.info(f"Beginning preprocessing of {experiment}")


########################################################################################################################
"""
# First part of the script: Computing the mne.Raw from the .xdf-file and saving it
"""

## Get mne raw object from xdf
## Load .xdf-file for this subject
logger.info(f".xdf-file found at {experiment.xdf_raw_filepath}")

logger.info("Building raw from .xdf-file")
raw_original = experiment.build_raw_from_xdf()

## Enhance annotations to indicate condition, block and feedback and have the proper duration
annotations: mne.annotations.Annotations = raw_original.annotations
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

## Save raw to file
filepath_fif_raw = experiment.basepath / f"{experiment.get_basename()}_{_suffix_mne_raw_original}"
raw_original.save(filepath_fif_raw, overwrite=False)
logger.info(f"Raw .fif derived from original xdf file saved in {filepath_fif_raw}")


########################################################################################################################
"""
Second part of the script: Manually inspecting the data to find artifacts and noisy sensors  
"""

## Load raw from files (makes this script independent of the steps executed previously)
filepath_fif_raw = experiment.basepath / experiment.get_raw_original_filenames()[-1]
raw: mne.io.RawArray = mne.io.read_raw_fif(filepath_fif_raw, preload=True)
logger.info(f"Loaded mne raw .fif file {filepath_fif_raw}: {raw}")


## Find and annotate periods in which the data exceeds range_threshold to avoid non-linearities in measurements
range_threshold = 2.0e-9
logger.info("Computing periods where sensor signal exceeds linear range +/-{range_threshold*1e9}nT")
annotations_nonlinear = get_nonlinear_annotations(raw, threshold=range_threshold, picks=None, duration_threshold_exceeded=0.2, description="BAD_NONLINEAR")
raw.set_annotations(raw.annotations + annotations_nonlinear)
logger.info(f"Found total {annotations_nonlinear.duration.sum()}s of signal exceeding threshold of +/-{range_threshold*1e9}nT")

## Crop data to contain data from continuous mode
t_continuous_start = raw.annotations[raw.annotations.description == 'CONTINUOUS_MODE/START/ON'].onset[0]
t_data_crop = t_continuous_start + 20
raw.crop(tmax=t_data_crop)
logger.info(f"End of experiment at t={t_continuous_start} s. Cropping data at t={t_data_crop}")

## User interaction:
# Inspect the annotated data to verify periods of railing and slow drifts or jumps
scalings = dict(mag=0.05e-9)  # 100pT range

logger.info(f"Showing PSD and unfiltered raw time series for user to review with scaling of {scalings['mag']*2*1e12} pT")

raw.plot(duration=10, remove_dc=True, scalings=scalings, clipping=4, title=f"Unprocessed data (s{experiment.get_participant_id():03})", block=True)
logger.info("User finished reviewing the data. Continue...")


##
annotation_label_artifact = 'BAD_ARTIFACT'
annotations_artifact_pre_bandpass = raw.annotations[raw.annotations.description == annotation_label_artifact]
logger.info(f"User annotated total of {annotations_artifact_pre_bandpass.duration.sum()}s as '{annotation_label_artifact}'")

## Obtain filtered data (remove harmonics of line noise and high pass filter) to find spiking channels (on copy to avoid shifts/delays induced by filters)
# Apply notch-filter to data copy to remove line-noise at 50 Hz and harmonics
notch_freqs = np.arange(50, 251, 50)
params_filter_notch = dict(freqs=notch_freqs, filter_length='auto', method='fir')
raw_filtered = raw.copy().notch_filter(**params_filter_notch)
logger.info(f"Applied notch filter to data with params {params_filter_notch}")

l_freq = 4
params_filter_highpass = dict(l_freq=l_freq, h_freq=None, method='iir', iir_params=dict(ftype='butter', order=5))
raw_filtered.filter(**params_filter_highpass)
logger.info(f"Created high-pass filtered copy of data, bidirectionally (to achieve zero-phase shift) to suppress movement artifacts and other linear trends under {l_freq} Hz (see Seymour et al. 2022) with params {params_filter_highpass}")

## Compute kurtosis in a sliding window over all channels to find spiking channels
window_length = 1000
kurtosis = compute_kurtosis_per_channel(raw_filtered.get_data(reject_by_annotation='omit'), window_length=window_length)
kurtosis_zscores = zscore(kurtosis)
excess_kurtosis_threshold = 1.96
logger.info(f"Computed kurtosis values for dataset in sliding windows of {window_length} samples to find spiking.")
logger.info(f"Channels with excess kurtosis (zscore>{excess_kurtosis_threshold}) suggested for further inspection: {np.array(raw.ch_names)[kurtosis_zscores > excess_kurtosis_threshold]}")

##
l_freq = 4
scalings = dict(mag=5e-12)  # 10pT range
logger.info(f"Showing raw time series for user to review and find high-frequency artifacts with scaling of {scalings['mag']*2*1e12} pT"
            "(identifying transient (less than 100 ms) and very large shifts in the OPM data that "
            f"appeared on all channels (see Seymour et al. 2022)) with applied high-pass filter at {l_freq} Hz")
raw.plot(duration=10, remove_dc=True, scalings=scalings, highpass=l_freq, filtorder=5, clipping=3, title=f"High-pass filtered >{l_freq} Hz", block=True)
logger.info("User finished reviewing the data. Continue...")


## Save the data
filepath_fif_annotated = experiment.basepath / f"{experiment.get_basename()}_{_suffix_mne_raw_annotated}"
raw.save(filepath_fif_annotated)
