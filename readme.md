# Optically pumped magnetometers for a brain-computer interface based on event-related desynchronization
This repository contains the publicly available code associated with the Master's thesis project by [Jan Zerfowski](https://janzerfowski.de), conducted as a cooperation between the [Donders Institute for Brain, Cognition and Behaviour](https://www.ru.nl/donders/), the [Clinical Neurotechnology Lab](http://www.clinical-neurotechnology.com/) at [Charité – Universitätsmedizin Berlin](https://www.charite.de/) and the [Working Group 8.21 - Optical Magnetometry](https://www.ptb.de/cms/en/ptb/fachabteilungen/abt8/ag-821.html) at the [Physikalisch-Technische Bundesanstalt](https://www.ptb.de/cms/en.html) in Berlin.

The work was supervised by Surjo Soekadar and Michael Tangermann. The thesis is also available under DOI [10.5281/zenodo.10998007](https://doi.org/10.5281/zenodo.10998007)

## Content of the repository
- [readme.md](./readme.md) This readme
- [thesis.pdf](./thesis.pdf) The thesis .pdf-file
- [requirements.txt](./requirements.txt) file to reconstruct the virtual environment (venv) that has been used with python 3.8 to run all the source code
- [analysis/](./analysis) contains the code that has been used to perform all analyses and create the figures presented in this thesis
  - [characterization/](./analysis/characterization) Code to process and analyze the characterization measurements
  - [miscellaneous/](./analysis/miscellaneous) Code to generate statistics of the data and plot the filter settings
  - [offline/](./analysis/offline) Code for the offline analysis
  - [online/](./analysis/online) Code for the online analysis
  - [preprocessing/](./analysis/preprocessing) Code that was used to manually preprocess the participant data
- [create_participant/](./create_participant) is a utility program that uses template files to create all necessary data folders and files to for a new participant in the experiment
- [data_organizer/](./data_organizer) provides an interface to the folders with the recorded data for easy access to raw and epoched data. It has been used extensively in the analysis scripts
- [DataProcessors/](./DataProcessors) contains a selection of the pipeline nodes that were also required in the offline analysis. All nodes are available in the BeamBCI module in the non-public repository
- [fieldline_lsl/](./fieldline_lsl) (submodule), containing the used version of the public [fieldline_lsl](https://github.com/jzerfowski/fieldline_lsl) module
- [fif2csv/](./fif2csv) (submodule), containing the most recent version of the public [fif2csv](https://github.com/jzerfowski/fif2csv/) module
- [for_participant/](./for_participant) contains the documents and forms participants were provided with before the experiment and which were used to acquire relevant information
- [Triggerduino/](./Triggerduino) (submodule), containing the most recent version of the public [Triggerduino](https://github.com/jzerfowski/Triggerduino) software
- [xdf2mne/](./xdf2mne) (submodule), containing the used version of the public [xdf2mne](https://github.com/jzerfowski/xdf2mne)

## Abstract
### Background
Stroke is one of the leading causes of disability worldwide and often responsible for impairments of hand motor function.
Rehabilitation and restoration of motor functions can be significantly improved using devices controlled by brain signals, so called brain-computer interfaces (BCIs).
Most current BCI systems are based on electroencephalography (EEG), which provides only limited spatial resolution and thus limited versatility of control commands.
Compared to EEG, optically pumped magnetometers (OPMs) measure cortical magnetic fields without contact to the scalp and provide a higher spatial resolution and bandwidth.
In contrast to superconducting quantum interference device (SQUID)-based magnetoencephalography (MEG), OPMs have low maintenance cost and allow movement in the scanner, making them more applicable in clinical contexts.

### Methods
We quantify the signal characteristics of a commercially available OPM system (FieldLine Inc., USA) in terms of noise floor, dynamic range and bandwidth to verify its suitability for cortical measurements.
We then develop an experiment contrasting resting and right hand grasping imagery to measure modulations of the sensorimotor rhythm (SMR) with 17 OPMs over the left motor cortex of 18 healthy participants.
The BCI capabilities of the OPM acquisition system are evaluated with a modular near real-time classification pipeline, which provides visual feedback to the user.

### Results
The sensor characterization revealed a system noise floor of about 27 fT/√Hz at 10 Hz, a bandwidth of 400 Hz and a dynamic range of ±15 nT, fulfilling the minimum requirements for cortical measurements.
In 10 of 16 eligible participants, a difference in SMR power between resting and grasping condition could be identified.
We show that OPMs are suitable to measure SMR modulations in near real-time and that the classification performance of our pipeline significantly exceeds chance level.

### Discussion
OPMs allow for the online quantification of voluntary modulations of the sensorimotor rhythm on single-trial basis, a central requirement for many BCI systems used in the rehabilitation of stroke survivors.
With their higher spatial resolution compared to EEG, OPMs could be used for more complex classification paradigms and ultimately facilitate a development towards more versatile BCI applications.
The increasing availability and sensitivity of commercialized OPM systems allows for the exploration of MEG in new research areas. OPMs are projected to become an important tool in the field of cognitive neuroscience within the next few years.

**Keywords:** brain-computer interface, optically pumped magnetometer,
	event-related desynchronization, sensorimotor rhythm

![](./misc/logos_bar.png)
