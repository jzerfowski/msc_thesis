#!/usr/bin/env python

"""
Offline Analysis
Module to compute the spectra and compare them during relaxation and grasping motor imagery for all participants
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sns

from utils_analysis import epochs_selector, epochs_get_conditions, array_to_data_frame, \
    epochs_data_frame_fix_condition
from utils_latex import write_figures_to_file
from utils_plotting import save_fig, palette, mark_range_y, set_context
from data_organizer import ExperimentOrganizer
from CSPNode import CSPNode
from utils_analysis_offline import train_csp_on_epochs, apply_csp_on_epochs, erd_band_half_width, get_filter_params

organizer = ExperimentOrganizer()
experiments = organizer.get_experiments_for_analysis('analysis_erd_freq')


##
train_tmin, train_tmax = (1, 5)

params_welch = dict(window='hann', nperseg=2 ** 11, noverlap=2 ** 10 + 2 ** 9, axis=-1, scaling='density')

blocks = [1]
feedback = False

def compute_df_welch_csp(epochs, csp_n_components=1):
    sfreq = epochs.info['sfreq']

    params_csp_filter = get_filter_params(erd_freq=None, mode='generic')
    epochs_filtered = epochs.copy().filter(**params_csp_filter)

    csp = CSPNode(in_channel_labels=epochs.ch_names, n_components=csp_n_components, method='physiological')
    train_csp_on_epochs(epochs_filtered, csp, tmin=train_tmin, tmax=train_tmax)

    epochs_csp = apply_csp_on_epochs(epochs, csp)
    ch_names = epochs_csp.ch_names

    conditions = np.array(epochs_get_conditions(epochs_csp))

    f_welch, Pxx_welch = scipy.signal.welch(epochs_csp.get_data(tmin=train_tmin, tmax=train_tmax), fs=sfreq,
                                            **params_welch)

    df_welch = array_to_data_frame(Pxx_welch, axes=[np.arange(len(conditions)), ch_names, f_welch],
                                   axes_names=['epoch', 'ch_name', 'freq']).reset_index()
    df_welch['Condition'] = df_welch['epoch'].map({epoch: label for epoch, label in enumerate(conditions)}.get)
    df_welch = epochs_data_frame_fix_condition(df_welch)  # makes 3 columns out of 1 condition column
    df_welch.loc[:, 'value_log'] = np.log(df_welch['value'])

    return df_welch, csp


##
def plot_df_welch(df_welch, csp, ax=None, fmin_plot=6, fmax_plot=18, value_name='value', n_boot=10000, participant_id=None, freq_erd=None):
    if ax is None:
        fig, ax = plt.subplots()

    df_welch_plot = df_welch[(df_welch.freq > fmin_plot) & (df_welch.freq < fmax_plot)].copy()
    df_welch_plot.loc[:, 'normer'] = df_welch_plot.groupby(['Condition', 'freq'])[value_name].transform('mean')

    value_name_norm = f'{value_name}norm'
    df_welch_plot.loc[:, value_name_norm] = (df_welch_plot[value_name] - df_welch_plot['normer'].min()) / (df_welch_plot['normer'].max() - df_welch_plot['normer'].min())

    sns.lineplot(data=df_welch_plot, x='freq', y=value_name_norm, hue='Condition', palette=palette, ax=ax, ci=95,
                 n_boot=n_boot)
    ax.set(xlim=(fmin_plot, fmax_plot), xlabel='Frequency (Hz)', ylabel='Normalized signal power (a.u.)',
           title=f'Participant {participant_id}')

    mark_range_y(8, 16, text=r"", color='grey', alpha=0.1, zorder=0, ax=ax)
    ax.legend()

    # Load determined ERD frequency
    if freq_erd is not None:
        mark_range_y(freq_erd - erd_band_half_width, freq_erd + erd_band_half_width, text=r'', color='green', alpha=0.2, zorder=1, ax=ax)

    return ax

##

set_context('paper')

latex_figures = []

# Automatically generate plots for all experiments
for experiment in experiments[:None]:
    epochs = experiment.get_epochs()
    epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)

    participant_id = int(experiment.get_participant_id())
    erd_freq = experiment.get_erd_freq()

    # This will return the 'generic' csp without incorporating the participant's individual ERD freq
    df_welch, csp_generic = compute_df_welch_csp(epochs)

    # save csp settings to file
    csp_settings_generic = csp_generic.get_settings(include_weights=True, include_patterns=True)
    with open(experiment.basepath / 'erd_freq_csp_settings_generic.json', 'w') as fp:
        json.dump(csp_settings_generic, fp)


    erd_freq = experiment.get_erd_freq()
    if erd_freq is not None:
        params_csp_filter = get_filter_params(erd_freq=erd_freq, mode='erd')
        epochs_filtered = epochs.copy().filter(**params_csp_filter)

        csp_individual = CSPNode(in_channel_labels=epochs.ch_names, n_components=1, method='physiological')
        train_csp_on_epochs(epochs_filtered, csp_individual, tmin=train_tmin, tmax=train_tmax)

        csp_settings_individual = csp_individual.get_settings(include_weights=True, include_patterns=True)
        with open(experiment.basepath / 'erd_freq_csp_settings_individual.json', 'w') as fp:
            json.dump(csp_settings_individual, fp)

    # plot
    fig, ax = plt.subplots()
    plot_df_welch(df_welch, csp_generic, ax=ax, fmin_plot=6, fmax_plot=18, value_name='value', n_boot=1000, participant_id=participant_id, freq_erd=erd_freq)

    figname = f"erd_frequency_p{experiment.get_participant_id()}"

    latex_caption = r'Participant {{{pid}}}: \gls{{ERD}} frequency plot. '.format(pid=experiment.get_participant_id())
    if erd_freq is None:
        latex_caption += r'No \gls{ERD} frequency could be determined for this participant.'
    else:
        latex_caption += r"The \gls{{ERD}} frequency was determined to be \qty{{{erd_freq}}}{{\hertz}} ($\qty{{\pm 1}}{{\hertz}}$)".format(erd_freq=erd_freq)
    latex_figure = save_fig(fig, figname, 'erd_frequency', latex_caption=latex_caption, latex_label=f'app:fig:erd_frequency_p{experiment.get_participant_id()}')
    print(latex_figure)
    latex_figures.append(latex_figure)

write_figures_to_file(latex_figures, 'app_erd_frequencies.tex')

## Selected subjects for plotting in thesis to show in a single plot
set_context('paper')

experiment_visible = organizer.match('20211217_opm_motor_imagery_p012')[0]
erd_freq_visible = experiment_visible.get_erd_freq()
epochs_visible = epochs_selector(experiment_visible.get_epochs(), blocks=blocks, feedback=feedback)
df_welch_visible, csp_visible = compute_df_welch_csp(epochs_visible)

experiment_invisible = organizer.match('20211222_opm_motor_imagery_p020')[0]
erd_freq_invisible = experiment_invisible.get_erd_freq()
epochs_invisible = epochs_selector(experiment_invisible.get_epochs(), blocks=blocks, feedback=feedback)
df_welch_invisible, csp_invisible = compute_df_welch_csp(epochs_invisible)

fig, axes = plt.subplots(ncols=2)

plot_df_welch(df_welch_visible, csp_visible, ax=axes[0], fmin_plot=6, fmax_plot=18, value_name='value', n_boot=1000, participant_id=experiment_visible.get_participant_id(), freq_erd=erd_freq_visible)
plot_df_welch(df_welch_invisible, csp_invisible, ax=axes[1], fmin_plot=6, fmax_plot=18, value_name='value', n_boot=1000, participant_id=experiment_invisible.get_participant_id(), freq_erd=erd_freq_invisible)

axes[0].get_legend().remove()
axes[1].set_ylabel('')

save_fig(fig, 'results_comparison_erd_frequency', 'erd_frequency', figsize_pgf=(6.5, 2.5))