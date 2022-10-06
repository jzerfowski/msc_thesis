#!/usr/bin/env python

"""
Offline Analysis
Compute the ERD timecourse for each participant individually
"""

import json

import mne
import matplotlib.pyplot as plt
import seaborn as sns

from utils_latex import write_figures_to_file
from utils_plotting import save_fig, palette, plot_csp_pattern, set_context, mark_range_y

from utils_analysis import epochs_to_data_frame, \
    epochs_permutation_test, draw_clusters, epochs_selector
from data_organizer import ExperimentOrganizer
from utils_analysis_offline import compute_erd_timecourse_csp_causal

blocks = [2, 3, 4, 5]
feedback = None

tmin_crop, tmax_crop = -2.5, 7.0
tmin_baseline, tmax_baseline = -2, 0

erd_stats = []

##

def plot_subject_erd_timecourse(experiment, ax_erd, ax_csp=False, tmin_crop=tmin_crop, tmax_crop=tmax_crop, tmin_baseline=tmin_baseline, tmax_baseline=tmax_baseline):
    epochs: mne.Epochs = experiment.get_epochs()
    epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)

    erd_freq = experiment.get_erd_freq(default=None)

    if erd_freq is None:
        raise ValueError(f"Cannot compute without erd_freq")

    with open(experiment.basepath / 'erd_freq_csp_settings_individual.json', 'r') as fp:
        csp_settings = json.load(fp)

    epochs_erd_timecourse, csp = compute_erd_timecourse_csp_causal(epochs, erd_freq, csp_settings=csp_settings, train_csp=False)

    epochs_erd_timecourse.resample(sfreq=100, npad='auto', window='boxcar', pad='edge')
    epochs_erd_timecourse.crop(tmin_crop, tmax_crop)
    mne.baseline.rescale(epochs_erd_timecourse._data, epochs_erd_timecourse.times, copy=False, mode='percent', baseline=(tmin_baseline, tmax_baseline))


    df_erd_timecourse = epochs_to_data_frame(epochs_erd_timecourse, scalings=dict(mag=1))

    df_erd_timecourse['ERD'] = df_erd_timecourse['value'] * 100

    # Begin the plotting
    sns.lineplot(data=df_erd_timecourse, x='time', y='ERD', hue='Condition', palette=palette, ci=95, n_boot=1000, ax=ax_erd)
    ax_erd.set(xlabel='Time (s)', ylabel='ERD (\%)')

    mark_range_y(x1=0, x2=5, ax=ax_erd, color='grey', alpha=0.2, zorder=0)
    times = epochs_erd_timecourse.times

    F_obs, clusters, cluster_pv, H0 = epochs_permutation_test(epochs_erd_timecourse, channel_adjacency=False)
    erd_stats = draw_clusters(clusters, cluster_pv, ax_erd, times)

    ax_erd.set(title=f"Participant {experiment.get_participant_id()}")

    if ax_csp:
        plot_csp_pattern(csp, epochs.info, component_idx=0, ax=ax_csp)

    # Compute some stats
    stats = dict(name=experiment.name, participant_id=experiment.get_participant_id(), **erd_stats)

    return stats


##

organizer = ExperimentOrganizer()
experiments = organizer.get_experiments_for_analysis('analysis_erd_timecourse')

set_context('paper')
latex_figures = []

for experiment in experiments[:None]:
    # fig, axes = plt.subplots(ncols=2)
    # stats = plot_subject_erd_timecourse(experiment, ax_erd=axes[0], ax_csp=axes[1])
    # erd_stats.append(stats)

    fig, ax = plt.subplots(ncols=1)
    stats = plot_subject_erd_timecourse(experiment, ax_erd=ax, ax_csp=None)
    erd_stats.append(stats)

    filename = f'erd_timecourse_p{experiment.get_participant_id()}'

    caption = rf"Timecourse of the \gls{{ERD}} for participant {experiment.get_participant_id()}"


    latex_figure = save_fig(fig, filename, 'erd_timecourse', latex_caption=caption, latex_label=f'app:fig:erd_timecourse_p{experiment.get_participant_id()}', figsize_pgf=(None, 2.5))
    latex_figures.append(latex_figure)
    print(latex_figure)

write_figures_to_file(latex_figures, 'app_erd_timecourses.tex')

