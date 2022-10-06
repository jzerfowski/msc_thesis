import json

import mne
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils_analysis import epochs_get_labels, epochs_selector
from utils_latex import write_figures_to_file
from utils_plotting import save_fig, cmap_diverging, transform, draw_eeg_channels, set_context
from CSPNode import CSPNode
from ProcessingPipeline import ProcessingPipeline

from data_organizer import ExperimentOrganizer
from utils_analysis_offline import get_filter_params

organizer = ExperimentOrganizer()
experiments = organizer.get_experiments_for_analysis('analysis_erd_focality')

##
blocks = [2, 3, 4, 5]
feedback = False
cv = 5
tmin, tmax = 1, 5

df_arrows = pd.DataFrame()


def plot_focality(experiment, draw_cbar=True, ax=None, draw_arrow=False, eeg_picks=['C3', 'C4', 'Cz', 'C1']):
    participant_id = int(experiment.get_participant_id())
    erd_freq = experiment.get_erd_freq()

    epochs = experiment.get_epochs()

    _csp_settings = dict(in_channel_labels=epochs.ch_names, n_components=1, method='physiological')
    csp = CSPNode(**_csp_settings)


    epochs = experiment.get_epochs()
    epochs: mne.Epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)
    params_filter = get_filter_params(erd_freq, mode='erd')
    #
    epochs.filter(**params_filter)

    pipeline = ProcessingPipeline(in_channel_labels=csp.in_channel_labels, nodes=[csp])
    X = epochs.get_data(tmin=tmin, tmax=tmax, picks='all')
    times = epochs.times
    times = times[(times >= tmin) & (times < tmax)]

    y = epochs_get_labels(epochs, asarray=True)

    X_train_out, y_train, timestamps_train = pipeline.train(X, y, timestamps=times)

    pattern = csp.A[:, 0]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    with sns.axes_style('white'):
        im, cm = mne.viz.plot_topomap(pattern, epochs.info, axes=ax, show=False, res=256, cmap=cmap_diverging)

        divider = make_axes_locatable(ax)
        if draw_cbar:
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, orientation='vertical')

            cbar.ax.get_yaxis().set_ticks([])
            cbar.ax.get_yaxis()
            cbar.ax.set_title('Pattern (norm)')

            # Arbitrary units, so normalize to -1 to 1
            ylim = cbar.ax.get_ylim()
            cbar.set_ticks(list(ylim) + [0])
            cbar.set_ticklabels([-1, 1, 0])

    ax.set_title(f"Participant {participant_id}")

    pos = mne.viz.topomap._find_topomap_coords(epochs.info, picks='all', sphere=None)
    draw_eeg_channels(ax=ax, eeg_picks=eeg_picks, alpha=0.7)

    # Plot arrow between position of highest and lowest contribution
    im_arr = im.get_array()  # the interpolated array
    idx_max = np.unravel_index(im_arr.argmax(), im_arr.shape)
    idx_min = np.unravel_index(im_arr.argmin(), im_arr.shape)

    x_min, y_min = transform(*idx_min[::-1], im)
    x_max, y_max = transform(*idx_max[::-1], im)
    val_min = im_arr[idx_min]
    val_max = im_arr[idx_max]

    dx, dy = x_max - x_min, y_max - y_min
    _df_arrows = pd.DataFrame([dict(Participant=participant_id, x_min=x_min, x_max=x_max,
                                                         y_min=y_min, y_max=y_max, dx=dx, dy=dy, val_min=val_min,
                                                         val_max=val_max)])

    return fig, ax, _df_arrows


## Plot the comparison for the results section
set_context('paper')

fig_comparison, axes = plt.subplots(ncols=2)
plot_focality(experiments[1], draw_cbar=False, ax=axes.flat[0], eeg_picks=['C3', 'C4', 'Cz', 'C1', 'CP3'])
plot_focality(experiments[5], draw_cbar=True, ax=axes.flat[1], eeg_picks=['C3', 'C4', 'Cz', 'C1', 'FC1'])

fig_comparison.tight_layout()
save_fig(fig_comparison, f'focality_erd_comparison_retrained', 'erd_focality', figsize_pgf=(6.5, 2.5))


## Plots for all participants
set_context('paper')

latex_figures = []
for experiment in experiments[:None]:
    fig, ax, df_arrows_exp = plot_focality(experiment, draw_arrow=True)
    df_arrows = pd.concat([df_arrows, df_arrows_exp], ignore_index=True)
    latex_figure = save_fig(fig, f'p{experiment.get_participant_id()}_focality_erd_individual_retrained', 'erd_focality', figsize_pgf=(None, 2.5), latex_caption=f'Participant {experiment.get_participant_id()}: Visualization of CSP pattern corresponding to the lowest eigenvalue.', latex_label=f'app:fig:erd_focality_p{experiment.get_participant_id()}')
    latex_figures.append(latex_figure)

write_figures_to_file(latex_figures, 'app_erd_focality.tex')
