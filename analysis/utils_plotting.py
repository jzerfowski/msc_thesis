#!/usr/bin/env python

"""
Utility functions for plotting
Used throughout all analyses to standardize color schemes and provide utility functions for saving LaTeX compatible figures
"""
import inspect
import logging
import pathlib

from datetime import datetime

import matplotlib as mpl
import mne
import numpy as np
import pandas.io.formats.style

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from utils_latex import LatexFigure


_default_palette = sns.color_palette('muted')
palette = {'Relax': _default_palette[0],
           'Reference': _default_palette[0],
           'Close': _default_palette[1],
           'Closed': _default_palette[2],
           'Open': _default_palette[3],
           0: _default_palette[4],
           1: _default_palette[5],
           'ERD': _default_palette[8],
           'No ERD': _default_palette[9],
           }

cmap_diverging = sns.color_palette("vlag", as_cmap=True)
cmap_accuracies = sns.color_palette('vlag', as_cmap=True)

# latex_textwidth = 4.7
latex_textwidth = 4.7747
latex_plotheight = 2.5

figure_logfile = pathlib.Path(__file__).parent / 'utils_plotting.log'
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(figure_logfile, mode='a'))

basepath_figures_png = pathlib.Path(r'D:\Thesis\figures')
basepath_figures_pgf =  pathlib.Path(r"D:\Thesis\src\thesis") / './figures/'

_context = None
_mpl_rcParamsDefault = mpl.rcParams.copy()


def set_context(context='talk'):
    print(f"{__name__} setting {context=}")
    with mpl._api.suppress_matplotlib_deprecation_warning():
        mpl.rcParams.update(_mpl_rcParamsDefault)

    global _context

    if context == 'paper':
        _context = context

        sns.set_style('darkgrid')
        sns.set_context('paper')
        # matplotlib.use("pgf")

        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "figure.figsize": [12.8, 9.6],
        })

        plt.rcParams["text.latex.preamble"] += "\n".join([
            r"\usepackage{siunitx}",
            r"\usepackage{underscore}",
        ])
        #
        # plt.rcParams["text.latex.preamble"].join([
        #     r"\usepackage{siunitx}",
        #     r"\usepackage{underscore}",
        # ])

    elif context == 'talk':
        _context = context

        sns.set_style('darkgrid')
        sns.set_context('talk')

        mpl.rcParams.update({
            "figure.figsize": [12.8, 9.6],
        })

    elif context == 'poster':
        _context = context

        sns.set_style('darkgrid')
        sns.set_context('poster')

        mpl.rcParams.update({
            "figure.figsize": [20, 15],
        })

    else:
        _context = None
        pass


def save_df(s: pandas.io.formats.style.Styler, filename, subdir='.', basepath_html: pathlib.Path = basepath_figures_png, basepath_tex: pathlib.Path = basepath_figures_pgf):
    filepath_html = (basepath_html / subdir / filename).with_suffix('.html')
    filepath_tex = (basepath_tex / subdir / filename).with_suffix('.tex')

    filepath_html.parent.mkdir(exist_ok=True)
    filepath_tex.parent.mkdir(exist_ok=True)

    s.to_html(filepath_html)


def save_fig(fig: plt.Figure, filename, subdir='.', basepath_png: pathlib.Path = basepath_figures_png, basepath_pgf: pathlib.Path = basepath_figures_pgf, figsize_pgf=None, latex_caption='', latex_label='', **kwargs):

    global _context

    saved_to = []

    if figsize_pgf is None:
        figsize_pgf = (latex_textwidth, latex_plotheight)
    elif isinstance(figsize_pgf, tuple):
        if figsize_pgf[0] is None:
            figsize_pgf = (latex_textwidth, figsize_pgf[1])
        if figsize_pgf[1] is None:
            figsize_pgf = (figsize_pgf[0], latex_plotheight)
    else:
        raise ValueError("Need tuple or None")

    if _context == 'talk':
        filename += '_talk'
    elif _context == 'poster':
        filename += '_poster'

    if basepath_png is not None:
        basepath_png = pathlib.Path(basepath_png)
        filepath_png = (basepath_png / subdir / filename).with_suffix('.png')
        filepath_png.parent.mkdir(exist_ok=True)

        fig.savefig(filepath_png, bbox_inches='tight', format='png', **kwargs)

        saved_to.append(filepath_png)

    if basepath_pgf is not None:
        basepath_pgf = pathlib.Path(basepath_pgf)
        filepath_pgf = (basepath_pgf / subdir / filename).with_suffix('.pgf')

        filepath_pgf.parent.mkdir(exist_ok=True)

        figsize = fig.get_size_inches()

        fig.set_size_inches(*figsize_pgf)
        fig.savefig(filepath_pgf, bbox_inches='tight', format='pgf', **kwargs)

        fig.set_size_inches(*figsize)
        saved_to.append(filepath_pgf)

    frame,caller_filename,line_number,function_name,lines,index = inspect.stack()[1]

    logger.info(f"Saved figure to {' and '.join([str(p) for p in saved_to])} on {datetime.now():%d.%m.%Y %H:%M} by {caller_filename}:{line_number}")

    return LatexFigure(filename=f"{filename}.pgf", path=f"figures/{subdir}", caption=latex_caption, label=latex_label)


def transform(x_idx: int, y_idx: int, im: AxesImage):
    x_idx_max, y_idx_max = im.get_array().shape
    x_extent, y_extent = (im.get_extent()[1] - im.get_extent()[0]), (im.get_extent()[3] - im.get_extent()[2])
    x = (x_idx - x_idx_max / 2) / x_idx_max * x_extent
    y = (y_idx - y_idx_max / 2) / y_idx_max * y_extent

    return x, y


def draw_eeg_channels(ax, eeg_picks=['C3', 'C4', 'Cz'], sphere=(0, -0.065, 0, 0.095), radius=0.007, alpha=0.5):
    standard_1020 = mne.channels.make_standard_montage('standard_1020')
    ch_names = standard_1020.ch_names
    raw = mne.io.RawArray(np.zeros((len(ch_names), 1)),
                          mne.create_info(ch_names=ch_names, sfreq=1, ch_types='eeg')).set_montage(standard_1020)
    eeg_pos = mne.viz.topomap._find_topomap_coords(raw.info, picks=eeg_picks, sphere=sphere)

    for ch_name, ch_pos in zip(eeg_picks, eeg_pos):
        circle = mpl.patches.Circle(ch_pos, radius=radius, alpha=alpha, facecolor='darkgrey', edgecolor='dimgrey')
        ax.add_patch(circle)
        proj_sphere_radius = sphere[3]
        ch_pos_x, ch_pos_y = ch_pos
        ch_pos_x / 0.095 * proj_sphere_radius
        ch_pos_y / 0.095 * proj_sphere_radius
        ax.text(ch_pos_x, ch_pos_y, f"{ch_name}", horizontalalignment='center', verticalalignment='center', color='black', fontsize=plt.rcParams['font.size']*2/3)


def plot_csp_pattern(csp, info: mne.Info, component_idx=0, ax=None, show_cbar=True):
    if ax is None:
        ax = plt.gca()
    pattern = csp.A[:, component_idx]
    eigenvalue = csp.d[component_idx]
    with sns.axes_style("white"):
        im, cm = mne.viz.plot_topomap(pattern, info, axes=ax, show=False)
    ax.set_title(f"Component {component_idx}, EV={eigenvalue:.2}")

    if show_cbar:
        clb = ax.figure.colorbar(im, ax=ax)

    return ax, im, cm


def mark_range_y(x1, x2, text='', ax=None, y_text=0.95, fontdict={}, text_kwargs={}, **kwargs):
    # kwargs, e.g., color='grey', alpha=0.2, zorder=0,
    if ax is None:
        ax = plt.gca()
    ax.fill_betweenx(y=[0, 1], x1=x1, x2=x2, transform=ax.get_xaxis_transform(), **kwargs)
    if text is not None and not text == '':
        ax.text(x=(x1 + x2)/2, y=y_text, s=text, transform=ax.get_xaxis_transform(), horizontalalignment='center', fontdict=fontdict, **text_kwargs)