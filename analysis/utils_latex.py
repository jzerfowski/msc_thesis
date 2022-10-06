#!/usr/bin/env python

"""
Utility functions for LaTeX
Provides methods to save a list of latex figures into a file such that they can be \included into a .tex file
"""

import pathlib
import string
import random

basepath_latex_src = pathlib.Path(r"D:\Thesis\src\thesis")


def write_figures_to_file(figures_list, latex_filename):
    full_str = '\n'.join([str(latex_figure) for latex_figure in figures_list])
    print(f"Writing {len(figures_list)} figures to file {basepath_latex_src / latex_filename}:")
    print(full_str)

    with open(basepath_latex_src / latex_filename, 'w') as fp:
        fp.write(full_str)

    print(f"Wrote {len(figures_list)} figures to file {basepath_latex_src / latex_filename}")


class LatexFigure:
    def __init__(self, filename, path, caption, label=None):
        self.filename = filename
        self.path = path
        self.caption = caption
        if label is None:
            label = 'fig:no_label_' + ''.join(random.choices(string.ascii_lowercase +
                                   string.digits, k=8))
        self.label = label

    def get_str(self):
        return r'''\begin{{figure}}[!ht]
    \begin{{center}}
        \subimport{{{path}}}{{{filename}}}
    \end{{center}}
    \caption{{{caption}}}\label{{{label}}}
\end{{figure}}
'''

    def __str__(self):
        return self.get_str().format(filename=self.filename, path=self.path, caption=self.caption, label=self.label)
