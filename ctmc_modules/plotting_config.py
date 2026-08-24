# -*- coding: utf-8 -*-

"""

Plotting config and specifications.

"""


#### IMPORTS ####

from pathlib import Path


#### CONFIG CONSTANTS ####

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

PLOTS_ROOT_DIR = PROJECT_ROOT_DIR / "generated_plots"
PLOTS_ROOT_DIR.mkdir(exist_ok=True)

TEXTWIDTH_IN = 6.614  # Width of LaTeX document in inches
LATEX_FONT_SIZE = 12  # Font size of LaTeX document
FONT_SIZE = LATEX_FONT_SIZE - 1
TICK_FONT_SIZE = FONT_SIZE - 1

