"""Plotting functions."""

# -- Set default matplotlib configuration
from importlib.resources import files

import matplotlib
import matplotlib.pyplot as plt

matplotlibrc_path = files("flyvis.analysis.visualization").joinpath("matplotlibrc")
matplotlib.rc_file(matplotlibrc_path)

plt.set_loglevel(level="warning")
del matplotlib, plt
