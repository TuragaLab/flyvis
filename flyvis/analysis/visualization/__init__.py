"""Plotting functions."""

# -- Set default matplotlib figure dpi in notebooks and font

from pathlib import Path
from flyvis import repo_dir
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc_file(repo_dir / "matplotlibrc")

plt.set_loglevel(level="warning")
del matplotlib, plt, repo_dir, Path
