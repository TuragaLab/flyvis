"""Plotting functions."""

# -- Set default matplotlib figure dpi in notebooks and font

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import fontManager

for font in (Path(__file__).parent.parent.parent / ".fonts").glob("*.ttf"):
    fontManager.addfont(font)
matplotlib.rc("figure", dpi=300)
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": "Arial", "size": 6})

plt.set_loglevel(level="warning")
del matplotlib, plt
