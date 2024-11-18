from typing import List, Tuple, Union

import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    hex2color,
    to_rgba,
)

from flyvis.analysis.visualization.plt_utils import color_labels
from flyvis.utils.groundtruth_utils import polarity

ON = "#af0019"  # red
OFF = "#00b2b2"  # blue

ON_FR = "#c1b933"  # yellow
OFF_FR = "#b140cc"  # violett

# ON_OSI = "#feb24c"  # light orange
# OFF_OSI = "#a1d99b"  # light green
ON_OSI = "#B74F0D"  # reddish like ON
OFF_OSI = "#0089B9"  # blueish like OFF

PD = "#1b9e77"
ND = "#e7298a"

INH = "#0000ff"
EXC = "#FF0000"


def is_hex(color: Union[str, Tuple[float, float, float]]) -> bool:
    """
    Check if the given color is in hexadecimal format.

    Args:
        color: The color to check.

    Returns:
        True if the color is in hexadecimal format, False otherwise.
    """
    return "#" in color


def is_integer_rgb(color: Union[Tuple[float, float, float], List[float]]) -> bool:
    """
    Check if the given color is in integer RGB format (0-255).

    Args:
        color: The color to check.

    Returns:
        True if the color is in integer RGB format, False otherwise.
    """
    try:
        return any([c > 1 for c in color])
    except TypeError:
        return False


def single_color_cmap(color: Union[str, Tuple[float, float, float]]) -> ListedColormap:
    """
    Create a single color colormap.

    Args:
        color: The color to use for the colormap.

    Returns:
        A ListedColormap object with the specified color.
    """
    if is_hex(color):
        color = to_rgba(color)
    elif is_integer_rgb(color):
        color = np.array(color) / 255
    return ListedColormap(color)


def color_to_cmap(
    end_color: str,
    start_color: str = "#FFFFFF",
    name: str = "custom_cmap",
    N: int = 256,
) -> LinearSegmentedColormap:
    """
    Create a colormap from start and end colors.

    Args:
        end_color: The end color of the colormap.
        start_color: The start color of the colormap.
        name: The name of the colormap.
        N: The number of color segments.

    Returns:
        A LinearSegmentedColormap object.
    """
    return LinearSegmentedColormap.from_list(
        name,
        [hex2color(start_color), hex2color(end_color)],
        N=N,
    )


def get_alpha_colormap(
    saturated_color: Union[str, Tuple[float, float, float]], number_of_shades: int
) -> ListedColormap:
    """
    Create a colormap from a color and a number of shades.

    Args:
        saturated_color: The base color for the colormap.
        number_of_shades: The number of shades to create.

    Returns:
        A ListedColormap object with varying alpha values.
    """
    if is_hex(saturated_color):
        rgba = [*hex2color(saturated_color)[:3], 0]
    elif is_integer_rgb(saturated_color):
        rgba = [*list(np.array(saturated_color) / 255.0), 0]

    colors = []
    alphas = np.linspace(1 / number_of_shades, 1, number_of_shades)[::-1]
    for alpha in alphas:
        rgba[-1] = alpha
        colors.append(rgba.copy())

    return ListedColormap(colors)


def adapt_color_alpha(
    color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]],
    alpha: float,
) -> Tuple[float, float, float, float]:
    """
    Transform a color specification to RGBA and adapt the alpha value.

    Args:
        color: Color specification in various formats: hex string, RGB tuple, or
            RGBA tuple.
        alpha: New alpha value to be applied.

    Returns:
        The adapted color in RGBA format.
    """
    color_rgb = to_rgba(color)
    r, g, b, _ = color_rgb
    return r, g, b, alpha


def flash_response_color_labels(ax):
    """
    Apply color labels for ON and OFF flash responses.

    Args:
        ax: The matplotlib axis to apply the labels to.

    Returns:
        The modified matplotlib axis.
    """
    on = [key for key, value in polarity.items() if value == 1]
    off = [key for key, value in polarity.items() if value == -1]
    color_labels(on, ON_FR, ax)
    color_labels(off, OFF_FR, ax)
    return ax


single_blue_cmap = single_color_cmap("#2c7fb8")
single_orange_cmap = single_color_cmap("#d95f0e")

cell_type_colors = {
    "R1": "#000000",
    "R2": "#000000",
    "R3": "#000000",
    "R4": "#000000",
    "R5": "#000000",
    "R6": "#000000",
    "R7": "#000000",
    "R8": "#000000",
    "L1": "#df6e26",
    "L2": "#df6e26",
    "L3": "#df6e26",
    "L4": "#df6e26",
    "L5": "#df6e26",
    "Lawf1": "#df6e26",
    "Lawf2": "#df6e26",
    "Am": "#000000",
    "C2": "#000000",
    "C3": "#000000",
    "CT1(Lo1)": "#000000",
    "CT1(M10)": "#000000",
    "Mi1": "#5db5ac",
    "Mi2": "#5db5ac",
    "Mi3": "#5db5ac",
    "Mi4": "#5db5ac",
    "Mi9": "#5db5ac",
    "Mi10": "#5db5ac",
    "Mi11": "#5db5ac",
    "Mi12": "#5db5ac",
    "Mi13": "#5db5ac",
    "Mi14": "#5db5ac",
    "Mi15": "#5db5ac",
    "T4a": "#ed3533",
    "T4b": "#ed3533",
    "T4c": "#ed3533",
    "T4d": "#ed3533",
    "T1": "#ed3533",
    "T2": "#ed3533",
    "T2a": "#ed3533",
    "T3": "#ed3533",
    "T5a": "#ed3533",
    "T5b": "#ed3533",
    "T5c": "#ed3533",
    "T5d": "#ed3533",
    "Tm1": "#a454a0",
    "Tm2": "#a454a0",
    "Tm3": "#a454a0",
    "Tm4": "#a454a0",
    "Tm5Y": "#a454a0",
    "Tm5a": "#a454a0",
    "Tm5b": "#a454a0",
    "Tm5c": "#a454a0",
    "Tm9": "#a454a0",
    "Tm16": "#a454a0",
    "Tm20": "#a454a0",
    "Tm28": "#a454a0",
    "Tm30": "#a454a0",
    "TmY3": "#a454a0",
    "TmY4": "#a454a0",
    "TmY5a": "#a454a0",
    "TmY9": "#a454a0",
    "TmY10": "#a454a0",
    "TmY13": "#a454a0",
    "TmY14": "#a454a0",
    "TmY15": "#a454a0",
    "TmY18": "#a454a0",
}


def truncate_colormap(
    cmap: Union[LinearSegmentedColormap, ListedColormap],
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 100,
) -> LinearSegmentedColormap:
    """
    Truncate a colormap to a specific range.

    Args:
        cmap: The colormap to truncate.
        minval: The minimum value of the new range.
        maxval: The maximum value of the new range.
        n: The number of color segments in the new colormap.

    Returns:
        A new LinearSegmentedColormap with the truncated range.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, max(n, 2))),
    )
    return new_cmap.resampled(max(n, 2))


class cmap_iter:
    """
    An iterator for colormap colors.

    Attributes:
        i: The current index.
        cmap: The colormap to iterate over.
        stop: The number of colors in the colormap.
    """

    def __init__(self, cmap: Union[LinearSegmentedColormap, ListedColormap]):
        """
        Initialize the cmap_iter.

        Args:
            cmap: The colormap to iterate over.
        """
        self.i: int = 0
        self.cmap = cmap
        self.stop: int = cmap.N

    def __next__(self) -> Tuple[float, float, float, float]:
        """
        Get the next color from the colormap.

        Returns:
            The next color as an RGBA tuple.
        """
        if self.i < self.stop:
            self.i += 1
            return self.cmap(self.i - 1)

    def _repr_html_(self) -> str:
        """
        Return the HTML representation of the colormap.

        Returns:
            The HTML representation of the colormap.
        """
        return self.cmap._repr_html_()
