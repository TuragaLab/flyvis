import numpy as np
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    hex2color,
    to_rgba,
)

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


def is_hex(color):
    return "#" in color


def is_integer_rgb(color):
    try:
        return any([c > 1 for c in color])
    except TypeError:
        return False


def single_color_cmap(color):
    if is_hex(color):
        color = to_rgba(color)
    elif is_integer_rgb(color):
        color = np.array(color) / 255
    else:
        pass

    return ListedColormap(color)


def color_to_cmap(end_color, start_color="#FFFFFF", name="custom_cmap", N=256):
    return LinearSegmentedColormap.from_list(
        name,
        [hex2color(start_color), hex2color(end_color)],
        N=N,
    )


def get_alpha_colormap(saturated_color, number_of_shades):
    """To create a colormap from a color and a number of shades."""
    if is_hex(saturated_color):
        rgba = [*hex2color(saturated_color)[:3], 0]
    elif is_integer_rgb(saturated_color):
        rgba = [*list(np.array(saturated_color) / 255.0), 0]

    N = number_of_shades
    colors = []
    alphas = np.linspace(1 / N, 1, N)[::-1]
    for alpha in alphas:
        rgba[-1] = alpha
        colors.append(rgba.copy())

    return ListedColormap(colors)


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
