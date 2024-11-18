"""HexScatter animation."""

from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from flyvis import utils

from ..visualization import plots, plt_utils
from .animations import Animation

__all__ = ["HexScatter"]


class HexScatter(Animation):
    """Regular hex-scatter animation.

    For hexals not on a regular hex grid, use the function pad_to_regular_hex.

    Args:
        hexarray: Shape (n_samples, n_frames, 1, n_input_elements).
        u: List of u coordinates of elements to plot.
        v: List of v coordinates of elements to plot.
        cranges: Color minimal and maximal abs value (n_samples).
        vmin: Color minimal value.
        vmax: Color maximal value.
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        batch_sample: Batch sample to start from.
        cmap: Colormap for the hex-scatter.
        edgecolor: Edgecolor for the hexals. None for no edge.
        update_edge_color: Whether to update the edgecolor after an animation step.
        update: Whether to update the canvas after an animation step.
            Must be False if this animation is composed with others using
            AnimationCollector.
        label: Label of the animation. Formatted with the current sample and
            frame number per frame.
        labelxy: Location of the label.
        fontsize: Fontsize.
        cbar: Display colorbar.
        background_color: Background color.
        midpoint: Midpoint for diverging colormaps.

    Attributes:
        fig (Figure): Matplotlib figure instance.
        ax (Axes): Matplotlib axes instance.
        background_color (str): Background color.
        hexarray (np.ndarray): Hex array data.
        cranges (Optional[List[float]]): Color ranges.
        vmin (Optional[float]): Minimum value for color mapping.
        vmax (Optional[float]): Maximum value for color mapping.
        midpoint (Optional[float]): Midpoint for diverging colormaps.
        kwargs (dict): Additional keyword arguments.
        batch_sample (int): Batch sample index.
        cmap: Colormap for the hex-scatter.
        update (bool): Whether to update the canvas after an animation step.
        label (str): Label template for the animation.
        labelxy (Tuple[float, float]): Label position.
        label_text: Text object for the label.
        n_samples (int): Number of samples.
        frames (int): Number of frames.
        extent (int): Hex extent.
        edgecolor (Optional[str]): Edgecolor for the hexals.
        update_edge_color (bool): Whether to update the edgecolor.
        fontsize (float): Font size.
        cbar (bool): Whether to display colorbar.
        u (List[float]): U coordinates of elements to plot.
        v (List[float]): V coordinates of elements to plot.

    """

    def __init__(
        self,
        hexarray: np.ndarray,
        u: Optional[List[float]] = None,
        v: Optional[List[float]] = None,
        cranges: Optional[List[float]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        batch_sample: int = 0,
        cmap: Union[str, Colormap] = cm.get_cmap("binary_r"),
        edgecolor: Optional[str] = None,
        update_edge_color: bool = True,
        update: bool = False,
        label: str = "Sample: {}\nFrame: {}",
        labelxy: Tuple[float, float] = (0.1, 0.95),
        fontsize: float = 5,
        cbar: bool = True,
        background_color: str = "none",
        midpoint: Optional[float] = None,
        **kwargs,
    ):
        self.fig = fig
        self.ax = ax
        self.background_color = background_color
        self.hexarray = utils.tensor_utils.to_numpy(hexarray)
        self.cranges = cranges
        self.vmin = vmin
        self.vmax = vmax
        self.midpoint = midpoint
        self.kwargs = kwargs
        self.batch_sample = batch_sample
        self.cmap = cmap
        self.update = update
        self.label = label
        self.labelxy = labelxy
        self.label_text = None
        self.n_samples, self.frames = hexarray.shape[0:2]
        self.extent = utils.hex_utils.get_hextent(hexarray.shape[-1])
        self.edgecolor = edgecolor
        self.update_edge_color = update_edge_color
        self.fontsize = fontsize
        self.cbar = cbar
        if u is None or v is None:
            u, v = utils.hex_utils.get_hex_coords(self.extent)
        self.u = u
        self.v = v
        super().__init__(None, self.fig)

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Frame number to initialize.

        """
        if frame < 0:
            frame += self.frames
        u, v = utils.hex_utils.get_hex_coords(self.extent)
        _values = self.hexarray[self.batch_sample]
        _vmin = _values.min()
        _vmax = _values.max()
        values = _values[frame].squeeze()
        vmin = (
            -self.cranges[self.batch_sample]
            if self.cranges is not None
            else self.vmin
            if self.vmin is not None
            else _vmin
        )
        vmax = (
            +self.cranges[self.batch_sample]
            if self.cranges is not None
            else self.vmax
            if self.vmax is not None
            else _vmax
        )
        scalarmapper, norm = plt_utils.get_scalarmapper(
            scalarmapper=None,
            cmap=self.cmap,
            norm=None,
            vmin=vmin,
            vmax=vmax,
            midpoint=self.midpoint,
        )
        self.fig, self.ax, (self.label_text, _) = plots.hex_scatter(
            self.u,
            self.v,
            values,
            fig=self.fig,
            midpoint=None,
            scalarmapper=scalarmapper,
            norm=norm,
            ax=self.ax,
            cmap=self.cmap,
            annotate=False,
            labelxy=self.labelxy,
            label=self.label.format(self.batch_sample, frame),
            edgecolor=self.edgecolor,
            fill=False,
            cbar=False,
            fontsize=self.fontsize,
            **self.kwargs,
        )
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.patch.set_facecolor(self.background_color)
        if self.cbar:
            plt_utils.add_colorbar_to_fig(
                self.fig,
                [self.ax],
                label="",
                width=0.01,
                height=0.5,
                x_offset=-2,
                cmap=self.cmap,
                norm=norm,
                fontsize=self.fontsize - 1,
                tick_length=1,
                tick_width=0.25,
                rm_outline=True,
                n_ticks=5,
                n_decimals=0,
            )

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Frame number to animate.

        """
        if frame < 0:
            frame += self.frames
        _values = self.hexarray[self.batch_sample]
        _vmin = _values.min()
        _vmax = _values.max()
        values = _values[frame].squeeze()
        vmin = (
            -self.cranges[self.batch_sample]
            if self.cranges is not None
            else self.vmin
            if self.vmin is not None
            else _vmin
        )
        vmax = (
            +self.cranges[self.batch_sample]
            if self.cranges is not None
            else self.vmax
            if self.vmax is not None
            else _vmax
        )
        scalarmapper, norm = plt_utils.get_scalarmapper(
            scalarmapper=None,
            cmap=self.cmap,
            norm=None,
            vmin=vmin,
            vmax=vmax,
            midpoint=self.midpoint,
        )
        if self.cbar:
            for ax in self.fig.axes:
                if ax.get_label() == "cbar":
                    ax.remove()
            plt_utils.add_colorbar_to_fig(
                self.fig,
                [self.ax],
                label="",
                width=0.01,
                height=0.5,
                x_offset=-2,
                cmap=self.cmap,
                norm=norm,
                fontsize=self.fontsize - 1,
                tick_length=1,
                tick_width=0.25,
                rm_outline=True,
                n_ticks=5,
                n_decimals=0,
            )
        fcolors = scalarmapper.to_rgba(values)
        for i, fc in enumerate(fcolors):
            if self.update_edge_color:
                self.ax.patches[i].set_color(fc)
            else:
                self.ax.patches[i].set_facecolor(fc)

        if self.label:
            self.label_text.set_text(self.label.format(self.batch_sample, frame))

        if self.update:
            self.update_figure()
