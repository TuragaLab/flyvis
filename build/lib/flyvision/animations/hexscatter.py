"""HexScatter animation.
"""
import numpy as np
from matplotlib import colormaps as cm

from flyvision import utils
from flyvision.plots import plots, plt_utils
from flyvision.animations.animations import Animation, AnimationCollector


class HexScatter(Animation):
    """Regular hex-scatter animation.

    Note: for hexals not on a regular hex grid use the function
    pad_to_regular_hex.

    Args:
        hexarray (array or tensor): shape (n_samples, n_frames, 1,
            n_input_elements)
        u (list): list of u coordinates of elements to plot. Defaults to None.
        v (list): list of v coordinates of elements to plot. Defaults to None.
        cranges (List[float]): color minimal and maximal abs value (n_samples).

        vmin (float): color minimal value.
        vmax (flot): color maximal value.
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        batch_sample (int): batch sample to start from. Defaults to 0.
        cmap (colormap): colormap for the hex-scatter. Defaults to
            cm.get_cmap("binary_r") (greyscale).
        edgecolor (str): edgecolor for the hexals. Defaults to "k" displaying
            edges. None for no edge.
        update_edge_color (bool): whether to update the edgecolor after an
            animation step. Defaults to True.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others using Anim-
            ationcollector. Defaults to False.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): location of the label. Defaults to
            (0.1, 0.95), i.e. top-left corner.
        fontsize (float): fontsize. Defaults to 5.
        cbar (bool): display colorbar. Defaults to True.
        background_color (str): background color. Defaults to "none".
        midpoint (float): midpoint for diverging colormaps. Defaults to None.

    Kwargs:
        passed to ~dvs.plots.plots.hex_scatter.
    """

    def __init__(
        self,
        hexarray,
        u=None,
        v=None,
        cranges=None,
        vmin=None,
        vmax=None,
        fig=None,
        ax=None,
        batch_sample=0,
        cmap=cm.get_cmap("binary_r"),
        edgecolor=None,
        update_edge_color=True,
        update=False,
        label="Sample: {}\nFrame: {}",
        labelxy=(0.1, 0.95),
        fontsize=5,
        cbar=True,
        background_color="none",
        midpoint=None,
        **kwargs
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
        path = None
        self.fontsize = fontsize
        self.cbar = cbar
        if u is None or v is None:
            u, v = utils.hex_utils.get_hex_coords(self.extent)
        self.u = u
        self.v = v
        super().__init__(path, self.fig)

    def init(self, frame=0):
        # to allow negative indices
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
            **self.kwargs
        )
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.patch.set_facecolor(self.background_color)
        if self.cbar:
            cbar = plt_utils.add_colorbar_to_fig(
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
        # self.fig.tight_layout()

    def animate(self, frame):
        # to allow negative indices
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
            # r emove old cbars
            for ax in self.fig.axes:
                if ax.get_label() == "cbar":
                    ax.remove()
            cbar = plt_utils.add_colorbar_to_fig(
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
