"""Animations of traces."""

from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flyvis import utils

from ..visualization import plots, plt_utils
from .animations import Animation


class Trace(Animation):
    """Animates a trace.

    Args:
        trace: Trace of shape (n_samples, n_frames).
        dt: Time step in seconds for accurate time axis.
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        update: Whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
        color: Optional color of the trace.
        title: Optional title of the animation.
        batch_sample: Batch sample to start from.
        dynamic_ax_lims: Whether the ax limits of the trace are animated.
        ylims: Static y-limits for the trace for each sample.
        ylabel: Optional y-label of the trace.
        contour: Optional background contour for trace in x direction.
        label: Label of the animation. Formatted with the current sample and frame number.
        labelxy: Normalized x and y location of the label.
        fontsize: Fontsize.
        figsize: Figure size.

    Attributes:
        trace (np.ndarray): Trace data.
        n_samples (int): Number of samples.
        frames (int): Number of frames.
        fig (Optional[Figure]): Figure instance.
        ax (Optional[Axes]): Axes instance.
        update (bool): Update flag.
        color (Optional[Union[str, np.ndarray]]): Color of the trace.
        label (str): Label format string.
        labelxy (Tuple[float, float]): Label position.
        label_text: Label text object.
        batch_sample (int): Current batch sample.
        fontsize (float): Font size.
        dynamic_ax_lims (bool): Dynamic axis limits flag.
        ylabel (str): Y-axis label.
        ylims (Optional[List[Tuple[float, float]]]): Y-axis limits.
        title (str): Plot title.
        contour (Optional[np.ndarray]): Contour data.
        contour_lims (Optional[np.ndarray]): Contour limits.
        dt (float): Time step.
        figsize (Tuple[float, float]): Figure size.
    """

    def __init__(
        self,
        trace: np.ndarray,
        dt: float = 1,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        update: bool = False,
        color: Optional[Union[str, np.ndarray]] = None,
        title: str = "",
        batch_sample: int = 0,
        dynamic_ax_lims: bool = True,
        ylims: Optional[List[Tuple[float, float]]] = None,
        ylabel: str = "",
        contour: Optional[np.ndarray] = None,
        label: str = "Sample: {}\nFrame: {}",
        labelxy: Tuple[float, float] = (0.1, 0.95),
        fontsize: float = 5,
        figsize: Tuple[float, float] = (2, 2),
    ):
        self.trace = utils.tensor_utils.to_numpy(trace)
        self.n_samples, self.frames = self.trace.shape
        self.fig = fig
        self.ax = ax
        self.update = update
        self.color = color
        self.label = label
        self.labelxy = labelxy
        self.label_text = None
        self.batch_sample = batch_sample
        self.fontsize = fontsize
        self._initial_frame = 0
        self.dynamic_ax_lims = dynamic_ax_lims
        self.ylabel = ylabel
        self.ylims = ylims
        self.title = title
        self.contour = contour
        if self.contour is not None:
            self.contour_lims = np.array([
                plt_utils.get_lims(c, 0.01) for c in self.contour
            ])
        self.dt = dt
        self.figsize = figsize
        super().__init__(None, self.fig)

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Starting frame number.
        """
        if frame < 0:
            frame += self.frames
        trace = self.trace[self.batch_sample, self._initial_frame : frame + 1]
        x = np.arange(frame + 1) * self.dt
        self.fig, self.ax, _, self.label_text = plots.traces(
            trace,
            x=x,
            contour=None,
            smooth=None,
            fig=self.fig,
            ax=self.ax,
            label=self.label,
            color=self.color,
            labelxy=self.labelxy,
            xlabel="time in s",
            ylabel=self.ylabel,
            fontsize=self.fontsize,
            title=self.title,
            figsize=self.figsize,
        )

        self._plot_contour()

        if self.dynamic_ax_lims:
            if self.ylims is not None:
                ymin, ymax = self.ylims[self.batch_sample]
            else:
                ymin, ymax = plt_utils.get_lims(trace, 0.1)
            xmin, xmax = plt_utils.get_lims(x, 0.1)
            self.ax.axis([xmin, xmax, ymin, ymax])

        self._sample = self.batch_sample

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Current frame number.
        """
        if frame < 0:
            frame += self.frames
        trace = self.trace[self.batch_sample, self._initial_frame : frame]
        x = np.arange(self._initial_frame, frame) * self.dt
        self.ax.lines[0].set_data(x, trace)

        if self.batch_sample != self._sample:
            self._plot_contour()

        if self.dynamic_ax_lims:
            if self.ylims is not None:
                ymin, ymax = self.ylims[self.batch_sample]
            else:
                ymin, ymax = plt_utils.get_lims(trace, 0.1)
            xmin, xmax = plt_utils.get_lims(x, 0.1)
            self.ax.axis([xmin, xmax, ymin, ymax])

        if self.label:
            self.label_text.set_text(self.label.format(self.batch_sample))

        if self.update:
            self.update_figure()

        self._sample = self.batch_sample

    def _plot_contour(self) -> None:
        """Plot the contour if available."""
        if self.contour is None:
            return

        contour = self.contour[self.batch_sample]

        while self.ax.collections:
            for c in self.ax.collections:
                c.remove()

        x = np.arange(len(contour)) * self.dt
        _y = np.linspace(-2000, 2000, 100)
        Z = np.tile(contour, (len(_y), 1))

        self.ax.contourf(
            x,
            _y,
            Z,
            cmap=cm.get_cmap("binary_r"),
            levels=20,
            zorder=-100,
            alpha=0.2,
            vmin=self.contour_lims[self.batch_sample, 0],
            vmax=self.contour_lims[self.batch_sample, 1],
        )


class MultiTrace(Animation):
    """Animates multiple traces in single plot.

    Args:
        trace: Trace of shape (n_samples, n_frames, n_traces).
        dt: Time step in seconds.
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        update: Whether to update the figure after each frame.
        legend: Legends of the traces.
        colors: Optional colors of the traces.
        title: Optional title of the animation.
        batch_sample: Batch sample to start from.
        dynamic_ax_lims: Whether the ax limits of the trace are animated.
        ylims: Static y-limits for the trace for each sample.
        ylabel: Optional y-label of the trace.
        contour: Optional background contour for trace in x direction.
        label: Label of the animation. Formatted with the current sample and frame number.
        labelxy: Normalized x and y location of the label.
        fontsize: Fontsize.
        path: Path object to save animation to.

    Attributes:
        trace (np.ndarray): Trace data.
        n_samples (int): Number of samples.
        frames (int): Number of frames.
        n_trace (int): Number of traces.
        fig (Optional[Figure]): Figure instance.
        ax (Optional[Axes]): Axes instance.
        update (bool): Update flag.
        colors (Optional[List[Union[str, np.ndarray]]]): Colors of the traces.
        label (str): Label format string.
        labelxy (Tuple[float, float]): Label position.
        label_text: Label text object.
        legend (Optional[List[str]]): Legend labels.
        batch_sample (int): Current batch sample.
        fontsize (float): Font size.
        dynamic_ax_lims (bool): Dynamic axis limits flag.
        ylabel (str): Y-axis label.
        ylims (Optional[List[Tuple[float, float]]]): Y-axis limits.
        title (str): Plot title.
        contour (Optional[np.ndarray]): Contour data.
        dt (float): Time step.
    """

    def __init__(
        self,
        trace: np.ndarray,
        dt: float = 1,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        update: bool = False,
        legend: Optional[List[str]] = None,
        colors: Optional[List[Union[str, np.ndarray]]] = None,
        title: str = "",
        batch_sample: int = 0,
        dynamic_ax_lims: bool = True,
        ylims: Optional[List[Tuple[float, float]]] = None,
        ylabel: str = "",
        contour: Optional[np.ndarray] = None,
        label: str = "Sample: {}\nFrame: {}",
        labelxy: Tuple[float, float] = (0.1, 0.95),
        fontsize: float = 5,
        path: Optional[str] = None,
    ):
        self.trace = utils.tensor_utils.to_numpy(trace)
        self.n_samples, self.frames, self.n_trace = self.trace.shape
        self.fig = fig
        self.ax = ax
        self.update = update
        self.colors = colors
        self.label = label
        self.labelxy = labelxy
        self.label_text = None
        self.legend = legend
        self.batch_sample = batch_sample
        self.fontsize = fontsize
        self._initial_frame = 0
        self.dynamic_ax_lims = dynamic_ax_lims
        self.ylabel = ylabel
        self.ylims = ylims
        self.title = title
        self.contour = contour
        self.dt = dt
        super().__init__(path, self.fig)

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Starting frame number.
        """
        self._initial_frame = frame
        trace = self.trace[self.batch_sample, frame]
        x = np.arange(frame + 1) * self.dt
        self.fig, self.ax, _, self.label_text = plots.traces(
            trace[:, None],
            x=x,
            contour=self.contour,
            smooth=None,
            fig=self.fig,
            ax=self.ax,
            label=self.label,
            color=self.colors,
            labelxy=self.labelxy,
            xlabel="time (s)",
            ylabel=self.ylabel,
            fontsize=self.fontsize,
            title=self.title,
            legend=self.legend,
        )
        if not self.dynamic_ax_lims:
            if self.ylims is not None:
                ymin, ymax = self.ylims[self.batch_sample]
            else:
                ymin, ymax = plt_utils.get_lims(self.trace, 0.1)
            xmin, xmax = plt_utils.get_lims(
                np.arange(self._initial_frame, self.trace.shape[1]), 0.1
            )
            self.ax.axis([xmin, xmax, ymin, ymax])

        self._sample = self.batch_sample

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Current frame number.
        """
        trace = self.trace[self.batch_sample, self._initial_frame : frame]
        x = np.arange(self._initial_frame, frame) * self.dt

        for n in range(self.n_trace):
            self.ax.lines[n].set_data(x, trace[:, n])

        contour = self.contour[self.batch_sample] if self.contour is not None else None

        if self.batch_sample != self._sample and contour is not None:
            self.ax.collections = []
            _x = np.arange(len(contour))
            _y = np.linspace(-2000, 2000, 100)
            Z = np.tile(contour, (len(_y), 1))
            self.ax.contourf(
                _x,
                _y,
                Z,
                cmap=cm.get_cmap("bone"),
                levels=2,
                alpha=0.3,
                vmin=0,
                vmax=1,
            )

        if self.dynamic_ax_lims:
            if self.ylims is not None:
                ymin, ymax = self.ylims[self.batch_sample]
            else:
                ymin, ymax = plt_utils.get_lims(trace, 0.1)
            xmin, xmax = plt_utils.get_lims(x, 0.1)
            self.ax.axis([xmin, xmax, ymin, ymax])

        if self.label:
            self.label_text.set_text(self.label.format(self.batch_sample, frame))

        if self.update:
            self.update_figure()

        self._sample = self.batch_sample
