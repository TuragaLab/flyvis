"""Animations of traces."""
import numpy as np
from matplotlib import colormaps as cm

from flyvision import utils
from flyvision.plots import plots
from flyvision.plots import plt_utils
from flyvision.animations.animations import Animation


class Trace(Animation):
    """Animates a trace.

    Args:
        trace: trace of shape (n_samples, n_frames).
        dt (float): time step in seconds for accurate time axis. Defaults to 1.
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.
        color (str or array): optional color of the trace.
        title (str): optional title of the animation.
        batch_sample (int): batch sample to start from. Defaults to 0.
        dynamic_ax_lims (bool): whether the ax limits of the trace are animated.
            Defaults to True.
        ylims (List[tuple]): static y-limits for the trace for each sample.
        ylabel (str): optional y-label of the trace.
        contour (array): optional background contour for trace in x direction.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0.1, 0.95), i.e. top-left corner.
        fontsize (float): fontsize. Defaults to 5.
        figsize (list): figure size. Defaults to [2, 2].
    """

    def __init__(
        self,
        trace,
        dt=1,
        fig=None,
        ax=None,
        update=False,
        color=None,
        title="",
        batch_sample=0,
        dynamic_ax_lims=True,
        ylims=None,
        ylabel="",
        contour=None,
        label="Sample: {}\nFrame: {}",
        labelxy=(0.1, 0.95),
        fontsize=5,
        figsize=(2, 2),
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
        path = None
        self.fontsize = fontsize
        self._initial_frame = 0
        self.dynamic_ax_lims = dynamic_ax_lims
        self.ylabel = ylabel
        self.ylims = ylims
        self.title = title
        self.contour = contour
        if self.contour is not None:
            self.contour_lims = np.array(
                [plt_utils.get_lims(c, 0.01) for c in self.contour]
            )
        self.dt = dt
        self.figsize = figsize
        super().__init__(path, self.fig)

    def init(self, frame=0):
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

    def animate(self, frame):
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

    def _plot_contour(self):
        if self.contour is None:
            return

        contour = self.contour[self.batch_sample]

        # to remove all previously drawn contours to not stack them
        # this somehow requires multiple calls of remove sometimes
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
        trace: trace of shape (n_samples, n_frames, #traces)
        dt (float): time step in seconds. Defaults to 1.
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        update (bool): whether to update the figure after each frame.
            Defaults to False.
        legend (List[str]): legends of the traces.
        colors (List[str or array]): optional colors of the traces.
        title (str): optional title of the animation.
        batch_sample (int): batch sample to start from. Defaults to 0.
        dynamic_ax_lims (bool): whether the ax limits of the trace are animated.
            Defaults to True.
        ylims (List[tuple]): static y-limits for the trace for each sample.
        ylabel (str): optional y-label of the trace.
        contour (array): optional background contour for trace in x direction.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0.1, 0.95), i.e. top-left corner.
        fontsize (float): fontsize. Defaults to 5.
        path (Path): path object to save animation to.
    """

    def __init__(
        self,
        trace,
        dt=1,
        fig=None,
        ax=None,
        update=False,
        legend=None,
        colors=None,
        title="",
        batch_sample=0,
        dynamic_ax_lims=True,
        ylims=None,
        ylabel="",
        contour=None,
        label="Sample: {}\nFrame: {}",
        labelxy=(0.1, 0.95),
        fontsize=5,
        path=None,
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

    def init(self, frame=0):
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

    def animate(self, frame):
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
