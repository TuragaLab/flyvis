"""Color coded flow-field animation."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Text

from flyvis import utils

from ..visualization import plots, plt_utils
from .animations import Animation

__all__ = ["HexFlow"]


class HexFlow(Animation):
    """Hexscatter of a color encoded flow field.

    Args:
        flow: Optic flow of shape (n_samples, n_frames, 2, n_input_elements).
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        batch_sample: Batch sample to start from.
        cmap: Colormap for the hex-scatter.
        cwheel: Display colorwheel.
        cwheelxy: Colorwheel offset x and y.
        label: Label of the animation.
        labelxy: Normalized x and y location of the label.
        update: Whether to update the canvas after an animation step.
        path: Path to save the animation to.
        figsize: Figure size.
        fontsize: Font size.
        background_color: Background color of the figure and axis.

    Attributes:
        fig (Figure): Figure instance.
        ax (Axes): Axis instance.
        background_color (str): Background color of the figure and axis.
        batch_sample (int): Batch sample to start from.
        kwargs (dict): Additional keyword arguments.
        update (bool): Whether to update the canvas after an animation step.
            Must be False if this animation is composed with others using
            AnimationCollector.
        cmap: Colormap for the hex-scatter.
        cwheel (bool): Display colorwheel.
        cwheelxy (Tuple[float, float]): Colorwheel offset x and y.
        label (str): Label of the animation.
        labelxy (Tuple[float, float]): Normalized x and y location of the label.
        label_text (Text): Text instance for the label.
        sm (ScalarMappable): ScalarMappable instance for color mapping.
        fontsize (float): Font size.
        figsize (List[float, float]): Figure size.
        flow (np.ndarray): Optic flow data.
        n_samples (int): Number of samples in the flow data.
        frames (int): Number of frames in the flow data.
        extent (Tuple[float, float, float, float]): Extent of the hexagonal grid.

    Note:
        All kwargs are passed to ~flyvis.analysis.visualization.plots.hex_flow.
    """

    def __init__(
        self,
        flow: Union[np.ndarray, "torch.Tensor"],
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        batch_sample: int = 0,
        cmap: "Colormap" = plt_utils.cm_uniform_2d,
        cwheel: bool = False,
        cwheelxy: Tuple[float, float] = (),
        label: str = "Sample: {}\nFrame: {}",
        labelxy: Tuple[float, float] = (0, 1),
        update: bool = False,
        path: Optional[str] = None,
        figsize: List[float] = [2, 2],
        fontsize: float = 5,
        background_color: Literal["none"] = "none",
        **kwargs,
    ):
        self.fig = fig
        self.ax = ax
        self.background_color = background_color
        self.batch_sample = batch_sample
        self.kwargs = kwargs
        self.update = update
        self.cmap = cmap
        self.cwheel = cwheel
        self.cwheelxy = cwheelxy

        self.label = label
        self.labelxy = labelxy
        self.label_text: Optional[Text] = None
        self.sm: Optional[ScalarMappable] = None
        self.fontsize = fontsize
        self.figsize = figsize

        self.flow = utils.tensor_utils.to_numpy(flow)

        self.n_samples, self.frames = self.flow.shape[0:2]
        self.extent = utils.hex_utils.get_hextent(self.flow.shape[-1])
        super().__init__(path, self.fig)

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Frame number to initialize with.
        """
        u, v = utils.hex_utils.get_hex_coords(self.extent)
        self.fig, self.ax, (self.label_text, self.sm, _, _) = plots.hex_flow(
            u,
            v,
            self.flow[self.batch_sample, frame],
            fig=self.fig,
            ax=self.ax,
            cwheel=self.cwheel,
            cwheelxy=self.cwheelxy,
            cmap=self.cmap,
            annotate=False,
            labelxy=self.labelxy,
            label=self.label.format(self.batch_sample, frame),
            figsize=self.figsize,
            fontsize=self.fontsize,
            **self.kwargs,
        )
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.patch.set_facecolor(self.background_color)

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Frame number to animate.
        """
        flow = self.flow[self.batch_sample, frame]

        r = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
        r /= r.max()
        theta = np.arctan2(flow[1], flow[0])
        color = self.sm.to_rgba(theta)
        color[:, -1] = r

        for i, fc in enumerate(color):
            self.ax.patches[i].set_color(fc)

        if self.label:
            self.label_text.set_text(self.label.format(self.batch_sample, frame))

        if self.update:
            self.update_figure()
