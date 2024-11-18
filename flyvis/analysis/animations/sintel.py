"""Animations of the Sintel data."""

from typing import Optional, Tuple

import numpy as np
from matplotlib import colormaps

from flyvis import utils

from ..visualization import figsize_utils
from .animations import AnimationCollector
from .hexflow import HexFlow
from .hexscatter import HexScatter


class SintelSample(AnimationCollector):
    """Sintel-specific animation of input, target, and groundtruth data.

    Args:
        lum: Input of shape (n_samples, n_frames, n_hexals).
        target: Target of shape (n_samples, n_frames, n_dims, n_features).
        prediction: Optional prediction of shape
            (n_samples, n_frames, n_dims, n_features).
        target_cmap: Colormap for the target (depth).
        fontsize: Font size for labels and titles.
        labelxy: Normalized x and y location of the label.
        max_figure_height_cm: Maximum figure height in centimeters.
        panel_height_cm: Height of each panel in centimeters.
        max_figure_width_cm: Maximum figure width in centimeters.
        panel_width_cm: Width of each panel in centimeters.
        title1: Title for the input panel.
        title2: Title for the target panel.
        title3: Title for the prediction panel.

    Attributes:
        fig (Figure): Matplotlib figure instance.
        axes (List[Axes]): List of matplotlib axes instances.
        lum (np.ndarray): Input data.
        target (np.ndarray): Target data.
        prediction (Optional[np.ndarray]): Prediction data.
        extent (Tuple[float, float, float, float]): Extent of the hexagonal grid.
        n_samples (int): Number of samples.
        frames (int): Number of frames.
        update (bool): Whether to update the canvas after an animation step.
        labelxy (Tuple[float, float]): Normalized x and y location of the label.
        animations (List): List of animation objects.
        batch_sample (int): Batch sample to start from.
    """

    def __init__(
        self,
        lum: np.ndarray,
        target: np.ndarray,
        prediction: Optional[np.ndarray] = None,
        target_cmap: str = colormaps["binary_r"],
        fontsize: float = 5,
        labelxy: Tuple[float, float] = (-0.1, 1),
        max_figure_height_cm: float = 22,
        panel_height_cm: float = 3,
        max_figure_width_cm: float = 18,
        panel_width_cm: float = 3.6,
        title1: str = "input",
        title2: str = "target",
        title3: str = "prediction",
    ) -> None:
        figsize = figsize_utils.figsize_from_n_items(
            2 if prediction is None else 3,
            max_figure_height_cm=max_figure_height_cm,
            panel_height_cm=panel_height_cm,
            max_figure_width_cm=max_figure_width_cm,
            panel_width_cm=panel_width_cm,
        )
        self.fig, self.axes = figsize.axis_grid(
            hspace=0.0,
            wspace=0,
            fontsize=fontsize,
            unmask_n=2 if prediction is None else 3,
        )

        self.lum = lum
        self.target = target
        self.prediction = prediction
        self.extent = utils.hex_utils.get_hextent(self.lum.shape[-1])

        self.n_samples, self.frames = self.lum.shape[0:2]
        self.update = False
        self.labelxy = labelxy

        animations = []
        animations.append(
            HexScatter(
                self.lum,
                fig=self.fig,
                ax=self.axes[0],
                title=title1,
                edgecolor=None,
                update_edge_color=True,
                fontsize=fontsize,
                cbar=True,
                labelxy=labelxy,
            )
        )

        if self.target.shape[-2] == 2:
            animations.append(
                HexFlow(
                    flow=self.target,
                    fig=self.fig,
                    ax=self.axes[1],
                    cwheel=True,
                    cwheelxy=(-0.7, 0.7),
                    title=title2,
                    label="",
                    fontsize=fontsize,
                )
            )
            if prediction is not None:
                animations.append(
                    HexFlow(
                        flow=self.prediction,
                        fig=self.fig,
                        ax=self.axes[2],
                        cwheel=True,
                        cwheelxy=(-0.7, 0.7),
                        title=title3,
                        label="",
                        fontsize=fontsize,
                    )
                )
        else:
            animations.append(
                HexScatter(
                    self.target,
                    fig=self.fig,
                    ax=self.axes[1],
                    cmap=target_cmap,
                    title=title2,
                    edgecolor=None,
                    fontsize=fontsize,
                    cbar=True,
                    labelxy=labelxy,
                )
            )
            if prediction is not None:
                animations.append(
                    HexScatter(
                        self.prediction,
                        fig=self.fig,
                        ax=self.axes[2],
                        cmap=target_cmap,
                        title=title3,
                        edgecolor=None,
                        fontsize=fontsize,
                        cbar=True,
                        labelxy=labelxy,
                    )
                )
        self.animations = animations
        self.batch_sample = 0
        super().__init__(None, self.fig)
