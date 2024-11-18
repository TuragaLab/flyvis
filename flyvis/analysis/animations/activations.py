"""Animations of neural activations."""

from typing import List, Optional, Union

import numpy as np
from matplotlib import colormaps as cm

from flyvis import utils

from ..visualization import figsize_utils
from .animations import AnimationCollector
from .hexscatter import HexScatter

__all__ = ["StimulusResponse"]


class StimulusResponse(AnimationCollector):
    """Hex-scatter animations for input and responses.

    Args:
        stimulus: Hexagonal input.
        responses: Hexagonal activation of particular neuron type.
        batch_sample: Batch sample to start from.
        figsize_scale: Scale factor for figure size.
        fontsize: Font size for the plot.
        u: List of u coordinates of neurons to plot.
        v: List of v coordinates of neurons to plot.
        max_figure_height_cm: Maximum figure height in centimeters.
        panel_height_cm: Height of each panel in centimeters.
        max_figure_width_cm: Maximum figure width in centimeters.
        panel_width_cm: Width of each panel in centimeters.

    Attributes:
        stimulus (np.ndarray): Numpy array of stimulus data.
        responses (List[np.ndarray]): List of numpy arrays of response data.
        update (bool): Flag to indicate if update is needed.
        n_samples (int): Number of samples.
        frames (int): Number of frames.
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        axes (List[matplotlib.axes.Axes]): List of matplotlib axes objects.
        animations (List[HexScatter]): List of HexScatter animation objects.
        batch_sample (int): Batch sample index.

    Note:
        If u and v are not specified, all neurons are plotted.
    """

    def __init__(
        self,
        stimulus: np.ndarray,
        responses: Union[np.ndarray, List[np.ndarray]],
        batch_sample: int = 0,
        figsize_scale: float = 1,
        fontsize: int = 5,
        u: Optional[List[int]] = None,
        v: Optional[List[int]] = None,
        max_figure_height_cm: float = 22,
        panel_height_cm: float = 3,
        max_figure_width_cm: float = 18,
        panel_width_cm: float = 3.6,
    ) -> None:
        self.stimulus = utils.tensor_utils.to_numpy(stimulus)

        # case: multiple response
        if isinstance(responses, List):
            self.responses = [utils.tensor_utils.to_numpy(r) for r in responses]
        else:
            self.responses = [utils.tensor_utils.to_numpy(responses)]

        self.update = False
        self.n_samples, self.frames = self.responses[0].shape[:2]

        figsize = figsize_utils.figsize_from_n_items(
            1 + len(self.responses),
            max_figure_height_cm=max_figure_height_cm,
            panel_height_cm=panel_height_cm,
            max_figure_width_cm=max_figure_width_cm,
            panel_width_cm=panel_width_cm,
        )
        self.fig, self.axes = figsize.axis_grid(
            unmask_n=1 + len(self.responses), hspace=0.0, wspace=0, fontsize=fontsize
        )

        stimulus_samples = self.stimulus.shape[0]
        if stimulus_samples != self.n_samples and stimulus_samples == 1:
            self.stimulus = np.repeat(self.stimulus, self.n_samples, axis=0)

        animations = []

        animations.append(
            HexScatter(
                self.stimulus,
                fig=self.fig,
                ax=self.axes[0],
                title="stimulus",
                labelxy=(-0.1, 1),
                update=False,
                title_y=0.9,
            )
        )

        cranges = np.max(np.abs(self.responses), axis=(0, 2, 3, 4))

        for i, responses in enumerate(self.responses, 1):
            animations.append(
                HexScatter(
                    responses,
                    fig=self.fig,
                    ax=self.axes[i],
                    cmap=cm.get_cmap("seismic"),
                    title=f"response {i}" if len(self.responses) > 1 else "response",
                    label="",
                    midpoint=0,
                    update=False,
                    u=u,
                    v=v,
                    cranges=cranges,
                    cbar=i == len(self.responses),
                    title_y=0.9,
                )
            )

        self.animations = animations
        self.batch_sample = batch_sample
        super().__init__(None, self.fig)
