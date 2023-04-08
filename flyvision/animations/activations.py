"""Animations of neural activations."""
from typing import List

from matplotlib import colormaps as cm
import numpy as np

from flyvision import utils
from flyvision.plots import plt_utils, figsize_utils
from flyvision.animations.animations import AnimationCollector
from flyvision.animations.hexscatter import HexScatter


class StimulusResponse(AnimationCollector):
    """Hex-scatter animations for input and responses.

    Args:
        stimulus (tensor): hexagonal input of shape
            (n_samples, n_frames, n_input_elements).
        responses (tensor or List[tensor]): hexagonal activation of particular
            neuron type (n_samples, n_frames, n_input_elements).
        batch_sample (int): batch sample to start from. Defaults to 0.
        figsize (list): figure size in inches. Defaults to [2, 1].
        fontsize (int): fontsize. Defaults to 5.
        u (list): list of u coordinates of neurons to plot. Defaults to None.
        v (list): list of v coordinates of neurons to plot. Defaults to None.

    Note: if u and v are not specified, all neurons are plotted.
    """

    def __init__(
        self,
        stimulus,
        responses,
        batch_sample=0,
        figsize_scale=1,
        fontsize=5,
        u=None,
        v=None,
        max_figure_height_cm=22,
        panel_height_cm=3,
        max_figure_width_cm=18,
        panel_width_cm=3.6,
    ):
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
