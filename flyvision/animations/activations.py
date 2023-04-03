"""Animations of neural activations.
"""
from typing import List

from matplotlib import colormaps as cm
import numpy as np

from flyvision import utils
from flyvision.plots import plt_utils
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
    ):
        self.stimulus = utils.tensor_utils.to_numpy(stimulus)

        # case: multiple response
        if isinstance(responses, List):
            self.responses = [utils.tensor_utils.to_numpy(r) for r in responses]
        else:
            self.responses = [utils.tensor_utils.to_numpy(responses)]

        self.update = False
        self.n_samples, self.frames = self.responses[0].shape[:2]
        self.fig, self.axes, (gw, gh) = plt_utils.get_axis_grid(
            gridheight=1,
            gridwidth=1 + len(self.responses),
            scale=figsize_scale,
            figsize=None,
            fontsize=fontsize,
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
            )
        )

        cranges = [
            np.max(np.abs(plt_utils.get_lims([r[i] for r in self.responses], 0.1)))
            for i in range(1)
        ]

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
                )
            )

        self.animations = animations
        self.batch_sample = batch_sample
        super().__init__(None, self.fig)
