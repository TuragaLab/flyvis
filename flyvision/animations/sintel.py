"""Animations of the Sintel data."""
from matplotlib import colormaps

from flyvision import utils
from flyvision.plots import plt_utils, figsize_utils
from flyvision.animations.animations import AnimationCollector
from flyvision.animations.hexscatter import HexScatter
from flyvision.animations.hexflow import HexFlow


class SintelSample(AnimationCollector):
    """Sintel-specific animation of input, target, and groundtruth data.

    Args:
        lum (array): input of shape (#samples, #frames, #hexals).
        target (array): target of shape (#samples, #frames, #dims, #features).
        prediction (array): optional prediction of shape (#samples, #frames, #dims, #features).
        fig (Figure): existing Figure instance or None.
        batch_sample (int): batch sample to start from. Defaults to 0.
        target_cmap (colormap): colormap for the target (depth). Defaults to
            cm.get_cmap("binary") (inverse greyscale).
        mode (str): mode for egomotion as target. Either 'translation' or 'rotation'.
        figsize (list): size of initialied figure if no fig instance given.
            Defaults to [10, 5].
        axes (Axis): optional list of existing Axis instances or None.
        figprops (dict): kwargs for plt_utils.figure.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.
        fontsize (float): fontsize. Defaults to 10.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0, 1), i.e. top-left corner.
    """

    def __init__(
        self,
        lum,
        target,
        prediction=None,
        target_cmap=colormaps["binary_r"],
        fontsize=5,
        labelxy=(-0.1, 1),
        max_figure_height_cm=22,
        panel_height_cm=3,
        max_figure_width_cm=18,
        panel_width_cm=3.6,
    ):
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

        self.lum = lum  # lum[:, None, None]
        self.target = target  # breakpoint()
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
                title="input",
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
                    title="target",
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
                        title="prediction",
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
                    title="target",
                    edgecolor=None,
                    fontsize=fontsize,
                    cbar=True,
                    labelxy=labelxy,
                )
            )
            if prediction:
                animations.append(
                    HexScatter(
                        self.prediction,
                        fig=self.fig,
                        ax=self.axes[2],
                        cmap=target_cmap,
                        title="prediction",
                        edgecolor=None,
                        fontsize=fontsize,
                        cbar=True,
                        labelxy=labelxy,
                    )
                )
        self.animations = animations
        self.batch_sample = 0
        super().__init__(None, self.fig)
