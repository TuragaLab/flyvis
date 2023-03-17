"""Color coded flow-field animation.
"""
import matplotlib.pyplot as plt
import numpy as np

from flyvision import utils
from flyvision.plots import plots
from flyvision.plots import plt_utils
from flyvision.animations.animations import Animation, AnimationCollector


class HexFlow(Animation):
    """Hexscatter of a color encoded flow field.

    Args:
        flow (array or tensor): optic flow of shape (#samples, n_frames, 2, n_hexals).
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        batch_sample (int): batch sample to start from. Defaults to 0.
        cmap (colormap): colormap for the hex-scatter. Defaults to
            plt_utils.cm_uniform_2d (greyscale).
        cwheel (bool): display colorwheel. Defaults to True.
        cwheelxy (tuple): colorwheel offset x and y.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0, 1), i.e. top-left corner.
        type (str): 'groundtruth' or 'prediction'.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.

    Kwargs:
        passed to ~flyvision.plots.plots.hex_flow.
    """

    def __init__(
        self,
        flow,
        fig=None,
        ax=None,
        batch_sample=0,
        cmap=plt_utils.cm_uniform_2d,
        cwheel=False,
        cwheelxy=(),
        label="Sample: {}\nFrame: {}",
        labelxy=(0, 1),
        type="groundtruth",
        update=False,
        path=None,
        figsize=[2, 2],
        fontsize=5,
        **kwargs
    ):
        self.fig = fig
        self.ax = ax
        self.batch_sample = batch_sample
        self.kwargs = kwargs
        self.update = update
        self.cmap = cmap
        self.cwheel = cwheel
        self.cwheelxy = cwheelxy

        self.label = label
        self.labelxy = labelxy
        self.label_text = None
        self.sm = None
        self.fontsize = fontsize
        self.figsize = figsize

        self.flow = utils.tensor_utils.to_numpy(flow)

        self.n_samples, self.frames = self.flow.shape[0:2]
        self.extent = utils.hex_utils.get_hextent(self.flow.shape[-1])
        super().__init__(path, self.fig)

    def init(self, frame=0):
        u, v = utils.hex_utils.get_hex_coords(self.extent)
        # breakpoint()
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
            **self.kwargs
        )

    def animate(self, frame):

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


class TwoFlows(AnimationCollector):
    def __init__(self, flow1, flow2):
        """
        (n_frames, 2, n_hexals)
        """

        self.fig, self.axes, _ = plt_utils.get_axis_grid(gridwidth=2, gridheight=1)
        self.animations = [
            HexFlow(
                flow1[
                    None,
                ],
                fig=self.fig,
                ax=self.axes[0],
            ),
            HexFlow(
                flow2[
                    None,
                ],
                fig=self.fig,
                ax=self.axes[1],
            ),
        ]
        self.frames = self.animations[0].frames
        self.n_samples = self.animations[0].n_samples
