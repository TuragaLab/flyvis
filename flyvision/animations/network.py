import matplotlib.pyplot as plt

from flyvision.utils.activity_utils import LayerActivity
from flyvision.plots.network import WholeNetworkFigure
from flyvision.animations.animations import Animation
from flyvision.animations.hexscatter import HexScatter
from flyvision.animations.hexflow import HexFlow
from flyvision.animations.imshow import Imshow


class WholeNetworkAnimation(Animation):
    def __init__(
        self,
        connectome,
        responses,
        cartesian_input=None,
        rendered_input=None,
        predicted_flow=None,
        target_flow=None,
        batch_sample=0,
        update=False,
        color_norm_per="batch",
        label="Sample: {}\nFrame: {}",
        cmap=plt.get_cmap("binary_r"),
        labelxy=(0, 0.9),
        titlepad=1,
        fontsize=5,
        **kwargs
    ):
        self.fig_backbone = WholeNetworkFigure(
            connectome,
            video=cartesian_input is not None,
            rendering=rendered_input is not None,
            motion_decoder=predicted_flow is not None,
            decoded_motion=predicted_flow is not None,
            pixel_accurate_motion=target_flow is not None,
        )
        self.fig_backbone.init_figure()
        self.fig = self.fig_backbone.fig
        self.ax_dict = self.fig_backbone.ax_dict

        plt.rc("axes", titlepad=titlepad)
        self.batch_sample = batch_sample
        self.kwargs = kwargs
        self.update = update
        self.label = label
        self.labelxy = labelxy
        self.fontsize = fontsize
        self.cmap = cmap
        self.n_samples, self.frames = responses.shape[:2]

        self.responses = LayerActivity(responses, connectome, keepref=True)
        self.cartesian_input = cartesian_input
        self.rendered_input = rendered_input
        self.predicted_flow = predicted_flow
        self.target_flow = target_flow
        self.color_norm_per = color_norm_per
        path = None
        super().__init__(path, self.fig)

    def init(self, frame=0):
        if self.fig_backbone.video:
            ## Video
            self.cartesian_input = Imshow(
                self.cartesian_input,
                vmin=0,
                vmax=1,
                cmap=plt.cm.binary_r,
                fig=self.fig,
                ax=self.ax_dict["video"],
            )
            self.cartesian_input.init(frame)
            self.cartesian_input.update = False

        if self.fig_backbone.rendering:
            ## Rendering
            self.rendered_input = HexScatter(
                self.rendered_input,
                vmin=0,
                vmax=1,
                cmap=plt.cm.binary_r,
                fig=self.fig,
                ax=self.ax_dict["rendering"],
                edgecolor=None,
                cbar=False,
                label="",
                background_color=self.fig_backbone.facecolor,
            )
            self.rendered_input.init(frame)
            self.rendered_input.update = False

        if self.fig_backbone.decoded_motion:
            # Predicted flow
            self.predicted_flow = HexFlow(
                self.predicted_flow,
                fig=self.fig,
                ax=self.ax_dict["decoded motion"],
                label="",
                cwheel=True,
                cwheelradius=0.5,
                fontsize=5,
            )
            self.predicted_flow.init(frame)
            self.predicted_flow.update = False

        if self.fig_backbone.pixel_accurate_motion:
            # Target flow
            self.target_flow = HexFlow(
                self.target_flow,
                fig=self.fig,
                ax=self.ax_dict["pixel-accurate motion"],
                label="",
            )
            self.target_flow.init(frame)
            self.target_flow.update = False

        # responses
        self.voltage_axes = []
        for i, cell_type in enumerate(self.fig_backbone.cell_types):
            voltage = self.responses[cell_type][:, :, None]
            nodes = self.fig_backbone.nodes
            nodes = nodes[nodes.type == cell_type]
            u, v = nodes[["u", "v"]].values.T
            anim = HexScatter(
                voltage,
                u=u,
                v=v,
                label="",
                cbar=False,
                edgecolor=None,
                ax=self.ax_dict[cell_type],
                fig=self.fig,
                cmap=plt.cm.binary_r,
            )
            anim.init(frame)
            anim.update = False
            self.voltage_axes.append(anim)

    def animate(self, frame):
        if self.fig_backbone.video:
            self.cartesian_input.animate(frame)
        if self.fig_backbone.rendering:
            self.rendered_input.animate(frame)
        if self.fig_backbone.decoded_motion:
            self.predicted_flow.animate(frame)
        if self.fig_backbone.pixel_accurate_motion:
            self.target_flow.animate(frame)

        for anim in self.voltage_axes:
            anim.animate(frame)

        if self.update:
            self.update_figure()
