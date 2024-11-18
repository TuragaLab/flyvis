from typing import Any, Literal, Optional

import matplotlib.pyplot as plt

from flyvis.utils.activity_utils import LayerActivity

from ..visualization.network_fig import WholeNetworkFigure
from .animations import Animation
from .hexflow import HexFlow
from .hexscatter import HexScatter
from .imshow import Imshow

__all__ = ["WholeNetworkAnimation"]


class WholeNetworkAnimation(Animation):
    """
    Create an animation of the whole network activity.

    This class generates an animation that visualizes the activity of a neural network,
    including input, rendering, predicted flow, and target flow if provided.

    Attributes:
        fig_backbone (WholeNetworkFigure): The backbone figure for the animation.
        fig (matplotlib.figure.Figure): The main figure object.
        ax_dict (dict): Dictionary of axes for different components of the animation.
        batch_sample (int): The index of the batch sample to animate.
        kwargs (dict): Additional keyword arguments.
        update (bool): Whether to update the figure during animation.
        label (str): Label format string for the animation.
        labelxy (tuple[float, float]): Position of the label.
        fontsize (int): Font size for labels.
        cmap (matplotlib.colors.Colormap): Colormap for the animation.
        n_samples (int): Number of samples in the responses.
        frames (int): Number of frames in the responses.
        responses (LayerActivity): Layer activity data.
        cartesian_input (Optional[Any]): Cartesian input data.
        rendered_input (Optional[Any]): Rendered input data.
        predicted_flow (Optional[Any]): Predicted flow data.
        target_flow (Optional[Any]): Target flow data.
        color_norm_per (str): Color normalization method.
        voltage_axes (list): List of voltage axes for different cell types.
    """

    def __init__(
        self,
        connectome: Any,
        responses: Any,
        cartesian_input: Optional[Any] = None,
        rendered_input: Optional[Any] = None,
        predicted_flow: Optional[Any] = None,
        target_flow: Optional[Any] = None,
        batch_sample: int = 0,
        update: bool = False,
        color_norm_per: Literal["batch"] = "batch",
        label: str = "Sample: {}\nFrame: {}",
        cmap: Any = plt.get_cmap("binary_r"),
        labelxy: tuple[float, float] = (0, 0.9),
        titlepad: int = 1,
        fontsize: int = 5,
        **kwargs: Any,
    ) -> None:
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

    def init(self, frame: int = 0) -> None:
        """
        Initialize the animation components.

        Args:
            frame: The initial frame number.
        """
        if self.fig_backbone.video:
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
            self.target_flow = HexFlow(
                self.target_flow,
                fig=self.fig,
                ax=self.ax_dict["pixel-accurate motion"],
                label="",
            )
            self.target_flow.init(frame)
            self.target_flow.update = False

        self.voltage_axes = []
        for cell_type in self.fig_backbone.cell_types:
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

    def animate(self, frame: int) -> None:
        """
        Update the animation for a given frame.

        Args:
            frame: The current frame number.
        """
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
