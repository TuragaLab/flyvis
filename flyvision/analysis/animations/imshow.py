"""Cartesian frame animation."""

from time import sleep
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..visualization import plt_utils
from .animations import Animation


class Imshow(Animation):
    """Animates an array of images using imshow.

    Args:
        images: Array of images to animate (n_samples, n_frames, height, width).
        fig: Existing Figure instance or None.
        ax: Existing Axis instance or None.
        update: Whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
        figsize: Size of the figure.
        sleep: Time to sleep between frames.
        **kwargs: Additional arguments passed to plt.imshow.

    Attributes:
        fig (plt.Figure): The figure object.
        ax (plt.Axes): The axes object.
        kwargs (dict): Additional arguments for imshow.
        update (bool): Whether to update the canvas after each step.
        n_samples (int): Number of samples in the images array.
        frames (int): Number of frames in each sample.
        images (np.ndarray): Array of images to animate.
        sleep (float): Time to sleep between frames.
        img (plt.AxesImage): The image object created by imshow.

    Note:
        The `images` array should have shape (n_samples, n_frames, height, width).
    """

    def __init__(
        self,
        images: np.ndarray,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        update: bool = True,
        figsize: List[int] = [1, 1],
        sleep: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fig, self.ax = plt_utils.init_plot(
            figsize=figsize,
            fig=fig,
            ax=ax,
            position=[0, 0, 1, 1],
            set_axis_off=True,
        )
        self.kwargs = kwargs
        self.update = update
        self.n_samples, self.frames = images.shape[:2]
        self.images = images
        self.sleep = sleep
        super().__init__(None, self.fig)

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: The initial frame to display.
        """
        self.img = self.ax.imshow(self.images[self.batch_sample, frame], **self.kwargs)
        if self.sleep is not None:
            sleep(self.sleep)

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: The frame number to animate.
        """
        self.img.set_data(self.images[self.batch_sample, frame])

        if self.update:
            self.update_figure()

        if self.sleep is not None:
            sleep(self.sleep)
