"""Cartesian frame animation."""
from time import sleep

from flyvision.plots import plt_utils
from flyvision.animations.animations import Animation


class Imshow(Animation):
    """Animates an array of images using imshow.

    Args:
        images: (n_samples, n_frames, height, width)
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        dpi (int): dots per inch.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.

    Kwargs:
        passed to ~plt.imshow.
    """

    def __init__(
        self,
        images,
        fig=None,
        ax=None,
        update=True,
        figsize=[1, 1],
        sleep=0.01,
        **kwargs
    ):
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

    def init(self, frame=0):
        self.img = self.ax.imshow(self.images[self.batch_sample, frame], **self.kwargs)
        if self.sleep is not None:
            sleep(self.sleep)

    def animate(self, frame):
        self.img.set_data(self.images[self.batch_sample, frame])

        if self.update:
            self.update_figure()

        if self.sleep is not None:
            sleep(self.sleep)
