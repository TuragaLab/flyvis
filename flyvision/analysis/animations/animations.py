import logging
import os
import tempfile
from pathlib import Path
from time import sleep
from typing import Iterable

import ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

__all__ = ["Animation", "AnimationCollector", "convert", "TestAnimation"]

TESTING = os.getenv("TESTING", "False").lower() == "true"
COLAB = bool(os.getenv("COLAB_RELEASE_TAG"))
ANIMATION_DIR = os.getenv("ANIMATION_DIR", "animations")


class Animation:
    """Base class for animations.

    Subclasses must implement `init` and `animate` methods.
    Subclasses must store the number of frames and samples in `frames` and
    `n_samples` attributes, respectively. Also, `batch_sample` must be an
    integer indicating the sample to animate. `update` must be a boolean
    indicating whether to update the canvas after each animation step.

    Args:
        path (Path): path to save the animation.
        fig (Figure): existing Figure instance or None.
        suffix (str): suffix for the animation path. Defaults to
            'animations/{}', which is formatted with the class name.
    """

    fig: matplotlib.figure.Figure = None
    update = True
    batch_sample = 0
    frames = 0
    n_samples = 0

    def __init__(self, path=None, fig=None, suffix="{}"):
        self.path = Path(ANIMATION_DIR if path is None else path) / suffix.format(
            self.__class__.__name__
        )
        self.fig = fig

    def init(self, frame=0):
        raise NotImplementedError("Subclasses should implement this method.")

    def animate(self, frame):
        raise NotImplementedError("Subclasses should implement this method.")

    def update_figure(self, clear_output=True):
        """Updates the figure canvas.

        Args:
            clear_output (bool): Whether to clear the previous output.
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if matplotlib.get_backend().lower() != "nbagg" or COLAB:
            display.display(self.fig)
            if clear_output:
                display.clear_output(wait=True)

    def animate_save(self, frame, dpi=100):
        """Updates the figure to the given frame and saves it."""
        self.animate(frame)
        identifier = f"{self.batch_sample:04}_{frame:04}"
        self.fig.savefig(
            self._path / f"{identifier}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            edgecolor="none",
        )

    def _get_indices(self, key, input):
        total = getattr(self, key)
        _indices = set(range(total))
        if input == "all":
            indices = _indices
        elif isinstance(input, Iterable):
            indices = _indices.intersection(set(input))
        else:
            raise ValueError(f"Invalid input for {key}: {input}")
        return sorted(indices)

    def animate_in_notebook(self, frames="all", samples="all", repeat=1):
        """Play animation within a Jupyter notebook.

        Ensures proper backend setup and allows for repeating the animation.
        """
        # self._verify_backend()
        frames_list, samples_list = self._initialize_animation(frames, samples)

        if TESTING:
            # Only plot one frame from one sample
            frames_list = frames_list[:1]
            samples_list = samples_list[:1]
            repeat = 1

        try:
            for _ in range(repeat):
                for sample in samples_list:
                    self.batch_sample = sample
                    for frame in frames_list:
                        self.animate(frame)
                        sleep(0.1)  # Pause between frames
        except KeyboardInterrupt:
            print("Animation interrupted. Displaying last frame.")
            # Display the last frame without clearing the output
            self.update_figure(clear_output=False)
            # Exit the function to prevent further clearing
            return

    def _verify_backend(self):
        """Ensure the notebook backend is set correctly."""
        backend = matplotlib.get_backend().lower()
        if backend != "nbagg" and not COLAB:
            raise RuntimeError(
                "Matplotlib backend is not set to notebook. Use '%matplotlib notebook'."
            )

    def _initialize_animation(self, frames, samples):
        """Initialize the animation state."""
        self.update = True
        self.init()
        frames_list = self._get_indices("frames", frames)
        samples_list = self._get_indices("n_samples", samples)
        return frames_list, samples_list

    def plot(self, sample, frame):
        previous_sample = self.batch_sample
        self.update = True
        self.init()
        self.batch_sample = sample
        self.animate(frame)
        self.batch_sample = previous_sample

    def _create_temp_dir(self, path=None):
        """Creates a temporary directory as destination for the images."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self._path = Path(self._temp_dir.name)

    def to_vid(
        self,
        fname,
        frames="all",
        dpi=100,
        framerate=30,
        samples="all",
        delete_if_exists=False,
        source_path=None,
        dest_path=None,
        type="webm",
    ):
        """Animates, saves individual frames, and converts to video using ffmpeg."""
        self._create_temp_dir(path=source_path)
        self.update = True
        self.init()
        frames_list = self._get_indices("frames", frames)
        samples_list = self._get_indices("n_samples", samples)

        try:
            for sample in samples_list:
                self.batch_sample = sample
                for frame in frames_list:
                    self.animate_save(frame, dpi=dpi)
        except Exception as e:
            logging.error(f"Error during animation: {e}")
            raise

        self.convert(
            fname,
            delete_if_exists=delete_if_exists,
            framerate=framerate,
            source_path=source_path,
            dest_path=dest_path,
            type=type,
        )

        # Cleanup temporary directory
        self._temp_dir.cleanup()

    def convert(
        self,
        fname,
        delete_if_exists=False,
        framerate=30,
        source_path=None,
        dest_path=None,
        type="mp4",
    ):
        """Converts png files in the animations dir to video."""
        dest_path = Path(dest_path or self.path)
        dest_path.mkdir(parents=True, exist_ok=True)
        convert(
            source_path or self._path,
            dest_path / f"{fname}.{type}",
            framerate,
            delete_if_exists,
            type=type,
        )


def convert(dir, dest, framerate, delete_if_exists, type="mp4"):
    """Converts png files in dir to mp4 or webm."""
    video = Path(dest)

    if type == "mp4":
        kwargs = dict(
            vcodec="libx264",
            vprofile="high",
            vlevel="4.0",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # to make sizes even
            pix_fmt="yuv420p",
            crf=18,
        )
    elif type == "webm":
        kwargs = dict(
            vcodec="libvpx-vp9",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",
            pix_fmt="yuva420p",
            crf=18,
            threads=4,
        )
    else:
        raise ValueError(f"Unsupported video type: {type}")

    if video.exists():
        if delete_if_exists:
            video.unlink()
        else:
            raise FileExistsError(f"File {video} already exists.")

    try:
        (
            ffmpeg.input(f"{dir}/*_*.png", pattern_type="glob", framerate=framerate)
            .output(str(video), **kwargs)
            .run(
                overwrite_output=True,
                quiet=True,
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except FileNotFoundError as e:
        if "ffmpeg" in str(e):
            logging.warning(f"Check ffmpeg installation: {e}")
            return
        else:
            raise
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error: {e.stderr.decode('utf8')}")
        raise e

    logging.info(f"Created {video}")


class AnimationCollector(Animation):
    """Collects Animations and updates all axes at once.

    Subclasses must populate the `animations` attribute with Animation objects
    and adhere to the Animation interface.
    """

    animations = []

    def init(self, frame=0):
        for animation in self.animations:
            animation.init(frame)
            # Disable individual updates to update all axes at once
            animation.update = False

    def animate(self, frame):
        for animation in self.animations:
            animation.animate(frame)
        if self.update:
            self.update_figure()

    def __setattr__(self, key, val):
        """Set attributes for all Animation objects at once."""
        if key == "batch_sample" and hasattr(self, "animations"):
            for animation in self.animations:
                setattr(animation, key, val)
        super().__setattr__(key, val)


class TestAnimation(Animation):
    """A test subclass of Animation that displays random data using imshow."""

    def __init__(self, path=None, fig=None, suffix="animations/{}", data=None):
        super().__init__(path=path, fig=fig, suffix=suffix)
        # Generate random data if not provided
        self.data = data if data is not None else np.random.rand(10, 10, 20)
        self.frames = self.data.shape[2]
        self.n_samples = 1
        self.batch_sample = 0

    def init(self, frame=0):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = self.fig.axes[0]
        self.im = self.ax.imshow(self.data[:, :, frame], animated=True)
        self.fig.show()

    def animate(self, frame):
        self.im.set_array(self.data[:, :, frame])
        if self.update:
            self.update_figure()


# Minimal test with random data
if __name__ == "__main__":
    import os

    # Set the TESTING environment variable to 'True'
    os.environ["TESTING"] = "True"

    # Re-evaluate TESTING after setting the environment variable
    TESTING = os.getenv("TESTING", "False").lower() == "true"

    # Ensure the matplotlib backend is set to 'notebook' for Jupyter
    # Uncomment the following line when running in a Jupyter notebook
    # %matplotlib notebook

    # Create random data for testing
    data = np.random.rand(10, 10, 20)

    # Create an instance of TestAnimation
    anim = TestAnimation(data=data)

    # Set the number of frames and samples
    anim.frames = 20  # Number of frames in the data
    anim.n_samples = 1  # Only one sample for testing

    # Run the animation in the notebook
    anim.animate_in_notebook()
