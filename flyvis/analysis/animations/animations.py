import logging
import os
import tempfile
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Literal, Optional, Union

import ffmpeg
import matplotlib
from IPython import display

__all__ = ["Animation", "AnimationCollector", "convert"]

TESTING = os.getenv("TESTING", "False").lower() == "true"
COLAB = bool(os.getenv("COLAB_RELEASE_TAG"))
ANIMATION_DIR = os.getenv("ANIMATION_DIR", "animations")


class Animation:
    """Base class for animations.

    Subclasses must implement `init` and `animate` methods.

    Args:
        path: Path to save the animation.
        fig: Existing Figure instance or None.
        suffix: Suffix for the animation path.

    Attributes:
        fig (matplotlib.figure.Figure): Figure instance for the animation.
        update (bool): Whether to update the canvas after each animation step.
        batch_sample (int): Sample to animate.
        frames (int): Number of frames in the animation.
        n_samples (int): Number of samples in the animation.
        path (Path): Path to save the animation.
    """

    fig: Optional[matplotlib.figure.Figure] = None
    update: bool = True
    batch_sample: int = 0
    frames: int = 0
    n_samples: int = 0

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        suffix: str = "{}",
    ):
        self.path = Path(ANIMATION_DIR if path is None else path) / suffix.format(
            self.__class__.__name__
        )
        self.fig = fig

    def init(self, frame: int = 0) -> None:
        """Initialize the animation.

        Args:
            frame: Initial frame number.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def animate(self, frame: int) -> None:
        """Animate a single frame.

        Args:
            frame: Frame number to animate.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def update_figure(self, clear_output: bool = True) -> None:
        """Update the figure canvas.

        Args:
            clear_output: Whether to clear the previous output.
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if matplotlib.get_backend().lower() != "nbagg" or COLAB:
            display.display(self.fig)
            if clear_output:
                display.clear_output(wait=True)

    def animate_save(self, frame: int, dpi: int = 100) -> None:
        """Update the figure to the given frame and save it.

        Args:
            frame: Frame number to animate and save.
            dpi: Dots per inch for the saved image.
        """
        self.animate(frame)
        identifier = f"{self.batch_sample:04}_{frame:04}"
        self.fig.savefig(
            self._path / f"{identifier}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
            edgecolor="none",
        )

    def _get_indices(self, key: str, input: Union[str, Iterable]) -> list[int]:
        """Get sorted list of indices based on input.

        Args:
            key: Attribute name to get total number of indices.
            input: Input specifying which indices to return.

        Returns:
            Sorted list of indices.

        Raises:
            ValueError: If input is invalid.
        """
        total = getattr(self, key)
        _indices = set(range(total))
        if isinstance(input, str) and input == "all":
            indices = _indices
        elif isinstance(input, Iterable):
            indices = _indices.intersection(set(input))
        else:
            raise ValueError(f"Invalid input for {key}: {input}")
        return sorted(indices)

    def animate_in_notebook(
        self,
        frames: Union[str, Iterable] = "all",
        samples: Union[str, Iterable] = "all",
        repeat: int = 1,
    ) -> None:
        """Play animation within a Jupyter notebook.

        Args:
            frames: Frames to animate.
            samples: Samples to animate.
            repeat: Number of times to repeat the animation.
        """
        frames_list, samples_list = self._initialize_animation(frames, samples)

        if TESTING:
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
            self.update_figure(clear_output=False)
            return

    def _verify_backend(self) -> None:
        """Ensure the notebook backend is set correctly.

        Raises:
            RuntimeError: If matplotlib backend is not set to notebook.
        """
        backend = matplotlib.get_backend().lower()
        if backend != "nbagg" and not COLAB:
            raise RuntimeError(
                "Matplotlib backend is not set to notebook. Use '%matplotlib notebook'."
            )

    def _initialize_animation(
        self, frames: Union[str, Iterable], samples: Union[str, Iterable]
    ) -> tuple[list[int], list[int]]:
        """Initialize the animation state.

        Args:
            frames: Frames to animate.
            samples: Samples to animate.

        Returns:
            Tuple of frames list and samples list.
        """
        self.update = True
        self.init()
        frames_list = self._get_indices("frames", frames)
        samples_list = self._get_indices("n_samples", samples)
        return frames_list, samples_list

    def plot(self, sample: int, frame: int) -> None:
        """Plot a single frame for a specific sample.

        Args:
            sample: Sample number to plot.
            frame: Frame number to plot.
        """
        previous_sample = self.batch_sample
        self.update = True
        self.init()
        self.batch_sample = sample
        self.animate(frame)
        self.batch_sample = previous_sample

    def _create_temp_dir(self, path: Optional[Union[str, Path]] = None) -> None:
        """Create a temporary directory as destination for the images.

        Args:
            path: Path to create the temporary directory.
        """
        self._temp_dir = tempfile.TemporaryDirectory()
        self._path = Path(self._temp_dir.name)

    def to_vid(
        self,
        fname: str,
        frames: Union[str, Iterable] = "all",
        dpi: int = 100,
        framerate: int = 30,
        samples: Union[str, Iterable] = "all",
        delete_if_exists: bool = False,
        source_path: Optional[Union[str, Path]] = None,
        dest_path: Optional[Union[str, Path]] = None,
        type: Literal["mp4", "webm"] = "webm",
    ) -> None:
        """Animate, save individual frames, and convert to video using ffmpeg.

        Args:
            fname: Output filename.
            frames: Frames to animate.
            dpi: Dots per inch for saved images.
            framerate: Frame rate of the output video.
            samples: Samples to animate.
            delete_if_exists: Whether to delete existing output file.
            source_path: Source path for temporary files.
            dest_path: Destination path for the output video.
            type: Output video type.
        """
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
            logging.error("Error during animation: %s", e)
            raise

        self.convert(
            fname,
            delete_if_exists=delete_if_exists,
            framerate=framerate,
            source_path=source_path,
            dest_path=dest_path,
            type=type,
        )

        self._temp_dir.cleanup()

    def convert(
        self,
        fname: str,
        delete_if_exists: bool = False,
        framerate: int = 30,
        source_path: Optional[Union[str, Path]] = None,
        dest_path: Optional[Union[str, Path]] = None,
        type: Literal["mp4", "webm"] = "mp4",
    ) -> None:
        """Convert PNG files in the animations directory to video.

        Args:
            fname: Output filename.
            delete_if_exists: Whether to delete existing output file.
            framerate: Frame rate of the output video.
            source_path: Source path for input PNG files.
            dest_path: Destination path for the output video.
            type: Output video type.
        """
        dest_path = Path(dest_path or self.path)
        dest_path.mkdir(parents=True, exist_ok=True)
        convert(
            source_path or self._path,
            dest_path / f"{fname}.{type}",
            framerate,
            delete_if_exists,
            type=type,
        )


def convert(
    directory: Union[str, Path],
    dest: Union[str, Path],
    framerate: int,
    delete_if_exists: bool,
    type: Literal["mp4", "webm"] = "mp4",
) -> None:
    """Convert PNG files in directory to MP4 or WebM.

    Args:
        directory: Source directory containing PNG files.
        dest: Destination path for the output video.
        framerate: Frame rate of the output video.
        delete_if_exists: Whether to delete existing output file.
        type: Output video type.

    Raises:
        ValueError: If unsupported video type is specified.
        FileExistsError: If output file exists and delete_if_exists is False.
    """
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
            ffmpeg.input(f"{directory}/*_*.png", pattern_type="glob", framerate=framerate)
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
            logging.warning("Check ffmpeg installation: %s", e)
            return
        else:
            raise
    except ffmpeg.Error as e:
        logging.error("ffmpeg error: %s", e.stderr.decode("utf8"))
        raise e

    logging.info("Created %s", video)


class AnimationCollector(Animation):
    """Collects Animations and updates all axes at once.

    Subclasses must populate the `animations` attribute with Animation objects
    and adhere to the Animation interface.

    Attributes:
        animations (list[Animation]): List of Animation objects to collect.
    """

    animations: list[Animation] = []

    def init(self, frame: int = 0) -> None:
        """Initialize all collected animations.

        Args:
            frame: Initial frame number.
        """
        for animation in self.animations:
            animation.init(frame)
            animation.update = False

    def animate(self, frame: int) -> None:
        """Animate all collected animations for a single frame.

        Args:
            frame: Frame number to animate.
        """
        for animation in self.animations:
            animation.animate(frame)
        if self.update:
            self.update_figure()

    def __setattr__(self, key: str, val: Any) -> None:
        """Set attributes for all Animation objects at once.

        Args:
            key: Attribute name to set.
            val: Value to set for the attribute.
        """
        if key == "batch_sample" and hasattr(self, "animations"):
            for animation in self.animations:
                setattr(animation, key, val)
        super().__setattr__(key, val)
