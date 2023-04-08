"""Base class for animations.""" ""
import itertools
import shutil
from time import sleep
from contextlib import contextmanager
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

from pathlib import Path
from typing import Iterable
import weakref
import re
import logging

import ffmpeg

from datamate import get_root_dir

import os

colab = False
if os.getenv("COLAB_RELEASE_TAG"):
    colab = True


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

    fig = None
    update = True
    batch_sample = 0
    frames = 0
    n_samples = 0
    _finalize = []

    def __init__(self, path=None, fig=None, suffix="animations/{}"):
        path = get_root_dir() if path is None else path
        self.path = path / suffix.format(self.__class__.__name__)
        self.fig = fig

    def _cleanup(self):
        if self.path.exists():
            for p in sorted(self.path.iterdir()):
                if re.match(r"\.\d{4}$", p.name):
                    shutil.rmtree(p)

    def init(self, frame=0):
        return NotImplemented

    def animate(self, frame):
        return NotImplemented

    def update_figure(self):
        """Updates the figure canvas."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if matplotlib.get_backend().lower() != "nbagg" or colab:
            display.display(self.fig)
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
        _indices = set(range(self.__getattribute__(key)))
        if isinstance(input, str) and input == "all":
            indices = _indices
        elif isinstance(input, Iterable):
            indices = _indices.intersection(set(input))
        else:
            raise ValueError(key, input)
        return indices

    def animate_in_notebook(self, frames="all", samples="all", repeat=1):
        """Play animation within a jupyter notebook.

        Requires to set the backend to `notebook`, i.e. `%matplotlib notebook`.
        """
        self.update = True
        self.init()
        frames = self._get_indices("frames", frames)
        samples = self._get_indices("n_samples", samples)
        _break = False
        while repeat:
            for sample in sorted(samples):
                self.batch_sample = sample
                for frame in sorted(frames):
                    try:
                        self.animate(frame)
                        sleep(0.1)
                    except KeyboardInterrupt:
                        self.animate(frame)
                        sleep(0.1)
                        _break = True
                        break
                if _break is True:
                    repeat = 1
                    break
            repeat -= 1

    def plot(self, sample, frame):
        _sample = self.batch_sample
        self.update = True
        self.init()
        self.batch_sample = sample
        self.animate(frame)
        self.batch_sample = _sample

    def _mk_dest(self, path=None):
        """Creates a temporary subdir as destination for the images."""
        for i in itertools.count():
            self._path = (path or self.path) / f".{i:04}"
            try:
                self._path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                pass
        self._finalize.append(weakref.finalize(self, shutil.rmtree, self._path))

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
    ):
        """Animates, saves individual frames, and converts to mp4 using ffmpeg."""
        self._mk_dest(path=source_path)
        # Once self is garbage-collected, the temporary dir is rm'd.
        self.update = True
        self.init()
        frames = self._get_indices("frames", frames)
        samples = self._get_indices("n_samples", samples)
        try:
            for sample in sorted(samples):
                self.batch_sample = sample
                for frame in sorted(frames):
                    self.animate_save(frame, dpi=dpi)
        except Exception as e:
            try:
                self.convert(fname + "_bak", delete_if_exists, framerate)
            except Exception as ee:
                raise (Exception([e, ee]))
        self.convert(
            fname,
            delete_if_exists,
            framerate,
            source_path=source_path,
            dest_path=dest_path,
        )

    def convert(
        self,
        fname,
        delete_if_exists=False,
        framerate=30,
        source_path=None,
        dest_path=None,
        type="mp4",
    ):
        """Converts png files in the animations dir to mp4."""
        convert(
            source_path or self._path,
            ((dest_path or self.path) / f"{fname}").with_suffix(f".{type}"),
            framerate,
            delete_if_exists,
        )


def convert(dir, dest, framerate, delete_if_exists, type="mp4"):
    """Converts png files in dir to mp4 or webm."""
    video = dest.with_suffix(f".{type}")

    if type == "mp4":
        kwargs = dict(
            vcodec="libx264",
            vprofile="high",
            vlevel="4.0",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # to make sizes even
            pix_fmt="yuv420p",
            crf=18,
        )
    # for full transparency support and better compression but takes very long
    elif type == "webm":
        kwargs = dict(
            vcodec="libvpx-vp9",
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",
            pix_fmt="yuva420p",
            crf=18,
            threads=4,
        )
    else:
        raise ValueError(f"{type}")

    if video.exists() and not delete_if_exists:
        raise FileExistsError
    elif video.exists() and delete_if_exists:
        video.unlink()
    else:
        pass

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
            logging.warning("Check ffmpeg installation: {}.".format(e))
            return
    except ffmpeg.Error as e:
        logging.info(f"stdout: {e.stdout.decode('utf8')}")
        logging.info(f"stderr:, {e.stderr.decode('utf8')}")
        raise e

    logging.info(f"Created {video}")


class AnimationCollector(Animation):
    """Collects Animations and updates all axes at once.

    Subclasses must populate the `animations` attribute with Animation objects
    and adhere to the Animation interface.

    Args:
        ...
        animations (list): list of Animation objects.
        ...
    """

    animations = []

    def init(self, frame=0):
        for animation in self.animations:
            animation.init(frame)
            # disable update of figure for individual animations to update
            # all axes at once in animate
            animation.update = False

    def animate(self, frame):
        for animation in self.animations:
            animation.animate(frame)
        if self.update:
            self.update_figure()

    def __setattr__(self, key, val):
        """Set the batch_sample for all Animation objects at once."""
        if key == "batch_sample":
            if hasattr(self, "animations"):
                for animation in self.animations:
                    animation.__setattr__(key, val)
        object.__setattr__(self, key, val)
