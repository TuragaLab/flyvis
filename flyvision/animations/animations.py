import itertools
import shutil
from time import sleep
from pathlib import Path
import weakref
import re
import logging


import matplotlib.pyplot as plt

# import ffmpeg

from flyvision import animation_dir


logging = logging.getLogger()


class Animation:
    """Base class for animations."""

    fig = None
    update = True
    batch_sample = 0
    frames = 0
    n_samples = 0
    _finalize = []

    def __init__(self, path=None, fig=None, suffix="animations/{}"):
        path = animation_dir if path is None else path
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

    def notebook_animation(self, frames="all", samples="all", repeat=1):
        """Play animation within a jupyter notebook."""
        self.update = True
        self.init()
        _frames = set(range(self.frames))
        if frames != "all":
            frames = _frames.intersection(set(frames))
        else:
            frames = _frames
        _samples = set(range(self.n_samples))
        if samples != "all":
            samples = _samples.intersection(set(samples))
        else:
            samples = _samples
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
        """Creates a temporary subfolder as destination for the images."""
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
        """Animates and saves the animation as mp4 video."""
        self._mk_dest(path=source_path)
        # Once self is garbage-collected, the temporary folder is rm'd.
        self.update = True
        self.init()
        _frames = set(range(self.frames))
        if frames != "all":
            frames = _frames.intersection(set(frames))
        else:
            frames = _frames
        _samples = set(range(self.n_samples))
        if samples != "all":
            samples = _samples.intersection(set(samples))
        else:
            samples = _samples
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
    ):
        """Converts png files in the animations directory to mp4."""
        convert(
            source_path or self._path,
            ((dest_path or self.path) / f"{fname}").with_suffix(".mp4"),
            framerate,
            delete_if_exists,
        )

    def saveanim(
        self,
        fname,
        frames="all",
        dpi=100,
        framerate=30,
        samples="all",
        delete_if_exists=False,
    ):
        """Save animation at specified path.

        Mimics fig.savefig(fname) interface to save animation at specified path.
        Note: fname must be a Path object.
        """
        if not isinstance(fname, Path):
            raise ValueError
        self.to_vid(
            fname=fname.name,
            frames=frames,
            dpi=dpi,
            framerate=framerate,
            samples=samples,
            delete_if_exists=delete_if_exists,
            dest_path=fname.parent,
        )


def convert(folder, dest, framerate, delete_if_exists):
    _convert(folder, dest, framerate, delete_if_exists, type="mp4")
    # _convert(folder, dest, framerate, delete_if_exists, type="webm")


def _convert(folder, dest, framerate, delete_if_exists, type="mp4"):
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
            ffmpeg.input(f"{folder}/*_*.png", pattern_type="glob", framerate=framerate)
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


class AnimatePlotFn(Animation):
    def __init__(self, plot_fn, path, fname, delete_if_exists, dpi, framerate):
        super().__init__(path, None, "")
        self.plot_fn = plot_fn
        self.frame_count = 0
        self.dpi = dpi
        self.fname = fname
        self.delete_if_exists = delete_if_exists
        self.framerate = framerate
        self._mk_dest()
        self._finalize.append(
            weakref.finalize(
                self,
                self.convert,
                self.fname,
                self.delete_if_exists,
                self.framerate,
            )
        )
        self.title = ""

    def __call__(self, *args, **kwargs):
        self.fig = self.plot_fn(*args, **kwargs)[0]
        self.fig.suptitle(self.title, fontsize=10)
        self.animate_save(self.frame_count, self.dpi)
        self.frame_count += 1

    def animate(self, _):
        pass

    def finalize(self):
        self.convert(self.fname, self.delete_if_exists, self.framerate)


class AnimationCollector(Animation):
    """Collects axes of animations into a single animation.

    Args:
        ...
        animations (list): list of Animation objects.
        ...
    """

    animations = []

    def init(self, frame=0):
        for animation in self.animations:
            animation.init(frame)

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
