from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Iterable, Dict, List, Any, Optional, Union, Callable, Tuple
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as nnf
import numpy as np
import pandas as pd
from PIL import Image

from datamate import Directory, root, Namespace

import flyvision
from flyvision.augmentation.temporal import (
    CropFrames,
    Interpolate,
)
from flyvision.datasets.datasets import MultiTaskDataset
from flyvision.augmentation.hex import (
    ContrastBrightness,
    GammaCorrection,
    HexRotate,
    HexFlip,
    PixelNoise,
)
from flyvision.rendering import BoxEye
from flyvision.rendering.utils import split
from flyvision.utils.dataset_utils import download_sintel
from flyvision.datasets.datasets import MultiTaskDataset

logging = logging.getLogger()


@root(flyvision.sintel_dir)
class RenderedSintel(Directory):
    """Rendering and referencing rendered sintel data."""

    def __init__(
        self,
        tasks: List[str] = ["flow"],
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        n_frames: int = 19,
        center_crop_fraction: float = 0.7,
        unittest=False,
    ):
        """Render and reference sintel data.

        Args:
            tasks: list of tasks to include in the rendering. One of
                ["flow"] or ["flow", "depth"]. Default: ["flow"].
            boxfilter: key word arguments for the BoxEye filter.
                Default: dict(extent=15, kernel_size=13).
            vertical_splits: number of vertical splits of each frame.
                Default: 3.
            n_frames: number of frames to render for each sequence.
                Default: 19.
            center_crop_fraction: fraction of the image to keep after
                cropping. Default: 0.7.
            unittest: if True, only renders a single sequence.
        """
        # always downloads and renders flow data, but optionally also depth
        render_depth = "depth" in tasks
        sintel_path = download_sintel(depth=render_depth)
        boxfilter = BoxEye(**boxfilter)

        lum_paths = (sintel_path / "training/final").iterdir()

        for i, lum_path in enumerate(tqdm(sorted(lum_paths), desc="Rendering")):
            # renders all frames for all sequences which have more than n_frames
            if len(list(lum_path.iterdir())) - 1 >= n_frames:
                flow_path = sintel_path / "training/flow" / lum_path.name
                depth_path = sintel_path / "training/depth" / lum_path.name

                # -- Flow from naturalistic input ------------------------------
                # Y[n] = f(X[1], ..., X[n])
                # n X   Y
                # 0 [x]  n.e.  # not in data
                # 1 [1]  [1]
                # 2 [2]  [2]
                # ...
                # n [n]  [n]

                # (frames, height, width)
                lum = load_sequence(
                    lum_path, sample_lum, start=1, end=None if not unittest else 4
                )
                # (splits, frames, height, width)
                lum_split = split(
                    lum,
                    boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                    vertical_splits,
                    center_crop_fraction,
                )
                # (frames, splits, 1, #hexals)
                lum_hex = boxfilter(lum_split).cpu()

                # (frames, 2, height, width)
                flow = load_sequence(
                    flow_path, sample_flow, end=None if not unittest else 3
                )
                # (splits, frames, 2, height, width)
                flow_split = split(
                    flow,
                    boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                    vertical_splits,
                    center_crop_fraction,
                )
                # (frames, splits, 2, #hexals)
                flow_hex = torch.cat(
                    (
                        boxfilter(flow_split[:, :, 0], ftype="sum"),
                        boxfilter(flow_split[:, :, 1], ftype="sum"),
                    ),
                    dim=2,
                ).cpu()
                if render_depth:
                    # (frames, height, width)
                    depth = load_sequence(
                        depth_path,
                        sample_depth,
                        start=1,
                        end=None if not unittest else 4,
                    )
                    # (splits, frames, height, width)
                    depth_splits = split(
                        depth,
                        boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                        vertical_splits,
                        center_crop_fraction,
                    )
                    # (frames, splits, 1, #hexals)
                    depth_hex = boxfilter(depth_splits, ftype="median").cpu()

                # -- store -----------------------------------------------------
                for j in range(lum_hex.shape[0]):
                    path = f"sequence_{i:02d}_{lum_path.name}_split_{j:02d}"

                    self[f"{path}/lum"] = lum_hex[j]

                    self[f"{path}/flow"] = flow_hex[j]

                    if render_depth:
                        self[f"{path}/depth"] = depth_hex[j]
            if unittest:
                break

    def __call__(self, seq_id):
        # Retrieve sequence Directory.
        data = self[sorted(self)[seq_id]]
        # Load all stored h5 files into memory.
        return {key: data[key][:] for key in sorted(data)}


def load_sequence(path, sample_function, start=0, end=None, as_tensor=True):
    """Calls sample_function on each file in the sorted path and returns
    a concatenation of the results."""
    samples = []
    for p in sorted(path.iterdir())[start:end]:
        samples.append(sample_function(p))
    samples = np.array(samples)
    if as_tensor:
        return torch.Tensor(samples)
    return samples


def sample_lum(path):
    lum = np.float32(Image.open(path).convert("L")) / 255
    return lum


def sample_flow(path):
    """Flow in units of pixel / image_height and with inverted y coordinate.

    Note: I don't know why dividing by image_height but this should decrease the
        gradient magnitude. I inverted y for convenience because I did not know
        that the image plane convention is y downwards.
    """
    with open(path, "rb") as f:
        _, w, h = np.fromfile(f, np.int32, count=3)
        data = np.fromfile(f, np.float32, count=(h * w * 2))
        uv = np.reshape(data, (h, w, 2)) / h  # why are we dividing by h?
        # we invert the y coordinate, which points from the top of the
        # image plane to the bottom
        return uv.transpose(2, 0, 1) * np.array([1, -1])[:, None, None]


def sample_depth(filename):
    with open(filename, "rb") as f:
        _, width, height = np.fromfile(f, dtype=np.int32, count=3)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def sintel_meta(rendered, sintel_path, n_frames, vertical_splits, render_depth):
    @dataclass
    class Meta:
        lum_paths: List[Path]
        flow_paths: List[Path]
        depth_paths: List[Path]
        sequence_indices: np.ndarray
        frames_per_scene: np.ndarray
        sequence_index_to_splits: Dict[int, np.ndarray]

    lum_paths = []
    sequence_indices = []
    frames_per_scene = []
    sequence_index_to_splits = {}
    for i, p in enumerate(sorted((sintel_path / "training/final").iterdir())):
        if len(list(p.iterdir())) - 1 >= n_frames and any(
            p.name in key for key in rendered.keys()
        ):
            lum_paths.append(p)
            sequence_indices.append(i)
            frames_per_scene.append(len(list(p.iterdir())))
        sequence_index_to_splits[i] = vertical_splits * i + np.arange(vertical_splits)
    sequence_indices = np.array(sequence_indices)
    frames_per_scene = np.array(frames_per_scene)

    flow_paths = [sintel_path / "training/flow" / path.name for path in lum_paths]
    depth_paths = (
        [sintel_path / "training/depth" / path.name for path in lum_paths]
        if render_depth
        else None
    )
    return Meta(
        lum_paths=lum_paths,
        flow_paths=flow_paths,
        depth_paths=depth_paths,
        sequence_indices=sequence_indices,
        frames_per_scene=frames_per_scene,
        sequence_index_to_splits=sequence_index_to_splits,
    )


class MultiTaskSintel(MultiTaskDataset):
    """Sintel dataset.

    Args:
        tasks: tasks to be used. Implemented are 'lum', 'flow', and 'depth'.
        interpolate: to interpolate targets. Defaults to False.
        n_frames: number of sequence frames to sample from.
        sample_all: to sample from all sequence frames.
        resampling: to supersample in time in case dt < 1/framerate.
        boxfilter: boxfilter configuration that is used to fly eye render the
            raw data as a preprocessing.
        vertical_splits: number of vertical splits to augment the raw data.
        p_flip: probability to flip under augmentation.
        p_rot: probability to rotate under augmentation.
        contrast: a contrast for augmentation is sampled
            from exp(N(0, contrast)).
        brightness: a brightness for augmentation is sampled
            from N(0, brightness).
        gamma_std: gamma correction augmentation in N(1, gamma_std).
        gaussian_white_noise: standard deviation for hexal-wise gaussian noise
            ~ N(0, gaussian_white_noise).
        cache: 'gpu' or 'cpu' to cache all preprocessed sequences for fast
            dataloading. Has currently no effect.
        init_cache: to turn on or off the loading of all preprocessed sequences
            to the device specifid by 'cache'.

    Attributes (overriding MultiTaskDataset):
        framerate: framerate of the original sequences.
        dt: sampling and integration time constant.
        n_sequences: number of sequences in the dataset.
        augment: turns augmentation on and off.
        t_pre: warmup time.
        t_post cooldown time.
        get_item: return an item of the dataset.
        tasks: a list of all tasks.
        task_weights: a weighting of each task.
        task_weights_sum: sum of all indicated task weights to normalize loss.
        losses: a loss function for each task.

    Additional attributes:
        spec: configuration.
        sintel_path: path to the raw Sintel data.
        n_frames: number of sequence frames to sample from.
        sample_all: to sample from all sequence frames.
        lum_paths: paths to all luminosity, i.e. input data, for sequences
            under taining/final.
        flow_paths: paths to all flow data for sequences under training/final.
        depth_paths: paths to all depth data for sequences under training/final.
        segmentation_paths: paths to all segmentation data for sequences under
            training/final.
        cam_paths: paths to all camera data for sequences under training/final.
        frames_per_scene: number of frames that each sequence contains.
        interpolate: to interpolate targets.
        resampling: to supersample in time in case dt < 1/framerate.
        boxfilter: boxfilter configuration that is used to fly eye render the
            raw data as a preprocessing.
        vertical_splits: number of vertical splits to augment the raw data.
        p_flip: probability to flip under augmentation.
        p_rot: probability to rotate under augmentation.
        contrast: a contrast for augmentation is sampled
            from exp(N(0, contrast)).
        brightness: a brightness for augmentation is sampled
            from N(0, brightness).
        noise_std: standard deviation for hexal-wise gaussian noise.
        cached_samples: all preprocessed sequences for fast dataloading cached on
            'gpu' or 'cpu'.
        fixed_sampling: deprecated. Causes to sample input and target frames in
            in a temporally covering way in preprocessing.
        original_flow_units: to sample flow target in original units (Pixels)
            and with downwards facing y of the image plane.
        depth_augment_contrast: to scale the contrast of objects by the
            inverse of their square distances. This removes the background from
            sequences.
        rendered: Directory pointing to preprocessed sequences.
        map_seq_id_to_splits: a dictionary to map raw sequence indices to the
            vertically split sequence indices.
        arg_df: a table with index, name, and frame information on each sequence.
        hmin, hmax: extreme values of all cached luminosity.
        extent: extent of the boxfilter, i.e. the fly eye.
        depth_augment: augmentation callable.
        jitter: augmentation callable.
        rotate: augmentation callable.
        flip: augmentation callable.
        noise: augmentation callable.

    Raises:
        ValueError: if any element in tasks is invalid.
    """

    # TODO: check if class attributes must be instance attributes
    # to implement attributes of abstract base class MultiTaskDataset
    framerate: int = 24
    dt: float = 1 / 50
    n_sequences: int = 0
    augment: bool = True
    t_pre: float = 0.0
    t_post: float = 0.0
    tasks: List[str] = []
    task_weights: Dict[str, float] = dict()
    task_weights_sum: float = 1.0
    losses: Dict[str, Callable] = dict()
    loss_kwargs: Dict[str, Any] = dict()

    # augmentation callables
    jitter: ContrastBrightness
    rotate: HexRotate
    flip: HexFlip
    noise: PixelNoise

    # other non-trivial structures
    arg_df: pd.DataFrame
    rendered: RenderedSintel
    cached_samples: List[Dict[str, torch.Tensor]]

    valid_tasks = ["lum", "flow", "depth"]

    def __init__(
        self,
        tasks=["lum", "flow"],
        interpolate=True,
        n_frames=19,
        dt=1 / 50,
        augment=True,
        all_frames=False,
        resampling=True,
        boxfilter=dict(extent=15, kernel_size=13),
        vertical_splits=3,
        p_flip=0.5,
        p_rot=0.5,
        contrast_std=0.2,
        brightness_std=0.1,
        gaussian_white_noise=0.08,
        gamma_std=None,
        center_crop_fraction=0.7,
        _init_cache=True,
        unittest=False,
        **kwargs,
    ):
        def check_tasks(tasks):
            invalid_tasks = [x for x in tasks if x not in self.valid_tasks]
            if invalid_tasks:
                raise ValueError(f"invalid tasks {invalid_tasks}")

            tasks = [v for v in self.valid_tasks if v in tasks]  # sort
            # because the input 'lum' is always required
            data_keys = tasks if "lum" in tasks else ["lum", *tasks]
            return tasks, data_keys

        self.tasks, self.data_keys = check_tasks(tasks)
        self.interpolate = interpolate
        self.n_frames = n_frames if not unittest else 3
        self.dt = dt

        self.all_frames = all_frames
        self.resampling = resampling

        self.boxfilter = boxfilter
        self.extent = boxfilter["extent"]
        assert vertical_splits >= 1
        self.vertical_splits = vertical_splits
        self.center_crop_fraction = center_crop_fraction

        self.augment = augment
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.gaussian_white_noise = gaussian_white_noise
        self.gamma_std = gamma_std
        self.init_augmentation()

        self.unittest = unittest

        self.sintel_path = download_sintel(depth="depth" in tasks)
        self.rendered = RenderedSintel(
            tasks=tasks,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            n_frames=n_frames,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
        )
        self.meta = sintel_meta(
            self.rendered, self.sintel_path, n_frames, vertical_splits, "depth" in tasks
        )

        self.config = Namespace(
            tasks=tasks,
            interpolate=interpolate,
            n_frames=n_frames,
            all_frames=all_frames,
            resampling=resampling,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            p_flip=p_flip,
            p_rot=p_rot,
            contrast_std=contrast_std,
            brightness_std=brightness_std,
            gaussian_white_noise=gaussian_white_noise,
            gamma_std=gamma_std,
            center_crop_fraction=center_crop_fraction,
        )

        self.n_sequences = len(self.rendered)

        self.arg_df = pd.DataFrame(
            dict(
                index=np.arange(self.n_sequences),
                original_index=self.meta.sequence_indices.repeat(vertical_splits),
                name=sorted(self.rendered.keys()),
                original_n_frames=self.meta.frames_per_scene.repeat(vertical_splits),
            )
        )

        if _init_cache:
            self.init_cache()

    def init_cache(self):
        self.cached_samples = [
            {
                key: torch.Tensor(val)
                for key, val in self.rendered(seq_id).items()
                if key in self.data_keys
            }
            for seq_id in range(self.n_sequences)
        ]

    def __setattr__(self, name, value):
        if hasattr(self, "renderer") and name in self.renderer.config.keys():
            raise AttributeError(
                "cannot change attribute of renderer past initialization"
            )
        super().__setattr__(name, value)
        augmentation_attributes = [
            "contrast_std",
            "brightness_std",
            "p_rot",
            "p_flip",
            "gaussian_white_noise",
            "dt",
            "all_frames",
            "gamma_std",
        ]
        if (
            all(hasattr(self, attr) for attr in augmentation_attributes)
            and name in augmentation_attributes
        ):
            self.init_augmentation()

    def init_augmentation(
        self,
    ) -> None:
        """Initialize augmentation callables."""

        self.random_crop = CropFrames(
            self.n_frames, all_frames=self.all_frames, random=True
        )

        self.jitter = ContrastBrightness(
            contrast_std=self.contrast_std, brightness_std=self.brightness_std
        )
        self.rotate = HexRotate(self.extent, p_rot=self.p_rot)
        self.flip = HexFlip(self.extent, p_flip=self.p_flip)
        self.noise = PixelNoise(self.gaussian_white_noise)

        self.piecewise_resample = Interpolate(
            self.framerate, 1 / self.dt, mode="nearest-exact"
        )
        self.linear_interpolate = Interpolate(
            self.framerate,
            1 / self.dt,
            mode="linear",
        )
        self.gamma_correct = GammaCorrection(1, self.gamma_std)

    def set_augmentation_params(
        self,
        n_rot: Optional[int] = None,
        flip_axis: Optional[int] = None,
        contrast_factor: Optional[float] = None,
        brightness_factor: Optional[float] = None,
        gaussian_white_noise: Optional[float] = None,
        gamma: Optional[float] = None,
        start_frame: Optional[int] = None,
        total_sequence_length: Optional[int] = None,
    ) -> None:
        """To set augmentation callable parameters at each call of get item."""
        self.rotate.set_or_sample(n_rot)
        self.flip.set_or_sample(flip_axis)
        self.jitter.set_or_sample(contrast_factor, brightness_factor)
        self.noise.set_or_sample(gaussian_white_noise)
        self.gamma_correct.set_or_sample(gamma)
        self.random_crop.set_or_sample(
            start=start_frame, total_sequence_length=total_sequence_length
        )

    def get_item(self, key: int) -> Dict[str, torch.Tensor]:
        """Returns a dataset sample.

        Note: usually invoked with indexing of self, e.g. self[0:10].
        """
        if self.augment:
            return self.apply_augmentation(self.cached_samples[key])
        return self.cached_samples[key]

    def apply_augmentation(
        self,
        data: Dict[str, torch.Tensor],
        n_rot: Optional[int] = None,
        flip_axis: Optional[int] = None,
        contrast_factor: Optional[float] = None,
        brightness_factor: Optional[float] = None,
        gaussian_white_noise: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """To augment a sample from the dataset."""

        self.set_augmentation_params(
            n_rot=n_rot,
            flip_axis=flip_axis,
            contrast_factor=contrast_factor,
            brightness_factor=brightness_factor,
            gaussian_white_noise=gaussian_white_noise,
            gamma=gamma,
            start_frame=None,
            total_sequence_length=data["lum"].shape[0],
        )
        if not self.resampling:
            self.piecewise_resample.augment = False
        if not self.interpolate:
            self.linear_interpolate.augment = False

        def transform_lum(lum):
            return self.piecewise_resample(
                self.rotate(
                    self.flip(
                        self.jitter(
                            self.noise(self.random_crop(lum)),
                        ),
                    )
                )
            )

        def transform_target(target):
            if self.interpolate:
                return self.linear_interpolate(
                    self.random_crop(self.rotate(self.flip(target)))
                )
            return self.piecewise_resample(
                self.random_crop(self.rotate(self.flip(target)))
            )

        return {
            **{"lum": transform_lum(data["lum"])},
            **{
                target: transform_target(data[target])
                for target in self.tasks
                if target in ["flow", "depth"]
            },
        }

    def original_sequence_index(self, key):
        """Get the original sequence index from an index of the split."""
        for index, splits in self.meta.sequence_index_to_splits.items():
            if key in splits:
                break
        return index

    def cartesian_sequence(
        self,
        key,
        vertical_splits=None,
        outwidth=417,
        center_crop_fraction=None,
        sampling=slice(1, None, None),
    ):
        """To return the cartesian sequence of a fly eye rendered sequence."""

        # we want to retrieve the original scene which is possibly split
        # into multiple ones
        key = self.original_sequence_index(key)
        lum_path = self.meta.lum_paths[key]
        images = np.array(
            [sample_lum(path) for path in sorted(lum_path.iterdir())[sampling]]
        )
        return split(
            images,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def cartesian_flow(
        self,
        key,
        vertical_splits=None,
        outwidth=417,
        center_crop_fraction=None,
        sampling=slice(None, None, None),
    ):
        """To return the cartesian flow of a fly eye rendered flow."""
        key = self.original_sequence_index(key)
        flow_path = self.meta.flow_paths[key]
        flow = np.array(
            [sample_flow(path) for path in sorted(flow_path.iterdir())[sampling]]
        )

        return split(
            flow,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def cartesian_depth(
        self,
        key,
        vertical_splits=None,
        outwidth=417,
        center_crop_fraction=None,
        sampling=slice(1, None, None),
    ):
        """To return the cartesian depth of a fly eye rendered depth."""
        key = self.original_sequence_index(key)
        flow_path = self.meta.depth_paths[key]
        depth = np.array(
            [sample_depth(path) for path in sorted(flow_path.iterdir())[sampling]]
        )

        return split(
            depth,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def original_train_and_validation_indices(self):
        """For the VanillaCNN that uses this dataset, this method returns the
        indices required for the dataloader.
        """
        _validation = [
            "ambush_2",
            "bamboo_1",
            "bandage_1",
            "cave_4",
            "market_2",
            "mountain_1",
        ]

        train = [
            "alley_1",
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "ambush_7",
            "bamboo_2",
            "bandage_2",
            "cave_2",
            "market_5",
            "market_6",
            "shaman_2",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ]

        train_indices = [
            i
            for i, name in enumerate(self.arg_df.name)
            if any([scene_name in name for scene_name in train])
        ]
        val_indices = [
            i
            for i, name in enumerate(self.arg_df.name)
            if any([scene_name in name for scene_name in _validation])
        ]
        return train_indices, val_indices


# class AugmentedPaddedSintelLum(MultiTaskSintel):
#     """Nan-padded sintel luminosity with rich augmentation.

#     To evoke rich network responses, performs controlled augmentation (flips,
#     rotations). Nan-padding is applied to sequences shorter than the max
#     length sequence.

#     Expands MultiTaskSintel with methods to hold a trained network directory
#     and return responses for specific augmentation parameters.

#     Note: This child of the Sintel dataset returns only the luminosity and
#         samples all raw frames by default. It turns augmentation at sampling
#         off.

#     Args:
#         flip_axes: list of axes to flip over.
#         n_rotations: list of number of rotations to perform.
#         build_stim_on_init: to build the augmented stimulus in cache.
#         dt: integration and sampling time constant.

#     Kwargs:
#         See list of arguments for MultiTaskSintel.
#         Overrides tasks, sample_all, resampling, init_cache, and augment.

#     Attributes:
#         sequences: augmented, cached sequences.
#         ~ see MultiTasksintel
#     """

#     sequences: List[Dict[str, torch.Tensor]]
#     _valid_flip_axes = [None, 0, 1, 2]
#     _valid_rotations = [0, 1, 2, 3, 4, 5]

#     def __init__(
#         self,
#         flip_axes=[None, 0, 1, 2],
#         n_rotations=[0, 1, 2, 3, 4, 5],
#         build_stim_on_init=True,
#         dt=1 / 50,
#         **kwargs,
#     ):
#         _invalid_flip_axes = [p for p in flip_axes if p not in self._valid_flip_axes]
#         _invalid_rotations = [p for p in n_rotations if p not in self._valid_rotations]
#         if _invalid_flip_axes or _invalid_rotations:
#             raise ValueError(
#                 "invalid augmentation " f"{_invalid_flip_axes}, {_invalid_rotations}"
#             )

#         kwargs.update(
#             tasks=["lum"], sample_all=True, resampling=True, init_cache=True
#         )
#         super().__init__(**kwargs)

#         self.augment = False
#         self.rotate = HexRotate(15, 0)
#         self.flip = HexFlip(15, None)
#         self.flip_axes = flip_axes
#         self.n_rotations = n_rotations

#         self._built = False
#         if build_stim_on_init:
#             self._build()
#             self._built = True

#         self.dt = dt

#     def _build(self):
#         seq_to_frames = {
#             i: d["lum"].shape[0] for i, d in enumerate(self.cached_samples)
#         }
#         self.n_frames = max([d["lum"].shape[0] for d in self.cached_samples])

#         self.params = [
#             (p[0][0], p[1], p[2], p[0][1])
#             for p in list(
#                 product(
#                     zip(list(seq_to_frames.keys()), list(seq_to_frames.values())),
#                     self.flip_axes,
#                     self.n_rotations,
#                 )
#             )
#         ]
#         self.arg_df = pd.DataFrame(
#             self.params, columns=["sequence", "flip_ax", "n_rot", "frames"]
#         )

#         self.sequences = {}
#         for i, (sample, flip_ax, n_rot, _) in enumerate(self.params):
#             self.flip.axis = flip_ax
#             self.rotate.n_rot = n_rot
#             self.sequences[i] = self.rotate(
#                 self.flip(self.cached_samples[sample]["lum"], "lum"), "lum"
#             )
#         self.n_sequences = len(self.sequences)

#     def init_responses(self, tnn, subdir="augmented_sintel"):
#         self.tnn = tnn
#         with exp_path_context():
#             if isinstance(tnn, (str, Path)):
#                 self.tnn, _ = init_network_dir(tnn, None, None)
#             self.central_activity = utils.CentralActivity(
#                 self.tnn[subdir].network_states.nodes.activity_central[:],
#                 self.tnn.ctome,
#                 keepref=True,
#             )

#     def get_item(self, key):
#         stim = self.sequences[key]

#         stim = nnf.pad(
#             stim,
#             pad=(0, 0, 0, 0, 0, self.n_frames - stim.shape[0]),
#             mode="constant",
#             value=np.nan,
#         ).squeeze()  # squeezing out the feat dim

#         if self.resampling:
#             frame_samples = self.get_temporal_sample_indices(
#                 self.n_frames, self.n_frames
#             )
#             return stim[frame_samples]
#         return stim

#     def get(self, sequence, flip_ax, n_rot):
#         key = self._key(sequence, flip_ax, n_rot)
#         return self[key]

#     def _key(self, sequence, flip_ax, n_rot):
#         try:
#             mask = self.mask(sequence, flip_ax, n_rot)
#             return np.arange(len(self))[mask].item()
#         except ValueError:
#             raise ValueError(
#                 f"sequence: {sequence}, flip_ax: {flip_ax}, n_rot: {n_rot} invalid."
#             )

#     def _params(self, key):
#         return self.arg_df.iloc[key].values

#     def mask(self, sequence=None, flip_ax=None, n_rot=None):
#         values = self.arg_df.values
#         _nans = np.isnan(values)
#         values = values.astype(object)
#         values[_nans] = "None"

#         def iterparam(param, name, axis, and_condition):
#             condition = np.zeros(len(values)).astype(bool)
#             if isinstance(param, Iterable) and not isinstance(param, str):
#                 for p in param:
#                     _new = values.take(axis, axis=1) == p
#                     assert any(_new), f"{name} {p} not in dataset."
#                     condition = np.logical_or(condition, _new)
#             else:
#                 _new = values.take(axis, axis=1) == param
#                 assert any(_new), f"{name} {param} not in dataset."
#                 condition = np.logical_or(condition, _new)
#             return condition & and_condition

#         condition = np.ones(len(values)).astype(bool)
#         if sequence is not None:
#             condition = iterparam(sequence, "sequence", 0, condition)
#         if flip_ax is not None:
#             condition = iterparam(flip_ax, "flip_ax", 1, condition)
#         if n_rot is not None:
#             condition = iterparam(n_rot, "n_rot", 2, condition)
#         return condition

#     def response(
#         self,
#         node_type=None,
#         sequence=None,
#         flip_ax=None,
#         n_rot=None,
#         rm_nans=False,
#     ):
#         assert self.tnn
#         mask = self.mask(sequence=sequence, flip_ax=flip_ax, n_rot=n_rot)

#         if node_type is not None:
#             responses = self.central_activity[node_type][mask][:, :, None]
#         else:
#             responses = self.central_activity[:][mask]

#         if rm_nans:
#             return remove_nans(responses)

#         return responses.squeeze()


# class AugmentedSintel(MultiTaskSintel):
#     """Sintel dataset with controlled, rich augmentation.

#     No nan-padding is applied.

#     Note: returns all data and can be used to evaluate networks on a richer
#         dataset.

#     Expands MultiTaskSintel with methods to hold a trained network directory
#     and return responses for specific augmentation parameters.

#     Args:
#         n_frames: number of sequence frames to sample from.
#         flip_axes: list of axes to flip over.
#         n_rotations: list of number of rotations to perform.
#         temporal_split: to enable temporally controlled augmentation
#             (experimental).
#         build_stim_on_init: to build the augmented stimulus in cache.
#         dt: integration and sampling time constant.

#     Kwargs:
#         See list of arguments for MultiTaskSintel.
#         Overrides resampling, init_cache, and augment.

#     Attributes:
#         sequences: augmented, cached sequences.
#         ~ see MultiTasksintel
#     """

#     sequences: List[Dict[str, torch.Tensor]]
#     _valid_flip_axes = [None, 0, 1, 2]
#     _valid_rotations = [0, 1, 2, 3, 4, 5]

#     def __init__(
#         self,
#         n_frames=19,
#         flip_axes=[None, 0, 1, 2],
#         n_rotations=[0, 1, 2, 3, 4, 5],
#         build_stim_on_init=True,
#         temporal_split=False,
#         dt=1 / 50,
#         **kwargs,
#     ):
#         _invalid_flip_axes = [p for p in flip_axes if p not in self._valid_flip_axes]
#         _invalid_rotations = [p for p in n_rotations if p not in self._valid_rotations]
#         if _invalid_flip_axes or _invalid_rotations:
#             raise ValueError(
#                 "invalid augmentation " f"{_invalid_flip_axes}, {_invalid_rotations}"
#             )

#         kwargs.update(resampling=True, init_cache=True)
#         super().__init__(**kwargs)

#         self.augment = False
#         self.rotate = HexRotate(15, 0)
#         self.flip = HexFlip(15, None)
#         self.flip_axes = flip_axes
#         self.n_rotations = n_rotations

#         self.dt = dt
#         # self.pad = pad
#         self.n_frames = n_frames

#         (
#             self.cached_samples,
#             self._original_repeats,
#         ) = _temporal_split_cached_samples(
#             self.cached_samples, n_frames, split=temporal_split
#         )

#         # disabling rotation and flips to use augment function for
#         # contrast / brightness and noise if wanted.
#         self.p_flip = 0
#         self.p_rot = 0

#         self._built = False
#         if build_stim_on_init:
#             self._build()
#             self._built = True

#     def _build(self):
#         vsplit_index, original_index, name = (
#             self.arg_df[["index", "original_index", "name"]]
#             .values.repeat(self._original_repeats, axis=0)
#             .T
#         )

#         tsplit_index = np.arange(len(self.cached_samples))

#         n_frames = [d["lum"].shape[0] for d in self.cached_samples]

#         self.params = [
#             (*p[0], p[1], p[2])
#             for p in list(
#                 product(
#                     zip(
#                         name,
#                         original_index,
#                         vsplit_index,
#                         tsplit_index,
#                         n_frames,
#                     ),
#                     self.flip_axes,
#                     self.n_rotations,
#                 )
#             )
#         ]

#         self.arg_df = pd.DataFrame(
#             self.params,
#             columns=[
#                 "name",
#                 "original_index",
#                 "vertical_split_index",
#                 "temporal_split_index",
#                 "frames",
#                 "flip_ax",
#                 "n_rot",
#             ],
#         )

#         self.sequences = {}
#         for i, (_, _, _, sample, _, flip_ax, n_rot) in enumerate(self.params):
#             self.flip.axis = flip_ax
#             self.rotate.n_rot = n_rot
#             self.sequences[i] = self._augment(
#                 self.cached_samples[sample],
#                 n_rot=n_rot,
#                 flip_axis=flip_ax,
#                 contrast_factor=0,
#                 brightness_factor=0,
#                 noise_std=0,
#             )

#     def init_responses(self, tnn, subdir="augmented_sintel"):
#         self.tnn = tnn
#         with exp_path_context():
#             if isinstance(tnn, (str, Path)):
#                 self.tnn, _ = init_network_dir(tnn, None, None)
#             self.central_activity = utils.CentralActivity(
#                 self.tnn[subdir].network_states.nodes.activity_central[:],
#                 self.tnn.ctome,
#                 keepref=True,
#             )

#     def __len__(self):
#         return len(self.sequences)

#     def _pad(self, data):
#         _data = {}
#         for key, value in data.items():
#             pad_length = self.n_frames - value.shape[0]
#             _data[key] = nnf.pad(
#                 value,
#                 pad=(0, 0, 0, 0, 0, pad_length),
#                 mode="constant",
#                 value=np.nan,
#             )
#         return _data

#     def get_item(self, key):
#         data = self.sequences[key]

#         if self.augment:
#             data = self._augment(data)

#         # if self.pad:
#         #     data = self._pad(data)

#         if self.resampling:
#             return self._dt_sample(data)
#         return data

#     def get(self, sequence, flip_ax, n_rot):
#         key = self._key(sequence, flip_ax, n_rot)
#         return self[key]

#     def _key(self, sequence, flip_ax, n_rot):
#         try:
#             mask = self.mask(sequence, flip_ax, n_rot)
#             return np.arange(len(self))[mask].item()
#         except ValueError:
#             raise ValueError(
#                 f"sequence: {sequence}, flip_ax: {flip_ax}, n_rot: {n_rot} invalid."
#             )

#     def _params(self, key):
#         return self.arg_df.iloc[key].values

#     def mask(self, sequence=None, flip_ax=None, n_rot=None):
#         values = self.arg_df.iloc[:, 1:].values
#         _nans = np.isnan(values)
#         values = values.astype(object)
#         values[_nans] = "None"

#         def iterparam(param, name, axis, and_condition):
#             condition = np.zeros(len(values)).astype(bool)
#             if isinstance(param, Iterable) and not isinstance(param, str):
#                 for p in param:
#                     _new = values.take(axis, axis=1) == p
#                     assert any(_new), f"{name} {p} not in dataset."
#                     condition = np.logical_or(condition, _new)
#             else:
#                 _new = values.take(axis, axis=1) == param
#                 assert any(_new), f"{name} {param} not in dataset."
#                 condition = np.logical_or(condition, _new)
#             return condition & and_condition

#         condition = np.ones(len(values)).astype(bool)
#         if sequence is not None:
#             condition = iterparam(sequence, "temporal_split_index", -4, condition)
#         if flip_ax is not None:
#             condition = iterparam(flip_ax, "flip_ax", -2, condition)
#         if n_rot is not None:
#             condition = iterparam(n_rot, "n_rot", -1, condition)
#         return condition

#     def response(
#         self,
#         node_type=None,
#         sequence=None,
#         flip_ax=None,
#         n_rot=None,
#         rm_nans=False,
#     ):
#         assert self.tnn
#         mask = self.mask(sequence=sequence, flip_ax=flip_ax, n_rot=n_rot)

#         if node_type is not None:
#             responses = self.central_activity[node_type][mask][:, :, None]
#         else:
#             responses = self.central_activity[:][mask]

#         if rm_nans:
#             return remove_nans(responses)

#         return responses.squeeze()


# class AugmentedSintelLum(AugmentedSintel):
#     """AugmentedSintel but returning only luminosty to be compatible with
#     network.stimulus_responses"""

#     __doc__ = """Overriding get_item to return only luminosity to be compatible with
#     network.stimulus_responses.
#     From parent class:
#     {}""".format(
#         AugmentedSintel.__doc__
#     )

#     def get_item(self, key):
#         data = super().get_item(key)
#         return data["lum"].squeeze(1)


# def _temporal_split_cached_samples(cached_sequences, max_frames, split=True):
#     """To split sequences in temporal dimension into regularly binned sequences.

#     Note: overlapping splits of sequences which
#         lengths are not an integer multiple of max_frames contain repeating frames.

#     Args:
#         cached_sequences (List[Dict[str, Tensor]]): ordered list of
#             dicts of sequences of shape (n_frames, n_features, n_hexals).
#         n_frames (int)

#     Returns:
#         List[Dict[str, Tensor]]: sames sequences but split along
#             temporal dimension.
#         array: original index of each new split.
#     """
#     if split:
#         _seq_lists = {k: [] for k in cached_sequences[0].keys()}

#         splits_per_seq = []
#         for i, sequence in enumerate(cached_sequences):
#             for key, value in sequence.items():
#                 _splits = temporal_split_sequence(value, max_frames)
#                 _seq_lists[key].extend([*_splits])
#             splits_per_seq.append([i, len(_splits)])

#         _split_cached_sequences = []
#         for i in range(len(_seq_lists["lum"])):
#             _split_cached_sequences.append({k: v[i] for k, v in _seq_lists.items()})

#         index, repeats = np.array(splits_per_seq).T
#         return _split_cached_sequences, repeats
#     else:
#         return cached_sequences, np.ones(len(cached_sequences)).astype(int)


# def temporal_split_sequence(sequence, max_frames):
#     """
#     Args:
#         sequence (array, Tensor): of shape (n_frames, n_features, n_hexals)

#     Returns:
#         array, Tensor: of shape (splits, max_frames, n_features, n_hexals)

#     Notes: the number of splits computes as int(np.round(n_frames / max_frames)).
#     """
#     n_frames, _, _ = sequence.shape
#     splits = np.round(n_frames / max_frames).astype(int)
#     if splits <= 1:
#         return sequence[:max_frames][None]
#     else:
#         return split(
#             sequence.transpose(0, -1),
#             max_frames,
#             splits,
#             center_crop_fraction=None,
#         ).transpose(1, -1)


# def remove_nans(responses):
#     """Removes nans in (sample, frames, channels)-like array.

#     Returns list of differently sized sequences.
#     """
#     _resp = []
#     for r in responses:
#         _isnan = np.isnan(r).any(axis=1)
#         _resp.append(r[~_isnan].squeeze())
#     return _resp
