import logging
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from datamate import Directory, Namespace, root
from tqdm import tqdm

from flyvis import renderings_dir

from .augmentation.hex import (
    ContrastBrightness,
    GammaCorrection,
    HexFlip,
    HexRotate,
    PixelNoise,
)
from .augmentation.temporal import (
    CropFrames,
    Interpolate,
)
from .datasets import MultiTaskDataset
from .rendering import BoxEye
from .rendering.utils import split
from .sintel_utils import (
    download_sintel,
    load_sequence,
    original_train_and_validation_indices,
    sample_depth,
    sample_flow,
    sample_lum,
    sintel_meta,
    temporal_split_cached_samples,
)

logger = logging.getLogger(__name__)

__all__ = ["RenderedSintel", "MultiTaskSintel", "AugmentedSintel"]


@root(renderings_dir)
class RenderedSintel(Directory):
    """Rendering and referencing rendered sintel data.

    Args:
        tasks: List of tasks to include in the rendering. May include 'flow' or 'depth'.
        boxfilter: Key word arguments for the BoxEye filter.
        vertical_splits: Number of vertical splits of each frame.
        n_frames: Number of frames to render for each sequence.
        center_crop_fraction: Fraction of the image to keep after cropping.
        unittest: If True, only renders a single sequence.

    Attributes:
        config: Configuration parameters used for rendering.
        sequence_<id>_<name>_split_<j>/lum (ArrayFile):
            Rendered luminance data (frames, 1, hexals).
        sequence_<id>_<name>_split_<j>/flow (ArrayFile):
            Rendered flow data (frames, 2, hexals).
        sequence_<id>_<name>_split_<j>/depth (ArrayFile):
            Rendered depth data (frames, 1, hexals).
    """

    def __init__(
        self,
        tasks: List[str] = ["flow"],
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        n_frames: int = 19,
        center_crop_fraction: float = 0.7,
        unittest: bool = False,
        sintel_path: Optional[Union[str, Path]] = None,
    ):
        # Convert sintel_path to Path object if it's not None
        sintel_path = (
            Path(sintel_path) if sintel_path else download_sintel(depth="depth" in tasks)
        )
        boxfilter = BoxEye(**boxfilter)

        lum_paths = (sintel_path / "training/final").iterdir()

        for i, lum_path in enumerate(tqdm(sorted(lum_paths), desc="Rendering")):
            # Renders all frames for all sequences which have more than n_frames
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
                    lum_path,
                    sample_lum,
                    start=1,
                    end=None if not unittest else 4,
                )
                # (splits, frames, height, width)
                lum_split = split(
                    lum,
                    boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                    vertical_splits,
                    center_crop_fraction,
                )
                # (splits, frames, 1, #hexals)
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
                # (splits, frames, 2, #hexals)
                flow_hex = torch.cat(
                    (
                        boxfilter(flow_split[:, :, 0], ftype="sum"),
                        boxfilter(flow_split[:, :, 1], ftype="sum"),
                    ),
                    dim=2,
                ).cpu()
                if "depth" in tasks:
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
                    # (splits, frames, 1, #hexals)
                    depth_hex = boxfilter(depth_splits, ftype="median").cpu()

                # -- store -----------------------------------------------------
                for j in range(lum_hex.shape[0]):
                    path = f"sequence_{i:02d}_{lum_path.name}_split_{j:02d}"

                    self[f"{path}/lum"] = lum_hex[j]

                    self[f"{path}/flow"] = flow_hex[j]

                    if "depth" in tasks:
                        self[f"{path}/depth"] = depth_hex[j]
            if unittest:
                break

    def __call__(self, seq_id: int) -> Dict[str, np.ndarray]:
        """Returns all rendered data for a given sequence index.

        Args:
            seq_id: Index of the sequence to retrieve.

        Returns:
            Dictionary containing the rendered data for the specified sequence.
        """
        # Load all stored h5 files into memory.
        data = self[sorted(self)[seq_id]]
        return {key: data[key][:] for key in sorted(data)}


class MultiTaskSintel(MultiTaskDataset):
    """Sintel dataset.

    Args:
        tasks: List of tasks to include. May include 'flow', 'lum', or 'depth'.
        boxfilter: Key word arguments for the BoxEye filter.
        vertical_splits: Number of vertical splits of each frame.
        n_frames: Number of frames to render for each sequence.
        center_crop_fraction: Fraction of the image to keep after cropping.
        dt: Sampling and integration time constant.
        augment: Turns augmentation on and off.
        random_temporal_crop: Randomly crops a temporal window of length `n_frames` from
            each sequence.
        all_frames: If True, all frames are returned. If False, only `n_frames`. Takes
            precedence over `random_temporal_crop`.
        resampling: If True, piecewise-constant resamples the input sequence to the
            target framerate (1/dt).
        interpolate: If True, linearly interpolates the target sequence to the target
            framerate (1/dt).
        p_flip: Probability of flipping the sequence across hexagonal axes.
        p_rot: Probability of rotating the sequence by n*60 degrees.
        contrast_std: Standard deviation of the contrast augmentation.
        brightness_std: Standard deviation of the brightness augmentation.
        gaussian_white_noise: Standard deviation of the pixel-wise gaussian white noise.
        gamma_std: Standard deviation of the gamma augmentation.
        _init_cache: If True, caches the dataset in memory.
        unittest: If True, only renders a single sequence.
        flip_axes: List of axes to flip over.

    Attributes:
        dt (float): Sampling and integration time constant.
        t_pre (float): Warmup time.
        t_post (float): Cooldown time.
        tasks (List[str]): List of all tasks.
        valid_tasks (List[str]): List of valid task names.

    Raises:
        ValueError: If any element in tasks is invalid.
    """

    original_framerate: int = 24
    dt: float = 1 / 50
    t_pre: float = 0.0
    t_post: float = 0.0
    tasks: List[str] = []
    valid_tasks: List[str] = ["lum", "flow", "depth"]

    def __init__(
        self,
        tasks: List[str] = ["flow"],
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        n_frames: int = 19,
        center_crop_fraction: float = 0.7,
        dt: float = 1 / 50,
        augment: bool = True,
        random_temporal_crop: bool = True,
        all_frames: bool = False,
        resampling: bool = True,
        interpolate: bool = True,
        p_flip: float = 0.5,
        p_rot: float = 5 / 6,
        contrast_std: float = 0.2,
        brightness_std: float = 0.1,
        gaussian_white_noise: float = 0.08,
        gamma_std: Optional[float] = None,
        _init_cache: bool = True,
        unittest: bool = False,
        flip_axes: List[int] = [0, 1],
        sintel_path: Optional[Union[str, Path]] = None,
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

        self.p_flip = p_flip
        self.p_rot = p_rot
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.gaussian_white_noise = gaussian_white_noise
        self.gamma_std = gamma_std
        self.random_temporal_crop = random_temporal_crop
        self.flip_axes = flip_axes
        self.fix_augmentation_params = False

        self.init_augmentation()
        self._augmentations_are_initialized = True
        # note: self.augment is a property with a setter that relies on
        # _augmentations_are_initialized
        self.augment = augment

        self.unittest = unittest

        # Download Sintel once and reuse the path
        self.sintel_path = (
            Path(sintel_path) if sintel_path else download_sintel(depth="depth" in tasks)
        )

        self.rendered = RenderedSintel(
            tasks=tasks,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            n_frames=n_frames,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
            sintel_path=self.sintel_path,
        )

        self.meta = sintel_meta(
            self.rendered, self.sintel_path, n_frames, vertical_splits, "depth" in tasks
        )

        self.config = Namespace(
            tasks=tasks,
            interpolate=interpolate,
            n_frames=n_frames,
            dt=dt,
            augment=augment,
            all_frames=all_frames,
            resampling=resampling,
            random_temporal_crop=random_temporal_crop,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            p_flip=p_flip,
            p_rot=p_rot,
            contrast_std=contrast_std,
            brightness_std=brightness_std,
            gaussian_white_noise=gaussian_white_noise,
            gamma_std=gamma_std,
            center_crop_fraction=center_crop_fraction,
            flip_axes=flip_axes,
        )

        self.arg_df = pd.DataFrame(
            dict(
                index=np.arange(len(self.rendered)),
                original_index=self.meta.sequence_indices.repeat(vertical_splits),
                name=sorted(self.rendered.keys()),
                original_n_frames=self.meta.frames_per_scene.repeat(vertical_splits),
            )
        )

        if _init_cache:
            self.init_cache()

    def init_cache(self) -> None:
        """Initialize the cache with preprocessed sequences."""
        self.cached_sequences = [
            {
                key: torch.tensor(val, dtype=torch.float32)
                for key, val in self.rendered(seq_id).items()
                if key in self.data_keys
            }
            for seq_id in range(len(self))
        ]

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} with {len(self)} sequences.\n"
        repr += "See docs, arg_df and meta for more details.\n"
        return repr

    @property
    def docs(self) -> str:
        print(self.__doc__)

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom attribute setter to handle special cases and update augmentation.

        Args:
            name: Name of the attribute to set.
            value: Value to set the attribute to.

        Raises:
            AttributeError: If trying to change framerate or rendered initialization
                attributes.
        """
        # some changes have no effect cause they are fixed, or set by the pre-rendering
        if name == "framerate":
            raise AttributeError("cannot change framerate")
        if hasattr(self, "rendered") and name in self.rendered.config:
            raise AttributeError("cannot change attribute of rendered initialization")
        super().__setattr__(name, value)
        # also update augmentation because it may already be initialized
        if getattr(self, "_augmentations_are_initialized", False):
            self.update_augmentation(name, value)

    def init_augmentation(self) -> None:
        """Initialize augmentation callables."""
        self.temporal_crop = CropFrames(
            self.n_frames, all_frames=self.all_frames, random=self.random_temporal_crop
        )
        self.jitter = ContrastBrightness(
            contrast_std=self.contrast_std, brightness_std=self.brightness_std
        )
        self.rotate = HexRotate(self.extent, p_rot=self.p_rot)
        self.flip = HexFlip(self.extent, p_flip=self.p_flip, flip_axes=self.flip_axes)
        self.noise = PixelNoise(self.gaussian_white_noise)

        self.piecewise_resample = Interpolate(
            self.original_framerate, 1 / self.dt, mode="nearest-exact"
        )
        self.linear_interpolate = Interpolate(
            self.original_framerate,
            1 / self.dt,
            mode="linear",
        )
        self.gamma_correct = GammaCorrection(1, self.gamma_std)

    def update_augmentation(self, name: str, value: Any) -> None:
        """Update augmentation parameters based on attribute changes.

        Args:
            name: Name of the attribute that changed.
            value: New value of the attribute.
        """
        if name == "dt":
            self.piecewise_resample.target_framerate = 1 / value
            self.linear_interpolate.target_framerate = 1 / value
        if name in ["all_frames", "random_temporal_crop"]:
            self.temporal_crop.all_frames = value
            self.temporal_crop.random = value
        if name in ["contrast_std", "brightness_std"]:
            self.jitter.contrast_std = value
            self.jitter.brightness_std = value
        if name == "p_rot":
            self.rotate.p_rot = value
        if name == "p_flip":
            self.flip.p_flip = value
        if name == "gaussian_white_noise":
            self.noise.std = value
        if name == "gamma_std":
            self.gamma_correct.std = value

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
        """Set augmentation callable parameters.

        Info:
            Called for each call of get_item.

        Args:
            n_rot: Number of rotations to apply.
            flip_axis: Axis to flip over.
            contrast_factor: Contrast factor for jitter augmentation.
            brightness_factor: Brightness factor for jitter augmentation.
            gaussian_white_noise: Standard deviation for noise augmentation.
            gamma: Gamma value for gamma correction.
            start_frame: Starting frame for temporal crop.
            total_sequence_length: Total length of the sequence.
        """
        if not self.fix_augmentation_params:
            self.rotate.set_or_sample(n_rot)
            self.flip.set_or_sample(flip_axis)
            self.jitter.set_or_sample(contrast_factor, brightness_factor)
            self.noise.set_or_sample(gaussian_white_noise)
            self.gamma_correct.set_or_sample(gamma)
            self.temporal_crop.set_or_sample(
                start=start_frame, total_sequence_length=total_sequence_length
            )

    def get_item(self, key: int) -> Dict[str, torch.Tensor]:
        """Return a dataset sample.

        Args:
            key: Index of the sample to retrieve.

        Returns:
            Dictionary containing the augmented sample data.
        """
        return self.apply_augmentation(self.cached_sequences[key])

    @contextmanager
    def augmentation(self, abool: bool):
        """Context manager to turn augmentation on or off in a code block.

        Args:
            abool: Boolean value to set augmentation state.

        Example:
            ```python
            with dataset.augmentation(True):
                for i, data in enumerate(dataloader):
                    ...  # all data is augmented
            ```
        """
        augmentations = [
            "temporal_crop",
            "jitter",
            "rotate",
            "flip",
            "noise",
            "piecewise_resample",
            "linear_interpolate",
            "gamma_correct",
        ]
        states = {key: getattr(self, key).augment for key in augmentations}
        _augment = self.augment
        try:
            self.augment = abool
            yield
        finally:
            self.augment = _augment
            for key in augmentations:
                getattr(self, key).augment = states[key]

    @property
    def augment(self) -> bool:
        """Get the current augmentation state."""
        return self._augment

    @augment.setter
    def augment(self, value: bool) -> None:
        """Set the augmentation state and update augmentation callables.

        Args:
            value: Boolean value to set augmentation state.
        """
        self._augment = value
        if not self._augmentations_are_initialized:
            return
        # note: random_temporal_crop can override augment=True
        self.temporal_crop.random = self.random_temporal_crop if value else False
        self.jitter.augment = value
        self.rotate.augment = value
        self.flip.augment = value
        self.noise.augment = value
        # note: these two are not affected by augment
        self.piecewise_resample.augment = self.resampling
        self.linear_interpolate.augment = self.interpolate
        self.gamma_correct.augment = value

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
        """Apply augmentation to a sample from the dataset.

        Args:
            data: Dictionary containing the sample data.
            n_rot: Number of rotations to apply.
            flip_axis: Axis to flip over.
            contrast_factor: Contrast factor for jitter augmentation.
            brightness_factor: Brightness factor for jitter augmentation.
            gaussian_white_noise: Standard deviation for noise augmentation.
            gamma: Gamma value for gamma correction.

        Returns:
            Dictionary containing the augmented sample data.
        """

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

        def transform_lum(lum):
            return self.piecewise_resample(
                self.rotate(
                    self.flip(
                        self.jitter(
                            self.noise(self.temporal_crop(lum)),
                        ),
                    )
                )
            )

        def transform_target(target):
            if self.interpolate:
                return self.linear_interpolate(
                    self.rotate(self.flip(self.temporal_crop(target)))
                )
            return self.piecewise_resample(
                self.rotate(self.flip(self.temporal_crop(target)))
            )

        return {
            **{"lum": transform_lum(data["lum"])},
            **{
                target: transform_target(data[target])
                for target in self.tasks
                if target in ["flow", "depth"]
            },
        }

    def original_sequence_index(self, key: int) -> int:
        """Get the original sequence index from an index of the split.

        Args:
            key: Index of the split.

        Returns:
            Original sequence index.

        Raises:
            ValueError: If the key is not found in splits.
        """
        for index, splits in self.meta.sequence_index_to_splits.items():
            if key in splits:
                return index
        raise ValueError(f"key {key} not found in splits")

    def cartesian_sequence(
        self,
        key: int,
        vertical_splits: Optional[int] = None,
        outwidth: int = 716,
        center_crop_fraction: Optional[float] = None,
        sampling: slice = slice(1, None, None),
    ) -> np.ndarray:
        """Return the cartesian sequence of a fly eye rendered sequence.

        Args:
            key: Index of the sequence.
            vertical_splits: Number of vertical splits to apply.
            outwidth: Output width of the sequence.
            center_crop_fraction: Fraction of the image to keep after cropping.
            sampling: Slice object for sampling frames.

        Returns:
            Numpy array containing the cartesian sequence.
        """
        # we want to retrieve the original scene which is possibly split
        # into multiple ones
        key = self.original_sequence_index(key)
        lum_path = self.meta.lum_paths[key]
        images = np.array([
            sample_lum(path) for path in sorted(lum_path.iterdir())[sampling]
        ])
        return split(
            images,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def cartesian_flow(
        self,
        key: int,
        vertical_splits: Optional[int] = None,
        outwidth: int = 417,
        center_crop_fraction: Optional[float] = None,
        sampling: slice = slice(None, None, None),
    ) -> np.ndarray:
        """Return the cartesian flow of a fly eye rendered flow.

        Args:
            key: Index of the sequence.
            vertical_splits: Number of vertical splits to apply.
            outwidth: Output width of the flow.
            center_crop_fraction: Fraction of the image to keep after cropping.
            sampling: Slice object for sampling frames.

        Returns:
            Numpy array containing the cartesian flow.
        """
        key = self.original_sequence_index(key)
        flow_path = self.meta.flow_paths[key]
        flow = np.array([
            sample_flow(path) for path in sorted(flow_path.iterdir())[sampling]
        ])

        return split(
            flow,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def cartesian_depth(
        self,
        key: int,
        vertical_splits: Optional[int] = None,
        outwidth: int = 417,
        center_crop_fraction: Optional[float] = None,
        sampling: slice = slice(1, None, None),
    ) -> np.ndarray:
        """Return the cartesian depth of a fly eye rendered depth.

        Args:
            key: Index of the sequence.
            vertical_splits: Number of vertical splits to apply.
            outwidth: Output width of the depth.
            center_crop_fraction: Fraction of the image to keep after cropping.
            sampling: Slice object for sampling frames.

        Returns:
            Numpy array containing the cartesian depth.
        """
        key = self.original_sequence_index(key)
        flow_path = self.meta.depth_paths[key]
        depth = np.array([
            sample_depth(path) for path in sorted(flow_path.iterdir())[sampling]
        ])

        return split(
            depth,
            outwidth,
            vertical_splits or self.vertical_splits,
            center_crop_fraction or self.center_crop_fraction,
        )

    def original_train_and_validation_indices(self) -> Tuple[List[int], List[int]]:
        """Get original training and validation indices for the dataloader.

        Returns:
            Tuple containing lists of train and validation indices.
        """
        return original_train_and_validation_indices(self)


class AugmentedSintel(MultiTaskSintel):
    """Sintel dataset with controlled, rich augmentation.

    Info:
        Returns deterministic augmented dataset to evaluate networks on a richer dataset.

    Args:
        n_frames: Number of sequence frames to sample from.
        flip_axes: List of axes to flip over.
        n_rotations: List of number of rotations to perform.
        temporal_split: Enable temporally controlled augmentation (experimental).
        build_stim_on_init: Build the augmented stimulus in cache.
        dt: Integration and sampling time constant.
        tasks: List of tasks to include. May include 'flow', 'lum', or 'depth'.
        interpolate: If True, linearly interpolates the target sequence to the target
            framerate.
        all_frames: If True, all frames are returned. If False, only `n_frames`.
        random_temporal_crop: Randomly crops a temporal window of length `n_frames`
            from each sequence.
        boxfilter: Key word arguments for the BoxEye filter.
        vertical_splits: Number of vertical splits of each frame.
        contrast_std: Standard deviation of the contrast augmentation.
        brightness_std: Standard deviation of the brightness augmentation.
        gaussian_white_noise: Standard deviation of the pixel-wise gaussian white noise.
        gamma_std: Standard deviation of the gamma augmentation.
        center_crop_fraction: Fraction of the image to keep after cropping.
        indices: Indices of the sequences to include.
        unittest: If True, only renders a single sequence.

    Attributes:
        cached_sequences (List[Dict[str, torch.Tensor]]): List of preprocessed sequences
            for fast dataloading.
        valid_flip_axes (List[int]): List of valid flip axes.
        valid_rotations (List[int]): List of valid rotation values.
        flip_axes (List[int]): List of axes to flip over.
        n_rotations (List[int]): List of number of rotations to perform.
        temporal_split (bool): Flag for temporally controlled augmentation.
        _built (bool): Flag indicating if the dataset has been built.
        params (List): List of augmentation parameters for each sequence.
        arg_df (pd.DataFrame): DataFrame containing augmentation parameters for each
            sequence.
    """

    cached_sequences: List[Dict[str, torch.Tensor]]
    valid_flip_axes: List[int] = [0, 1, 2, 3]
    valid_rotations: List[int] = [0, 1, 2, 3, 4, 5]

    def __init__(
        self,
        n_frames: int = 19,
        flip_axes: List[int] = [0, 1],
        n_rotations: List[int] = [0, 1, 2, 3, 4, 5],
        build_stim_on_init: bool = True,
        temporal_split: bool = False,
        augment: bool = True,
        dt: float = 1 / 50,
        tasks: List[Literal["flow", "depth", "lum"]] = ["flow"],
        interpolate: bool = True,
        all_frames: bool = False,
        random_temporal_crop: bool = False,
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        contrast_std: Optional[float] = None,
        brightness_std: Optional[float] = None,
        gaussian_white_noise: Optional[float] = None,
        gamma_std: Optional[float] = None,
        center_crop_fraction: float = 0.7,
        indices: Optional[List[int]] = None,
        unittest: bool = False,
        **kwargs,
    ):
        if any([arg not in self.valid_flip_axes for arg in flip_axes]):
            raise ValueError(f"invalid flip axes {flip_axes}")

        if any([arg not in self.valid_rotations for arg in n_rotations]):
            raise ValueError(f"invalid rotations {n_rotations}")

        super().__init__(
            tasks=tasks,
            interpolate=interpolate,
            n_frames=n_frames,
            dt=dt,
            augment=augment,
            all_frames=all_frames,
            resampling=True,
            random_temporal_crop=random_temporal_crop,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            p_flip=0,
            p_rot=0,
            contrast_std=contrast_std,
            brightness_std=brightness_std,
            gaussian_white_noise=gaussian_white_noise,
            gamma_std=gamma_std,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
            _init_cache=True,
        )
        self.indices = np.array(indices) if indices is not None else None
        self.flip_axes = flip_axes
        self.n_rotations = n_rotations
        self.temporal_split = temporal_split

        self.config.update({
            'flip_axes': self.flip_axes,
            'n_rotations': self.n_rotations,
            'temporal_split': self.temporal_split,
            'indices': self.indices,
        })

        self._built = False
        if build_stim_on_init:
            self._build()
            self._built = True

    def _build(self):
        """Build augmented dataset with temporal splits and geometric augmentations."""
        # to deterministically apply temporal augmentation/binning of sequences
        # into ceil(sequence_length / n_frames) bins
        (
            self.cached_sequences,
            self.original_repeats,
        ) = temporal_split_cached_samples(
            self.cached_sequences, self.n_frames, split=self.temporal_split
        )

        vsplit_index, original_index, name = (
            self.arg_df[["index", "original_index", "name"]]
            .values.repeat(self.original_repeats, axis=0)
            .T
        )
        tsplit_index = np.arange(len(self.cached_sequences))

        n_frames = [d["lum"].shape[0] for d in self.cached_sequences]

        self.params = [
            (*p[0], p[1], p[2])
            for p in list(
                product(
                    zip(
                        name,
                        original_index,
                        vsplit_index,
                        tsplit_index,
                        n_frames,
                    ),
                    self.flip_axes,
                    self.n_rotations,
                )
            )
        ]

        self.arg_df = pd.DataFrame(
            self.params,
            columns=[
                "name",
                "original_index",
                "vertical_split_index",
                "temporal_split_index",
                "frames",
                "flip_ax",
                "n_rot",
            ],
        )
        # breakpoint()
        # apply deterministic geometric augmentation
        cached_sequences = {}
        for i, (_, _, _, sample, _, flip_ax, n_rot) in enumerate(self.params):
            self.flip.axis = flip_ax
            self.rotate.n_rot = n_rot
            cached_sequences[i] = {
                key: self.rotate(self.flip(value))
                for key, value in self.cached_sequences[sample].items()
            }
        self.cached_sequences = cached_sequences

        if self.indices is not None:
            self.cached_sequences = [self.cached_sequences[i] for i in self.indices]
            self.arg_df = self.arg_df.iloc[self.indices]
            self.params = [self.params[i] for i in self.indices]

        # disable deterministically applied augmentation, such that in case
        # self.augment is True, the other augmentation types can be applied
        # randomly
        self.flip.augment = False
        self.rotate.augment = False
        # default to cropping 0 to n_frames
        self.temporal_crop.random = False
        if self.temporal_split:
            self.temporal_crop.augment = False

    def _original_length(self) -> int:
        """Return the original number of sequences before splitting."""
        return len(self) // self.vertical_splits

    def pad_nans(
        self, data: Dict[str, torch.Tensor], pad_to_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Pad the data with NaNs to a specified length.

        Args:
            data: Dictionary containing the data to pad.
            pad_to_length: Length to pad the data to.

        Returns:
            Padded data dictionary.
        """
        if pad_to_length is not None:
            data = {}
            for key, value in data.items():
                # pylint: disable=not-callable
                data[key] = nnf.pad(
                    value,
                    pad=(0, 0, 0, 0, 0, pad_to_length),
                    mode="constant",
                    value=np.nan,
                )
            return data
        return data

    def get_item(
        self, key: int, pad_to_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            key: Index of the item to retrieve.
            pad_to_length: Length to pad the data to.

        Returns:
            Dictionary containing the retrieved data.
        """
        if self.augment:
            return self.pad_nans(
                self.apply_augmentation(self.cached_sequences[key], n_rot=0, flip_axis=0),
                pad_to_length,
            )
        return self.pad_nans(self.cached_sequences[key], pad_to_length)
