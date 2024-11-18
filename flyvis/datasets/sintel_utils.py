import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import flyvis
from flyvis.utils.dataset_utils import download_url_to_file

from .rendering.utils import split

logger = logging.getLogger(__name__)


def load_sequence(
    path: Path,
    sample_function: Callable,
    start: int = 0,
    end: Optional[int] = None,
    as_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calls sample_function on each file in the sorted path and returns
    a concatenation of the results.

    Args:
        path: Path to the directory containing the sequence files.
        sample_function: Function to apply to each file in the sequence.
        start: Starting index for file selection.
        end: Ending index for file selection.
        as_tensor: If True, returns a PyTorch tensor; otherwise, returns a NumPy array.

    Returns:
        Concatenated sequence data as either a PyTorch tensor or NumPy array.
    """
    samples = []
    for p in sorted(path.iterdir())[start:end]:
        samples.append(sample_function(p))
    samples = np.array(samples)
    if as_tensor:
        return torch.tensor(samples, dtype=torch.float32)
    return samples


def sample_lum(path: Path) -> np.ndarray:
    """Sample luminance data from an image file.

    Args:
        path: Path to the image file.

    Returns:
        Normalized luminance data as a NumPy array.
    """
    lum = np.float32(Image.open(path).convert("L")) / 255
    return lum


def sample_flow(path: Path) -> np.ndarray:
    """Sample optical flow data from a file.

    Note: Flow is in units of pixel / image_height and with inverted negative y
    coordinate (i.e. y-axis pointing upwards in image plane).

    Args:
        path: Path to the flow data file.

    Returns:
        Optical flow data as a NumPy array.
    """
    with open(path, "rb") as f:
        _, w, h = np.fromfile(f, np.int32, count=3)
        data = np.fromfile(f, np.float32, count=(h * w * 2))
        uv = np.reshape(data, (h, w, 2)) / h  # why are we dividing by h?
        # we invert the y coordinate, which points from the top of the
        # image plane to the bottom
        return uv.transpose(2, 0, 1) * np.array([1, -1])[:, None, None]


def sample_depth(filename: Path) -> np.ndarray:
    """Sample depth data from a file.

    Args:
        filename: Path to the depth data file.

    Returns:
        Depth data as a NumPy array.
    """
    with open(filename, "rb") as f:
        _, width, height = np.fromfile(f, dtype=np.int32, count=3)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def temporal_split_cached_samples(
    cached_sequences: List[Dict[str, torch.Tensor]], max_frames: int, split: bool = True
) -> Tuple[List[Dict[str, torch.Tensor]], np.ndarray]:
    """Deterministically split sequences in time dimension into regularly binned
        sequences.

    Note:
        Overlapping splits of sequences which lengths are not an integer multiple of
        `max_frames` contain repeating frames.

    Args:
        cached_sequences: Ordered list of dicts of sequences of shape
            (n_frames, n_features, n_hexals).
        max_frames: Maximum number of frames per split.
        split: Whether to perform the temporal split.

    Returns:
        Tuple containing:
        - List of dictionaries with temporally split sequences.
        - Array of original indices for each new split.
    """
    if split:
        seq_lists = {k: [] for k in cached_sequences[0]}

        splits_per_seq = []
        for i, sequence in enumerate(cached_sequences):
            for key, value in sequence.items():
                splits = temporal_split_sequence(value, max_frames)
                seq_lists[key].extend([*splits])
            splits_per_seq.append([i, len(splits)])

        split_cached_sequences = []
        for i in range(len(seq_lists["lum"])):
            split_cached_sequences.append({k: v[i] for k, v in seq_lists.items()})

        index, repeats = np.array(splits_per_seq).T
        return split_cached_sequences, repeats
    return cached_sequences, np.ones(len(cached_sequences)).astype(int)


def temporal_split_sequence(
    sequence: Union[np.ndarray, torch.Tensor], max_frames: int
) -> Union[np.ndarray, torch.Tensor]:
    """Split a sequence along the temporal dimension.

    Args:
        sequence: Array or tensor of shape (n_frames, n_features, n_hexals).
        max_frames: Maximum number of frames per split.

    Returns:
        Array or tensor of shape (splits, max_frames, n_features, n_hexals).

    Notes:
        The number of splits is computed as int(np.round(n_frames / max_frames)).
    """
    n_frames, _, _ = sequence.shape
    splits = np.round(n_frames / max_frames).astype(int)
    if splits <= 1:
        return sequence[:max_frames][None]
    return split(
        sequence.transpose(0, -1),  # splits along last axis
        max_frames,
        splits,
        center_crop_fraction=None,
    ).transpose(1, -1)  # cause first will be splits, second will be frames


def remove_nans(responses: np.ndarray) -> List[np.ndarray]:
    """Remove NaNs from responses array.

    Args:
        responses: Array of shape (sample, frames, channels).

    Returns:
        List of arrays with NaNs removed, potentially of different sizes.
    """
    _resp = []
    for r in responses:
        _isnan = np.isnan(r).any(axis=1)
        _resp.append(r[~_isnan].squeeze())
    return _resp


@dataclass
class SintelMeta:
    lum_paths: List[Path]
    flow_paths: List[Path]
    depth_paths: List[Path]
    sequence_indices: np.ndarray
    frames_per_scene: np.ndarray
    sequence_index_to_splits: Dict[int, np.ndarray]


def sintel_meta(
    rendered: "flyvis.RenderedSintel",
    sintel_path: Path,
    n_frames: int,
    vertical_splits: int,
    render_depth: bool,
) -> SintelMeta:
    """Returns a dataclass with meta information about the (rendered) sintel dataset.

    Args:
        rendered: RenderedSintel object containing the rendered data.
        sintel_path: Path to the Sintel dataset.
        n_frames: Number of frames to consider for each sequence.
        vertical_splits: Number of vertical splits for each frame.
        render_depth: Whether depth data is rendered.

    Returns:
        Meta dataclass containing metadata about the Sintel dataset.
    """

    lum_paths = []
    sequence_indices = []
    frames_per_scene = []
    sequence_index_to_splits = {}
    for i, p in enumerate(sorted((sintel_path / "training/final").iterdir())):
        if len(list(p.iterdir())) - 1 >= n_frames and any(
            p.name in key for key in rendered
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
    return SintelMeta(
        lum_paths=lum_paths,
        flow_paths=flow_paths,
        depth_paths=depth_paths,
        sequence_indices=sequence_indices,
        frames_per_scene=frames_per_scene,
        sequence_index_to_splits=sequence_index_to_splits,
    )


def original_train_and_validation_indices(
    dataset: "flyvis.MultiTaskSintel",
) -> Tuple[List[int], List[int]]:
    """Get original training and validation indices for the dataloader.

    Returns:
        Tuple containing lists of train and validation indices.
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
        for i, name in enumerate(dataset.arg_df.name)
        if any([scene_name in name for scene_name in train])
    ]
    val_indices = [
        i
        for i, name in enumerate(dataset.arg_df.name)
        if any([scene_name in name for scene_name in _validation])
    ]
    # these were dropped by the pytorch dataload because of the chosen
    # batchsize in the original training run
    val_indices.remove(37)
    val_indices.remove(38)
    return train_indices, val_indices


def download_sintel(delete_if_exists: bool = False, depth: bool = False) -> Path:
    """Download the sintel dataset.

    Args:
        delete_if_exists: If True, delete the dataset if it exists and download again.
        depth: If True, download the depth dataset as well.

    Returns:
        Path to the sintel dataset.
    """
    sintel_dir = flyvis.sintel_dir
    sintel_dir.mkdir(parents=True, exist_ok=True)

    def exists(depth: bool = False) -> bool:
        try:
            assert sintel_dir.exists()
            assert (sintel_dir / "training").exists()
            assert (sintel_dir / "test").exists()
            assert (sintel_dir / "training/flow").exists()
            if depth:
                assert (sintel_dir / "training/depth").exists()
            return True
        except AssertionError:
            return False

    def download_and_extract(url: str, depth: bool = False) -> None:
        sintel_zip = sintel_dir / Path(url).name

        if not exists(depth=depth) or delete_if_exists:
            logger.info("Downloading Sintel dataset.")
            assert not sintel_zip.exists()
            download_url_to_file(url, sintel_zip)
            logger.info("Extracting Sintel dataset.")
            with zipfile.ZipFile(sintel_zip, "r") as zip_ref:
                zip_ref.extractall(sintel_dir)
        else:
            logger.info("Found Sintel at %s", sintel_dir)

    download_and_extract(
        "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip", depth=False
    )
    if depth:
        download_and_extract(
            "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip",
            depth=True,
        )

    assert exists(depth)

    return sintel_dir
