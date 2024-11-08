"""Base classes for all dynamic stimuli datasets."""

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import torch

from flyvis.datasets.augmentation.temporal import get_temporal_sample_indices
from flyvis.utils.dataset_utils import get_random_data_split
from flyvis.utils.df_utils import where_dataframe

__all__ = ["SequenceDataset", "StimulusDataset", "MultiTaskDataset"]


class SequenceDataset(torch.utils.data.Dataset):
    """Base class for all sequence datasets.

    All sequence datasets can subclass this class. They are expected to implement
    the following attributes and methods.

    Attributes:
        framerate (int): Framerate of the original sequences.
        dt (float): Sampling and integration time constant.
        t_pre (float): Warmup time.
        t_post (float): Cooldown time.
        arg_df (pd.DataFrame): required DataFrame containing the dataset parameters.
    """

    arg_df: pd.DataFrame = None
    dt: float = None
    t_pre: float = None
    t_post: float = None

    def get_item(self, key: int) -> Any:
        """Return an item of the dataset.

        Args:
            key: Index of the item to retrieve.

        Returns:
            The dataset item at the specified index.
        """

    def __len__(self) -> int:
        """Size of the dataset."""
        return len(self.arg_df)

    def __getitem__(self, key: Union[slice, Iterable, int, np.int_]) -> Any:
        """Implements advanced indexing.

        Args:
            key: Index, slice, or iterable of indices.

        Returns:
            The dataset item(s) at the specified index/indices.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the key type is invalid.
        """
        return getitem(self, key)

    def get_temporal_sample_indices(
        self, n_frames: int, total_seq_length: int, augment: bool = None
    ) -> torch.Tensor:
        """Returns temporal indices to sample from a sequence.

        Args:
            n_frames: Number of sequence frames to sample from.
            total_seq_length: Total sequence length.
            augment: If True, picks the start frame at random. If False, starts at 0.

        Returns:
            Tensor of temporal indices.

        Note:
            Interpolates between start_index and start_index + n_frames and rounds the
            resulting float values to integer to create indices. This can lead to
            irregularities in terms of how many times each raw data frame is sampled.
        """
        augment = augment if augment is not None else getattr(self, "augment", False)
        framerate = getattr(self, "original_framerate", 1 / self.dt)
        return get_temporal_sample_indices(
            n_frames, total_seq_length, framerate, self.dt, augment
        )


class StimulusDataset(SequenceDataset):
    """Base class for stimulus datasets."""

    def get_stimulus_index(self, kwargs: Dict[str, Any]) -> int:
        """Get the sequence id for a set of arguments.

        Args:
            kwargs: Dictionary containing independent arguments or parameters
                describing the sample of the dataset.

        Returns:
            The sequence id for the given arguments.

        Raises:
            ValueError: If arg_df attribute is not specified.

        Note:
            The child dataset implements the specific method:
            ```python
            def get_stimulus_index(self, arg1, arg2, ...):
                return StimulusDataset.get_stimulus_index(locals())
            ```
            with locals() specifying kwargs in terms of `arg1`, `arg2`, ...
            to index arg_df.
        """
        if getattr(self, "arg_df", None) is None:
            raise ValueError("arg_df attribute not specified.")

        if "self" in kwargs:
            del kwargs["self"]

        return where_dataframe(self.arg_df, **kwargs).item()


class MultiTaskDataset(SequenceDataset):
    """Base class for all (multi-)task sequence datasets.

    All (multi-)task sequence datasets can subclass this class. They are expected
    to implement the following additional attributes and methods.

    Attributes:
        tasks (List[str]): A list of all tasks.
        augment (bool): Turns augmentation on and off.
    """

    tasks: List[str] = []
    augment: bool = False

    @contextmanager
    def augmentation(self, abool: bool) -> None:
        """Contextmanager to turn augmentation on or off in a code block.

        Args:
            abool: Boolean value to set augmentation.

        Example:
            ```python
            with dataset.augmentation(True):
                for i, data in enumerate(dataloader):
                    ...  # all data is augmented
            ```
        """
        _prev = self.augment
        self.augment = abool
        try:
            yield
        finally:
            self.augment = _prev

    def get_random_data_split(
        self, fold: int, n_folds: int, shuffle: bool = True, seed: int = 0
    ) -> np.ndarray:
        """Returns a random data split.

        Args:
            fold: Current fold number.
            n_folds: Total number of folds.
            shuffle: Whether to shuffle the data.
            seed: Random seed for reproducibility.

        Returns:
            Array of indices for the data split.
        """
        return get_random_data_split(
            fold,
            n_samples=len(self),
            n_folds=n_folds,
            shuffle=shuffle,
            seed=seed,
        )


def getitem(cls: Any, key: Union[slice, Iterable, int, np.int_]) -> Any:
    """Implements advanced indexing for a dataset.

    Args:
        cls: The dataset instance.
        key: Index, slice, or iterable of indices.

    Returns:
        The item(s) at the specified index/indices.

    Raises:
        IndexError: If the index is out of range.
        TypeError: If the key type is invalid.
    """
    if isinstance(key, slice):
        return [cls[ii] for ii in range(*key.indices(len(cls)))]
    elif isinstance(key, Iterable):
        return [cls[i] for i in key]
    elif isinstance(key, (int, np.int_)):
        if key < 0:
            key += len(cls)
        if key < 0 or key >= len(cls):
            raise IndexError(f"The index ({key}) is out of range.")
        return cls.get_item(key)
    else:
        raise TypeError("Invalid argument type.")
