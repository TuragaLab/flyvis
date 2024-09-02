"""Base classes for all dynamic stimuli datasets."""

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import torch

from flyvision.augmentation import temporal
from flyvision.objectives import Loss
from flyvision.utils.dataset_utils import get_random_data_split

__all__ = ["SequenceDataset", "StimulusDataset", "MultiTaskDataset"]


class SequenceDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    """Base class for all sequence datasets.

    All sequence datasets should subclass this class.

    Subclasses must override:
        framerate: framerate of the original sequences.
        dt: sampling and integration time constant.
        n_sequences: number of sequences in the dataset.
        augment: turns augmentation on and off.
        t_pre: warmup time.
        t_post cooldown time.
        get_item: return an item of the dataset.
    """

    @property
    @abstractmethod
    def framerate(self) -> int:
        """Framerate of the original sequences."""
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """Sampling and integration time constant."""
        pass

    @property
    @abstractmethod
    def n_sequences(self) -> int:
        """Number of sequences in the dataset."""
        pass

    @property
    @abstractmethod
    def augment(self) -> bool:
        """Turns augmentation on and off."""
        pass

    @contextmanager
    def augmentation(self, abool: bool) -> None:
        """Contextmanager to turn augmentation on or off in a code block.

        Example usage:
            >>> with dataset.augmentation(True):
            >>>    for i, data in enumerate(dataloader):
            >>>        ...  # all data is augmented
        """
        _prev = self.augment
        self.augment = abool
        try:
            yield
        finally:
            self.augment = _prev

    @property
    @abstractmethod
    def t_pre(self) -> float:
        """Warmup time."""
        pass

    @property
    @abstractmethod
    def t_post(self) -> float:
        """Cooldown time."""
        pass

    @abstractmethod
    def get_item(self, key: int) -> Any:
        """Return an item of the dataset."""
        pass

    def __len__(self) -> int:
        """Size of the dataset."""
        return self.n_sequences

    def __getitem__(self, key: Union[slice, Iterable, int, np.int_]) -> Any:
        """Implements advanced indexing."""
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, Iterable):
            return [self[i] for i in key]
        elif isinstance(key, (int, np.int_)):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item(key)
        else:
            raise TypeError("Invalid argument type.")

    def get_temporal_sample_indices(
        self, n_frames: int, total_seq_length: int, augment: bool = None
    ) -> torch.Tensor:
        """Returns temporal indices to sample from a sequence.

        Args:
            n_frames: number of sequence frames to sample from.
            total_seq_length: total sequence length.
            framerate: original framerate of the sequence.
            dt: sampling time constant.
            augment: if True, picks the start frame at random. If False,
                starts at 0.

        Note: interpolates between start_index and start_index + n_frames and
        rounds the resulting float values to integer to create indices. This can
        lead to irregularities in terms of how many times each raw data frame is
        sampled.
        """
        augment = augment if augment is not None else self.augment
        return temporal.get_temporal_sample_indices(
            n_frames, total_seq_length, self.framerate, self.dt, augment
        )


class StimulusDataset(SequenceDataset, metaclass=ABCMeta):
    @property
    @abstractmethod
    def arg_df(self) -> pd.DataFrame:
        """Table storing sequence id and associated arguments, or parameters."""
        pass

    @staticmethod
    def param_indices(all_params, requested_params):
        """Returns the indices for where requested_params occur in all_params."""
        if requested_params is None:
            return slice(None)
        return np.array([
            [i] for i, param in enumerate(all_params) if param in requested_params
        ])

    def _get_sequence_id_from_arguments(self, args: Dict[str, Any]) -> int:
        """Get the sequence id for a set of arguments.

        Note: that is the key passed to get_item.

        args (Dict): dictionary that contains independent arguments,
            or parameters, describing the sample of the dataset.
            Those arguments must be specified in self.arg_df.

        The child dataset implements a specific method
            ```def get_arg_df(arg1, arg2, ...):
                return self._get_arg_index(locals())
            ``` with locals() specifying args in terms of `arg1`, `arg2`, ...
            to index arg_df.
        """

        if getattr(self, "arg_df", None) is None:
            raise ValueError("arg_df attribute not specified.")

        def _query_from_args(args):
            if "self" in args:
                del args["self"]
            _query_start = "{}=={}"
            _query_append = "& {}=={}"
            return "".join(
                (
                    _query_start.format(key, value)
                    if i == 0
                    else _query_append.format(key, value)
                )
                for i, (key, value) in enumerate(args.items())
            )

        query = _query_from_args(args)
        return self.arg_df.query(query).index.item()

    @abstractmethod
    def get_sequence_id_from_arguments(self) -> int:
        """Return the index for a stimulus sample by parameter.

        Note, a child of StimulusDataset implements this function and the
        function parameters are the independent parameters defining the stimulus
        sample.
        """
        pass


class MultiTaskDataset(SequenceDataset):
    """Base class for all (multi-)task sequence datasets.

    All (multi-)task sequence datasets should subclass this class.

    Subclasses must override (from SequenceDataset):
        framerate: framerate of the original sequences.
        dt: sampling and integration time constant.
        n_sequences: number of sequences in the dataset.
        augment: turns augmentation on and off.
        t_pre: warmup time.
        t_post cooldown time.
        get_item: return an item of the dataset.

    Additional:
        tasks: a list of all tasks.
        task_weights: a weighting of each task.
        task_weights_sum: sum of all indicated task weights to normalize loss.
        losses: a loss function for each task.
    """

    @property
    @abstractmethod
    def tasks(self) -> List[str]:
        """A list of all tasks."""
        pass

    @property
    @abstractmethod
    def task_weights(self) -> Dict[str, float]:
        """A weighting of each task."""
        pass

    @property
    @abstractmethod
    def task_weights_sum(self) -> float:
        """Sum of all indicated task weights to normalize loss."""
        pass

    def _init_task_weights(self, task_weights: Dict[str, float]) -> None:
        if task_weights is not None:
            self.task_weights = {task: task_weights[task] for task in self.tasks}
            self.task_weights_sum = sum(self.task_weights.values())
        else:
            self.task_weights = {task: 1 for task in self.tasks}
            self.task_weights_sum = len(self.tasks)

    @property
    @abstractmethod
    def losses(self) -> Dict[str, Loss]:
        """A loss function for each task."""
        pass

    # pylint: disable=redefined-builtin
    def loss(
        self, input: torch.Tensor, target: torch.Tensor, task: str, **kwargs
    ) -> torch.Tensor:
        """Returns the task loss multiplied with the task weight."""
        return (
            self.task_weights[task]
            * self.losses[task](input, target, **kwargs)
            / self.task_weights_sum
        )

    def _original_length(self) -> int:
        """Override to return the original length of the dataset, e.g. in case
        its derived from splitting sequences."""
        return self.__len__()

    def get_random_data_split(self, fold, n_folds, shuffle=True, seed=0):
        """Returns a random data split."""
        return get_random_data_split(
            fold,
            n_samples=self._original_length(),
            n_folds=n_folds,
            shuffle=shuffle,
            seed=seed,
        )


def get_temporal_sample_indices(
    n_frames: int,
    total_seq_length: int,
    framerate: int,
    dt: float,
    augment: bool,
) -> torch.Tensor:
    """Returns temporal indices to sample from a sequence.

    Args:
        n_frames: number of sequence frames to sample from.
        total_seq_length: total sequence length.
        framerate: original framerate of the sequence.
        dt: sampling time constant.
        augment: if True, picks the start frame at random. If False,
            starts at 0.

    Note: interpolates between start_index and start_index + n_frames and
    rounds the resulting float values to integer to create indices. This can
    lead to irregularities in terms of how many times each raw data frame is
    sampled.
    """
    if n_frames > total_seq_length:
        raise ValueError(
            f"cannot interpolate between {n_frames} frames from a total"
            f" of {total_seq_length} frames"
        )
    start = 0
    if augment:
        last_valid_start = total_seq_length - n_frames or 1
        start = np.random.randint(low=0, high=last_valid_start)
    return torch.arange(start, start + n_frames - 1e-6, dt * framerate).long()


def getitem(cls, key):
    if isinstance(key, slice):
        # get the start, stop, and step from the slice
        return [cls[ii] for ii in range(*key.indices(len(cls)))]
    elif isinstance(key, Iterable):
        return [cls[i] for i in key]
    elif isinstance(key, (int, np.int_)):
        # handle negative indices
        if key < 0:
            key += len(cls)
        if key < 0 or key >= len(cls):
            raise IndexError("The index (%d) is out of range." % key)
        # get the data from direct index
        return cls.get_item(key)
    else:
        raise TypeError("Invalid argument type.")
