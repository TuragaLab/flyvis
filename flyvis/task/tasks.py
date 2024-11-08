import logging
from typing import Dict

import torch
from datamate import Namespace
from toolz import valmap
from torch.utils.data import DataLoader, sampler

from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.datasets.datasets import MultiTaskDataset
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.dataset_utils import IndexSampler

from . import objectives
from .decoder import ActivityDecoder


class Task:
    """Defines a task for a multi-task dataset from configurations.

    Args:
        dataset: Configuration for the dataset.
        decoder: Configuration for the decoder.
        loss: Configuration for the loss functions.
        batch_size: Size of each batch. Defaults to 4.
        n_iters: Number of iterations. Defaults to 250,000.
        n_folds: Number of folds for cross-validation. Defaults to 4.
        fold: Current fold number. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to 0.
        original_split: Whether to use the original data split. Defaults to False.

    Attributes:
        batch_size: Size of each batch.
        n_iters: Number of iterations.
        n_folds: Number of folds for cross-validation.
        fold: Current fold number.
        seed: Random seed for reproducibility.
        decoder: Configuration for the decoder.
        dataset (MultiTaskDataset): The initialized multi-task dataset.
        losses (Namespace): Loss functions for each task.
        train_seq_index (List[int]): Indices of training sequences.
        val_seq_index (List[int]): Indices of validation sequences.
        train_data (DataLoader): DataLoader for training data.
        train_batch (DataLoader): DataLoader for a single training batch.
        val_data (DataLoader): DataLoader for validation data.
        val_batch (DataLoader): DataLoader for a single validation batch.
        overfit_data (DataLoader): DataLoader for overfitting on a single sample.
    """

    def __init__(
        self,
        dataset: Namespace,
        decoder: Namespace,
        loss: Namespace,
        task_weights: Dict[str, float] = None,
        batch_size: int = 4,
        n_iters: int = 250_000,
        n_folds: int = 4,
        fold: int = 1,
        seed: int = 0,
        original_split: bool = False,
    ):
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.n_folds = n_folds
        self.fold = fold
        self.seed = seed
        self.decoder = decoder

        # Initialize dataset.
        self.dataset = forward_subclass(MultiTaskDataset, dataset)
        self.task_weights, self.task_weights_sum = self.init_task_weights(task_weights)

        self.losses = Namespace({
            task: getattr(objectives, config) for task, config in loss.items()
        })

        if original_split:
            self.train_seq_index, self.val_seq_index = (
                self.dataset.original_train_and_validation_indices()
            )
        else:
            self.train_seq_index, self.val_seq_index = self.dataset.get_random_data_split(
                fold, n_folds, seed
            )

        # Initialize dataloaders.
        self.train_data = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler.SubsetRandomSampler(self.train_seq_index),
            drop_last=True,
        )
        self.train_batch = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=IndexSampler(self.train_seq_index[:batch_size]),
            drop_last=False,
        )
        logging.info(
            "Initialized dataloader with training sequence indices \n%s",
            self.train_seq_index,
        )

        self.val_data = DataLoader(
            self.dataset,
            batch_size=1,
            sampler=IndexSampler(self.val_seq_index),
        )
        self.val_batch = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=IndexSampler(self.val_seq_index[:batch_size]),
        )
        logging.info(
            "Initialized dataloader with validation sequence indices \n%s",
            self.val_seq_index,
        )

        # Initialize overfitting loader.
        self.overfit_data = DataLoader(self.dataset, sampler=IndexSampler([0]))

    def init_decoder(
        self, connectome: ConnectomeFromAvgFilters
    ) -> Dict[str, ActivityDecoder]:
        """Initialize the decoder.

        Args:
            connectome: The connectome directory.

        Returns:
            A dictionary of initialized decoders.
        """
        return init_decoder(self.decoder, connectome)

    def loss(
        self, input: torch.Tensor, target: torch.Tensor, task: str, **kwargs
    ) -> torch.Tensor:
        """Returns the task loss multiplied with the task weight.

        Args:
            input: Input tensor.
            target: Target tensor.
            task: Task name.
            **kwargs: Additional keyword arguments for the loss function.

        Returns:
            Weighted task loss.
        """
        return (
            self.task_weights[task]
            * self.losses[task](input, target, **kwargs)
            / self.task_weights_sum
        )

    def init_task_weights(self, task_weights: Dict[str, float]) -> Dict[str, float]:
        """Returns the task weights.

        Returns:
            A dictionary of task weights.
        """
        task_weights = (
            task_weights
            if task_weights is not None
            else {task: 1 for task in self.dataset.tasks}
        )

        return task_weights, sum(task_weights.values())


def init_decoder(
    config: Dict, connectome: ConnectomeFromAvgFilters
) -> Dict[str, ActivityDecoder]:
    """Initialize decoders.

    Args:
        config: Configuration for the decoders.
        connectome: The connectome directory.

    Returns:
        A dictionary of decoders.

    Example:
        ```python
        decoder = Namespace(
            flow=Namespace(
                type="DecoderGAVP",
                shape=[8, 2],
                kernel_size=5,
                const_weight=0.001,
                p_dropout=0.5,
            ),
            depth=Namespace(
                type="DecoderGAVP",
                shape=[8, 1],
                kernel_size=5,
                const_weight=0.001,
                p_dropout=0.5,
            ),
            lum=Namespace(
                type="DecoderGAVP",
                shape=[8, 3],
                n_out_features=2,
                kernel_size=5,
                const_weight=0.001,
                p_dropout=0.5,
            ),
            shared=False,
        )
        ```
    """
    config = config.deepcopy()

    def init(conf):
        return forward_subclass(ActivityDecoder, {**conf, "connectome": connectome})

    decoder = valmap(init, config)

    return decoder
