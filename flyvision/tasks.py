from typing import Dict
from torch.utils.data import DataLoader, sampler
from toolz import valmap
import logging

from datamate import Namespace

from flyvision.connectome import ConnectomeDir
from flyvision.utils.dataset_utils import IndexSampler
from flyvision.utils.class_utils import forward_subclass
from flyvision.datasets.datasets import MultiTaskDataset
from flyvision.decoder import ActivityDecoder
from flyvision.objectives import Objective


class Task:
    """Defines a task for a multi-task dataset from configurations."""

    def __init__(
        self,
        dataset: Namespace,
        decoder: Namespace,
        loss: Namespace,
        batch_size=4,
        n_iters=250_000,
        n_folds=4,
        fold=1,
        seed=0,
    ):

        logging.info("Initializing task with configurations:")
        logging.info(f"dataset: {dataset}")
        logging.info(f"decoder: {decoder}")
        logging.info(f"loss: {loss}")
        logging.info(f"batch_size: {batch_size}")
        logging.info(f"n_iters: {n_iters}")
        logging.info(f"n_folds: {n_folds}")
        logging.info(f"fold: {fold}")
        logging.info(f"seed: {seed}")

        self.batch_size = batch_size
        self.n_iters = n_iters
        self.n_folds = n_folds
        self.fold = fold
        self.seed = seed
        self.decoder = decoder

        # Initialize dataset.
        self.dataset = forward_subclass(
            MultiTaskDataset, dataset
        )  # type: MultiTaskDataset
        self.dataset.losses = Namespace(
            {task: forward_subclass(Objective, config) for task, config in loss.items()}
        )

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
        logging.info(
            f"Initialized dataloader with training sequence indices \n {self.train_seq_index}"
        )

        self.val_data = DataLoader(
            self.dataset,
            batch_size=1,
            sampler=IndexSampler(self.val_seq_index),
        )
        logging.info(
            f"Initialized dataloader with validation sequence indices \n {self.val_seq_index}"
        )

        # Initialize overfitting loader.
        self.overfit_data = DataLoader(self.dataset, sampler=IndexSampler([0]))

    def init_decoder(self, connectome):
        return _init_decoder(self.decoder, connectome)


class IndexSampler(sampler.Sampler):
    """Samples the provided indices in sequence."""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def _init_decoder(
    config: Dict, connectome: ConnectomeDir
) -> Dict[str, ActivityDecoder]:
    """Initialize decoders.

    Returns:
        Dict[str, ActivityDecoder]: A dictionary of decoders.

    Example config:

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
    """

    config = config.deepcopy()

    def init(conf):
        return forward_subclass(ActivityDecoder, {**conf, "connectome": connectome})

    decoder = valmap(init, config)

    return decoder
