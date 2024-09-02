import logging
from typing import Dict

from datamate import Namespace
from toolz import valmap
from torch.utils.data import DataLoader, sampler

from flyvision.connectome import ConnectomeDir
from flyvision.datasets.datasets import MultiTaskDataset
from flyvision.decoder import ActivityDecoder
from flyvision.objectives import Loss
from flyvision.utils.class_utils import forward_subclass
from flyvision.utils.dataset_utils import IndexSampler


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
        original_split=False,
    ):
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.n_folds = n_folds
        self.fold = fold
        self.seed = seed
        self.decoder = decoder

        # Initialize dataset.
        self.dataset = forward_subclass(MultiTaskDataset, dataset)  # type: MultiTaskDataset
        self.dataset.losses = Namespace({
            task: forward_subclass(Loss, config) for task, config in loss.items()
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
            "Initialized dataloader with training sequence indices \n"
            "{self.train_seq_index}"
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
            "Initialized dataloader with validation sequence indices \n"
            "{self.val_seq_index}"
        )

        # Initialize overfitting loader.
        self.overfit_data = DataLoader(self.dataset, sampler=IndexSampler([0]))

    def init_decoder(self, connectome):
        return _init_decoder(self.decoder, connectome)


def _init_decoder(config: Dict, connectome: ConnectomeDir) -> Dict[str, ActivityDecoder]:
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
