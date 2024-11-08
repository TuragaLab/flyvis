import inspect
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import flyvis
from flyvis.datasets import MultiTaskDataset
from flyvis.network import Network, NetworkView
from flyvis.task.objectives import epe, l2norm
from flyvis.utils.class_utils import forward_subclass

logging = logging.getLogger(__name__)

__all__ = ["validate", "validate_all_checkpoints"]


@torch.no_grad()
def validate(
    network: Network,
    decoder,
    dataloader,
    loss_fns,
    dt,
    t_pre=0.0,
    loss_kwargs={},
):
    """Tests the network and decoder on the dataloader with the given loss functions."""
    network.eval()

    for _decoder in decoder.values():
        _decoder.eval()

    dataset = dataloader.dataset
    steady_state = network.steady_state(
        t_pre=t_pre,
        dt=dt,
        batch_size=dataloader.batch_size,
        value=0.5,
        state=None,
        grad=False,
    )
    losses = {task: [] for task in dataset.tasks}  # type: Dict[str, List]
    stimulus = network.stimulus

    with dataset.augmentation(False):
        for _, data in enumerate(dataloader):
            # Resets the stimulus buffer (#frames, #samples, #neurons).
            # The number of frames and samples can change, but the number of nodes is
            # constant.
            n_samples, n_frames, _, _ = data["lum"].shape
            stimulus.zero(n_samples, n_frames)

            # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor
            # stimuli.
            stimulus.add_input(data["lum"])

            # Run stimulus through network.
            activity = network(stimulus(), dt, state=steady_state)

            # Decode activity and evaluate loss.
            for task in dataset.tasks:
                y = data[task]
                y_est = decoder[task](activity)
                losses[task].append([
                    fn(y_est, y, **loss_kwargs).detach().cpu().item() for fn in loss_fns
                ])

    summed_loss = 0
    task_loss = {}
    # Record loss per task (+rec).
    for task in losses:
        # (#samples, #loss_functions)
        loss = np.array(losses[task])
        sample_average_loss = np.mean(loss, axis=0)
        if f"loss_{task}" not in task_loss:
            task_loss[f"loss_{task}"] = []
        task_loss[f"loss_{task}"].append(sample_average_loss)
        summed_loss += sample_average_loss
    # Record average loss.
    val_loss = summed_loss / len(losses)

    network.train()
    for _decoder in decoder.values():
        _decoder.train()

    return val_loss, task_loss


def validate_all_checkpoints(
    network_view: NetworkView,
    loss_fns=None,
    dt=1 / 50,
    t_pre=0.5,
    validation_subdir="validation",
):
    dataset = forward_subclass(MultiTaskDataset, network_view.dir.config.task.dataset)
    loss_fns = get_loss_fns(loss_fns)
    network_view.init_network()
    network_view.init_decoder()
    _, val_sequences = dataset.original_train_and_validation_indices()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        sampler=flyvis.utils.dataset_utils.IndexSampler(val_sequences),
        drop_last=False,
    )

    dataloader.dataset.dt = dt

    loss = []
    progress = tqdm(total=len(network_view.dir.chkpt_index))
    for chkpt in network_view.checkpoints.indices:
        network = network_view.network(checkpoint=chkpt)
        decoder = network_view.init_decoder(
            checkpoint=chkpt, decoder=network_view.decoder
        )
        loss.append(
            validate(
                network=network.network,
                decoder=decoder,
                dataloader=dataloader,
                loss_fns=loss_fns,
                dt=dt,
                t_pre=t_pre,
            )[0]
        )
        progress.update(1)
    progress.close()

    loss = np.array(loss)

    for i, fn in enumerate(loss_fns):
        network_view.dir[validation_subdir][fn.__name__] = loss[:, i]

    network_view.dir[validation_subdir].config = dict(
        dt=dt,
        t_pre=t_pre,
        loss_fns=[fn.__name__ for fn in loss_fns],
        validation_subdir=validation_subdir,
        validation_function=inspect.currentframe().f_code.co_name,
    )

    return loss


def get_loss_fns(loss_fns):
    if loss_fns:
        return loss_fns
    return [l2norm, epe]
