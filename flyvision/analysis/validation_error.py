import inspect
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

import flyvision

logging = logging.getLogger(__name__)


@torch.no_grad()
def test(
    network,
    decoder,
    dataloader,
    loss_fns,
    dt,
    t_pre=0.0,
    loss_kwargs={},
    t_fade_in=1.0,
    only_original_frames=False,  # evaluate at framerate of the movie and skip the resampled frames
    # to make a fair comparison to vanillaCNN who start at n_frames - 1
    start_frame=None,
):
    """Tests the network on another dataloader.

    Args:
        network
        decoer
        dataloader
        loss_fns (List): list of objective functions


    Returns:
        val_loss (array): (1, #loss functions)
        task_loss (Dict[str, array]): validation losses per task
    """
    network.eval()

    for _decoder in decoder.values():
        _decoder.eval()

    dataset = dataloader.dataset
    steady_state = network.steady_state(
        t_pre=t_pre,
        dt=dt,
        batch_size=dataloader.batch_size,
        value=0.5,
        initial_frames=None,
        state=None,
        no_grad=True,
    )
    losses = {task: [] for task in dataset.tasks}  # type: Dict[str, List]
    stimulus = flyvision.Stimulus(network.connectome, 0, 0, _init=False)
    stimulus.skip_network = network.skip_network

    with dataset.augmentation(False):
        for i, data in enumerate(dataloader):
            if only_original_frames:
                samples = data["frame_samples"]
                original_indices = torch.Tensor([
                    torch.where(samples == s)[0][0] for s in torch.unique(samples)
                ]).long()
            else:
                # to sample all indices
                original_indices = slice(start_frame, None)

            fade_in_state = network.fade_in_state(
                t_fade_in=t_fade_in,
                dt=dt,
                initial_frames=data["lum"][:, 0],
                state=steady_state,
            )

            # Resets the stimulus buffer (#frames, #samples, #neurons).
            # The number of frames and samples can change, but the number of nodes is constant.
            n_samples, n_frames, _, _ = data["lum"].shape
            stimulus.zero(n_samples, n_frames)

            # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor stimuli.
            stimulus.add_input(data["lum"])

            with stimulus.memory_friendly():
                # Run stimulus through network.
                activity = network(stimulus(), dt, state=fade_in_state)

            # Decode activity and evaluate loss.
            for task in dataset.tasks:
                y = data[task][:, original_indices]
                y_est = decoder[task](activity)[:, original_indices]
                losses[task].append([
                    fn(y, y_est, **loss_kwargs).detach().cpu().item() for fn in loss_fns
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


def get_loss_fns(loss_fns):
    if loss_fns:
        return loss_fns
    return [
        flyvision.objectives.L2Norm(),
        flyvision.objectives.EPE(),
    ]


def compute_original_validation_error(
    solver,
    validation_subwrap="validation_v2",
    loss_fns=None,
    dt=1 / 50,
    t_pre=0.25,
    t_fade_in=0.0,
    drop_last=True,
    start_frame=None,
):
    """Compute the validation error for the hardcoded fixed set of sequences
    below.
    """
    loss_fns = get_loss_fns(loss_fns)
    # original sorting
    if drop_last:
        val_sequences = [
            39,
            40,
            41,
            48,
            49,
            50,
            6,
            7,
            8,
            27,
            28,
            29,
            21,
            22,
            23,
            36,
            # 37,  # was dropped with bs 4 and drop_last
            # 38,  # was dropped with bs 4 and drop_last
        ]
    else:
        val_sequences = [
            39,
            40,
            41,
            48,
            49,
            50,
            6,
            7,
            8,
            27,
            28,
            29,
            21,
            22,
            23,
            36,
            37,  # was dropped with bs 4 and drop_last
            38,  # was dropped with bs 4 and drop_last
        ]
    dataloader = DataLoader(
        solver.task.dataset,
        batch_size=1 if not drop_last else solver.task.batch_size,
        num_workers=0,
        sampler=dvs.datasets.IndexSampler(val_sequences),
        # drop_last=True,
    )

    dataloader.dataset.dt = dt

    loss = []
    for chkpt in solver.wrap.chkpt_index[:]:
        solver.recover(checkpoint=chkpt, recover_optimizer=False)
        network = solver.network
        decoder = solver.decoder
        loss.append(
            test(
                network=network,
                decoder=decoder,
                dataloader=dataloader,
                loss_fns=loss_fns,
                dt=dt,
                t_pre=t_pre,
                t_fade_in=t_fade_in,
                start_frame=start_frame,
            )[0]
        )

    loss = np.array(loss)

    for i, fn in enumerate(loss_fns):
        # if fn.__name__ in ["epe", "l2"]:
        #     solver.wrap[validation_subwrap].loss = loss[:, i]
        # else:
        solver.wrap[validation_subwrap][fn.__name__] = loss[:, i]

    dvs.utils.write_meta(
        solver.wrap[validation_subwrap].path,
        dict(
            spec=dvs.Namespace(
                dt=dt,
                t_pre=t_pre,
                t_fade_in=t_fade_in,
                loss_fns=[fn.__name__ for fn in loss_fns],
                validation_subwrap=validation_subwrap,
                validation_function=inspect.currentframe().f_code.co_name,
            ),
            status="done",
        ),
    )
    return loss
