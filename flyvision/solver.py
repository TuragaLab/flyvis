"""
#TODO: docstring
"""
import math
import time
from pathlib import Path
import shutil
from typing import Mapping, List, Dict, Any, Union, Optional, Iterable
import json
import logging


logging = logging.getLogger("dvs")

import numpy as np
import torch
from toolz import valmap
from torch.utils.data import DataLoader
from tqdm import tqdm

import dvs
from dvs.networks import Network
from dvs.tasks import Task, Conditions
from dvs.datasets.wraps import init_network_wrap, past
from dvs.datasets import IndexSampler, Gratings, Flashes, TwoBarFlashes
from dvs.penalizer import Penalty
from dvs.scheduler import HyperParamScheduler
from dvs.utils.datawrapper import Datawrap, Namespace, namespacify
from dvs.stimulus import Stimulus
from dvs.utils.wrap_utils import detach_cpu, write_meta
from dvs import analysis
from dvs.animations.animations import AnimatePlotFn
import dvs.analysis


class BaseSolver:
    """Initialize or recover a solver for a type of NetworkWrap.

    A solver aggregates required wraps and modules.
    It implements methods for training and testing,
    checkpointing and recovering checkpoints.

    Args:
        name (str)
        config (dict or Namespace): contains all required configurations.
                Required if an existing NetworkWrap is initialized.
                Optional if an existing NetworkWrap is recovered (will be overwritten by calling 'recover'.)
    """

    def __init__(self, name="", config=None):
        NotImplemented

    def __setattr__(self, key, value):
        if key == "dt":
            if self.task is not None:
                self.task.dataset.dt = value
            elif hasattr(self, "conditions"):
                self.conditions.dt = value
        object.__setattr__(self, key, value)

    def __getattr__(self, key):
        if key == "dt":
            if self.task is not None:
                return self.task.dataset.dt
            elif hasattr(self, "conditions"):
                return self.conditions.dt
        return object.__getattribute__(self, key)


class MultiTaskSolver(BaseSolver):
    """
    Args:
        name (str or path)
        config (dict or Namespace): contains all required configurations.
                Required if an existing NetworkWrap is initialized.
                Optional if an existing NetworkWrap is recovered.
        init_network (bool): defaults toT
        init_decoder (bool)

    Attributes:
        wrap (Datawrap): network specific datawrap.
        network (nn.Module): the fly visual system network module.
        decoder (Mapping[str, torch.nn.Module]): decoder specified for the tasks.
        task (Task): task Configurable.
        optimizer (torch.optim.Optimizer): optimizer for both network and decoder parameters.
        checkpoints (List[int]): a list of all integer identifiers for checkpoints.

    Note: change root dir by dvs.set_root_dir(dvs.exp_dir / <task>) before
    initialization.
    """

    wrap: Datawrap = None
    network: Network = None
    decoder: Mapping[str, torch.nn.Module] = Namespace()
    task: Task = None
    optimizer: object = None
    penalty: object = None
    scheduler: object = None
    checkpoints: List[int] = []
    iteration: int = 0
    stimulus: object = None

    def __init__(
        self,
        name="",
        config=None,
        duplicate="",
        init_network=True,
        init_decoder=True,
        init_task=True,
        init_optim=True,
        init_penalties=True,
        init_scheduler=True,
        delete_if_exists=False,  # never change!
        parse_past_config=False,
        use_wandb=False,
    ):
        if duplicate:
            # Load configuration from another experiment.
            wrap, _ = init_network_wrap(duplicate, None, delete_if_exists=False)
            config = wrap.meta.spec
        self.wrap, self.wrap_dir = init_network_wrap(
            name, config, delete_if_exists=delete_if_exists
        )
        self.config = (
            namespacify(config) or self.wrap.spec
        )  # given config or stored config

        if parse_past_config:
            parse_past_config(self.config)

        self.iteration = 0
        self.finetune_by_iter = getattr(self.config, "finetune_by_iter", False)
        self._val_loss = 1.0e15
        self._last_chkpt_ind = -1
        self._curr_chkpt_ind = -1
        self._initialized = _init_solver(
            self,
            init_network=init_network,
            init_decoder=init_decoder,
            init_task=init_task,
            init_optim=init_optim,
            init_penalties=init_penalties,
            init_scheduler=init_scheduler,
        )
        logging.info("Initialized solver.")
        logging.info(repr(self.config))
        self.checkpoints, _ = _stored_checkpoints(self.wrap.chkpts.path)

        self.nnv = analysis.NetworkViews(tnn=self.wrap)
        # self.anim_flash_responses = AnimatePlotFn(self.nnv.flash_response_grid,
        #                                           path=self.nnv.tnn.path / "analysis",
        #                                           dpi=100,
        #                                           fname="flash_responses_over_chkps",
        #                                           delete_if_exists=False,
        #                                           framerate=4)
        # self.anim_motion_tuning = AnimatePlotFn(self.nnv.motion_tuning_grid,
        #                                          path=self.nnv.tnn.path / "analysis",
        #                                          dpi=100,
        #                                          fname="motion_tuning_over_chkpts",
        #                                          delete_if_exists=False,
        #                                          framerate=4)
        self.use_wandb = use_wandb
        if self.use_wandb:
            try:
                # to save some time at library import, because
                # loading takes relativelt long
                import wandb

                project = f"dvs_{'_'.join(self.task.dataset.tasks)}"
                if len(name.split("/")) == 2:
                    group = name.split("/")[0]
                    id = name.split("/")[1]
                    resume = "None" if delete_if_exists else "allow"
                    if not (self.wrap.path / "wandb").exists():
                        (self.wrap.path / "wandb").mkdir()
                    self.wandb = wandb.init(
                        project=project,
                        group=group,
                        job_type="_".join(self.task.dataset.tasks),
                        id=id,
                        resume=resume,
                        dir=self.wrap.path,
                        notes=self.config.comment,
                        config=self.config,
                    )
                    self.wandb.watch(
                        (self.network, *(d for d in self.decoder.values()))
                    )
                    logging.info("Initialized wandb.")
                else:
                    logging.info("Could not initialize wandb.")
                    self.use_wandb = False
            except:
                self.use_wandb = False

        logging.info(
            f"Network parameter requiring grad: {[p[0] for p in self.network.named_parameters() if p[1].requires_grad]}"
        )
        logging.info(
            f"Decoder parameter requiring grad: { {task:[p[0] for p in self.decoder[task].named_parameters() if p[1].requires_grad] for task in self.decoder} }"
        )

    def recover(
        self,
        recover_network=True,
        recover_decoder=True,
        recover_optimizer=True,
        recover_penalty=True,
        strict_recover=True,
        checkpoint=-1,  # -1 for last, 'best' for best based on validation
        other=None,
        force=False,
        validation_subwrap="original_validation_v2",  # required if checkpoint == 'best'
        loss_name="epe",
    ):
        """
        Recovers a solver state.
        """
        if checkpoint == "best":
            loss_name = dvs.analysis.validation_error._check_loss_name(
                self.wrap[validation_subwrap], loss_name
            )
            checkpoint = np.argmin(self.wrap[validation_subwrap][loss_name][:])
            logging.info(
                f"Checkpoint {checkpoint} found best performing based on {validation_subwrap}."
            )
        # to store them later in the stimulus response configs
        self._chkpt_validation_subwrap = validation_subwrap
        self._recovered_chkpt = checkpoint
        checkpoint = int(checkpoint)
        _recover_solver(
            self,
            checkpoint=checkpoint,
            recover_network=recover_network,
            recover_decoder=recover_decoder,
            recover_optimizer=recover_optimizer,
            recover_penalty=recover_penalty,
            strict=strict_recover,
            other=None,
            force=force,
        )

    def train(self, overfit=False, initial_checkpoint=True):
        """Trains the network by backprop through time.
        Args:
            overfit (bool): If true the dataloader is substituted by a
                single-sequence loader and augmentation is turned off. Defaults to False.
            initial_checkpoint (bool): to disable the initial checkpoint when debugging.
        Raises:
            OverflowError: raised if the activity or loss reports Nan values for more than 100 iterations.
        Stores:
            wrap.loss
            wrap.loss_<task>
            wrap.activity
            wrap.activity_min
            wrap.activity_max
        """
        # return None is iterations have already been trained.
        if self.iteration >= self.n_iters:
            return

        # to debug code within the training loop the initial checkpoint should be
        # disabled
        if initial_checkpoint:
            self.checkpoint()

        logging.info("Starting training.")
        # The overfit_seq dataloader only contains single sequence and
        # can thus be used for overfitting. This is just meant to debug the model architecture.
        dataloader = self.task.overfit_seq if overfit else self.task.train_data
        # For overfitting we also turn the augmentation off.
        augment = False if overfit else True

        # The number of full presentations of the training data is derived from the
        # preset number of training iterations, the length of the dataloader and the current iteration.
        n_epochs = np.ceil((self.n_iters - self.iteration) / len(dataloader)).astype(
            int
        )

        chkpt_every = _chkpt_every_epoch(self.n_iters - self.iteration, len(dataloader))

        # Workaround, to be able to resume and don't loose any data to wrap attributes that
        # are not extended here, this code first retrieves the potentially stored lists
        def _get_scalar(wrap, key):
            return wrap[key][()] if key in wrap else 0

        def _get_array(wrap, key):
            return wrap[key][:] if key in wrap else []

        loss_over_iters = [*_get_array(self.wrap, "loss")]
        activity_over_iters = [*_get_array(self.wrap, "activity")]
        activity_min_over_iters = [*_get_array(self.wrap, "activity_min")]
        activity_max_over_iters = [*_get_array(self.wrap, "activity_max")]
        loss_per_task = {
            f"loss_{task}": [*_get_array(self.wrap, f"loss_{task}")]
            for task in self.task.dataset.tasks
        }

        _use_initial_frame_state = self.config.get("use_initial_frame_state", False)
        _steady_state_per_iter = self.config.get("steady_state_per_iter", False)

        start_time = time.time()
        with self.task.dataset.augmentation(augment):
            for epoch in range(n_epochs):
                steady_state = None
                # the default is to compute a steady state for each epoch, then
                # it's computed here. otherwise inside the closure handle_batch
                # TODO: this should be outsourced to keep track of actual
                # parameter updates inside networks that require recomputation
                # of the steady state to make decoder finetuning more efficient.
                if not (_steady_state_per_iter or _use_initial_frame_state):
                    steady_state = self.network.steady_state(
                        t_pre=self.config.get("t_pre_train", 0.5),
                        dt=self.task.dataset.dt,
                        batch_size=dataloader.batch_size,
                        value=0.5,
                    )

                for i, data in enumerate(dataloader):

                    def handle_batch():
                        """Closure (in the hope) to free memory for checkpointing efficiently."""

                        # Resets the stimulus buffer (samples, frames, neurons).
                        n_samples, n_frames, _, _ = data["lum"].shape
                        if _steady_state_per_iter:
                            initial_state = self.network.steady_state(
                                t_pre=self.config.get("t_pre_train", 0.5),
                                dt=self.task.dataset.dt,
                                batch_size=dataloader.batch_size,
                                value=0.5,
                            )
                        elif _use_initial_frame_state:
                            # this uses self.stimulus, so it's important that the stimulus.zero() is called afterwards again
                            initial_state = self.network.steady_state(
                                t_pre=self.config.get("t_pre_train", 0.5),
                                dt=self.task.dataset.dt,
                                batch_size=n_samples,
                                value=None,
                                initial_frames=data["lum"][:, 0],
                            )
                        else:
                            initial_state = steady_state

                        self.stimulus.zero(n_samples, n_frames)

                        # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor stimuli.
                        self.stimulus.add_input(data["lum"])

                        # Reset gradients.
                        self.optimizer.zero_grad()

                        with self.stimulus.memory_friendly():
                            # Run stimulus through network.
                            activity = self.network(
                                self.stimulus(),
                                self.task.dataset.dt,
                                state=initial_state,
                            )

                        losses = {task: 0 for task in self.task.dataset.tasks}
                        for task in self.task.dataset.tasks:
                            y = data[task]
                            y_est = self.decoder[task](activity)
                            loss_kwargs = {
                                **getattr(
                                    self.decoder[task], "loss_kwargs", {}
                                ),  # learnable loss kwargs such as std are part of the decoding.
                                **data.get("loss_kwargs", {}),
                            }
                            losses[task] = self.task.dataset.loss(
                                y, y_est, task, **loss_kwargs
                            )

                        loss = sum(losses.values()) / self.task.dataset.task_weights_sum
                        # Update parameters.
                        loss.backward(retain_graph=True)

                        if getattr(self.config, "clip_grad", None):
                            logging.info(
                                "Clipping gradients with tc: 200, bias: 100, syn_strength: 5000."
                            )
                            torch.nn.utils.clip_grad_value_(
                                self.network.nodes_time_const, 2500
                            )
                            torch.nn.utils.clip_grad_value_(
                                self.network.nodes_bias, 750
                            )
                            torch.nn.utils.clip_grad_value_(
                                self.network.edges_syn_strength, 100_000
                            )

                        self.optimizer.step()

                        # logging.info(("Epoch/iteration:"
                        #                f"{epoch}/{self.iteration} - "
                        #                f"loss: {loss.detach().cpu()}"))

                        # Activity and parameter dependent penalties.
                        self.penalty(activity)
                        # self.penalty(steady_state)

                        # Log results.
                        loss = loss.detach().cpu()
                        # self.wrap.extend("loss", [loss])
                        # self.wrap.extend("activity", [activity])

                        for task in self.task.dataset.tasks:
                            loss_per_task[f"loss_{task}"].append(
                                losses[task].detach().cpu()
                            )
                            if self.use_wandb:
                                self.wandb.log(
                                    {
                                        "training loss": {
                                            task: losses[task].detach().cpu()
                                        }
                                    }
                                )

                        loss_over_iters.append(loss)
                        activity = activity.detach().cpu()
                        mean_activity = activity.mean()
                        activity_over_iters.append(mean_activity)
                        activity_min_over_iters.append(activity.min())
                        activity_max_over_iters.append(activity.max())
                        if self.use_wandb:
                            self.wandb.log(
                                {
                                    "activity_mean": mean_activity,
                                    "activity_min": activity.min(),
                                    "activity_max": activity.max(),
                                }
                            )

                        return loss, mean_activity

                    # Call closure.
                    loss, activity = handle_batch()

                    # Increment iteration count.
                    self.iteration += 1

                    # _tqdm.update()

                # logging.info(f"Finished epoch {epoch}.")

                # Interrupt training if the network explodes.
                # TODO: make this optimal
                if torch.isnan(loss) or torch.isnan(activity):
                    logging.warning("Network exploded.")
                    raise OverflowError("Invalid values encountered in trace.")

                # The scheduling of hyperparams are functions of the iteration
                # However, we allow steps only after full presentations of the data.
                if epoch + 1 != n_epochs:
                    self.scheduler(self.iteration)
                    logging.info(
                        f"Scheduled paremeters for iteration {self.iteration}."
                    )

                if (epoch % chkpt_every == 0) or (epoch + 1 == n_epochs):
                    self.wrap.loss = loss_over_iters
                    self.wrap.activity = activity_over_iters
                    self.wrap.activity_min = activity_min_over_iters
                    self.wrap.activity_max = activity_max_over_iters

                    for task in self.task.dataset.tasks:
                        self.wrap[f"loss_{task}"] = loss_per_task[f"loss_{task}"]

                    logging.info("Storing gradients from last batch.")
                    # Store leftover gradients from last batch.
                    for name, param in self.network.named_parameters():
                        if param.grad is not None:
                            self.wrap.gradients.extend(name, [detach_cpu(param.grad)])
                            self.wrap.gradients.extend(
                                name + "_norm", [detach_cpu(param.grad.norm(2))]
                            )
                            self.wrap.parameters.extend(name, [detach_cpu(param)])

                    self.checkpoint()

                if (
                    self.finetune_by_iter is not False
                    and self.iteration >= self.finetune_by_iter
                ):
                    _init_finetuning(self)
                    self.finetune_by_iter = False
                    logging.info("Initialized finetuning.")

                logging.info("Finished epoch.")

            # _tqdm.close()

        time_elapsed = time.time() - start_time
        self.wrap.time_trained = time_elapsed + _get_scalar(self.wrap, "time_trained")

        # self.anim_flash_responses.finalize()
        # self.anim_motion_tuning.finalize()
        logging.info("Finished training.")

    # @torch.no_grad()
    # def initial_state(
    #     self,
    #     t_pre=None,
    #     dt=None,
    #     batch_size=None,
    #     value=0.5,
    #     initial_frames=None,
    #     state=None,
    #     no_grad=True,
    # ):
    #     """Returns the network state after grey scale or initial frame stimulus."""
    #     t_pre = t_pre or self.task.dataset.t_pre
    #     dt = dt or self.task.dataset.dt
    #     batch_size = batch_size or self.task.batch_size

    #     if value is not None and initial_frames is None:
    #         return self.network.steady_state(
    #             t_pre,
    #             self.dt,
    #             batch_size,
    #             state=state,
    #             no_grad=no_grad,
    #             value=value,
    #             initial_frames=initial_frames,
    #         )
    #     # case 2: initial frames to accomodate to are specified
    #     elif initial_frames is not None and value is None:
    #         if self.config.get("use_initial_frame_state", False):
    #             return self.network.steady_state(
    #                 t_pre,
    #                 self.dt,
    #                 batch_size,
    #                 state=state,
    #                 no_grad=no_grad,
    #                 value=value,
    #                 initial_frames=initial_frames,
    #             )
    #     else:
    #         return state

    def checkpoint(self):
        """Creates a checkpoint.

        Validates on the validation data calling ~self.test.
        Validates on a training batch calling ~self.track_batch.
        Tracks a gradient if self.config.track_grad is True (experimental).
        Stores a checkpoint of the network, decoder and optimizer parameters using
        pytorch's pickle function.

        Stores:
            wrap.chkpt_index (List): numerical identifier of the checkpoint.
            wrap.chkpt_iter (List): iteration at which this checkpoint was recorded.
            wrap.best_chkpt_index (int): chkpt index at which the val loss is minimal.
        """
        torch.cuda.synchronize()
        logging.info("Synchronized cuda.")
        self._last_chkpt_ind += 1
        self._curr_chkpt_ind += 1

        # Tracking of validation loss and training batch loss.
        logging.info("Test on validation data.")
        val_loss = self.test(
            dataloader=self.task.val_data, mode="validation", track_loss=True
        )
        if self.use_wandb:
            self.wandb.log({"validation loss": val_loss})

        logging.info("Test on training data.")
        _ = self.test(dataloader=self.task.train_data, mode="training", track_loss=True)

        # logging.info("Tracking batch.")
        # self.track_batch(dataloader=self.task.tracked_train_batch,
        #                  mode="tracked_train_batch",
        #                  track_loss=True)

        # Store gradients.
        # if self.config.track_grad:
        #     self.track_grad(dataloader=self.task.tracked_train_batch,
        #                     mode="tracked_train_batch")

        logging.info("Saving state dicts.")
        # Store state of pytorch modules.
        nn_state_dict = self.network.state_dict()
        dec_state_dict = {}
        if self.decoder:
            dec_state_dict = valmap(lambda x: x.state_dict(), self.decoder)
        chkpt = {
            "network": nn_state_dict,
            "decoder": dec_state_dict,
            "optim": self.optimizer.state_dict(),
            "time": time.ctime(),
            "val_loss": val_loss,
            "iteration": self.iteration - 1,
            "dt": self.task.dataset.dt,
        }
        if hasattr(self, "penalty"):
            chkpt.update(self.penalty._chkpt())
        torch.save(chkpt, self.wrap.path / f"chkpts/chkpt_{self._last_chkpt_ind:05}")

        # Store network params in wrap.
        logging.info("Updating network params in wrap.")
        _wrap_network_params(self.network, self.wrap)
        if self.decoder:
            _wrap_decoder_params(self.decoder, self.wrap)

        # Append chkpt index.
        self.checkpoints.append(self._last_chkpt_ind)
        self.wrap.extend("chkpt_index", [self._last_chkpt_ind])
        self.wrap.extend("chkpt_iter", [self.iteration - 1])
        self.wrap.dt = self.task.dataset.dt

        # Overwrite best val loss.
        if val_loss < self._val_loss:
            self.wrap.best_chkpt_index = self._last_chkpt_ind
            self._val_loss = val_loss

        if self._curr_chkpt_ind > 0:
            # Plotting losses.
            try:
                analysis.network_views.loss(self.nnv)
            except:
                pass

        # self.stimulus_response(
        #     stimulus="flashes",
        #     save_whole_layers=False,
        #     delete_if_exists=True,
        # )
        # self.nnv.init_flashes()
        # # self.anim_flash_responses(0, 1, 6)
        # # plot_flash_responses(self.nnv,
        # #                      filename=f"flashes/flash_responses_chkpt_{self._last_chkpt_ind}")

        # movingedge_config = Namespace(
        #     widths=[80],  # in 1 * radians(2.25)
        #     offsets=(-6, 7),  # in 1 * radians(2.25)
        #     intensities=[0, 1],
        #     speeds=[4.8, 13, 19],  # in 1 * radians(5.8) / s
        #     height=80,  # in 1 * radians(2.25)
        #     bar_loc_horizontal=np.radians(0),
        #     post_pad_mode="continue",
        # )

        # movingedge_data = dvs.datasets.Movingbar(
        #     **movingedge_config,
        #     dt=1 / 200,
        #     device="cuda",
        #     t_pre=1.0,
        #     t_post=1.0,
        # )

        # self.stimulus_response(
        #     stimulus="movingbar",
        #     subwrap=f"movingedge",
        #     save_whole_layers=False,
        #     delete_if_exists=True,
        #     t_pre=None,
        #     dataset=movingedge_data,
        #     config=movingedge_config,
        # )

        # self.nnv.init_movingbar("movingedge")
        # # self.anim_motion_tuning()
        # # plot_motion_tuning(self.nnv,
        # #                    filename=f"motion_tuning/chkpt_{self._last_chkpt_ind}")

        # analysis.network_views.stimuli_response_ON_OFF_pathway(
        #     self.nnv, filename=f"pathway_response/chkpt_{self._last_chkpt_ind}"
        # )
        logging.info("Checkpointed.")

    @torch.no_grad()
    def test(
        self,
        dataloader,
        mode="validation",
        track_loss=False,
        t_pre=0.25,  # don't change if still comparing to 92
    ):
        """Tests the network on another dataloader.

        Args:
            dataloader (Dataloader): pytorch Dataloader to test on.
            mode (str): identifier for subwrap. Defaults to 'validation'.
                    Meaning items will be stored under wrap.validation
            track_loss (bool): whether to store the loss.

        Returns:
            float: validation loss

        Stores:
            wrap.<mode>.loss_<task> (List): loss per task averaged over whole dataset.
            wrap.<mode>.iteration (List): iteration, when this was called.
            wrap.<mode>.loss (List): average loss over tasks.
        """
        self._eval()

        # Update hypterparams.
        self.scheduler(self.iteration)

        initial_state = self.network.steady_state(
            t_pre=t_pre,
            dt=self.task.dataset.dt,
            batch_size=dataloader.batch_size,
            value=0.5,
            initial_frames=None,
            state=None,
            no_grad=True,
        )
        losses = {
            task: () for task in self.task.dataset.tasks
        }  # type: Dict[str, Tuple]
        with self.task.dataset.augmentation(False):
            for i, data in enumerate(dataloader):
                # Resets the stimulus buffer (#frames, #samples, #neurons).
                # The number of frames and samples can change, but the number of nodes is constant.
                n_samples, n_frames, _, _ = data["lum"].shape
                # this uses self.stimulus, so it's important that the stimulus.zero() is called afterwards again

                self.stimulus.zero(n_samples, n_frames)

                # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor stimuli.
                self.stimulus.add_input(data["lum"])

                with self.stimulus.memory_friendly():
                    # Run stimulus through network.
                    activity = self.network(
                        self.stimulus(),
                        self.task.dataset.dt,
                        state=initial_state,
                    )

                # Decode activity and evaluate loss.
                for task in self.task.dataset.tasks:
                    y = data[task]
                    y_est = self.decoder[task](activity)
                    loss_kwargs = {
                        **getattr(
                            self.decoder[task], "loss_kwargs", {}
                        ),  # learnable loss kwargs such as std are part of the decoding.
                        **data.get("loss_kwargs", {}),
                    }
                    losses[task] += (
                        self.task.dataset.loss(y, y_est, task, **loss_kwargs)
                        .detach()
                        .cpu()
                        .item(),
                    )

        # Store results.
        if track_loss:
            summed_loss = 0
            # Record loss per task (+rec).
            for task in losses:
                loss = np.mean(losses[task])
                # dvs.utils.wrap_utils._extend(
                #     self.wrap[mode], "loss" + "_" + task, [loss]
                # )
                self.wrap[mode].extend("loss" + "_" + task, [loss])
                summed_loss += loss
            # Record average loss.
            self.wrap[mode].extend("iteration", [self.iteration])
            val_loss = summed_loss / len(losses)
            self.wrap[mode].extend("loss", [val_loss])

        self.track_batch(dataloader, mode, track_loss=False)

        self._train()

        return val_loss

    @torch.no_grad()
    def track_batch(self, dataloader, mode, track_loss=True):
        """Tests the network on another dataloader.

        Args:
            dataloader (Dataloader): pytorch Dataloader to take first batch from.
            mode (str): identifier for subwrap. Defaults to 'tracked_train_batch'.
                    Meaning items will be stored under wrap.tracked_train_batch.
            track_loss (bool): whether to store the loss.

        Stores:
            wrap.<mode>.loss_<task> (List): loss per task on the tracked batch.
            wrap.<mode>.iteration (List): iteration, when this was called.
            wrap.<mode>.loss (List): average loss over tasks.
            wrap.<mode>.x
            wrap.<mode>.y
            wrap.<mode>.y_est
            wrap.<mode>.network_states.nodes
            wrap.<mode>.network_states.edges
            wrap.<mode>.dt
        """
        self._eval()

        # Update hypterparams.
        self.scheduler(self.iteration)

        data_indices = [
            sorted(dataloader.sampler.indices)[i] for i in range(dataloader.batch_size)
        ]
        with dataloader.dataset.augmentation(False):
            data = dataloader.collate_fn([dataloader.dataset[i] for i in data_indices])

        # Resets the stimulus buffer (#frames, #samples, #neurons).
        # The number of frames and samples can change, but the number of nodes is constant.
        n_samples, n_frames, _, _ = data["lum"].shape

        steady_state = self.network.steady_state(
            t_pre=self.config.get("t_pre_train", 0.5),
            dt=self.task.dataset.dt,
            batch_size=dataloader.batch_size,
            value=0.5
            if not self.config.get("use_initial_frame_state", False)
            else None,
            initial_frames=None,
            state=None,
            no_grad=True,
        )
        initial_state = self.network.steady_state(
            t_pre=self.config.get("t_pre_train", 0.5),
            dt=self.task.dataset.dt,
            batch_size=n_samples,
            value=None,
            initial_frames=data["lum"][:, 0]
            if self.config.get("use_initial_frame_state", False)
            else None,
            state=steady_state,
            no_grad=True,
        )

        with self.stimulus.memory_friendly():
            self.stimulus.zero(n_samples, n_frames)
            # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor stimuli.
            self.stimulus.add_input(data["lum"])
            # Run stimulus through network.
            activity = self.network(
                self.stimulus(),
                self.task.dataset.dt,
                state=initial_state,
            )
        summed_loss = 0.0
        y_all = {}
        y_est_all = {}
        for task in self.task.dataset.tasks:
            y = data[task]
            y_est = self.decoder[task](activity)
            loss_kwargs = {
                **getattr(
                    self.decoder[task], "loss_kwargs", {}
                ),  # learnable loss kwargs such as std are part of the decoding.
                **data.get("loss_kwargs", {}),
            }
            loss = (
                self.task.dataset.loss(y, y_est, task, **loss_kwargs)
                .detach()
                .cpu()
                .item()
            )
            summed_loss += loss
            y_all[task] = detach_cpu(y)
            y_est_all[task] = detach_cpu(y_est)

            # Record loss per task.
            if track_loss:
                self.wrap[mode].extend("loss" + "_" + task, [loss])

        # Record average loss.
        if track_loss:
            self.wrap[mode].extend("iteration", [self.iteration])
            self.wrap[mode].extend("loss", [summed_loss / len(self.task.dataset.tasks)])

        self.wrap[mode].x = data["lum"].detach().cpu()
        self.wrap[mode].y = y_all
        self.wrap[mode].y_est = y_est_all
        self.wrap[mode].network_states.nodes.activity = activity.detach().cpu()
        self.wrap[mode].dt = self.task.dataset.dt

        # self.extend_stored_activity(activity,
        #                             save_whole_layers=False,
        #                             subwrap=mode,
        #                             identifier=f"activity_over_chkpts_{self.task.dataset.dt*1000:.0F}ms")

        # self.wrap[mode].extend("dt_over_chkpts", [self.task.dataset.dt])

        self._train()

    def track_grad(self, dataloader, mode="tracked_train_batch"):
        return NotImplementedError
        """Tracks the gradient on a data batch.

        Note, this is experimental. Mason recommends backward hooks.

        Args:
            dataloader (Dataloader): pytorch Dataloader to take first batch from.
            mode (str): identifier for subwrap. Defaults to 'tracked_train_batch'.
                    Meaning items will be stored under wrap.tracked_train_batch.

        Stores:
            wrap.<mode>.gradients.<task>.<parameter>
        """
        self._eval()
        self.scheduler(self.iteration)

        with self.task.dataset.augmentation(False):
            data = next(iter(dataloader))

        # Resets the stimulus buffer (#frames, #samples, #neurons).
        # The number of frames and samples can change, but the number of nodes is constant.
        n_samples, n_frames, _, _ = data["lum"].shape
        self.stimulus.zero(n_samples, n_frames)

        # Add batch of hex-videos (#frames, #samples, #hexals) as photorecptor stimuli.
        self.stimulus.add_input(data["lum"])

        with self.stimulus.memory_friendly():
            # Run stimulus through network.
            activity = self.network(
                self.stimulus(),
                self.task.dataset.dt,
                state=self.initial_state(n_samples),
            )

        # For each task decode the activity of the visual system and measure the loss.
        gradients = {}
        for task in self.task.dataset.tasks:
            gradients[task] = {}
            self.optimizer.zero_grad()
            y = data[task]
            y_est = self.decoder[task](activity)
            loss_kwargs = {
                **getattr(
                    self.decoder[task], "loss_kwargs", {}
                ),  # learnable loss kwargs such as std are part of the decoding.
                **data.get("loss_kwargs", {}),
            }
            loss = self.task.dataset.loss(y, y_est, task, **loss_kwargs)
            loss.backward(retain_graph=True)

            for key, params in self.network.named_parameters():
                if params.grad is not None:
                    self.wrap[mode].gradients[task].extend(
                        key, [detach_cpu(params.grad)]
                    )
        self._train()

    def _train(self):
        """Calls the train method of all involved modules."""
        self.network.train()
        if self.decoder is not None:
            for decoder in self.decoder.values():
                decoder.train()

    def _eval(self):
        """Calls the eval method of all involved modules."""
        self.network.eval()
        if self.decoder is not None:
            for decoder in self.decoder.values():
                decoder.eval()

    @torch.no_grad()
    def stimulus_response(
        self,
        stimulus=None,
        dt=1 / 200,
        t_pre=1.0,
        t_fade_in=0.0,
        subwrap=None,
        dataset=None,
        save_pre_stim_activity=True,
        save_whole_layers=True,
        delete_if_exists=False,
        config=None,
    ):
        """Records a stimulus response.

        Args:
            stimulus (str): stimulus name, e.g. gratings or flashes.
                A dataset named stimulus.capitalize() must exist in
                dvs.datasets.
            save_pre_stim_activity (bool): if True stores the activity during grey stimuli.
                    Otherwise, removes it. Defaults to True.
            save_whole_layers (bool): if True, saves to state of the entire network.
                    Otherwise, saves only the central nodes activity. Defaults to False.
            subwrap (str): activity is stored under wrap.<stimulus> or wrap.<subwrap>
                if not None. Defaults to None.
            delete_if_exists (bool): if the subwrap exists, delete it.
                Defaults to False.

        Stores:
            wrap.<stimulus>.network_states.nodes.activity
            wrap.<stimulus>._meta.yaml
        """

        self._eval()

        subwrap = subwrap or stimulus.lower()

        path = self.wrap[subwrap].path  # type: Path
        if path.exists() and delete_if_exists:
            shutil.rmtree(path)
        if not path.exists():
            path.mkdir(parents=True)

        if dataset is None:
            config = config or getattr(self.config, stimulus)
            if config is None:
                logging.info(f"{stimulus.capitalize()} not configured.")
                return

            config.update(dt=dt)
            dataset = dvs.datasets.__dict__[stimulus.capitalize()](**config)

        if config is not None:
            write_meta(path, Namespace(spec=config.deepcopy(), status="done"))

        if not save_pre_stim_activity and dataset is None:
            time_slice = slice(int(config.t_pre / dt), None)
        else:
            time_slice = slice(None)

        dataset.dt = dt

        for activity, _ in self.network.stimulus_response(
            dataset, dt, t_pre=t_pre, t_fade_in=t_fade_in
        ):
            self.extend_stored_activity(
                activity, save_whole_layers, subwrap, time_slice=time_slice
            )

        self._train()

    def extend_stored_activity(
        self,
        activity,
        save_whole_layers,
        subwrap,
        identifier="activity",
        time_slice=slice(None),
    ):
        """
        activity : n_samples, n_frames, n_neurons
        """
        activity = detach_cpu(activity)[:, time_slice]

        if not save_whole_layers:
            identifier += "_central"
            activity = activity[:, :, self.network.ctome.central_nodes_index[:]]

        self.wrap[subwrap].network_states.nodes.extend(
            identifier, [activity[:].squeeze()]
        )


# ---- INITIALIZATION FUNCTIONS


def trained_network(
    name,
    checkpoint=-1,
    validation_subwrap="original_validation_v2",
    validation_loss_name="epe",
    network=None,
):
    """Recover trained network from experiment name."""
    if isinstance(name, Path):
        wrap = Datawrap(name)
    else:
        wrap = Datawrap(dvs.get_exp_dir() / name)
    network = network or Network(
        **dvs.datasets.wraps.past.parse_solver_config(wrap.spec).network
    )
    ids, paths = _stored_checkpoints(wrap.chkpts.path)

    # to return a network even if no checkpoints were found
    if checkpoint is None or (not ids and not paths and checkpoint == 0):
        return network, None, wrap

    if checkpoint == "best":
        checkpoint = np.argmin(wrap[validation_subwrap][validation_loss_name][:])

    checkpoint = int(checkpoint)
    logging.info(f"Checkpoint {paths[checkpoint]} loaded.")

    _recover_network(network, paths[checkpoint])
    if "validation" in wrap:
        dt = wrap.validation.dt[()]
    else:
        dt = wrap.dt[()]
    return network, dt, wrap


def trained_decoder(
    name,
    checkpoint=-1,
    validation_subwrap="original_validation_v2",
    validation_loss_name="epe",
):
    if isinstance(name, Path):
        wrap = Datawrap(name)
    else:
        wrap = Datawrap(dvs.get_exp_dir() / name)

    decoder = _init_decoder(wrap.spec.task.decoder, wrap.ctome)
    ids, paths = _stored_checkpoints(wrap.chkpts.path)

    # to return a network even if no checkpoints were found
    if checkpoint is None or (not ids and not paths and checkpoint == 0):
        return decoder, wrap

    if checkpoint == "best":
        checkpoint = np.argmin(wrap[validation_subwrap][validation_loss_name][:])
    checkpoint = int(checkpoint)

    _recover_decoder(decoder, paths[checkpoint])
    logging.info(f"Checkpoint {paths[checkpoint]} loaded.")
    return decoder, wrap


def trained_network_task(name):
    if isinstance(name, Path):
        wrap = Datawrap(name)
    else:
        wrap = Datawrap(dvs.get_exp_dir() / name)
    task = dvs.solver._init_task(wrap.spec.task)
    return task


def _init_solver(
    cls: MultiTaskSolver,
    init_network=False,
    init_decoder=False,
    init_task=False,
    init_conditions=False,
    init_optim=False,
    init_penalties=False,
    init_scheduler=False,
    checkpoint=-1,
):
    """Initialize solver."""
    initialized = []
    if init_network:
        cls.network = _init_network(cls.config.network)
        # Store prior parameter api for later comparison to trained parameters.
        if not cls.wrap.prior_param_api.path.exists():
            with torch.no_grad():
                cls.wrap.prior_param_api = detach_cpu(
                    cls.network.param_api(as_reftensor=True)
                )
        # It is just good to keep track of the number of free parameters.
        if not (cls.wrap.path / "num_network_params.h5").exists():
            cls.wrap.num_network_params = cls.network.num_parameters
        # To access the ctome data from the Network datawrap,
        # the ctome datawrap and it's meta is copied to it.
        if not (cls.wrap.path / "ctome").exists():
            cls.wrap.ctome = cls.network.ctome
            write_meta(cls.wrap.ctome.path, cls.network.ctome.meta)
        # The stimulus class.
        cls.stimulus = cls.network._stimulus
        # Stimulus(
        #     n_samples=1, n_frames=1, ctome=cls.network.ctome
        # )
        initialized.append("network")

    assert not all([init_task, init_conditions])
    if init_task:
        cls.task = _init_task(cls.config.task)
        cls.task_keys = cls.task.dataset.tasks
        cls.task_weights = cls.task.dataset.task_weights
        cls.n_iters = cls.config.task.n_iters
        if not (cls.wrap.path / "n_train_samples.h5").exists():
            cls.wrap.n_train_samples = len(cls.task.train_data)
        if not (cls.wrap.path / "train_data_index.h5").exists():
            cls.wrap.train_data_index = cls.task.train_seq_index
        if not (cls.wrap.path / "val_data_index.h5").exists():
            cls.wrap.val_data_index = cls.task.val_seq_index
        initialized.append("task")

        if init_decoder:
            cls.decoder = _init_decoder(cls.config.task.decoder, cls.wrap.ctome)
            # It is just good to keep track of the number of parameters.
            if not (cls.wrap.path / "num_decoder_params.h5").exists():
                cls.wrap.num_decoder_params = [
                    decoder.num_parameters for decoder in cls.decoder.values()
                ]
            initialized.append("decoder")

    if init_conditions:
        cls.conditions = _init_conditions(cls.config.conditions)
        cls.task_keys = cls.conditions.task_keys
        cls.task_weights = cls.conditions.task_weights
        cls.n_iters = cls.conditions.n_iters
        initialized.append("conditions")

        if init_decoder:
            cls.decoder = _init_decoder(cls.conditions.decoder_confs, cls.wrap.ctome)
            # It is just good to keep track of the number of parameters.
            if not (cls.wrap.path / "num_decoder_params").exists():
                cls.wrap.num_decoder_params = [
                    decoder.num_parameters for decoder in cls.decoder.values()
                ]
            initialized.append("decoder")

    if init_optim:
        cls.optimizer = _init_optimizer(cls.config, cls.network, cls.decoder)
        initialized.append("optim")

    if init_penalties:
        cls.penalty = _init_penalty(cls)
        initialized.append("penalties")

    if init_scheduler:
        # TODO: because of n_iters this depends on task initialization
        # TODO: therefore init_task=False and init_scheduler=True won't work
        cls.scheduler = HyperParamScheduler(cls)
        cls.scheduler(cls.iteration)
        initialized.append("scheduler")

    return initialized


def _init_network(config):
    """
    See ~dvs.networks.Network.
    """
    return Network(**config)

def _init_task(config):
    """
    See ~dvs.tasks.GenericTask.
    """
    return Task(config)


def _init_conditions(config):
    return Conditions(config)


def _init_optimizer(config, network, decoder):
    """Initializes the mutual optim of network and decoders."""

    def _unique_dec_params(modules):
        """Returns unique decoder parameters, i.e. shared parameters only once."""

        def equals_none(x, parameters):
            return not any([id(x) == id(_x) for _x in parameters])

        optim_params = [
            dict(
                params=list(next(iter(modules.values())).parameters()),
                **config.optim_dec,
            )
        ]
        for i, module in enumerate(modules.values()):
            if i == 0:
                continue
            optim_params.append(
                dict(
                    params=[
                        x
                        for x in module.parameters()
                        if equals_none(x, optim_params[0]["params"])
                    ],
                    **config.optim_dec,
                )
            )
        return optim_params

    config = config.deepcopy()

    # if the optimizer type is specified in both network
    # and decoder and equal, a different one than Adam is used
    network_optim = getattr(config.optim_net, "type", None)
    decoder_optim = getattr(config.optim_dec, "type", None)
    if (
        not any(optim is None for optim in (network_optim, decoder_optim))
        and network_optim == decoder_optim
    ):
        config.optim_dec.pop("type")
        _optim = torch.optim.__dict__[config.optim_net.pop("type")]
    else:
        logging.warning(
            f"Falling back to Adam optimizer (shared). Decoder optimizer was configured as {decoder_optim}. Network optimizer was configured as {network_optim}."
        )
        _optim = torch.optim.Adam
    logging.info(f"Initializing {_optim.__name__} for network and decoder.")

    param_groups = [dict(params=network.parameters(), **config.optim_net)]

    if decoder:
        param_groups.extend(_unique_dec_params(decoder))

    return _optim(param_groups)


def _init_penalty(solver):
    """See ~dvs.penalizer.Penalty."""
    return Penalty(solver)


def _init_scheduler(solver):
    """See ~dvs.scheduler.Scheduler."""
    return HyperParamScheduler(solver)


def _init_gratings(config, dt):
    """See ~dvs.datasets.Gratings."""
    if config:
        config.update(dt=dt)
        gratings_dataset = Gratings(**config)
        return DataLoader(
            gratings_dataset,
            batch_size=1,
            sampler=IndexSampler(np.arange(len(gratings_dataset))),
        )


def _init_flashes(config, dt):
    """See ~dvs.datasets.Flashes."""
    if config:
        config.update(dt=dt)
        flashes_dataset = Flashes(**config)
        return DataLoader(
            flashes_dataset,
            batch_size=1,
            sampler=IndexSampler(np.arange(len(flashes_dataset))),
        )


def _init_two_bar_flashes(config, dt):
    """See ~dvs.datasets.Gratings."""
    if config:
        config.update(dt=dt)
        tbf_dataset = TwoBarFlashes(**config)
        return DataLoader(
            tbf_dataset,
            batch_size=1,
            sampler=IndexSampler(np.arange(len(tbf_dataset))),
        )


def _init_finetuning(cls):
    """Enables or disables gradients to particular parameters."""
    cls.network.edges_syn_strength.requires_grad = False
    cls.network.edges_syn_count.requires_grad = True


# ---- CHECKPOINT RECOVERY FUNCTIONS


def _recover_solver(
    cls,
    checkpoint=-1,
    recover_network=True,
    recover_decoder=True,
    recover_optimizer=True,
    recover_penalty=True,
    strict=True,
    other=None,
    force=False,
):
    if other is not None:
        other_wrap, other_wrap_dir = init_network_wrap(name=other, config=None)
        checkpoints, paths = _stored_checkpoints(other_wrap.chkpts.path)
    else:
        checkpoints, paths = _stored_checkpoints(cls.wrap.chkpts.path)

    if not checkpoints or not any(
        (recover_network, recover_decoder, recover_optimizer, recover_penalty)
    ):
        logging.info("No checkpoint found. Continuing with initialized parameters.")
        return
    # The last checkpoint could be different from the current checkpoint now.
    # The next checkpoints name is infered from _last_chkpt_ind, therefore it's important
    # to keep track of this correctly to not overwrite checkpoints by accident.

    if checkpoints[checkpoint] == cls._curr_chkpt_ind and not force:
        logging.info("Checkpoint already recovered.")
        return

    cls._last_chkpt_ind = checkpoints[-1]
    cls._curr_chkpt_ind = checkpoints[checkpoint]

    checkpoint_data = torch.load(paths[checkpoint])
    logging.info(f"Checkpoint {paths[checkpoint]} loaded.")
    past.parse_penalty_optims_checkpoint(checkpoint_data)

    # TODO!: the last case is wrong since chkpt_every relates to epochs not iterations any more
    cls.iteration = checkpoint_data.pop("iteration", None) or checkpoint_data.pop(
        "t", None
    )
    if "scheduler" in cls._initialized:
        # The scheduler keeps track of the integration time step as a function of the
        # iteration number.
        cls.scheduler(cls.iteration)

    # The _val_loss variable is used to keep track of the best checkpoint,
    # and must therefore be set correctly.
    cls._val_loss = checkpoint_data.pop("val_loss", 1e15)

    if recover_network and "network" in cls._initialized:
        _recover_network(cls.network, checkpoint_data.pop("network"))
    if recover_decoder and "decoder" in cls._initialized:
        _recover_decoder(cls.decoder, checkpoint_data.pop("decoder"), strict=strict)
    if recover_optimizer and "optim" in cls._initialized:
        _recover_optimizer(cls.optimizer, checkpoint_data.pop("optim"))
    if recover_penalty and "penalties" in cls._initialized:
        _recover_penalty_optimizer(
            cls.penalty.optimizers, checkpoint_data.pop("penalty_optims")
        )

    logging.info("Recovered modules.")


@torch.no_grad()
def _wrap_network_params(network, wrap):
    try:
        wrap.param_api = detach_cpu(network.param_api(as_reftensor=True))
        wrap.param_api.edges.weight_trained = (
            wrap.param_api.edges.sign[:]
            * wrap.param_api.edges.syn_strength[:]
            * wrap.param_api.edges.syn_count[:]
        )
    except:
        pass


@torch.no_grad()
def _wrap_decoder_params(decoder, wrap):
    wrap.decoder_params = detach_cpu(valmap(lambda x: x.state_dict(), decoder))


def _recover_network(network, state_or_path):
    """Loads network parameters from state dict.

    Args:
        network (nn.Module): dvs network.
        state_or_path (dict or Path): state or path to checkpoint,
                         which contains the "network" parameters.
    """
    state = _state(state_or_path, "network")
    if state is not None:
        network.load_state_dict(state)
        logging.info("Recovered network state.")
    else:
        logging.warning("Could not recover network state.")


def _recover_decoder(decoder, state_or_path, strict=True):
    """Same as _recover_network for multiple decoders."""
    states = _state(state_or_path, "decoder")
    if states is not None:
        for key, decoder in decoder.items():
            state = states.pop(key, None)
            if state is not None:
                decoder.load_state_dict(state, strict=strict)
                logging.info(f"Recovered {key} decoder state.")
            else:
                logging.warning(f"Could not recover state of {key} decoder.")
    else:
        logging.warning("Could not recover decoder states.")


def _recover_optimizer(optimizer, state_or_path):
    """Same as _recover_network for optimizer."""
    state = _state(state_or_path, "optim")
    if state is not None:
        optimizer.load_state_dict(state)
        logging.info(f"Recovered optimizer state.")
    else:
        logging.warning("Could not recover optimizer state.")


def _recover_penalty_optimizer(optimizers, state_or_path):
    """Same as _recover_network for penalty optimizers."""
    states = _state(state_or_path, "penalty_optims")
    if states is not None:
        for key, optim in optimizers.items():
            state = states.pop(key, None)
            if state is not None:
                optim.load_state_dict(state)
                logging.info(f"Recovered {key} optimizer state.")
            else:
                logging.warning(f"Could not recover state of {key} optimizer.")
    else:
        logging.warning("Could not recover penalty optimizer states.")


def _state(state_or_path, key):
    """Returns a state dictionary.

    Args:
        state_or_path (dict or Path): state dict, or path to a checkpoint.
        key (str): key to look for in state dict recovered from path.

    Raises:
        ValueError: if state_or_path neither dict or Path.

    Returns:
        dict: state dictionary.
    """
    if state_or_path is None:
        return
    if isinstance(state_or_path, Path):
        state = torch.load(state_or_path).pop(key, None)
    elif isinstance(state_or_path, dict):
        state = state_or_path
    else:
        raise ValueError(f"{type(state_or_path)} invalid type.")
    return state


# -- Supporting definitions ----------------------------------------------------


# def write_meta(path, meta):
#     meta_path = path / "_meta.yaml"
#     meta_path.write_text(json.dumps(meta.to_dict()))


def _interp_num_frames(n_frames, framerate, dt):
    """Returns the number of frames after interpolation.

    Args:
        n_frames (int): number of frames before interpolation.
        framerate (int): frame rate of the dataset in 1/s.
        dt (float): integration time constant in s.

    Returns:
        int: number of frames after interpolation.
    """
    return math.ceil(n_frames / (framerate * dt))


def _stored_checkpoints(path, glob="chkpt_*"):
    """Returns all numerical identifier and paths to checkpoints stored in path.

    Args:
        path (Path): checkpoint directory.

    Returns:
        tuple: identifier, paths
            with
                identifier (List[int]): numerical identifier.
                paths (List[Path]): paths.
    """
    import re

    if not path.exists():
        path.mkdir()
    _paths = sorted(list((path).glob(glob)))
    try:
        _identifier = [int(re.findall("\d{1,10}", p.parts[-1])[0]) for p in _paths]
        # if e.g. the identifier was epoch with up to 6 digits, but the string format
        # of the identifier only allowed five digits, then the sorting is messed
        # up but can be undone. therefore, resorting here using the identifier.
        _sorting_index = np.argsort(_identifier)
        _paths = np.array(_paths)[_sorting_index].tolist()
        _identifier = np.array(_identifier)[_sorting_index].tolist()
        return _identifier, _paths
    except IndexError:
        return None, _paths


def _copy_wrap(source: Path, target: Path):
    # TODO: add meta flag
    if not target.exists():
        _source_wrap = Datawrap(source)
        _target_wrap = Datawrap(target.parent)
        _target_wrap[target.name] = _source_wrap


def _chkpt_every_epoch(iters, len_loader, fraction=0.00025):
    chkpts_per_all_iters = iters * fraction
    chkpt_per_iter = chkpts_per_all_iters / iters
    chkpt_per_epoch = chkpt_per_iter * len_loader
    x = 1 / chkpt_per_epoch
    return int(np.ceil(x))  # , chkpts_per_all_iters


def _chkpt_every_iter(iters, len_loader, fraction=0.00025):
    chkpts_per_all_iters = iters * fraction
    chkpt_per_iter = chkpts_per_all_iters / iters
    x = 1 / chkpt_per_iter
    return int(np.ceil(x))  # , chkpts_per_all_iters


recover_network = _recover_network
recover_decoder = _recover_decoder
stored_checkpoints = _stored_checkpoints
