import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import flyvis
from flyvis.datasets.sintel import MultiTaskSintel
from flyvis.network.directories import NetworkDir
from flyvis.task import objectives as task_objectives

from .models import VanillaHexCNNBaseline, VanillaHexCNNBaselineHexSpace

logger = logging.getLogger(__name__)

__all__ = ["train_baseline", "build_baseline_network_name"]


MODEL_REGISTRY = {
    "VanillaHexCNNBaseline": VanillaHexCNNBaseline,
    "VanillaHexCNNBaselineHexSpace": VanillaHexCNNBaselineHexSpace,
}


def _prepare_lum_inputs(lum: torch.Tensor, n_frames: int) -> torch.Tensor:
    # Legacy VanillaHexCNN expects a single luminance channel.
    if lum.ndim == 4 and lum.shape[2] != 1:
        lum = lum.mean(dim=2, keepdim=True)
    if lum.ndim == 4 and lum.shape[1] != n_frames:
        lum = lum[:, -n_frames:]
    return lum


@dataclass
class TrainingResult:
    network_name: str
    model_path: Path
    n_train_steps: int
    n_checkpoints: int


def build_baseline_network_name(task_name: str, ensemble_and_network_id: str) -> str:
    return f"baselines/{task_name}/{ensemble_and_network_id}"


def _build_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = dict(config["dataset"])
    dataset_cfg.pop("type", None)
    dataset = MultiTaskSintel(**dataset_cfg)
    try:
        train_indices, val_indices = dataset.original_train_and_validation_indices()
    except ValueError:
        # Small/unit-test datasets may not contain the canonical validation indices.
        n_samples = len(dataset)
        split = max(1, int(0.8 * n_samples))
        train_indices = list(range(split))
        val_indices = list(range(split, n_samples))
        if len(val_indices) == 0:
            val_indices = train_indices[-1:]
            train_indices = train_indices[:-1] or train_indices

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        pin_memory=False,
        shuffle=False,
        sampler=SubsetRandomSampler(train_indices),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        pin_memory=False,
        shuffle=False,
        sampler=SubsetRandomSampler(val_indices),
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module, val_loader: DataLoader, metric_fn, device: torch.device
) -> float:
    model.eval()
    values: List[float] = []
    if isinstance(model.conv[0], nn.Sequential):
        n_frames = model.conv[0][0].in_channels
    else:
        n_frames = model.conv[0].in_channels
    with val_loader.dataset.augmentation(False):
        for data in val_loader:
            inputs = _prepare_lum_inputs(data["lum"], n_frames=n_frames).to(device)
            targets = data["flow"][:, -1].unsqueeze(1).to(device)
            outputs = model(inputs)
            values.append(metric_fn(outputs, targets).detach().cpu().item())
    return float(sum(values) / max(len(values), 1))


def train_baseline(config: Dict) -> TrainingResult:
    network_name = build_baseline_network_name(
        config["task_name"], config["ensemble_and_network_id"]
    )
    ensemble_id, model_id = config["ensemble_and_network_id"].split("/")
    dir_config = {
        "network_name": network_name,
        "network": {
            "connectome": dict(
                type="ConnectomeFromAvgFilters",
                file="fib25-fib19_v2.2.json",
                extent=15,
                n_syn_fill=1,
            )
        },
        "baseline": {
            "name": "vanilla_hex_cnn",
            "arch": config["arch"],
            "arch_kwargs": config["arch_kwargs"],
            "ensemble_id": ensemble_id,
            "model_id": model_id,
            "task_name": config["task_name"],
            "loss_name": config["loss_name"],
            "validation_metric": config["validation_metric"],
        },
    }

    model_dir = NetworkDir(
        network_name,
        {**dir_config, "delete_if_exists": config.get("delete_if_exists", False)},
    )
    model_dir.path.mkdir(parents=True, exist_ok=True)
    (model_dir.path / "chkpts").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_dataloaders(config)
    model = MODEL_REGISTRY[config["arch"]](**config["arch_kwargs"]).to(flyvis.device)
    expected_n_frames = int(config["arch_kwargs"]["n_frames"])

    param_groups = [
        {"params": model.bias_parameters(), "weight_decay": config["bias_decay"]},
        {"params": model.weight_parameters(), "weight_decay": config["weight_decay"]},
    ]
    if config["solver"] == "adam":
        optimizer = torch.optim.Adam(
            param_groups, config["lr"], betas=(config["momentum"], config["beta"])
        )
    elif config["solver"] == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, config["lr"], momentum=config["momentum"]
        )
    else:
        raise ValueError(f"Unsupported solver '{config['solver']}'")

    milestones = config.get("milestones")
    scheduler = None
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=config["scheduler_gamma"]
        )

    loss_fn = getattr(task_objectives, config["loss_name"])
    val_metric_fn = getattr(task_objectives, config["validation_metric"])

    training_losses: List[float] = []
    validation_metric_values: List[float] = []
    best_metric = float("inf")
    chkpt_index = -1
    global_step = 0
    max_steps = config.get("max_steps")
    start = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        for data in train_loader:
            inputs = _prepare_lum_inputs(
                data["lum"], n_frames=expected_n_frames
            ).to(flyvis.device)
            targets = data["flow"][:, -1].unsqueeze(1).to(flyvis.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.detach().cpu().item())
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        if scheduler is not None:
            scheduler.step()

        should_checkpoint = ((epoch + 1) % config["chkpt_every"] == 0) or (
            epoch + 1 == config["epochs"]
        )
        if should_checkpoint:
            val_metric = _evaluate(model, val_loader, val_metric_fn, flyvis.device)
            validation_metric_values.append(val_metric)
            chkpt_index += 1
            torch.save(
                {
                    "network": model.state_dict(),
                    "arch": config["arch"],
                    "arch_kwargs": config["arch_kwargs"],
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "iteration": global_step,
                    "validation_metric": val_metric,
                    "metric_name": config["validation_metric"],
                },
                model_dir.path / "chkpts" / f"chkpt_{chkpt_index:05}",
            )
            model_dir.extend("chkpt_index", [chkpt_index])
            model_dir.extend("chkpt_iter", [global_step])
            model_dir["validation"].extend(config["validation_metric"], [val_metric])
            model_dir["validation"].extend("loss", [val_metric])
            if val_metric < best_metric:
                best_metric = val_metric
                model_dir.best_chkpt_index = chkpt_index
            logger.info(
                "Epoch %d checkpointed: %s=%.6f",
                epoch + 1,
                config["validation_metric"],
                val_metric,
            )

        if max_steps is not None and global_step >= max_steps:
            break

    model_dir.loss = training_losses
    model_dir.time_trained = time.time() - start
    logger.info(
        "Finished baseline training at %s with %d steps.",
        network_name,
        len(training_losses),
    )
    return TrainingResult(
        network_name=network_name,
        model_path=model_dir.path,
        n_train_steps=len(training_losses),
        n_checkpoints=len(validation_metric_values),
    )

