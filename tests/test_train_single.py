import os
import subprocess
import sys
from importlib import resources

from flyvis.network.directories import NetworkDir

TRAIN_SINGLE = str(resources.files("flyvis_cli") / "training" / "train_single.py")


def train_single(root_dir, sintel_path, *overrides):
    return subprocess.run(
        [
            sys.executable,
            TRAIN_SINGLE,
            "task_name=flow",
            "ensemble_and_network_id=0000/000",
            "description=test",
            "task.n_iters=4",
            f"+task.dataset.sintel_path={sintel_path}",
            "task.original_split=false",
            "task.dataset.boxfilter.extent=1",
            "task.dataset.n_frames=4",
            "task.dataset.dt=0.041",
            "task.batch_size=2",
            "network.connectome.extent=1",
            "scheduler.chkpt_every_epoch=1",
            *overrides,
        ],
        cwd=root_dir,
        env={**os.environ, "FLYVIS_ROOT_DIR": str(root_dir)},
        capture_output=True,
        text=True,
    )


def test_train_single_resume(mock_sintel_data, tmp_path):
    # only the initial checkpoint, as left by a job stopped before training
    run = train_single(tmp_path, mock_sintel_data, "train=false", "checkpoint_only=true")
    assert run.returncode == 0, run.stderr

    # the same command again would train over the existing checkpoints
    run = train_single(tmp_path, mock_sintel_data)
    assert run.returncode != 0
    assert "already has checkpoints" in run.stderr

    run = train_single(tmp_path, mock_sintel_data, "resume=true")
    assert run.returncode == 0, run.stderr

    network_dir = NetworkDir(tmp_path / "results" / "flow" / "0000" / "000")
    assert len(network_dir.loss[:]) == 4
    assert network_dir.chkpt_iter[:].tolist()[-1] == 3
