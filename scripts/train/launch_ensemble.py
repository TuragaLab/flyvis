"""Submit runs to the cluster.

Example:
    $ python launch_ensemble.py \
        --start 0 \
        --end 5 \
        --launch_training \
        --nP 4 \
        --gpu num=1 \
        --q gpu_rtx \
        --ensemble_id 9999 \
        --task_name flow \
        --script train.py \
        description='test' \
        train=true \
        +delete_if_exists=true
"""

import argparse
import logging
import re
import socket
import subprocess
import sys
from time import sleep

from flyvision import results_dir

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


def run_job(command, dry):
    """Runs command in subprocess. Assumes submission to LSF cluster."""
    if dry:
        job_id = "dry run"
        logging.info(command)
        return job_id
    answer = subprocess.getoutput(command)
    print(answer)
    job_id = re.findall(r"(?<=<)\d+(?=>)", answer)
    print(job_id)
    assert len(job_id) == 1
    return job_id[0]


def is_running(job_id, dry):
    """Check if job is running on LSF cluster."""
    if dry:
        return False
    job_info = subprocess.getoutput("bjobs -w")
    return job_id in job_info


def kill_job(job_id, dry):
    """Kill job on LSF cluster."""
    return f"bkill {job_id}" if dry else subprocess.getoutput(f"bkill {job_id}")


def wait_for_single(job_id, job_name, dry=False):
    """Wait for job to finish on LSF cluster."""
    try:
        if not dry:
            sleep(60)
        while is_running(job_name, dry):
            if not dry:
                sleep(60)
    except KeyboardInterrupt as e:
        logging.info(kill_job(job_id, dry))
        raise KeyboardInterrupt from e


def wait_for_many(job_id_names, dry=False):
    """Wait for multiple jobs to finish on LSF cluster."""
    try:
        if not dry:
            print("Jobs launched.. waiting 60s..")
            sleep(60)
        while any(is_running(job_name, dry) for job_name in job_id_names.values()):
            if not dry:
                print("Jobs still running.. waiting 60s..")
                sleep(60)
    except KeyboardInterrupt as e:
        for job_id in job_id_names:
            logging.info(kill_job(job_id, dry))
        raise KeyboardInterrupt from e


def check_valid_host(blacklist):
    """Prevent running on certain blacklisted hosts, e.g., login nodes."""
    host = socket.gethostname()
    if any([h in host for h in blacklist]):
        raise ValueError(f"This script should not be run from {host}!")


def launch_range(
    start: int,
    end: int,
    ensemble_id: str,
    task_name: str,
    nP: int,
    gpu: str,
    q: str,
    script: str,
    dry: bool,
    # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...] or following
    # hydra syntax, i.e. ["kw=val", ...]
    kwargs: list,
):
    """Launch a range of jobs."""

    LSF_PART = "bsub -J {}_{} -n {} -o {} -gpu '{}' -q {} "
    SCRIPT_PART = "{} {} {}"  # --{}"

    job_id_names = {}
    for i in range(start, end):
        kw = kwargs.copy()
        network_id = f"{ensemble_id:03}/{i:04}"
        assert "_" not in network_id
        network_dir = results_dir / task_name / network_id
        if not network_dir.exists():
            network_dir.mkdir(parents=True)
        log_file = network_dir.parent / f"{i:04}_{script.split('.')[0]}.log"
        if log_file.exists():
            log_file.unlink()

        kw.extend([f"network_id={network_id}"])
        kw.extend([f"task_name={task_name}"])

        LSF_CMD = LSF_PART.format(
            task_name,  # -J {}_ job name prefix
            network_id,  # -J _{} job name suffix
            nP,  # -n {} num proc
            log_file,  # -o {} output file name
            gpu,  # -gpu '{}' number of gpus
            q,  # -q {} queue
        )
        SCRIPT_CMD = SCRIPT_PART.format(
            sys.executable,  # {} python
            script,  # {} script
            " ".join(kw),  # {} all other key word arguments
        )
        command = LSF_CMD + SCRIPT_CMD
        logging.info(command)
        job_id = run_job(command, dry)
        job_id_names[job_id] = f"{task_name}_{network_id}"

    wait_for_many(job_id_names, dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ensemble of networks.")
    parser.add_argument("--start", type=int, default=0, help="Start id of ensemble.")
    parser.add_argument("--end", type=int, default=50, help="End id of ensemble.")

    parser.add_argument("--nP", type=int, default=4, help="Numnber of processors.")
    parser.add_argument("--gpu", type=str, default="num=1")
    parser.add_argument("--q", type=str, default="gpu_rtx")
    parser.add_argument(
        "--ensemble_id",
        type=int,
        default="",
        help=(
            "Id of the ensemble, e.g. 0045. The ultimate name of a model will be of "
            "the form task/ensemble_id/<start-end>"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Name given to the task, e.g. opticflow.",
    )

    parser.add_argument("--script", type=str, default="", help="Script to run.")
    parser.add_argument(
        "--launch_training",
        dest="launch_training",
        action="store_true",
        help="Train models.",
    )

    parser.add_argument(
        "--dry",
        dest="dry",
        action="store_true",
    )

    # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    args, kwargs = parser.parse_known_intermixed_args()
    print(args, kwargs)
    if args.launch_training:
        if "individual_model_ids" in kwargs:
            # launch one by one way to run individual model ids, e.g. when a few failed
            # because of server side errors
            idx = kwargs.index("individual_model_ids")
            individual_model_ids = kwargs[idx + 1]
            model_ids = individual_model_ids.split(",")
            for _id in model_ids:
                launch_range(
                    _id,
                    _id + 1,
                    args.ensemble_id,
                    args.task_name,
                    args.nP,
                    args.gpu,
                    args.q,
                    args.script,
                    args.dry,
                    kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw",
                    # "val", ...]
                )
        else:
            launch_range(
                args.start,
                args.end,
                args.ensemble_id,
                args.task_name,
                args.nP,
                args.gpu,
                args.q,
                args.script,
                args.dry,
                kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw",
                #   "val", ...]
            )
