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
logging = logging.getLogger(__name__)


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
    """Launch a range of models."""

    LSF_PART = "bsub -J {}_{} -n {} -o {} -gpu '{}' -q {} "
    SCRIPT_PART = "{} {} {}"  # --{}"

    job_id_names = {}
    for i in range(start, end):
        kw = kwargs.copy()
        ensemble_and_network_id = f"{ensemble_id:04}/{i:03}"
        assert "_" not in ensemble_and_network_id
        network_dir = results_dir / task_name / ensemble_and_network_id
        if not network_dir.exists():
            network_dir.mkdir(parents=True)
        log_file = (
            network_dir.parent / f"{i:04}_{script.split('/')[-1].split('.')[0]}.log"
        )
        if log_file.exists():
            log_file.unlink()

        kw.extend([f"ensemble_and_network_id={ensemble_and_network_id}"])
        kw.extend([f"task_name={task_name}"])

        LSF_CMD = LSF_PART.format(
            task_name,  # -J {}_ job name prefix
            ensemble_and_network_id,  # -J _{} job name suffix
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
        job_id_names[job_id] = f"{task_name}_{ensemble_and_network_id}"

    wait_for_many(job_id_names, dry)


def launch_single(
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
    """Launch a single job for an ensemble."""

    LSF_PART = "bsub -J {}_{} -n {} -o {} -gpu '{}' -q {} "
    SCRIPT_PART = "{} {} {}"  # --{}"

    job_id_names = {}
    kw = kwargs.copy()
    ensemble_id = f"{ensemble_id:04}"
    assert "_" not in ensemble_id
    ensemble_dir = results_dir / task_name / ensemble_id

    assert ensemble_dir.exists()
    log_file = ensemble_dir / f"{script.split('/')[-1].split('.')[0]}.log"
    if log_file.exists():
        log_file.unlink()

    kw.extend([f"ensemble_id={ensemble_id}"])
    kw.extend([f"task_name={task_name}"])

    LSF_CMD = LSF_PART.format(
        task_name,  # -J {}_ job name prefix
        ensemble_id,  # -J _{} job name suffix
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
    job_id_names[job_id] = f"{task_name}_{ensemble_id}"

    wait_for_many(job_id_names, dry)
