import logging
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from time import sleep
from typing import Dict, List

from flyvis import results_dir

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ClusterManager(ABC):
    """Abstract base class for cluster management operations."""

    @abstractmethod
    def run_job(self, command: str) -> str:
        """
        Run a job on the cluster.

        Args:
            command: The command to run.

        Returns:
            The job ID as a string.
        """
        pass

    @abstractmethod
    def is_running(self, job_id: str) -> bool:
        """
        Check if a job is running.

        Args:
            job_id: The ID of the job to check.

        Returns:
            True if the job is running, False otherwise.
        """
        pass

    @abstractmethod
    def kill_job(self, job_id: str) -> str:
        """
        Kill a running job.

        Args:
            job_id: The ID of the job to kill.

        Returns:
            The output of the kill command.
        """
        pass

    @abstractmethod
    def get_submit_command(
        self, job_name: str, n_cpus: int, output_file: str, gpu: str, queue: str
    ) -> str:
        """
        Get the command to submit a job to the cluster.

        Args:
            job_name: The name of the job.
            n_cpus: The number of CPUs to request.
            output_file: The file to write job output to.
            gpu: The GPU configuration.
            queue: The queue to submit the job to.

        Returns:
            The submit command as a string.
        """
        pass

    @abstractmethod
    def get_script_part(self, command: str) -> str:
        """
        Get the script part of the command.

        Args:
            command: The command to wrap.

        Returns:
            The wrapped command as a string.
        """
        pass


class LSFManager(ClusterManager):
    """Cluster manager for LSF (Load Sharing Facility) systems."""

    def run_job(self, command: str) -> str:
        answer = subprocess.getoutput(command)
        job_id = re.findall(r"(?<=<)\d+(?=>)", answer)
        assert len(job_id) == 1
        return job_id[0]

    def is_running(self, job_id: str) -> bool:
        job_info = subprocess.getoutput("bjobs -w")
        return job_id in job_info

    def kill_job(self, job_id: str) -> str:
        return subprocess.getoutput(f"bkill {job_id}")

    def get_submit_command(
        self, job_name: str, n_cpus: int, output_file: str, gpu: str, queue: str
    ) -> str:
        return f"bsub -J {job_name} -n {n_cpus} -o {output_file} -gpu '{gpu}' -q {queue} "

    def get_script_part(self, command: str) -> str:
        return command


class SLURMManager(ClusterManager):
    """Cluster manager for SLURM systems.

    Warning:
        This is untested.
    """

    def run_job(self, command: str) -> str:
        answer = subprocess.getoutput(command)
        job_id = re.findall(r"\d+", answer)
        assert len(job_id) == 1
        return job_id[0]

    def is_running(self, job_id: str) -> bool:
        cmd = f"sacct -j {job_id} --format=State --noheader -X"
        state = subprocess.getoutput(cmd).strip()
        return state in ["PENDING", "RUNNING", "REQUEUED"]

    def kill_job(self, job_id: str) -> str:
        return subprocess.getoutput(f"scancel {job_id}")

    def get_submit_command(
        self, job_name: str, n_cpus: int, output_file: str, gpu: str, queue: str
    ) -> str:
        return (
            f"sbatch --job-name={job_name} "
            f"--cpus-per-task={n_cpus} "
            f"--output={output_file} "
            f"--gres=gpu:{gpu} "
            f"--partition={queue} "
        )

    def get_script_part(self, command: str) -> str:
        return f"--wrap '{command}'"


class VirtualClusterManager(ClusterManager):
    """Simulated cluster manager for local execution."""

    def __init__(self):
        warnings.warn(
            "VirtualClusterManager is being used. This is not recommended for "
            "production use and may not accurately represent cluster behavior.",
            UserWarning,
            stacklevel=2,
        )
        self.running_jobs = {}
        self._current_output_file = None

    def get_submit_command(
        self, job_name: str, n_cpus: int, output_file: str, gpu: str, queue: str
    ) -> str:
        self._current_output_file = output_file
        return f"echo 'Running {job_name} locally' && "

    def run_job(self, command: str) -> str:
        job_id = str(len(self.running_jobs) + 1)
        if self._current_output_file:
            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(self._current_output_file)
            logger.info(f"Writing output to: {self._current_output_file}")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self._current_output_file, 'w') as f:
                process = multiprocessing.Process(
                    target=subprocess.run,
                    args=(command,),
                    kwargs={'shell': True, 'stdout': f, 'stderr': subprocess.STDOUT},
                )
                process.start()
                self.running_jobs[job_id] = process
        else:
            logger.warning("No output file specified for job")
            process = multiprocessing.Process(target=os.system, args=(command,))
            process.start()
            self.running_jobs[job_id] = process
        return job_id

    def is_running(self, job_id: str) -> bool:
        return job_id in self.running_jobs and self.running_jobs[job_id].is_alive()

    def kill_job(self, job_id: str) -> str:
        if job_id in self.running_jobs:
            process = self.running_jobs[job_id]
            os.kill(process.pid, signal.SIGTERM)
            process.join(timeout=5)
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
            del self.running_jobs[job_id]
            return f"Job {job_id} terminated"
        return f"Job {job_id} not found"

    def get_script_part(self, command: str) -> str:
        return command


def get_cluster_manager(dry: bool = False) -> ClusterManager:
    """
    Autodetect the cluster type and return the appropriate ClusterManager.

    Args:
        dry: If True, return LSFManager even if no cluster is detected.

    Returns:
        An instance of the appropriate ClusterManager subclass.
    """
    virtual = os.environ.get("VIRTUAL_CLUSTER", "").lower() in ("true", "1", "yes", "on")
    dry = dry or os.environ.get("DRYRUN_ONLY", "").lower() in ("true", "1", "yes", "on")

    if subprocess.getoutput("command -v bsub"):
        return LSFManager()
    elif subprocess.getoutput("command -v sbatch"):
        return SLURMManager()
    else:
        if dry:
            return LSFManager()
        elif virtual:
            warnings.warn(
                "No cluster management system detected. Using VirtualClusterManager for "
                "local execution. This is not recommended for production use.",
                UserWarning,
                stacklevel=2,
            )
            return VirtualClusterManager()
        else:
            raise RuntimeError("No cluster management system detected.")


class LazyClusterManager:
    """Lazy initialization of the cluster manager."""

    def __init__(self):
        self._instance = None
        # TODO: dry run handling currently not elegant but works for now
        self._dry = False

    def set_dry(self, dry: bool):
        """Set dry run mode."""
        self._dry = dry
        self._instance = None  # Reset instance to force re-initialization

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = get_cluster_manager(dry=self._dry)
        return getattr(self._instance, name)


CLUSTER_MANAGER = LazyClusterManager()


def run_job(command: str, dry: bool) -> str:
    """
    Run a job on the cluster.

    Args:
        command: The command to run.
        dry: If True, perform a dry run without actually submitting the job.

    Returns:
        The job ID as a string, or "dry run" for dry runs.
    """
    # TODO: dry handling currently not elegant but works for now
    env_dry = os.environ.get("DRYRUN_ONLY", "").lower() in ("true", "1", "yes", "on")
    is_dry = dry or env_dry

    if is_dry:
        job_id = "dry run"
        logger.info("Dry run command: %s", command)
        return job_id

    return CLUSTER_MANAGER.run_job(command)


def is_running(job_id: str, dry: bool) -> bool:
    """
    Check if a job is running.

    Args:
        job_id: The ID of the job to check.
        dry: If True, always return False.

    Returns:
        True if the job is running, False otherwise.
    """
    if dry:
        return False
    return CLUSTER_MANAGER.is_running(job_id)


def kill_job(job_id: str, dry: bool) -> str:
    """
    Kill a running job.

    Args:
        job_id: The ID of the job to kill.
        dry: If True, return a message without actually killing the job.

    Returns:
        The output of the kill command or a dry run message.
    """
    if dry:
        return f"Would kill job {job_id}"
    return CLUSTER_MANAGER.kill_job(job_id)


def wait_for_single(job_id: str, dry: bool = False) -> None:
    """
    Wait for a single job to finish on the cluster.

    Args:
        job_id: The ID of the job to wait for.
        job_name: The name of the job.
        dry: If True, skip actual waiting.

    Raises:
        KeyboardInterrupt: If the waiting is interrupted by the user.
    """
    try:
        if not dry:
            sleep(60)
        while is_running(job_id, dry):
            if not dry:
                sleep(60)
    except KeyboardInterrupt as e:
        logger.info("Killing job %s", kill_job(job_id, dry))
        raise KeyboardInterrupt from e


def wait_for_many(job_id_names: Dict[str, str], dry: bool = False) -> None:
    """
    Wait for multiple jobs to finish on the cluster.

    Args:
        job_id_names: A dictionary mapping job IDs to job names.
        dry: If True, skip actual waiting.

    Raises:
        KeyboardInterrupt: If the waiting is interrupted by the user.
    """
    try:
        if not dry:
            print("Jobs launched.. waiting 60s..")
            sleep(60)
        while any(is_running(job_id, dry) for job_id in job_id_names):
            if not dry:
                print("Jobs still running.. waiting 60s..")
                sleep(60)
    except KeyboardInterrupt as e:
        for job_id in job_id_names:
            logger.info("Killing job %s", kill_job(job_id, dry))
        raise KeyboardInterrupt from e


def check_valid_host(blacklist: List[str]) -> None:
    """
    Prevent running on certain blacklisted hosts, e.g., login nodes.

    Args:
        blacklist: A list of blacklisted hostnames or substrings.

    Raises:
        ValueError: If the current host is in the blacklist.
    """
    host = socket.gethostname()
    if any(h in host for h in blacklist):
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
    kwargs: List[str],
) -> None:
    """
    Launch a range of models.

    Args:
        start: The starting index for the range.
        end: The ending index for the range.
        ensemble_id: The ID of the ensemble.
        task_name: The name of the task.
        nP: The number of processors to use.
        gpu: The GPU configuration.
        q: The queue to submit the job to.
        script: The script to run.
        dry: If True, perform a dry run without actually submitting jobs.
        kwargs: A list of additional keyword arguments for the script.

    Note:
        kwargs is an ordered list of strings, either in the format ["-kw", "val", ...]
        or following hydra syntax, i.e. ["kw=val", ...].
    """
    SCRIPT_PART = "{} {} {}"

    CLUSTER_MANAGER.set_dry(dry)

    job_id_names = {}
    for i in range(start, end):
        kw = kwargs.copy()
        ensemble_and_network_id = f"{ensemble_id:04}/{i:03}"
        assert "_" not in ensemble_and_network_id
        network_dir = results_dir / task_name / ensemble_and_network_id
        if not network_dir.parent.exists():
            network_dir.parent.mkdir(parents=True)
        log_file = (
            network_dir.parent / f"{i:04}_{script.split('/')[-1].split('.')[0]}.log"
        )
        if log_file.exists():
            log_file.unlink()

        kw.extend([f"ensemble_and_network_id={ensemble_and_network_id}"])
        kw.extend([f"task_name={task_name}"])

        LSF_CMD = CLUSTER_MANAGER.get_submit_command(
            f"{task_name}_{ensemble_and_network_id}", nP, log_file, gpu, q
        )
        SCRIPT_CMD = SCRIPT_PART.format(sys.executable, script, " ".join(kw))
        command = LSF_CMD + CLUSTER_MANAGER.get_script_part(SCRIPT_CMD)
        logger.info("Launching command: %s", command)
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
    kwargs: List[str],
) -> None:
    """
    Launch a single job for an ensemble.

    Args:
        ensemble_id: The ID of the ensemble.
        task_name: The name of the task.
        nP: The number of processors to use.
        gpu: The GPU configuration.
        q: The queue to submit the job to.
        script: The script to run.
        dry: If True, perform a dry run without actually submitting the job.
        kwargs: A list of additional keyword arguments for the script.

    Note:
        kwargs is an ordered list of strings, either in the format ["-kw", "val", ...]
        or following hydra syntax, i.e. ["kw=val", ...].
    """
    SCRIPT_PART = "{} {} {}"

    CLUSTER_MANAGER.set_dry(dry)

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

    LSF_CMD = CLUSTER_MANAGER.get_submit_command(
        f"{task_name}_{ensemble_id}", nP, log_file, gpu, q
    )
    SCRIPT_CMD = SCRIPT_PART.format(sys.executable, script, " ".join(kw))
    command = LSF_CMD + CLUSTER_MANAGER.get_script_part(SCRIPT_CMD)
    logger.info("Launching command: %s", command)
    job_id = run_job(command, dry)
    job_id_names[job_id] = f"{task_name}_{ensemble_id}"

    wait_for_many(job_id_names, dry)
