"""Submit runs to the cluster.


Example:
    $ python launch.py \
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
import os
import sys

import subprocess
from copy import deepcopy
import re
from time import sleep
import argparse
from pathlib import Path
import logging

from flyvision import results_dir 

logging.basicConfig(format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
logging = logger = logging.getLogger(__name__)


def ensure_shebang(script_path, conda_env):

    python_path = f"{conda_env}/bin/python"

    # Read the existing content of the script
    with open(script_path, 'r') as file:
        lines = file.readlines()
    
    # Check if the first line is a shebang
    if lines and lines[0].startswith('#!'):
        shebang = lines[0].strip()
        if shebang == f"#!{python_path}":
            print(f"Shebang already correct: {shebang}")
            return
        else:
            print(f"Replacing shebang {shebang} with #!{python_path}")
            lines[0] = f"#!{python_path}\n"
    else:
        print(f"Adding shebang #!{python_path}")
        lines.insert(0, f"#!{python_path}\n")
    
    # Write the updated content back to the script
    with open(script_path, 'w') as file:
        file.writelines(lines)

def run_job(command, dry):
    if dry:
        job_id = "dry run"
        logging.info(command)
        return job_id
    answer = subprocess.getoutput(command)
    print(answer)
    job_id = re.findall("(?<=<)\d+(?=>)", answer)
    print(job_id)
    assert len(job_id) == 1
    return job_id[0]


def is_running(job_id, dry):
    if dry:
        return False
    job_info = subprocess.getoutput("bjobs -w")
    if job_id in job_info:
        return True
    return False


def kill_job(job_id, dry):
    return f"bkill {job_id}" if dry else subprocess.getoutput(f"bkill {job_id}")


def wait_for_single(job_id, job_name, dry=False):
    try:
        if not dry:
            sleep(60)
        while is_running(job_name, dry):
            if not dry:
                sleep(60)
    except KeyboardInterrupt as e:
        logging.info(kill_job(job_id, dry))
        raise KeyboardInterrupt from e


def wait_for_all(job_id_names, dry=False):
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


def check_valid_host():
    import socket

    host = socket.gethostname()
    if any([h in host for h in ["e02u30", "e05u15", "e05u16"]]):
        raise ValueError("This script should be run from an interactive session!")

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
    """Launch a range of jobs.
    """
    # make sure we are in the same conda environment
    conda_env = os.environ.get('CONDA_PREFIX')      
    if not conda_env:
        print("Error: This script should be run within a conda environment.")
        sys.exit(1)
    ensure_shebang(script, conda_env)

    LSF_PART = "bsub -J {}_{} -n {} -o {} -gpu '{}' -q {} "
    SCRIPT_PART = "./{} {}"  # --{}"

    job_id_names = {}
    for i in range(start, end):
        kw = kwargs.copy()
        network_id = f"{ensemble_id:03}/{i:04}"
        assert "_" not in network_id
        network_dir = (
            results_dir / task_name / network_id
        )
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
            script,  # ./{} script
            " ".join(kw),  # {} all other key word arguments
        )
        command = LSF_CMD + SCRIPT_CMD
        logging.info(command)
        job_id = run_job(command, dry)
        job_id_names[job_id] = f"{task_name}_{network_id}"

    wait_for_all(job_id_names, dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ensemble of networks.")
    parser.add_argument(
        "--start", type=int, default=0, help="Start id of ensemble."
    )
    parser.add_argument(
        "--end", type=int, default=50, help="End id of ensemble."
    )

    parser.add_argument(
        "--nP", type=int, default=4, help="Numnber of processors."
    )
    parser.add_argument("--gpu", type=str, default="num=1")
    parser.add_argument("--q", type=str, default="gpu_rtx")
    parser.add_argument(
        "--ensemble_id",
        type=int,
        default="",
        help="Id of the ensemble, e.g. 0045. The ultimate name of a model will be of the form task/ensemble_id/<start-end>",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Name given to the task, e.g. opticflow.",
    )

    parser.add_argument("--script", type=str, default="", help="Script to run.")
    parser.add_argument("--launch_training", dest="launch_training", action="store_true", help="Train models.")

    # parser.add_argument(
    #     "--execute_model_validation",
    #     dest="execute_model_validation",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_model_synthetic_recordings",
    #     dest="execute_model_synthetic_recordings",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_model_synthetic_recordings_manuscript",
    #     dest="execute_model_synthetic_recordings_manuscript",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_ensemble_clustering",
    #     dest="execute_ensemble_clustering",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_ensemble_notebook",
    #     dest="execute_ensemble_notebook",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_training_and_tuning_overview",
    #     dest="execute_training_and_tuning_overview",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_vanilla_cnn_validation",
    #     dest="execute_vanilla_cnn_validation",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--execute_single_model_notebook",
    #     dest="execute_single_model_notebook",
    #     action="store_true",
    # )

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
                    kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
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
                kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
            )

    # if args.execute_model_validation:
    #     _kwargs = ["-task", args.task_name]
    #     run_ensemble_jobs(
    #         args.start,
    #         args.end,
    #         args.ensemble_id,
    #         args.task_name,
    #         args.nP,
    #         args.gpu,
    #         args.q,
    #         "execute_model_validation.py",
    #         None,
    #         args.dry,
    #         args.optional_name,
    #         _kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    #     )

    # if args.execute_model_synthetic_recordings:
    #     _kwargs = deepcopy(kwargs)
    #     _kwargs.extend(["-chkpt", "best", "-task", args.task_name])
    #     run_ensemble_jobs(
    #         args.start,
    #         args.end,
    #         args.ensemble_id,
    #         args.task_name,
    #         args.nP,
    #         args.gpu,
    #         args.q,
    #         "execute_model_synthetic_recordings.py",
    #         None,
    #         args.dry,
    #         args.optional_name,
    #         _kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    #     )

    # if args.execute_model_synthetic_recordings_manuscript:
    #     _kwargs = ["-chkpt", "best", "-task", args.task_name, *kwargs]
    #     run_ensemble_jobs(
    #         args.start,
    #         args.end,
    #         args.ensemble_id,
    #         args.task_name,
    #         args.nP,
    #         args.gpu,
    #         args.q,
    #         "execute_model_synthetic_recordings_manuscript.py",
    #         None,
    #         args.dry,
    #         args.optional_name,
    #         _kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    #     )

    # if args.execute_ensemble_clustering:
    #     ensemble_name = f"{args.task_name}/{args.ensemble_id:04}"
    #     run_per_ensemble_job(
    #         ensemble_name,
    #         args.nP,
    #         args.gpu,
    #         args.q,
    #         "execute_ensemble_clustering.py",
    #         args.dry,
    #         kwargs,
    #     )

    # if args.execute_ensemble_notebook:
    #     ensemble_name = f"{args.task_name}/{args.ensemble_id:04}"
    #     run_per_ensemble_job(
    #         ensemble_name,
    #         4,
    #         "num=1",
    #         "gpu_rtx",
    #         "execute_ensemble_notebook.py",
    #         # add an argument for the notebook name
    #         args.dry,
    #         kwargs,
    #     )

    # if args.execute_training_and_tuning_overview:
    #     ensemble_name = f"{args.task_name}/{args.ensemble_id:04}"
    #     run_per_ensemble_job(
    #         ensemble_name,
    #         args.nP,
    #         "num=1",
    #         "gpu_rtx",
    #         "execute_training_and_tuning_overview.py",
    #         args.dry,
    #         kwargs,
    #     )

    # if args.execute_vanilla_cnn_validation:
    #     run_job(f"./submit.py vanilla_hex_cnn_evaluation.py --id {args.ensemble_id}", False)

    # if args.execute_single_model_notebook:
    #     run_ensemble_jobs(
    #         args.start,
    #         args.end,
    #         args.ensemble_id,
    #         args.task_name,
    #         args.nP,
    #         args.gpu,
    #         args.q,
    #         "execute_single_model_notebook.py",
    #         None,
    #         args.dry,
    #         args.optional_name,
    #         kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    #     )
