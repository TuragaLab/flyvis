import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import flyvision
from flyvision.ensemble import model_paths_from_parent


@dataclass
class Status:
    """Status object from log files of model runs.

    Attributes:
        log_files (List[Path]): list of all log files
        train_logs (List[Path]): list of train logs
        model_id_to_train_log_file (Dict[str, Path]): model id to log file
        status (Dict[str, Path]): model id to status
        user_input (Dict[str, Path]): model id to user input (behind lsf command)
        hosts (Dict): model id to host
        rerun_failed_runs (List[str]): formatted submission commands to restart failed
            models

    """

    ensemble_name: str
    log_files: List[Path]
    train_logs: List[Path]
    model_id_to_train_log_file: Dict[str, Path]
    status: Dict[str, Path]
    user_input: Dict[str, Path]
    hosts: Dict
    rerun_failed_runs: Dict[str, List[str]]
    lsf_part: str

    def print_for_rerun(self, exclude_failed_hosts=True, model_ids: List[str] = None):
        """Formatted submission commands to restart failed models."""
        model_ids = model_ids or list(self.rerun_failed_runs.keys())
        for model_id in model_ids:
            command = ""
            subcmds = self.rerun_failed_runs[model_id]
            if exclude_failed_hosts:
                command += f"{subcmds[0]}{subcmds[1]}{subcmds[2]}"
            else:
                command += f"{subcmds[0]}{subcmds[2]}"
            print(command, end="\n")

    def get_hosts(self):
        """Hosts on which the job was executed."""
        return list(set(flatten_list(self.hosts.values())))

    def bad_hosts(self):
        """Hosts on which the job failed."""
        host_lists = [
            host
            for model_id, host in self.hosts.items()
            if self.status[model_id] not in ["Successfully completed.", "running"]
        ]
        return list(set(flatten_list(host_lists)))

    def successful_runs(self):
        """Number of successful runs."""
        return sum([1 for k, v in self.status.items() if v == "Successfully completed."])

    def running_runs(self):
        """Number of running runs."""
        return sum([1 for k, v in self.status.items() if v == "running"])

    def failed_runs(self):
        """Number of failed runs."""
        return sum([1 for k, v in self.status.items() if "Exited with exit code" in v])

    def successful_model_ids(self):
        """Model ids of successful runs."""
        return [k for k, v in self.status.items() if v == "Successfully completed."]

    def running_model_ids(self):
        """Model ids of running runs."""
        return [k for k, v in self.status.items() if v == "running"]

    def failed_model_ids(self):
        """Model ids of failed runs."""
        return [k for k, v in self.status.items() if "Exited with exit code" in v]

    def lookup_log(self, model_id, log_type="train_single", last_n_lines=20):
        """Lookup log for a model id."""
        log_file = [
            p
            for p in self.log_files
            if log_type in str(p) and p.name.split("_")[0] == model_id
        ][0]
        return log_file.read_text().split("\n")[-last_n_lines:]

    def extract_error_trace(
        self, model_id, check_last_n_lines=100, log_type="train_single"
    ):
        """Extracts the Python error message and traceback from a given log string."""
        log_string = "\n".join(
            self.lookup_log(model_id, last_n_lines=check_last_n_lines, log_type=log_type)
        )
        # regex pattern to match the error message and traceback
        pattern = r"Traceback \(most recent call last\):(.+?)(?=\n\n|\Z)"

        # search for the pattern in the log string
        match = re.search(pattern, log_string, re.DOTALL)

        if match:
            return match.group(
                0
            ).strip()  # return the matched group without leading/trailing whitespace
        else:
            return "No Python error found in the log."

    def extract_error_type(
        self, model_id, log_type="train_single", check_last_n_lines=100
    ):
        """Extracts the type of error from a given log string."""
        log_string = "\n".join(
            self.lookup_log(model_id, last_n_lines=check_last_n_lines, log_type=log_type)
        )
        # regex pattern to match the error type
        pattern = r"\b[A-Z]\w*Error\b"

        # search for the pattern in the log string
        match = re.search(pattern, log_string)

        if match:
            return match.group(0)
        else:
            return "No specific error type found."

    def all_errors(self, check_last_n_lines=100, log_type="train_single"):
        errors = []
        for model_id in self.failed_model_ids():
            errors.append(
                self.extract_error_trace(
                    model_id,
                    check_last_n_lines=check_last_n_lines,
                    log_type=log_type,
                )
            )
        return set(errors)

    def all_error_types(self, log_type="train_single"):
        error_types = []
        for model_id in self.failed_model_ids():
            error_types.append(self.extract_error_type(model_id, log_type=log_type))
        return set(error_types)

    def print_all_errors(self, check_last_n_lines=100, log_type="train_single"):
        for model_id in self.failed_model_ids():
            print(
                f"Model {model_id} failed with the following error message "
                "and traceback:\n"
            )
            print(
                self.extract_error_trace(
                    model_id,
                    check_last_n_lines=check_last_n_lines,
                    log_type=log_type,
                )
            )
            print("\n")

    def __getitem__(self, key):
        if key in self.status:
            return self.status[key]
        return object.__getitem__(self, key)

    def __repr__(self):
        _repr = f"Status of ensemble {self.ensemble_name}."
        _repr += f"\n{len(self.status)} models."
        _repr += f"\nHosts: {','.join(self.get_hosts())}."
        _repr += f"\n  {self.successful_runs()} successful runs."
        _repr += f"\n  {self.running_runs()} running runs."
        _repr += f"\n  {self.failed_runs()} failed runs."
        if self.failed_runs() > 0:
            _repr += f"\n  Bad hosts: {','.join(self.bad_hosts())}."
            _repr += "\n  Use .print_for_rerun() to print formatted submission commands"
            _repr += " to restart failed models."
            _repr += "\nError types:"
            for error in self.all_error_types():
                _repr += f"\n  {error}"
            _repr += (
                "\n  Run .print_all_errors() to print the error messages and tracebacks."
            )
        return _repr


def find_host(log_string):
    """Find the host on which the job was executed."""
    pattern = r"executed on host\(s\) <(?:\d*\*)?(.+?)>,"
    # find all occurrences of the pattern
    hosts = re.findall(pattern, log_string)
    # assert len(hosts) == 1
    return hosts


def get_exclude_host_part(log_string, exclude_hosts: Union[str, List[str]]):
    """Get the part of the lsf command that excludes hosts."""
    if exclude_hosts is None:
        return ""

    exclude_host_part = '-R "select[{}]" '

    # case 1: exclude_hosts is a str and 'auto'
    if isinstance(exclude_hosts, str) and exclude_hosts == "auto":
        exclude_hosts = find_host(log_string)
    # case 2: exclude_hosts is a str and not 'auto', assume its a host name
    elif isinstance(exclude_hosts, str):
        exclude_hosts = [exclude_hosts]
    # case 3: exclude hosts is a list of str
    else:
        pass

    exclusion_strings = []
    for host in exclude_hosts:
        exclusion_string = "hname!='{}'".format(host)
        exclusion_strings.append(exclusion_string)

    exclude_host_part = exclude_host_part.format(" && ".join(exclusion_strings))
    return exclude_host_part


def get_status(
    ensemble_name: str,
    nP=4,
    gpu="num=1",
    queue="gpu_l4",
    force=False,
    exclude_hosts: Union[str, List[str]] = "auto",
) -> Status:
    """Get Status object for the ensemble of models with formmatting for rerun.

    Args:
        ensemble_name (str): ensemble name (e.g. "flow/<id>")
        nP (int, optional): number of processors. Defaults to 4.
        gpu (str, optional): number of gpus. Defaults to "num=1".
        queue (str, optional): queue. Defaults to "gpu_l4".
        force (bool, optional): force rerun. Defaults to False.
        exclude_hosts (Union[str, List[str]], optional): host to exclude.
            Defaults to "auto". I.e. hosts on which the run failed will be
            excluded in the next run.
    """

    _lsf_part = "bsub -J {} -n {} -o {} -gpu '{}' -q {} "

    tnn_paths, path = model_paths_from_parent(flyvision.results_dir / ensemble_name)
    log_files = [p for p in path.iterdir() if p.suffix == ".log"]
    train_logs = [p for p in log_files if "train_single" in str(p)]
    model_id_to_train_log_file = {p.name.split("_")[0]: p for p in train_logs}

    status = {}
    user_input = {}
    hosts = {}
    log_strings = {}
    for p in train_logs:
        model_id = p.name.split("_")[0]
        log_str = p.read_text()
        log_strings[model_id] = log_str
        # make sure the job has finished, else its running
        if log_str.split("\n")[-3] == "The output (if any) is above this job summary.":
            # line -18 is LSF's return status
            status[model_id] = log_str.split("\n")[-18]
            # line -21 is user input to LSF job
            user_input[model_id] = log_str.split("\n")[-21]

        else:
            status[model_id] = "running"
            user_input[model_id] = ""

        hosts[model_id] = find_host(log_str)

    # if all failed for the same reason, it is probably not a cluster f*** but
    # a bug in the codebase, so we make sure they failed for different reasons
    _lfs_cmd = _lsf_part
    rerun_failed_runs = {}
    for model_id, stat in status.items():
        if stat not in ["Successfully completed.", "running"]:
            _lsf_cmd = _lsf_part.format(
                f"{ensemble_name}/{model_id}",
                nP,  # -n {} num proc
                model_id_to_train_log_file[model_id],  # -o {} output file name
                gpu,  # -gpu '{}' number of gpus
                queue,  # -q {} queue
            )
            exclude_host_part = get_exclude_host_part(
                log_strings[model_id], exclude_hosts
            )
            rerun_failed_runs[model_id] = [
                _lsf_cmd,
                exclude_host_part,
                user_input[model_id],
            ]
    return Status(
        ensemble_name,
        log_files,
        train_logs,
        model_id_to_train_log_file,
        status,
        user_input,
        hosts,
        rerun_failed_runs,
        _lfs_cmd,
    )


def flatten_list(nested_list):
    """
    Flatten a nested list of lists into a single list with all elements.

    Args:
    nested_list (list): A nested list of lists to be flattened.

    Returns:
    list: A single flattened list with all elements.
    """
    # use a recursive approach to handle arbitrary nesting levels
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened
