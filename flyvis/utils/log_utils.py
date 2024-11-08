import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import flyvis
from flyvis.network.ensemble import model_paths_from_parent


@dataclass
class Status:
    """Status object from log files of model runs.

    Attributes:
        ensemble_name: Name of the ensemble.
        log_files: List of all log files.
        train_logs: List of train logs.
        model_id_to_train_log_file: Mapping of model ID to log file.
        status: Mapping of model ID to status.
        user_input: Mapping of model ID to user input (behind LSF command).
        hosts: Mapping of model ID to host.
        rerun_failed_runs: Formatted submission commands to restart failed models.
        lsf_part: LSF command part.
    """

    ensemble_name: str
    log_files: List[Path]
    train_logs: List[Path]
    model_id_to_train_log_file: Dict[str, Path]
    status: Dict[str, str]
    user_input: Dict[str, str]
    hosts: Dict[str, List[str]]
    rerun_failed_runs: Dict[str, List[str]]
    lsf_part: str

    def print_for_rerun(
        self, exclude_failed_hosts: bool = True, model_ids: List[str] = None
    ) -> None:
        """Print formatted submission commands to restart failed models.

        Args:
            exclude_failed_hosts: Whether to exclude failed hosts.
            model_ids: List of model IDs to rerun. If None, all failed models are
                included.
        """
        model_ids = model_ids or list(self.rerun_failed_runs.keys())
        for model_id in model_ids:
            command = ""
            subcmds = self.rerun_failed_runs[model_id]
            if exclude_failed_hosts:
                command += f"{subcmds[0]}{subcmds[1]}{subcmds[2]}"
            else:
                command += f"{subcmds[0]}{subcmds[2]}"
            print(command)

    def get_hosts(self) -> List[str]:
        """Get hosts on which the job was executed."""
        return list(set(flatten_list(self.hosts.values())))

    def bad_hosts(self) -> List[str]:
        """Get hosts on which the job failed."""
        host_lists = [
            host
            for model_id, host in self.hosts.items()
            if self.status[model_id] not in ["Successfully completed.", "running"]
        ]
        return list(set(flatten_list(host_lists)))

    def successful_runs(self) -> int:
        """Get number of successful runs."""
        return sum(1 for v in self.status.values() if v == "Successfully completed.")

    def running_runs(self) -> int:
        """Get number of running runs."""
        return sum(1 for v in self.status.values() if v == "running")

    def failed_runs(self) -> int:
        """Get number of failed runs."""
        return sum(1 for v in self.status.values() if "Exited with exit code" in v)

    def successful_model_ids(self) -> List[str]:
        """Get model IDs of successful runs."""
        return [k for k, v in self.status.items() if v == "Successfully completed."]

    def running_model_ids(self) -> List[str]:
        """Get model IDs of running runs."""
        return [k for k, v in self.status.items() if v == "running"]

    def failed_model_ids(self) -> List[str]:
        """Get model IDs of failed runs."""
        return [k for k, v in self.status.items() if "Exited with exit code" in v]

    def lookup_log(
        self, model_id: str, log_type: str = "train_single", last_n_lines: int = 20
    ) -> List[str]:
        """Lookup log for a model ID.

        Args:
            model_id: ID of the model.
            log_type: Type of log to lookup.
            last_n_lines: Number of lines to return from the end of the log.

        Returns:
            List of log lines.
        """
        log_file = [
            p
            for p in self.log_files
            if log_type in str(p) and p.name.split("_")[0] == model_id
        ][0]
        return log_file.read_text().split("\n")[-last_n_lines:]

    def extract_error_trace(
        self, model_id: str, check_last_n_lines: int = 100, log_type: str = "train_single"
    ) -> str:
        """Extract the Python error message and traceback from a given log string.

        Args:
            model_id: ID of the model.
            check_last_n_lines: Number of lines to check from the end of the log.
            log_type: Type of log to extract error from.

        Returns:
            Extracted error message and traceback, or a message if no error is found.
        """
        log_string = "\n".join(
            self.lookup_log(model_id, last_n_lines=check_last_n_lines, log_type=log_type)
        )
        pattern = r"Traceback \(most recent call last\):(.+?)(?=\n\n|\Z)"
        match = re.search(pattern, log_string, re.DOTALL)
        return match.group(0).strip() if match else "No Python error found in the log."

    def extract_error_type(
        self, model_id: str, log_type: str = "train_single", check_last_n_lines: int = 100
    ) -> str:
        """Extract the type of error from a given log string.

        Args:
            model_id: ID of the model.
            log_type: Type of log to extract error from.
            check_last_n_lines: Number of lines to check from the end of the log.

        Returns:
            Extracted error type, or a message if no specific error type is found.
        """
        log_string = "\n".join(
            self.lookup_log(model_id, last_n_lines=check_last_n_lines, log_type=log_type)
        )
        pattern = r"\b[A-Z]\w*Error\b"
        match = re.search(pattern, log_string)
        return match.group(0) if match else "No specific error type found."

    def all_errors(
        self, check_last_n_lines: int = 100, log_type: str = "train_single"
    ) -> set:
        """Get all unique errors from failed runs.

        Args:
            check_last_n_lines: Number of lines to check from the end of the log.
            log_type: Type of log to extract errors from.

        Returns:
            Set of unique error messages.
        """
        return set(
            self.extract_error_trace(
                model_id,
                check_last_n_lines=check_last_n_lines,
                log_type=log_type,
            )
            for model_id in self.failed_model_ids()
        )

    def all_error_types(self, log_type: str = "train_single") -> set:
        """Get all unique error types from failed runs.

        Args:
            log_type: Type of log to extract error types from.

        Returns:
            Set of unique error types.
        """
        return set(
            self.extract_error_type(model_id, log_type=log_type)
            for model_id in self.failed_model_ids()
        )

    def print_all_errors(
        self, check_last_n_lines: int = 100, log_type: str = "train_single"
    ) -> None:
        """Print all errors and tracebacks from failed runs.

        Args:
            check_last_n_lines: Number of lines to check from the end of the log.
            log_type: Type of log to extract errors from.
        """
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

    def __getitem__(self, key: str) -> str:
        """Get status for a specific model ID."""
        if key in self.status:
            return self.status[key]
        return object.__getitem__(self, key)

    def __repr__(self) -> str:
        """Return a string representation of the Status object."""
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


def find_host(log_string: str) -> List[str]:
    """Find the host(s) on which the job was executed.

    Args:
        log_string: The log string to search for host information.

    Returns:
        List of host names.
    """
    pattern = r"executed on host\(s\) <(?:\d*\*)?(.+?)>,"
    return re.findall(pattern, log_string)


def get_exclude_host_part(log_string: str, exclude_hosts: Union[str, List[str]]) -> str:
    """Get the part of the LSF command that excludes hosts.

    Args:
        log_string: The log string to search for host information.
        exclude_hosts: Host(s) to exclude. Can be 'auto', a single host name, or a list
            of host names.

    Returns:
        The LSF command part for excluding hosts.
    """
    if exclude_hosts is None:
        return ""

    exclude_host_part = '-R "select[{}]" '

    if isinstance(exclude_hosts, str) and exclude_hosts == "auto":
        exclude_hosts = find_host(log_string)
    elif isinstance(exclude_hosts, str):
        exclude_hosts = [exclude_hosts]

    exclusion_strings = [f"hname!='{host}'" for host in exclude_hosts]
    return exclude_host_part.format(" && ".join(exclusion_strings))


def get_status(
    ensemble_name: str,
    nP: int = 4,
    gpu: str = "num=1",
    queue: str = "gpu_l4",
    exclude_hosts: Union[str, List[str]] = "auto",
) -> Status:
    """Get Status object for the ensemble of models with formatting for rerun.

    Args:
        ensemble_name: Ensemble name (e.g. "flow/<id>").
        nP: Number of processors.
        gpu: Number of GPUs.
        queue: Queue name.
        exclude_hosts: Host(s) to exclude. Can be 'auto', a single host name, or a
            list of host names.

    Returns:
        Status object containing information about the ensemble runs.
    """
    _lsf_part = "bsub -J {} -n {} -o {} -gpu '{}' -q {} "

    tnn_paths, path = model_paths_from_parent(flyvis.results_dir / ensemble_name)
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
        if log_str.split("\n")[-3] == "The output (if any) is above this job summary.":
            status[model_id] = log_str.split("\n")[-18]
            user_input[model_id] = log_str.split("\n")[-21]
        else:
            status[model_id] = "running"
            user_input[model_id] = ""

        hosts[model_id] = find_host(log_str)

    _lfs_cmd = _lsf_part
    rerun_failed_runs = {}
    for model_id, stat in status.items():
        if stat not in ["Successfully completed.", "running"]:
            _lsf_cmd = _lsf_part.format(
                f"{ensemble_name}/{model_id}",
                nP,
                model_id_to_train_log_file[model_id],
                gpu,
                queue,
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


def flatten_list(nested_list: List) -> List:
    """Flatten a nested list of lists into a single list with all elements.

    Args:
        nested_list: A nested list of lists to be flattened.

    Returns:
        A single flattened list with all elements.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened
