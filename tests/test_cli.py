from unittest.mock import patch

import pytest

from flyvis_cli.flyvis_cli import (
    SCRIPT_COMMANDS,
    filter_args,
    handle_help_request,
    main,
)


def test_filter_args():
    # Test filtering out commands from arguments
    argv = ["--ensemble_id", "0001", "train", "--task_name", "flow", "validate"]
    commands = ["train", "validate"]
    filtered = filter_args(argv, commands)
    assert filtered == ["--ensemble_id", "0001", "--task_name", "flow"]

    # Test with no commands to filter
    argv = ["--ensemble_id", "0001", "--task_name", "flow"]
    commands = ["train", "validate"]
    filtered = filter_args(argv, commands)
    assert filtered == argv

    # Test with empty input
    assert filter_args([], []) == []


def test_handle_help_request():
    # Test help request for a specific command
    with patch('flyvis_cli.flyvis_cli.run_script') as mock_run:
        argv = ["flyvis", "train", "--help"]
        assert handle_help_request(argv) is True
        mock_run.assert_called_once_with(SCRIPT_COMMANDS["train"], ["--help"])

    # Test no help request
    with patch('flyvis_cli.flyvis_cli.run_script') as mock_run:
        argv = ["flyvis", "train", "--ensemble_id", "0001"]
        assert handle_help_request(argv) is False
        mock_run.assert_not_called()


@pytest.mark.parametrize(
    "args,expected_return",
    [
        (["--ensemble_id", "0001", "--task_name", "flow", "train"], 0),
        (["--help"], 0),  # Help message exits with 0
        ([""], 2),  # Invalid flag exits with 2
    ],
)
def test_main_argument_parsing(args, expected_return):
    with (
        patch('sys.argv', ['flyvis'] + args),
        patch('flyvis_cli.flyvis_cli.run_script') as mock_run,
    ):
        if expected_return != 0:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == expected_return
        else:
            try:
                assert main() == expected_return
            except SystemExit as exc:
                assert exc.code == expected_return
            if "--help" not in args:
                mock_run.assert_called()


def test_main_runs_multiple_commands():
    args = ["--ensemble_id", "0001", "--task_name", "flow", "train", "validate"]
    with (
        patch('sys.argv', ['flyvis'] + args),
        patch('flyvis_cli.flyvis_cli.run_script') as mock_run,
    ):
        main()
        assert mock_run.call_count == 2
        # Verify calls were made in the correct order
        expected_calls = [
            (
                (
                    SCRIPT_COMMANDS["train"],
                    ["--ensemble_id", "0001", "--task_name", "flow"],
                ),
            ),
            (
                (
                    SCRIPT_COMMANDS["validate"],
                    ["--ensemble_id", "0001", "--task_name", "flow"],
                ),
            ),
        ]
        mock_run.assert_has_calls(expected_calls)
