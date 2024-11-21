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
    argv = [
        "--ensemble_id",
        "0001",
        "train",
        "--task_name",
        "flow",
        "validate",
        "task.n_iters=100",
        "foo:int=1",
        "record",
    ]
    commands = ["train", "validate", "record"]
    selected_commands, filtered_args = filter_args(argv, commands)
    assert selected_commands == ["train", "validate", "record"]
    assert filtered_args == [
        "--ensemble_id",
        "0001",
        "--task_name",
        "flow",
        "task.n_iters=100",
        "foo:int=1",
    ]

    # Test with no commands to filter
    argv = ["--ensemble_id", "0001", "--task_name", "flow"]
    commands = ["train", "validate"]
    selected_commands, filtered_args = filter_args(argv, commands)
    assert selected_commands == []
    assert filtered_args == argv

    # Test with empty input
    selected_commands, filtered_args = filter_args([], [])
    assert selected_commands == []
    assert filtered_args == []

    # Test order preservation
    argv = ["validate", "--ensemble_id", "0001", "train", "--task_name", "flow"]
    commands = ["train", "validate"]
    selected_commands, filtered_args = filter_args(argv, commands)
    assert selected_commands == ["validate", "train"]  # Commands preserved in order found
    assert filtered_args == ["--ensemble_id", "0001", "--task_name", "flow"]


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
    "args,expected_behavior",
    [
        (
            ["--ensemble_id", "0001", "--task_name", "flow", "train"],
            {"should_succeed": True, "expected_command": "train"},
        ),
        (
            ["--help"],
            {"should_succeed": False, "return_code": 1},
        ),
        (
            [],
            {"should_succeed": False, "return_code": 1},
        ),
    ],
)
def test_main_argument_parsing(args, expected_behavior):
    """Test CLI argument parsing behavior."""
    with (
        patch('sys.argv', ['flyvis'] + args),
        patch('flyvis_cli.flyvis_cli.run_script') as mock_run,
    ):
        result = main()

        if expected_behavior["should_succeed"]:
            assert result == 0
            # Verify the correct script was called
            mock_run.assert_called_once()
            script_path = mock_run.call_args[0][0]
            assert script_path.name == f"{expected_behavior['expected_command']}.py"
        else:
            # Help and usage errors return 1
            assert result == expected_behavior["return_code"]


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
