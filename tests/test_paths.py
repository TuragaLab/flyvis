import os
import tempfile
from pathlib import Path

import flyvis


def test_default_root_dir():
    """Test that the default root directory is correctly set relative to the package."""
    # Clear any existing FLYVIS_ROOT_DIR environment variable
    if "FLYVIS_ROOT_DIR" in os.environ:
        del os.environ["FLYVIS_ROOT_DIR"]

    # Get the resolved root directory
    root_dir = flyvis.resolve_root_dir()

    # Expected default is repo_dir/data
    expected_dir = flyvis.repo_dir / "data"
    assert root_dir == expected_dir.absolute()


def test_env_var_root_dir():
    """Test that FLYVIS_ROOT_DIR environment variable is respected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set environment variable to temp directory
        os.environ["FLYVIS_ROOT_DIR"] = temp_dir

        # Get the resolved root directory
        root_dir = flyvis.resolve_root_dir()

        assert root_dir == Path(temp_dir).absolute()


def test_dotenv_file():
    """Test that .env file is properly loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a .env file
        env_content = f'FLYVIS_ROOT_DIR={temp_dir}'
        with open(temp_path / '.env', 'w') as f:
            f.write(env_content)

        # Clear any existing environment variable
        if "FLYVIS_ROOT_DIR" in os.environ:
            del os.environ["FLYVIS_ROOT_DIR"]

        # Change to temp directory and reload environment
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            import dotenv

            dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

            # Get the resolved root directory
            root_dir = flyvis.resolve_root_dir()

            assert root_dir == temp_path.absolute()

        finally:
            # Restore original directory
            os.chdir(original_dir)


def test_path_expansion():
    """Test that user path expansion works correctly."""
    test_path = "~/flyvis_test"
    os.environ["FLYVIS_ROOT_DIR"] = test_path

    root_dir = flyvis.resolve_root_dir()
    expected_dir = Path(test_path).expanduser().absolute()

    assert root_dir == expected_dir
