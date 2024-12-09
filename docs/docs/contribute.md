# Contributing

Please submit an issue on
[GitHub](https://github.com/turagalab/flyvis/issues) for any suggestions for improving
the code, bug reports, or questions. We will attend to issues and pull requests as time allows
and appreciate any feedback! To get involved, please read the [developer guide](#developer-guide).


# Developer Guide

## Project Setup

This project uses Python and is built with setuptools. It requires Python 3.6 or higher.

### Installation

To set up the development environment:

1. Clone the repository:
   ```
   git clone https://github.com/turagalab/flyvis.git
   cd flyvision
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the package with development dependencies:
   ```
   pip install -e ".[dev]"
   ```

## Development Tools

### Code Formatting and Linting

We use the following tools for code quality:

- **Ruff**: For linting and code formatting. Configuration is in `pyproject.toml`.
- **Pre-commit**: To run checks before committing.

To set up pre-commit hooks:

```
pre-commit install
```


### Testing

We use pytest for testing. Run tests with:

```
pytest
```

Pytest configuration is in `pyproject.toml`. Notable markers:
- `slow`: For time-consuming tests
- `gpu`: For tests requiring a GPU
- `require_large_download`: For tests needing large data downloads
- `require_download`: For tests needing data downloads

To run tests excluding certain markers:

```
pytest -m "not slow and not gpu and not require_large_download and not require_download"
```


### Documentation

Documentation is built using MkDocs. To build and serve the documentation locally:

```
mkdocs serve
```

To build the documentation:

### Dependency Management

- Main dependencies are listed in `pyproject.toml` under `[project.dependencies]`.
- Development dependencies are under `[project.optional-dependencies.dev]`.
- Documentation dependencies are under `[project.optional-dependencies.docs]`.

To install all dependencies including development and documentation:

```
pip install -e ".[dev,docs]"
```


### Version Management

This project uses `setuptools_scm` for versioning. The version is automatically derived from git tags.

## Project Structure

- `flyvision/`: Main package directory
- `tests/`: Test files
- `docs/`: Documentation files
- `examples/`: Example scripts and notebooks
- `scripts/`: Utility scripts
- `config/`: Configuration files
- `data/`: Data files

## Contribution Guidelines

1. Fork the repository and create your branch from `main`.
2. Ensure the test suite passes by running `pytest`.
3. Make sure your code follows the project's style guide (enforced by Ruff).
4. Update documentation as necessary.
5. Create a pull request with a clear description of your changes.

## Additional Notes

- The project includes optional dependencies for development, documentation, and examples. Install them as needed.

For any questions or issues, please open an issue on the project's GitHub repository.
