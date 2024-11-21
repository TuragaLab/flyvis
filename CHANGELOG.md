# Changelog

## [v1.1.1] - 2024-11-21

### Distribution
- Moved configuration files into package structure
- Switched to `importlib.resources` for resource management
- Removed MANIFEST.in in favor of pyproject.toml configuration
- Enhanced root directory resolution logic

### Infrastructure
- Improved cluster management with better dry-run and slurm support

### CLI
- Enhanced command-line interface error handling
- Improved argument parsing for multiple commands
- Added `init_config` for config-based customization of network and training

### Documentation
- Updated package metadata for PyPI
- Added project URLs to package configuration
- Removed broken badges from README
- Added explanations for hydra-config-based customization of network and training in `CLI Reference`

## [v1.1.0] - 2024-11-20

### Breaking
- Renamed package from `flyvision` to `flyvis` for better consistency

### Features
- Added Command Line Interface `flyvis` for scripts
- Improved test suite performance and coverage
- Relaxed Python version dependency and added multi-Python version testing

### Documentation
- Improved docs
- Updated project metadata for PyPI
- Removed broken badges from README

### Infrastructure
- Removed strict version pins for better compatibility (particularly removed UMAP constraints)
- Updated GitHub workflows
- Updated README badges
- Updated package-data handling

[v1.1.0]: https://github.com/TuragaLab/flyvis/releases/tag/v1.1.0
