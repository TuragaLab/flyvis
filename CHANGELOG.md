# Changelog

## [Unreleased]

### Features
- Added precomputed response normalization constants for the released ensemble
  (`flyvis/data/responses_norm.h5`, ~40 kB, shipped with the package).
  `Ensemble.responses_norm` now loads them silently instead of simulating 30 minutes
  of naturalistic stimuli per model, which makes the paper figures reproducible on a
  laptop. Constants computed for custom ensembles are cached in the ensemble
  directory (`<ensemble_dir>/responses_norm.h5`). Constants are keyed by model name
  and validated against the SHA256 of the checkpoint they were computed from, so
  the order of an ensemble is irrelevant and a retrained checkpoint is recomputed
  rather than silently reused.
- Added `flyvis responses-norm` to compute and store the constants of an ensemble.
- Added `angular_tuning`, which returns the unnormalized speed- and width-averaged
  tuning that `plot_angular_tuning` plots, so that several curves can be put on a
  common scale.
- Added `model_reduction` to `plot_angular_tuning` to reduce across models with
  something other than the mean, e.g. the median, and `normalize_by` to override the
  per-curve normalization with a shared constant.
- Added `examples/figure_04_top_models.py`, which reproduces figure 4a,b for the
  models with the lowest task error instead of the task-optimal cluster.

### Infrastructure
- Added a `Release` workflow that builds and checks the distributions on every push
  to `main` and publishes them to PyPI via trusted publishing when a GitHub Release
  is published. It verifies that the version matches the release tag and that the
  precomputed response norms are present in both the wheel and the source
  distribution before uploading.
- Made the test suite deterministic. Five tests asserted on unseeded randomness and
  failed on roughly one run in seven between them. The global generators are now
  seeded before every test, `FLYVIS_TEST_SEED` re-runs the suite under a different
  seed, and the assertions that were only true for most draws were corrected.

## [v1.1.3] - 2026-03-07

### Bug Fixes
- Fixed `umap_embedding` `NotFittedError` when fewer than two rows have nonzero variance
- Fixed flash response index label alignment with filtered response index ordering
- Fixed `EnsembleView` `FileNotFoundError` in example notebooks by using explicit `best_checkpoint_fn_kwargs`

### Documentation
- Added pretrained model download step to example notebooks
- Updated installation documentation

### Infrastructure
- Updated package configuration in `pyproject.toml`

[v1.1.3]: https://github.com/TuragaLab/flyvis/releases/tag/v1.1.3

## [v1.1.2] - 2024-12-09

### Infrastructure
- Migrated to importlib.resources for file path handling
  - Improved package installation reliability
  - Standardized default script path definitions
  - Removed unused PROJECT_ROOT constant
- Added colorama dependency
- Added warning system for unrecognized CLI arguments with colored output
- Improved virtual cluster output handling
  - Added proper output file handling for virtual cluster jobs
  - Added automatic output directory creation
  - Improved process management for local execution
  - Fixed job status tracking to use job_id

### Bug Fixes
- Fixed file exists error
- Fixed notebook compatibility issues

### Documentation
- Renamed package from 'flyvision' to 'flyvis' across all documentation, examples
- Updated CLI documentation and usage examples
- Added optional notebook output clearing step to building docs
- Removed pyright and version pin notes from contribute.md
- Cleaned up CLI help text formatting
- Updated file paths in examples to use relative paths
- Removed duplicate CLI documentation from mkdocs.yml

[v1.1.2]: https://github.com/TuragaLab/flyvis/releases/tag/v1.1.2

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
