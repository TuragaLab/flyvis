# Command Line Interface

The command line interface provides a set of tools for managing training, validating, recording, and analyzing models.

## Basic Syntax

```bash
flyvis [COMMANDS] [OPTIONS]
```

## Basic Commands

The following ordered commands represent a complete pipeline:

- `train` - Train models
- `validate` - Run validation on trained models
- `record` - Record model responses
- `analysis` - Perform analysis on model results
- `notebook-per-model` - Generate individual model analysis notebooks
- `notebook-per-ensemble` - Generate ensemble analysis notebooks

Run as:
```
flyvis train validate record analysis notebook-per-model notebook-per-ensemble [OPTIONS]
```

### Common Options

- `--ensemble_id ENSEMBLE_ID` - Unique identifier for the model ensemble (e.g., "0001")
- `--task_name TASK_NAME` - Name of the task to execute (e.g., "flow")


## Other Commands
Other commands available are:

- `train-single` - Train a single model
- `val-single` - Validate a single model
- `synthetic-recordings-single` - Record responses for a single model
- `ensemble-analysis` - Perform analysis on an ensemble
- `download-pretrained-models` - Download pretrained models
- `notebook` - Run a notebook

See the [cli entry point](flyvis_cli/flyvis.md) page for more information or run `flyvis --help` for a full list of commands.

## Script Reference

The following scripts are called by the commands above:

#### Training Scripts
- [`train`](flyvis_cli/training/train.md) - Main training script for model ensembles
- [`train_single`](flyvis_cli/training/train_single.md) - Training script for individual models

#### Validation Scripts
- [`validate`](flyvis_cli/validation/validate.md) - Main validation script for model ensembles
- [`val_single`](flyvis_cli/validation/val_single.md) - Validation script for individual models

#### Analysis Scripts
- [`record`](flyvis_cli/analysis/record.md) - Record model responses
- [`synthetic_recordings_single`](flyvis_cli/analysis/synthetic_recordings_single.md) - Generate synthetic recordings for individual models
- [`analysis`](flyvis_cli/analysis/analysis.md) - Launch analysis script for model ensembles
- [`ensemble_analysis`](flyvis_cli/analysis/ensemble_analysis.md) - Analysis script for model ensembles

#### Notebook Generation
- [`notebook_per_model`](flyvis_cli/analysis/notebook_per_model.md) - Generate analysis notebooks for individual models
- [`notebook_per_ensemble`](flyvis_cli/analysis/notebook_per_ensemble.md) - Generate analysis notebooks for ensembles
- [`notebook`](flyvis_cli/analysis/notebook.md) - General notebook execution script

#### Utilities
- [`download_pretrained_models`](flyvis_cli/download_pretrained_models.md) - Download pre-trained models


## Example Usage

Some common usage patterns for the `flyvis` CLI:

```bash
# Example 1: Display the default training configuration
flyvis train --help

# Example 2: Full pipeline for an ensemble 0001 with 4 models on the flow task with defaults (dry run)
flyvis \
    train validate record analysis notebook-per-model notebook-per-ensemble \
    --ensemble_id 0001 \
    --task_name flow \
    --start 0 \
    --end 4 \
    --dry

# Example 3: (Re)run the recording of responses and analysis for the existing ensemble 0001
flyvis \
    validate record analysis \
    --ensemble_id 0001 \
    --task_name flow

# Example 4: Run analysis and generate notebooks for an existing ensemble 0003 with defaults
flyvis \
    analysis notebook-per-model notebook-per-ensemble \
    --ensemble_id 0003 \
    --task_name flow

# Example 5: Full pipeline for ensemble 0004 on the flow task, with custom ensemble analysis notebook
flyvis \
    train validate record analysis notebook-per-model notebook-per-ensemble \
    --ensemble_id 0004 \
    --task_name flow \
    --notebook_path custom_notebook.ipynb

# Example 6: Run only the record and analysis steps for existing ensemble 0005 on the depth task, with a dry run
flyvis \
    record analysis \
    --ensemble_id 0005 \
    --task_name depth \
    --dry

# Example 7: Full pipeline for ensemble 0006 on the flow task, with custom training settings
flyvis \
    train validate record analysis notebook-per-model notebook-per-ensemble \
    --ensemble_id 0006 \
    --task_name flow \
    --start 0 \
    --end 5 \
    --nP 8 \
    task.n_iters=1000  # note: this combines hydra and argparse syntax

# Example 8: Run a notebook for each model in ensemble 0001
flyvis notebook-per-model --notebook_per_model_path notebooks/record_custom_stimuli_responses.ipynb \
    --ensemble_id 0001 \
    --task_name flow \
    stim_height:int=4 \
    stim_width:int=2 \
    speed:float=25

# Example 9: Run a notebook for the entire ensemble 0001
flyvis notebook-per-ensemble --notebook_path notebooks/analyze_custom_stimuli_responses.ipynb \
    --ensemble_id 0001 \
    --task_name flow
```

## Additional Notes

- Note that all arguments are passed through from the entry point to the underlying
scripts, so you can use all scripts through the entry-point `flyvis`.
- The CLI combines the usage of argparse arguments like `--ensemble_id` and `--task_name`,
hydra arguments like `task.n_iters` and `solver.optim.lr`, and passes also typed
arguments, like `ensemble_id:str=0001` to paperpile for notebook execution (required arguments depend on the notebook definition).
- See the help menu for each script for more information. This is not exhaustively tested
in all available configurations, please report any issues on the GitHub repository.
