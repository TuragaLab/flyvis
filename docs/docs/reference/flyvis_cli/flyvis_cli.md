# Command Line Interface Entry Point


::: flyvis_cli.flyvis_cli
    options:
      heading_level: 4


```
usage: flyvis_cli.py [-h] [--ensemble_id ENSEMBLE_ID] [--task_name TASK_NAME]
                     command [command ...]

Run flyvis pipelines or individual scripts with compute cloud options.

positional arguments:
  command               Commands to run in order.

options:
  -h, --help            show this help message and exit
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g. flow.

Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the compute cloud.
Each command corresponds to a script that launches required jobs.

train                : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/training/train.py
train-single         : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/training/train_single.py
validate             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/validation/validate.py
val-single           : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/validation/val_single.py
record               : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/record.py
synthetic-recordings-single : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/synthetic_recordings_single.py
analysis             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/analysis.py
ensemble-analysis    : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/ensemble_analysis.py
notebook-per-model   : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook_per_model.py
notebook-per-ensemble : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook_per_ensemble.py
notebook             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook.py
download-pretrained  : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/download_pretrained_models.py

All additional arguments are passed directly to the respective scripts.
For detailed help on each command, run: flyvis <command> --help
usage: flyvis_cli.py [-h] [--ensemble_id ENSEMBLE_ID] [--task_name TASK_NAME]
                     command [command ...]

Run flyvis pipelines or individual scripts with compute cloud options.

positional arguments:
  command               Commands to run in order.

options:
  -h, --help            show this help message and exit
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g. flow.

Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the compute cloud.
Each command corresponds to a script that launches required jobs.

train                : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/training/train.py
train-single         : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/training/train_single.py
validate             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/validation/validate.py
val-single           : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/validation/val_single.py
record               : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/record.py
synthetic-recordings-single : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/synthetic_recordings_single.py
analysis             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/analysis.py
ensemble-analysis    : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/ensemble_analysis.py
notebook-per-model   : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook_per_model.py
notebook-per-ensemble : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook_per_ensemble.py
notebook             : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/notebook.py
download-pretrained  : Runs /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/download_pretrained_models.py

All additional arguments are passed directly to the respective scripts.
For detailed help on each command, run: flyvis <command> --help

```
