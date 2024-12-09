# Launch Ensemble Validation on Compute Cloud


::: flyvis_cli.validation.validate
    options:
      heading_level: 4


```
usage:
flyvis validate [-h] [--start START] [--end END] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [additional_options...]
       or
validate.py [-h] [--start START] [--end END] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [additional_options...]

For a full list of validation options and default arguments, run: flyvis val_single --help

Validate each model of an ensemble on the compute cloud.

options:
  -h, --help            show this help message and exit
  --start START         Start id of ensemble.
  --end END             End id of ensemble.
  --nP NP               Number of processors.
  --gpu GPU             Number of GPUs.
  --q Q                 Queue.
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., flow.
  --val_script VAL_SCRIPT
                        Script to run for validation. Default: /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/validation/val_single.py
  --dry                 Perform a dry run without actually launching jobs.

```
