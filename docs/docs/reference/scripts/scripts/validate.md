# Validate

`scripts/scripts/validate.py`

```
usage: validate.py [-h] [--start START] [--end END] [--nP NP] [--gpu GPU]
                   [--q Q] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME
                   [--val_script VAL_SCRIPT] [--dry]

Validate each model of an ensemble on the cluster.

optional arguments:
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
                        Script to run for validation.
  --dry                 Perform a dry run without actually launching jobs.

```
