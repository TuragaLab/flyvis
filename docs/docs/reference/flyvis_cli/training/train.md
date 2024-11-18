# Launch Ensemble Training on Compute Cloud


::: flyvis_cli.training.train
    options:
      heading_level: 4


```
usage:
flyvis train [-h] [--start START] [--end END] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [hydra_options...]
       or
train.py [-h] [--start START] [--end END] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [hydra_options...]

For a full list of hydra options and default arguments, run: flyvis train-single --help

Train an ensemble of models. Launches a job for each model on the compute cloud.

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
  --train_script TRAIN_SCRIPT
                        Script to run for training. Default: /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/training/train_single.py
  --dry                 Perform a dry run without actually launching jobs.

```
