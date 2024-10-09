# Record Help

Script: `scripts/scripts/record.py`

```
usage: record.py [-h] [--start START] [--end END] [--nP NP] [--gpu GPU]
                 [--q Q] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME
                 [--synthetic_recordings_script SYNTHETIC_RECORDINGS_SCRIPT]
                 [--dry]

Run synthetic recordings for each model of an ensemble on the cluster.

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
  --synthetic_recordings_script SYNTHETIC_RECORDINGS_SCRIPT
                        Script to run for synthetic recordings.
  --dry                 Perform a dry run without actually launching jobs.

```
