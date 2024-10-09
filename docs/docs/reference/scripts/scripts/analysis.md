# Analysis Help

Script: `scripts/scripts/analysis.py`

```
usage: analysis.py [-h] [--nP NP] [--gpu GPU] [--q Q] --ensemble_id
                   ENSEMBLE_ID --task_name TASK_NAME
                   [--ensemble_analysis_script ENSEMBLE_ANALYSIS_SCRIPT]
                   [--dry]

Run ensemble analysis on the cluster.

optional arguments:
  -h, --help            show this help message and exit
  --nP NP               Number of processors.
  --gpu GPU             Number of GPUs.
  --q Q                 Queue.
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., flow.
  --ensemble_analysis_script ENSEMBLE_ANALYSIS_SCRIPT
                        Script to run for ensemble analysis.
  --dry                 Perform a dry run without actually launching jobs.

```
