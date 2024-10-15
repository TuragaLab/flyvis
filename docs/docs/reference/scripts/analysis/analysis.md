# Launch Ensemble Analysis on Cluster

`analysis/analysis.py`

```
usage: analysis.py [-h] [--nP NP] [--gpu GPU] [--q Q] --ensemble_id
                   ENSEMBLE_ID --task_name TASK_NAME
                   [--ensemble_analysis_script ENSEMBLE_ANALYSIS_SCRIPT]
                   [--dry]

Run ensemble analysis on the cluster.

optional arguments:
  -h, --help            show this help message and exit
  --nP NP               Number of processors to use (default: 4).
  --gpu GPU             GPU configuration (default: 'num=1').
  --q Q                 Queue to submit the job to.
  --ensemble_id ENSEMBLE_ID
                        ID of the ensemble, e.g., 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., 'flow', 'depth', 'lum'.
  --ensemble_analysis_script ENSEMBLE_ANALYSIS_SCRIPT
                        Script to run for ensemble analysis (default: /groups/
                        turaga/home/lappalainenj/FlyVis/private/flyvision/scri
                        pts/analysis/ensemble_analysis.py).
  --dry                 Perform a dry run without actually launching jobs.

Examples:
    python analysis.py --ensemble_id 0045 --task_name flow


```
