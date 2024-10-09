# Notebook Per Model Help

Script: `scripts/scripts/notebook_per_model.py`

```
usage: notebook_per_model.py [-h] [--start START] [--end END] [--nP NP]
                             [--gpu GPU] [--q Q] --ensemble_id ENSEMBLE_ID
                             --task_name TASK_NAME
                             [--notebook_script NOTEBOOK_SCRIPT]
                             --notebook_path NOTEBOOK_PATH --output_path
                             OUTPUT_PATH [--dry]

Run a notebook for each model of an ensemble on the cluster.

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
  --notebook_script NOTEBOOK_SCRIPT
                        Script to run for executing notebooks.
  --notebook_path NOTEBOOK_PATH
                        Path of the notebook to execute, e.g.
                        /path/to/__main__.ipynb.
  --output_path OUTPUT_PATH
                        Path for the output notebook.
  --dry                 Perform a dry run without actually launching jobs.

```
