# Launch Notebook Per Ensemble on Cluster

`analysis/notebook_per_ensemble.py`

```
usage: notebook_per_ensemble.py [-h] [--nP NP] [--gpu GPU] [--q Q]
                                --ensemble_id ENSEMBLE_ID --task_name
                                TASK_NAME [--notebook_path NOTEBOOK_PATH]
                                [--dry]

Run ensemble notebook on the cluster.

optional arguments:
  -h, --help            show this help message and exit
  --nP NP               Number of processors.
  --gpu GPU             Number of GPUs.
  --q Q                 Queue.
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., flow.
  --notebook_path NOTEBOOK_PATH
                        Path of the notebook to execute.
  --dry                 Perform a dry run without actually launching jobs.

```
