# Launch Notebook Per Ensemble on Compute Cloud


::: flyvis_cli.analysis.notebook_per_ensemble
    options:
      heading_level: 4


```
usage:
flyvis notebook-per-ensemble [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME --notebook_path PATH
       or
notebook_per_ensemble.py [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME --notebook_path PATH

Run ensemble notebook on the compute cloud.

options:
  -h, --help            show this help message and exit
  --nP NP               Number of processors.
  --gpu GPU             Number of GPUs.
  --q Q                 Queue.
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., flow.
  --notebook_path NOTEBOOK_PATH
                        Path of the notebook to execute. Default: /groups/tura
                        ga/home/lappalainenj/FlyVis/private/flyvision/flyvis/a
                        nalysis/__main__.ipynb
  --dry                 Perform a dry run without actually launching jobs.

```
