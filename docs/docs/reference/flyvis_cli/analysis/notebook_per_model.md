# Launch Notebook Per Model on Compute Cloud


::: flyvis_cli.analysis.notebook_per_model
    options:
      heading_level: 4


```
usage:
flyvis notebook-per-model [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME --notebook_per_model_path PATH
       or
notebook_per_model.py [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME --notebook_per_model_path PATH

Run a notebook for each model of an ensemble on the compute cloud.

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
  --notebook_per_model_path NOTEBOOK_PER_MODEL_PATH
                        Path of the notebook to execute. Default: /groups/tura
                        ga/home/lappalainenj/FlyVis/private/flyvision/flyvis/a
                        nalysis/__main_per_model__.ipynb
  --dry                 Perform a dry run without actually launching jobs.

```
