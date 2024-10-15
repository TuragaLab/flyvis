# Pipeline Manager to Launch Multiple Scripts on Cluster

`pipeline_manager.py`

```
usage: pipeline_manager.py [-h] --ensemble_id ENSEMBLE_ID --task_name
                           TASK_NAME --command
                           {train,validate,record,analysis,notebook_per_model,notebook_per_ensemble}
                           [{train,validate,record,analysis,notebook_per_model,notebook_per_ensemble} ...]

Manage ensemble operations.

optional arguments:
  -h, --help            show this help message and exit
  --ensemble_id ENSEMBLE_ID
                        Id of the ensemble, e.g. 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g. flow.
  --command {train,validate,record,analysis,notebook_per_model,notebook_per_ensemble} [{train,validate,record,analysis,notebook_per_model,notebook_per_ensemble} ...]
                        Commands to run in order.

Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the cluster.
Each command corresponds to a script that launches required jobs.

train               : Runs training/train.py
validate            : Runs validation/validate.py
record              : Runs analysis/record.py
analysis            : Runs analysis/analysis.py
notebook_per_model  : Runs analysis/notebook_per_model.py
notebook_per_ensemble : Runs analysis/notebook_per_ensemble.py

All arguments after --command are passed directly to the respective scripts.
For detailed help on each command, run the individual script with --help.

```
