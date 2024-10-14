# Val Single

`scripts/validation/val_single.py`

```
usage: val_single.py [-h] [--ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID]
                     [--task_name TASK_NAME]

Validate a single network.

optional arguments:
  -h, --help            show this help message and exit

Hybrid Arguments:
  --ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID
                        ensemble_and_network_id=value: ID of the ensemble and
                        network to use, e.g. 0045/000 (Required)
  --task_name TASK_NAME
                        task_name=value: Name of the task. Resulting network
                        name will be task_name/ensemble_and_network_id.
                        (Required)

```
