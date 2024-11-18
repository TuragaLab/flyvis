# Run Validation for Single Model


::: flyvis_cli.validation.val_single
    options:
      heading_level: 4


```
usage:
flyvis val-single [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY
       or
val_single.py [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY

Validate a single network across all its checkpoints. Computes and stores validation metrics in the network's validation directory.

options:
  -h, --help            show this help message and exit

Hybrid Arguments:
  --ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID
                        ensemble_and_network_id=value: ID of the ensemble and
                        network to use, e.g. 0045/000 (Required)
  --task_name TASK_NAME
                        task_name=value: Name of the task. Resulting network
                        name will be task_name/ensemble_and_network_id.
                        (Required)

Examples:
--------
1. Validate a specific network:
    flyvis val-single task_name=flow ensemble_and_network_id=0045/000

2. Validate a network from a different task:
    flyvis val-single task_name=depth ensemble_and_network_id=0023/012

```
