# Run Synthetic Recordings


::: flyvis_cli.analysis.synthetic_recordings_single
    options:
      heading_level: 4


```
usage:
flyvis synthetic-recordings [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY [options]
       or
synthetic_recordings_single.py [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY [options]

This script generates and stores various types of synthetic responses for a given network, such as flash responses, moving edge responses, and impulse responses. The responses are automatically cached for later use in analysis.

options:
  -h, --help            show this help message and exit
  --validation_subdir VALIDATION_SUBDIR
  --loss_file_name LOSS_FILE_NAME
  --batch_size BATCH_SIZE
  --delete_recordings
  --functions FUNCTIONS [FUNCTIONS ...]
                        List of functions to run.

Hybrid Arguments:
  --task_name TASK_NAME
                        task_name=value: Name of the task (e.g., 'flow',
                        'depth') (Required)
  --ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID
                        ensemble_and_network_id=value: ID in the format
                        XXXX/YYY (ensemble/network) (Required)

Examples:
--------
1. Generate all default synthetic recordings:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000

2. Generate only specific response types:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000 \
       --functions spatial_impulses_responses central_impulses_responses

3. Generate with custom batch size and clear existing recordings:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000 \
       --batch_size 16 --delete_recordings

```
