# Synthetic Recordings Single

`scripts/analysis/synthetic_recordings_single.py`

```
usage: synthetic_recordings_single.py [-h] [--task_name TASK_NAME]
                                      [--ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID]
                                      [--chkpt CHKPT]
                                      [--validation_subdir VALIDATION_SUBDIR]
                                      [--loss_file_name LOSS_FILE_NAME]
                                      [--batch_size BATCH_SIZE]
                                      [--delete_recordings]
                                      [--functions FUNCTIONS [FUNCTIONS ...]]

Record synthetic responses.

optional arguments:
  -h, --help            show this help message and exit
  --chkpt CHKPT         checkpoint to evaluate.
  --validation_subdir VALIDATION_SUBDIR
  --loss_file_name LOSS_FILE_NAME
  --batch_size BATCH_SIZE
  --delete_recordings
  --functions FUNCTIONS [FUNCTIONS ...]
                        List of functions to run.

Hybrid Arguments:
  --task_name TASK_NAME
                        task_name=value: (Required)
  --ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID
                        ensemble_and_network_id=value: (Required)

Script for precomputing synthetic recordings for a single network.

Example Usage:
--------------
1. Generate all default synthetic recordings for a specific network:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
   ```

2. Generate only spatial and central impulse responses for a different network:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=9998/000    --functions spatial_impulses_responses central_impulses_responses
   ```

3. Generate default recordings with a custom batch size and delete existing recordings:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000    --batch_size 16 --delete_recordings
   ```

Available Functions:
--------------------
- flash_responses
- moving_edge_responses
- moving_edge_responses_currents
- moving_bar_responses
- naturalistic_stimuli_responses
- spatial_impulses_responses
- central_impulses_responses

```
