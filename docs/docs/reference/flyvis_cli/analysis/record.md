# Launch Synthetic Recordings on Compute Cloud


::: flyvis_cli.analysis.record
    options:
      heading_level: 4


```
usage:
flyvis record [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [options] [synthetic_recordings_script_args...]
       or
record.py [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [options] [synthetic_recordings_script_args...]

For a full list of options and default arguments, run: flyvis synthetic-recordings-single --help

Run synthetic recordings for each model of an ensemble on the compute cloud.

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
  --synthetic_recordings_script SYNTHETIC_RECORDINGS_SCRIPT
                        Script to run for synthetic recordings. Default: /grou
                        ps/turaga/home/lappalainenj/FlyVis/private/flyvision/f
                        lyvis_cli/analysis/synthetic_recordings_single.py
  --dry                 Perform a dry run without actually launching jobs.

```
