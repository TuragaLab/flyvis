# Launch Ensemble Analysis on Compute Cloud


::: flyvis_cli.analysis.analysis
    options:
      heading_level: 4


```
usage:
flyvis analysis [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [ensemble_analysis_script_options...]
       or
analysis.py [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [ensemble_analysis_script_options...]

For a full list of options and default arguments, run: flyvis ensemble-analysis --help

Run ensemble analysis on the compute cloud. Launches a single job to analyze all models in the ensemble.

options:
  -h, --help            show this help message and exit
  --nP NP               Number of processors to use (default: 4).
  --gpu GPU             GPU configuration (default: 'num=1').
  --q Q                 Queue to submit the job to.
  --ensemble_id ENSEMBLE_ID
                        ID of the ensemble, e.g., 0045.
  --task_name TASK_NAME
                        Name given to the task, e.g., 'flow', 'depth', 'lum'.
  --ensemble_analysis_script ENSEMBLE_ANALYSIS_SCRIPT
                        Script to run for ensemble analysis. Default: /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/flyvis_cli/analysis/ensemble_analysis.py
  --dry                 Perform a dry run without actually launching jobs.

```
