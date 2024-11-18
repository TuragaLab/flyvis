# Run Ensemble Analysis


::: flyvis_cli.analysis.ensemble_analysis
    options:
      heading_level: 4


```
usage:
flyvis ensemble-analysis [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [ensemble_analysis_script_options...]
       or
ensemble_analysis.py [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME [hydra_options...]

Example:
    Compute UMAP and clustering for the ensemble 0000:
    flyvis ensemble-analysis
        task_name=flow
        ensemble_and_network_id=0000/000
        --functions umap_and_clustering_main

Analysis for ensemble.

options:
  -h, --help            show this help message and exit
  --validation_subdir VALIDATION_SUBDIR
  --loss_file_name LOSS_FILE_NAME
  --functions {umap_and_clustering_main} [{umap_and_clustering_main} ...]
                        List of functions to run.
  --delete_umap_and_clustering

Hybrid Arguments:
  --task_name TASK_NAME
                        task_name=value:  (Required)
  --ensemble_id ENSEMBLE_ID
                        ensemble_id=value:  (Required)

```
