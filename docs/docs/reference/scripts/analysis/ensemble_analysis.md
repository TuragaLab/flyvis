# Run Ensemble Analysis

`analysis/ensemble_analysis.py`

```
usage: ensemble_analysis.py [-h] [--task_name TASK_NAME]
                            [--ensemble_id ENSEMBLE_ID] [--chkpt CHKPT]
                            [--validation_subdir VALIDATION_SUBDIR]
                            [--loss_file_name LOSS_FILE_NAME]
                            [--functions {umap_and_clustering_main} [{umap_and_clustering_main} ...]]
                            [--delete_umap_and_clustering]

Analysis for ensemble.

optional arguments:
  -h, --help            show this help message and exit
  --chkpt CHKPT         checkpoint to evaluate.
  --validation_subdir VALIDATION_SUBDIR
  --loss_file_name LOSS_FILE_NAME
  --functions {umap_and_clustering_main} [{umap_and_clustering_main} ...]
                        List of functions to run.
  --delete_umap_and_clustering

Hybrid Arguments:
  --task_name TASK_NAME
                        task_name=value: (Required)
  --ensemble_id ENSEMBLE_ID
                        ensemble_id=value: (Required)

Store analysis results of an ensemble. Example: Compute UMAP and clustering
for the ensemble 0000: ```bash python ensemble_analysis.py task_name=flow
ensemble_and_network_id=0000/000 --functions umap_and_clustering_main ```

```
