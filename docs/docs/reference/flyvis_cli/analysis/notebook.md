# Run Notebook


::: flyvis_cli.analysis.notebook
    options:
      heading_level: 4


```
usage:
flyvis notebook [-h] --notebook_path PATH [options] [hybrid_args...]
       or
notebook.py [-h] --notebook_path PATH [options] [hybrid_args...]

There are three modes of operation:
1. Basic: Only requires --notebook_path
2. Per-model: Requires --notebook_per_model_path, --notebook_per_model=true, --ensemble_and_network_id
3. Per-ensemble: Requires --notebook_path, --per_ensemble=true, --task_name, --ensemble_id

Run a Jupyter notebook using papermill. Required arguments depend on the specific notebook. Pass any additional arguments as key:type=value triplets.

options:
  -h, --help            show this help message and exit
  --notebook_path NOTEBOOK_PATH
                        Path of the notebook to execute, e.g.
                        /path/to/__main__.ipynb.
  --notebook_per_model_path NOTEBOOK_PER_MODEL_PATH
                        Path of the notebook to execute for each model, e.g.
                        /path/to/__main_per_model__.ipynb.
  --output_path OUTPUT_PATH
                        Path for the output notebook. If not provided, a
                        temporary file will be used and deleted after
                        execution.
  --dry                 Perform a dry run without actually executing the
                        notebook.

Hybrid Arguments:
  --notebook_per_model NOTEBOOK_PER_MODEL
                        notebook_per_model=value: Flag to set the output path
                        to the per-model notebook path. Requires
                        ensemble_and_network_id. (type: bool)
  --ensemble_and_network_id ENSEMBLE_AND_NETWORK_ID
                        ensemble_and_network_id=value: Id in form of
                        task_name/ensemble_id/network_id, e.g., flow/0000/000.
                        Required if notebook_per_model is True. (type: str)
  --per_ensemble PER_ENSEMBLE
                        per_ensemble=value: Flag to set the output path to the
                        per-ensemble notebook path. Requires ensemble_id and
                        task_name. (type: bool)
  --task_name TASK_NAME
                        task_name=value: Name of the task, e.g. flow. Required
                        if per_ensemble is True. (type: str)
  --ensemble_id ENSEMBLE_ID
                        ensemble_id=value: Id of the ensemble, e.g. 0045.
                        Required if per_ensemble is True. (type: int)

        Examples:
            # Basic usage
            flyvis notebook --notebook_path notebooks/analysis.ipynb
            # Per-model analysis
            flyvis notebook --notebook_per_model_path notebooks/per_model.ipynb \
                            notebook_per_model=true \
                            ensemble_and_network_id=flow/0045/000
            # Per-ensemble analysis
            flyvis notebook --notebook_path notebooks/ensemble.ipynb \
                            per_ensemble=true \
                            task_name=flow \
                            ensemble_id=45
            # With additional parameters passed to the notebook
            flyvis notebook --notebook_path notebooks/analysis.ipynb \
                            param1:str=value1 \
                            param2:int=42


```
