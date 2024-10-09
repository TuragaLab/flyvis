# Notebook Help

Script: `scripts/scripts/notebook.py`

```
usage: notebook.py [-h] [--notebook_path NOTEBOOK_PATH]
                   [--output_path OUTPUT_PATH] [--dry]

Run a Jupyter notebook using papermill. Required arguments depend on the
specific notebook. Pass any additional arguments as key:type=value triplets.

optional arguments:
  -h, --help            show this help message and exit
  --notebook_path NOTEBOOK_PATH
                        Path of the notebook to execute, e.g.
                        /path/to/__main__.ipynb.
  --output_path OUTPUT_PATH
                        Path for the output notebook. If not provided, a
                        temporary file will be used.
  --dry                 Perform a dry run without actually executing the
                        notebook.

```
