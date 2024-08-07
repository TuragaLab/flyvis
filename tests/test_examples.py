from pathlib import Path

import nbformat
import papermill as pm
import pytest

examples_path = Path(__file__).parent.parent / "examples"
# List of notebook names
notebooks = [
    "01_flyvision_connectome.ipynb",
    "02_flyvision_optic_flow_task.ipynb",
    "03_flyvision_flash_responses.ipynb",
    "04_flyvision_moving_edge_responses.ipynb",
    "05_flyvision_umap_and_clustering_models.ipynb",
    # "06_flyvision_tmy_predictions.ipynb",
    "07_flyvision_providing_custom_stimuli.ipynb",
]


# Define a parameterized test using pytest
@pytest.mark.parametrize("notebook", notebooks)
def test_notebook_execution(notebook, tmpdir):
    notebook_path = str(examples_path / notebook)
    output_notebook_path = tmpdir.join("output_" + notebook)

    # Run the notebook with papermill
    pm.execute_notebook(notebook_path, str(output_notebook_path))

    # Check if the notebook executed successfully
    with open(output_notebook_path, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Inspect the notebook's cells and outputs for specific validation
    for cell in nb.cells:
        if cell.cell_type == "code":
            for output in cell.outputs:
                if "ename" in output:  # error name
                    pytest.fail(f"Error found in cell:\n{output['evalue']}")
