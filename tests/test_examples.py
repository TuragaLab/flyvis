import os

os.environ["TESTING"] = "true"
from pathlib import Path

import nbformat
import papermill as pm
import pytest
import torch

pytestmark = pytest.mark.slow
examples_path = Path(__file__).parent.parent / "examples"
# List of notebook names
notebooks = [
    pytest.param("01_flyvision_connectome.ipynb", marks=[]),
    pytest.param(
        "02_flyvision_optic_flow_task.ipynb",
        marks=[
            pytest.mark.require_large_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "03_flyvision_flash_responses.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "04_flyvision_moving_edge_responses.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "05_flyvision_umap_and_clustering_models.ipynb",
        marks=[
            pytest.mark.require_large_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "06_flyvision_maximally_excitatory_stimuli.ipynb",
        marks=[
            pytest.mark.require_large_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "07_flyvision_providing_custom_stimuli.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "figure_01_fly_visual_system.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "figure_02_simple_stimuli_responses.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "figure_03_naturalistic_stimuli_responses.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
    pytest.param(
        "figure_04_mechanisms.ipynb",
        marks=[
            pytest.mark.require_download,
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
        ],
    ),
]


# Define a parameterized test using pytest
@pytest.mark.slow
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
