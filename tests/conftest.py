import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flyvis import connectome_file
from flyvis.connectome import ConnectomeFromAvgFilters

SEED = int(os.environ.get("FLYVIS_TEST_SEED", 0))
"""Seed used to make the test suite deterministic.

Override with `FLYVIS_TEST_SEED` to re-run the suite under a different seed. An
assertion that only holds for the default seed is a broken assertion, so sweeping
the seed is how to find one.
"""


@pytest.fixture(autouse=True)
def deterministic_rng():
    """Reset the global random number generators before every test.

    Augmentations draw from the global numpy stream, so without this a test's
    outcome depends on how many draws the tests before it happened to consume, and
    changes from run to run. Tests whose assertions only hold for most draws then
    fail sporadically, which in CI is indistinguishable from a real regression.

    Note:
        This makes a test reproducible, it does not make an assertion that only
        holds for most draws correct. Assert on what the code guarantees, or set
        the random parameters explicitly, rather than relying on a lucky seed.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture(scope="session")
def connectome(tmp_path_factory):
    return ConnectomeFromAvgFilters(
        tmp_path_factory.mktemp("tmp") / "test",
        dict(file=connectome_file.name, extent=1, n_syn_fill=1),
    )


@pytest.fixture(scope="session")
def sequence_path(tmp_path_factory):
    # session fixtures are built before the per-test seeding above runs, so they
    # seed their own generator to keep the mock data identical across runs
    rng = np.random.default_rng(SEED)
    sequences = rng.random((20, 10, 64, 64))
    sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    path = tmp_path_factory.mktemp("tmp") / "sequences.npy"
    np.save(path, sequences)
    return str(path)


@pytest.fixture(scope="session")
def mock_sintel_data():
    """Create a minimal mock Sintel dataset structure with original dimensions."""
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Original dimensions
        HEIGHT, WIDTH = 436, 1024

        for seq_name in ["alley_1", "alley_2"]:
            # Create directory structure
            (tmp_path / "training/final" / seq_name).mkdir(parents=True)
            (tmp_path / "training/flow" / seq_name).mkdir(parents=True)
            (tmp_path / "training/depth" / seq_name).mkdir(parents=True)

            # Create dummy files with original dimensions
            for i in range(5):
                # Luminance (final) - (436, 1024)
                img = (rng.uniform(0, 1, (HEIGHT, WIDTH)) * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    tmp_path / f"training/final/{seq_name}/frame_{i:04d}.png"
                )

                # Flow - (2, 436, 1024)
                with open(
                    tmp_path / f"training/flow/{seq_name}/frame_{i:04d}.flo", 'wb'
                ) as f:
                    # Write header
                    np.array([202021.25], dtype=np.float32).tofile(f)  # Magic number
                    np.array([WIDTH, HEIGHT], dtype=np.int32).tofile(f)  # Dimensions
                    # Write flow data
                    rng.standard_normal((HEIGHT, WIDTH, 2)).astype(np.float32).tofile(f)

                # Depth - (436, 1024)
                with open(
                    tmp_path / f"training/depth/{seq_name}/frame_{i:04d}.dpt", 'wb'
                ) as f:
                    # Write header
                    np.array([1, WIDTH, HEIGHT], dtype=np.int32).tofile(f)  # Dimensions
                    # Write depth data
                    rng.standard_normal((HEIGHT, WIDTH)).astype(np.float32).tofile(f)

        yield tmp_path
