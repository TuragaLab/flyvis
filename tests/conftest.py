import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flyvis import connectome_file
from flyvis.connectome import ConnectomeFromAvgFilters


@pytest.fixture(scope="session")
def connectome(tmp_path_factory):
    return ConnectomeFromAvgFilters(
        tmp_path_factory.mktemp("tmp") / "test",
        dict(file=connectome_file.name, extent=1, n_syn_fill=1),
    )


@pytest.fixture(scope="session")
def sequence_path(tmp_path_factory):
    sequences = np.random.rand(20, 10, 64, 64)
    sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    path = tmp_path_factory.mktemp("tmp") / "sequences.npy"
    np.save(path, sequences)
    return str(path)


@pytest.fixture(scope="session")
def mock_sintel_data():
    """Create a minimal mock Sintel dataset structure with original dimensions."""
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
                img = (np.random.uniform(0, 1, (HEIGHT, WIDTH)) * 255).astype(np.uint8)
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
                    np.random.randn(HEIGHT, WIDTH, 2).astype(np.float32).tofile(f)

                # Depth - (436, 1024)
                with open(
                    tmp_path / f"training/depth/{seq_name}/frame_{i:04d}.dpt", 'wb'
                ) as f:
                    # Write header
                    np.array([1, WIDTH, HEIGHT], dtype=np.int32).tofile(f)  # Dimensions
                    # Write depth data
                    np.random.randn(HEIGHT, WIDTH).astype(np.float32).tofile(f)

        yield tmp_path
