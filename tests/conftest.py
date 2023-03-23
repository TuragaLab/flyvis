import pytest
import numpy as np

from flyvision import connectome_file
from flyvision.connectome import ConnectomeDir


@pytest.fixture(scope="session")
def connectome(tmp_path_factory):
    return ConnectomeDir(
        tmp_path_factory.mktemp("tmp") / "test",
        dict(file=connectome_file, extent=1, n_syn_fill=1),
    )


@pytest.fixture(scope="session")
def sequence_path(tmp_path_factory):
    sequences = np.random.rand(20, 10, 64, 64)
    sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    path = tmp_path_factory.mktemp("tmp") / "sequences.npy"
    np.save(path, sequences)
    return str(path)
