from os import PathLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import urllib
import pytest
import requests
from pathlib import Path
import torch
import numpy as np
from datamate import Directory

import flyvision
from flyvision import connectome_file
from flyvision import ConnectomeDir, ConnectomeView
from flyvision.datasets.base import SequenceDataset
from flyvision.ensemble import EnsembleView
from flyvision.utils.activity_utils import LayerActivity


class RenderedData(Directory):
    class Config(dict):
        path: PathLike

    def __init__(self, config: Config):
        sequences = np.load(self.config.path)
        receptors = flyvision.rendering.BoxEye()

        rendered_sequences = []
        for sequence in sequences:
            rendered_sequences.append(receptors(sequence[None]).cpu().numpy())
            # break to only create one sequence
            break

        rendered_sequences = np.concatenate(rendered_sequences)
        self.sequences = rendered_sequences


class CustomStimuli(SequenceDataset):
    dt = 1 / 50
    framerate = 24
    t_pre = 0.5
    t_post = 0.5
    n_sequences = None
    augment = False

    def __init__(self, raw_data_path, rendered_path="auto"):
        if rendered_path == "auto":
            self.dir = RenderedData(dict(path=raw_data_path))
        else:
            self.dir = RenderedData(rendered_path, dict(path=raw_data_path))
        self.sequences = torch.Tensor(self.dir.sequences[:])
        self.n_sequences = self.sequences.shape[0]

    def get_item(self, key):
        sequence = self.sequences[key]
        resample = self.get_temporal_sample_indices(
            sequence.shape[0], sequence.shape[0]
        )
        return sequence[resample]


@pytest.fixture(scope="module")
def rendered_dir(tmpdir_factory, sequence_path):
    return RenderedData(
        Path(tmpdir_factory.mktemp("tmp")) / "rendered", dict(path=sequence_path)
    )


@pytest.fixture(scope="module")
def custom_dataset(sequence_path, rendered_dir):
    return CustomStimuli(sequence_path, rendered_dir.path)


def test_connectome(connectome):
    assert isinstance(connectome, Directory)
    assert connectome.path.name == "test"


def test_connectome_view(connectome):
    connectome_view = ConnectomeView(connectome)
    assert isinstance(connectome_view, ConnectomeView)

    fig = connectome_view.connectivity_matrix("n_syn")
    fig.show()
    assert len(fig.axes) == 2
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig = connectome_view.receptive_fields_grid("T4c")
    fig.show()
    assert (
        len([ax for ax in fig.axes if ax.get_visible()])
        == len(connectome_view.receptive_fields_df("T4c"))
        == len(connectome_view.sources_list("T4c"))
    )
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig = connectome_view.projective_fields_grid("T4c")
    fig.show()
    assert (
        len([ax for ax in fig.axes if ax.get_visible()])
        == len(connectome_view.projective_fields_df("T4c"))
        == len(connectome_view.targets_list("T4c"))
    )
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig = connectome_view.network_layout()
    fig.show()
    assert len(fig.axes) == len(connectome.unique_cell_types) + 2
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_moving_mnist_url():
    moving_mnist_url = (
        "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    )

    def url_exists(url):
        try:
            response = requests.head(url)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.ConnectionError as e:
            return e

    assert url_exists(moving_mnist_url) == True


def test_imshow(sequence_path):
    sequences = np.load(sequence_path)
    example_sequence_ids = np.random.randint(0, high=sequences.shape[0], size=10)
    animation = flyvision.animations.Imshow(sequences, cmap=plt.cm.binary_r)
    animation.animate_in_notebook(samples=example_sequence_ids, frames=[0])


def test_boxeye(sequence_path):
    receptors = flyvision.rendering.BoxEye()
    fig = receptors.illustrate()
    fig.show()
    assert isinstance(fig, Figure)
    plt.close(fig)

    sequences = np.load(sequence_path)

    single_frame = sequences[0, 0]
    single_frame = torch.Tensor(single_frame)
    single_frame = single_frame[None, None]
    rendered = receptors(single_frame)

    assert rendered.shape == (1, 1, 1, 721)


def test_rendering(rendered_dir):
    rendered_sequences = rendered_dir.sequences[:]

    # just checking the visualzation runs
    animation = flyvision.animations.HexScatter(rendered_sequences, vmin=0, vmax=1)
    animation.animate_in_notebook(samples="all", frames=[0])


def test_sequence_dataset(custom_dataset):
    assert custom_dataset[0].shape == (42, 1, 721)


def test_model_responses(custom_dataset):
    ensemble = EnsembleView(flyvision.results_dir / "opticflow/000")

    movie_input = custom_dataset[0]

    responses = np.array(list(ensemble.simulate(movie_input[None], custom_dataset.dt)))

    assert responses.shape == (50, 1, 42, 45669)

    voltages = LayerActivity(responses, ensemble[0].connectome, keepref=True)

    cell_type = "T4c"
    assert voltages[cell_type].shape == (50, 1, 42, 721)

    central_voltages = voltages.central

    assert central_voltages[cell_type].shape == (50, 1, 42)


def test_cluster():
    from datamate import namespacify

    cell_type = "T4c"
    ensemble = EnsembleView(flyvision.results_dir / "opticflow/000")

    cluster_indices = ensemble.cluster_indices(cell_type)

    assert namespacify(cluster_indices) == namespacify(
        {
            0: np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    6,
                    7,
                    9,
                    11,
                    12,
                    13,
                    14,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    27,
                    29,
                    30,
                    31,
                    35,
                    36,
                    37,
                    42,
                    44,
                    47,
                    48,
                ]
            ),
            1: np.array([4, 5, 26, 33, 38, 40, 43]),
            2: np.array([8, 10, 15, 25, 28, 32, 34, 39, 41, 45, 46, 49]),
        }
    )
